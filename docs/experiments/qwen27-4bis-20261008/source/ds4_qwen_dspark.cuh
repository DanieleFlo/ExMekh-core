/* Qwen DSpark GQA primitives. Independently implemented from the pinned
 * checkpoint's equations; target Qwen attention kernels stay unchanged. */
#include <new>
namespace qwen_dspark {
constexpr unsigned H = 32, KV = 8, D = 128, PART = 1024, TILE = 8192;

struct workspace {
    unsigned context, rows, partitions;
    __half *key, *value;
    float *partials;
    uint64_t uploaded = 0;
};

__global__ static void features_kernel(float *dst, const float *src, unsigned rows, unsigned slot) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)rows * 5120u)
        dst[(i / 5120u) * 25600u + slot * 5120u + i % 5120u] = src[i];
}

/* Learned head RMSNorm followed by half-rotation YaRN. All heads use 128
 * dimensions. inv_freq and attention scaling follow the checkpoint config. */
__global__ static void norm_rope_kernel(float *x, const float *weight,
        unsigned heads, unsigned position) {
    const unsigned d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
    float *p = x + ((uint64_t)row * heads + head) * D;
    __shared__ float input[D], square[D];
    input[d] = p[d]; square[d] = p[d] * p[d];
    __syncthreads();
    for (unsigned stride = D / 2; stride; stride >>= 1) {
        if (d < stride) square[d] += square[d + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(square[0] / D + 1e-6f);
    const unsigned pair = d & (D / 2 - 1);
    /* At distant positions fast-math powf/angle rounding exceeds the head
     * normalization error. Compute only the rotary phase in float64. */
    const double base = pow(1e7, (double)(2 * pair) / D);
    const double low = floor(D * log(8192.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(1e7)));
    const double high = ceil(D * log(8192.0 / (2.0 * M_PI)) / (2.0 * log(1e7)));
    const double ramp = fmin(1.0, fmax(0.0, ((double)pair - low) / (high - low)));
    const double frequency = (1.0 - ramp) / base + ramp / (32.0 * base);
    double sine64, cosine64; sincos((position + row) * frequency, &sine64, &cosine64);
    const float sine = (float)sine64, cosine = (float)cosine64;
    const float factor = 1.0f + 0.1f * logf(32.0f);
    const unsigned other = d ^ (D / 2);
    const float a = input[d] * inv * weight[d];
    const float b = input[other] * inv * weight[other];
    p[d] = factor * (a * cosine + (d < D / 2 ? -b : b) * sine);
}

/* Four Q heads share one KV load. Each partition has its own FP32 online
 * softmax; the final kernel merges partitions in increasing logical order.
 * Noise rows are fully visible: DSpark's draft block is bidirectional. */
__global__ static void partial_kernel(float *partials, const float *q,
        const __half *key, const __half *value, const float *noise_k,
        const float *noise_v, unsigned context, unsigned rows,
        unsigned origin, unsigned first_partition, unsigned partitions) {
    const unsigned kvh = blockIdx.x, partition = first_partition + blockIdx.y;
    const unsigned row = blockIdx.z, d = threadIdx.x;
    const unsigned begin = partition * PART, end = min(begin + PART, context + rows);
    float m[4], denominator[4] = {}, accum[4] = {};
    for (unsigned h = 0; h < 4; ++h) m[h] = -INFINITY;
    __shared__ float reduction[4][D];
    for (unsigned cell = begin; cell < end; ++cell) {
        const bool noise = cell >= context;
        const uint64_t i = ((uint64_t)(noise ? cell-context : cell-origin) * KV + kvh) * D + d;
        const float k = noise ? noise_k[i] : __half2float(key[i]);
        const float v = noise ? noise_v[i] : __half2float(value[i]);
        for (unsigned h = 0; h < 4; ++h)
            reduction[h][d] = q[((uint64_t)row * H + kvh * 4 + h) * D + d] * k;
        __syncthreads();
        for (unsigned stride = D / 2; stride; stride >>= 1) {
            if (d < stride) for (unsigned h = 0; h < 4; ++h)
                reduction[h][d] += reduction[h][d + stride];
            __syncthreads();
        }
        for (unsigned h = 0; h < 4; ++h) {
            const float score = reduction[h][0] * 0.08838834764831845f;
            const float next = fmaxf(m[h], score);
            const float old_scale = expf(m[h] - next), new_scale = expf(score - next);
            denominator[h] = denominator[h] * old_scale + new_scale;
            accum[h] = accum[h] * old_scale + v * new_scale;
            m[h] = next;
        }
        __syncthreads();
    }
    for (unsigned h = 0; h < 4; ++h) {
        float *p = partials + (((uint64_t)row * H + kvh * 4 + h) * partitions + partition) * (D + 2);
        if (!d) { p[0] = m[h]; p[1] = denominator[h]; }
        p[d + 2] = accum[h];
    }
}

__global__ static void merge_kernel(float *out, const float *partials, unsigned partitions) {
    const unsigned head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    const float *p = partials + ((uint64_t)row * H + head) * partitions * (D + 2);
    float m = -INFINITY;
    for (unsigned i = 0; i < partitions; ++i) m = fmaxf(m, p[(uint64_t)i * (D + 2)]);
    float value = 0, denominator = 0;
    for (unsigned i = 0; i < partitions; ++i) {
        const float *s = p + (uint64_t)i * (D + 2);
        const float scale = s[1] > 0 ? expf(s[0] - m) : 0;
        denominator += s[1] * scale; value += s[d + 2] * scale;
    }
    out[((uint64_t)row * H + head) * D + d] = value / denominator;
}
}

extern "C" int ds4_gpu_qwen_dspark_features_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *in, unsigned rows, unsigned slot) {
    if (!out || !in || slot >= 5 || !rows || in->bytes < (uint64_t)rows * 5120 * 4 ||
        out->bytes < (uint64_t)rows * 25600 * 4) return 0;
    qwen_dspark::features_kernel<<<(rows * 5120u + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)in->ptr, rows, slot);
    return cuda_ok(cudaGetLastError(), "Qwen DSpark target taps");
}

extern "C" int ds4_gpu_qwen_dspark_norm_rope_tensor(ds4_gpu_tensor *x,
        const void *model, uint64_t bytes, uint64_t offset, unsigned heads,
        unsigned rows, unsigned position) {
    if (!x || !model || (heads != 32 && heads != 8) || !rows ||
        x->bytes < (uint64_t)rows * heads * 128 * 4 || offset > bytes || bytes-offset < 128*4) return 0;
    const float *w = (const float *)cuda_resolve_weight_ptr(model, offset, 128*4, 0, "Qwen DSpark head norm");
    if (!w) return 0;
    qwen_dspark::norm_rope_kernel<<<dim3(rows, heads), 128>>>((float *)x->ptr, w, heads, position);
    return cuda_ok(cudaGetLastError(), "Qwen DSpark YaRN");
}

extern "C" void *ds4_gpu_qwen_dspark_workspace_alloc(unsigned context, unsigned rows) {
    if (!context || context > 262144 || !rows || rows > 16 || g_n_gpus != 1) return NULL;
    auto *w = new (std::nothrow) qwen_dspark::workspace{};
    if (!w) return NULL;
    w->context = context; w->rows = rows;
    w->partitions = (context + rows + qwen_dspark::PART-1) / qwen_dspark::PART;
    const uint64_t kv_bytes = (uint64_t)qwen_dspark::TILE * 1024 * sizeof(__half);
    if (!cuda_ok(cudaMalloc(&w->key, kv_bytes), "Qwen DSpark K tile") ||
        !cuda_ok(cudaMalloc(&w->value, kv_bytes), "Qwen DSpark V tile") ||
        !cuda_ok(cudaMalloc(&w->partials, (uint64_t)rows * 32 * w->partitions * 130 * 4), "Qwen DSpark partials")) {
        cudaFree(w->key); cudaFree(w->value); cudaFree(w->partials); delete w; return NULL;
    }
    return w;
}

extern "C" void ds4_gpu_qwen_dspark_workspace_free(void *workspace) {
    auto *w = (qwen_dspark::workspace *)workspace; if (!w) return;
    cudaDeviceSynchronize();
    fprintf(stderr, "ds4: Qwen DSpark draft KV H2D=%.3f GiB tile=%.3f MiB\n",
        (double)w->uploaded / 1073741824.0, (double)(2ull * qwen_dspark::TILE * 1024 * 2) / 1048576.0);
    cudaFree(w->key); cudaFree(w->value); cudaFree(w->partials); delete w;
}

extern "C" int ds4_gpu_qwen_dspark_attention_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *noise_k, const ds4_gpu_tensor *noise_v,
        const uint16_t *key, const uint16_t *value, unsigned context, unsigned rows, void *workspace) {
    auto *w = (qwen_dspark::workspace *)workspace;
    if (!w || !out || !q || !noise_k || !noise_v || !key || !value ||
        context > w->context || !rows || rows > w->rows ||
        out->bytes < (uint64_t)rows * 4096*4 || q->bytes < (uint64_t)rows * 4096*4 ||
        noise_k->bytes < (uint64_t)rows * 1024*4 || noise_v->bytes < (uint64_t)rows * 1024*4) return 0;
    const unsigned partitions = (context + rows + qwen_dspark::PART-1) / qwen_dspark::PART;
    /* Include a final tile holding only noise when context is tile-aligned. */
    for (unsigned origin = 0; origin < context + rows; origin += qwen_dspark::TILE) {
        const unsigned cells = origin < context ? std::min(qwen_dspark::TILE, context-origin) : 0;
        if (cells) {
            const uint64_t bytes = (uint64_t)cells * 1024 * 2;
            if (!qwen4_cuda::transfer_upload(w->key, key + (uint64_t)origin * 1024, bytes, 0) ||
                !qwen4_cuda::transfer_upload(w->value, value + (uint64_t)origin * 1024, bytes, 0)) return 0;
            w->uploaded += bytes * 2;
        }
        const unsigned first = origin / qwen_dspark::PART;
        const unsigned count = std::min(qwen_dspark::TILE / qwen_dspark::PART, partitions-first);
        qwen_dspark::partial_kernel<<<dim3(8, count, rows), 128>>>(w->partials, (const float *)q->ptr,
            w->key, w->value, (const float *)noise_k->ptr, (const float *)noise_v->ptr,
            context, rows, origin, first, partitions);
        if (!cuda_ok(cudaGetLastError(), "Qwen DSpark attention partition")) return 0;
    }
    qwen_dspark::merge_kernel<<<dim3(32, rows), 128>>>((float *)out->ptr, w->partials, partitions);
    return cuda_ok(cudaGetLastError(), "Qwen DSpark attention merge");
}
