/* Qwen3.8 Flash Next. Included by ds4_cuda.cu so model residency, streams
 * and temporary allocations have the same lifetime as the other CUDA paths. */

#include "ds4_qwen4_vision.h"
#include "ds4_qwen4_expert_cache.h"
#include "ds4_read_queue.h"

namespace qwen4_cuda {

__device__ __forceinline__ float sum(float x);
__device__ __forceinline__ float sigmoid(float x);
template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid = NULL, const uint8_t *signs = NULL);
__device__ float injection(const float *inj, unsigned hc, unsigned stream);

struct rope_args { float freq[32], scale; unsigned nrot; };
static float rope_freq[32];
static float rope_scale = 1;
static bool rope_set;

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = rope_set ? rope_scale : 1;
    for (unsigned i = 0; i < nrot / 2; i++)
        r.freq[i] = rope_set ? rope_freq[i] : powf(base, -2.0f * i / nrot);
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

__global__ void attn_prep(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp) {
    const unsigned slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        for (unsigned i = lane; i < Di; i += 32) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            kc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            vc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

__global__ void block_key(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, float eps, rope_args rp) {
    const unsigned b = block0 + blockIdx.x, lane = threadIdx.x, npt = D / 32;
    __shared__ float row[128];
    float ss = 0, v[4];
    for (unsigned i = 0; i < npt; i++) {
        float a = 0;
        for (unsigned t = 0; t < ratio; t++) a += ik[((uint64_t)b * ratio + t) * D + lane * npt + i];
        v[i] = a / ratio;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) row[lane * npt + i] = v[i] * inv * gamma[lane * npt + i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)b * ratio * 4, rp);
    for (unsigned i = lane; i < D; i += 32) out[(uint64_t)b * D + i] = __float2half_rn(row[i]);
}

__global__ void idx_score(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio) {
    const unsigned b = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, lane = threadIdx.x & 31;
    if (b >= N) return;
    float score = 0;
    if (b >= (pos0 + t + 1) / ratio) score = -3e38f;
    else for (unsigned h = 0; h < H; h++) {
        float a = 0;
        for (unsigned i = lane; i < D; i += 32)
            a += q[((uint64_t)t * H + h) * D + i] * __half2float(key[(uint64_t)b * D + i]);
        score += fmaxf(sum(a), 0);
    }
    if (!lane) out[(uint64_t)t * N + b] = score;
}

__global__ void tile_max(unsigned *out, const float *scores, unsigned N, unsigned tiles) {
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (tile >= tiles) return;
    unsigned v = 0;
    for (unsigned i = tile * 8; i < min(N, tile * 8 + 8); i++)
        v = max(v, __float_as_uint(fmaxf(scores[(uint64_t)t * N + i], 0)));
    out[(uint64_t)t * tiles + tile] = v;
}

/* Exact radix threshold and stable gather, with the same greater-than then
 * equal-score ordering as Metal. No context-dependent candidate truncation. */
__global__ void idx_select(int *out, const float *score, unsigned N, unsigned K) {
    const unsigned tid = threadIdx.x, t = blockIdx.x;
    __shared__ unsigned hist[256], gt[256], eq[256], threshold, need;
    const float *row = score + (uint64_t)t * N;
    if (!tid) { threshold = 0; need = K; }
    __syncthreads();
    for (unsigned pass = 0; pass < 4; pass++) {
        const unsigned shift = 24 - 8 * pass, mask = pass ? (0xffffffffu << (shift + 8)) : 0;
        hist[tid] = 0;
        __syncthreads();
        for (unsigned i = tid; i < N; i += 256) {
            const unsigned key = __float_as_uint(fmaxf(row[i], 0));
            if ((key & mask) == threshold) atomicAdd(hist + ((key >> shift) & 255), 1u);
        }
        __syncthreads();
        if (!tid) for (int d = 255; d >= 0; d--) {
            if (hist[d] >= need) { threshold |= (unsigned)d << shift; break; }
            need -= hist[d];
        }
        __syncthreads();
    }
    const unsigned chunk = (N + 255) / 256, begin = min(N, tid * chunk), end = min(N, begin + chunk);
    unsigned ng = 0, ne = 0;
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        ng += key > threshold; ne += key == threshold;
    }
    gt[tid] = ng; eq[tid] = ne;
    __syncthreads();
    if (!tid) {
        unsigned pg = 0, pe = 0;
        for (unsigned i = 0; i < 256; i++) {
            const unsigned g = gt[i], e = eq[i];
            gt[i] = pg; eq[i] = pe; pg += g; pe += e;
        }
    }
    __syncthreads();
    unsigned g = gt[tid], e = eq[tid];
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        if (key > threshold) out[(uint64_t)t * K + g++] = i;
        else if (key == threshold) { if (e < need) out[(uint64_t)t * K + K - need + e] = i; e++; }
    }
}

__global__ void idx_expand(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride) {
    const unsigned t = blockIdx.x, pos = pos0 + t, tail = (pos + 1) / ratio * ratio;
    for (unsigned i = threadIdx.x; i < K * ratio; i += blockDim.x)
        out[(uint64_t)t * stride + i] = blocks[(uint64_t)t * K + i / ratio] * ratio + i % ratio;
    for (unsigned i = tail + threadIdx.x; i <= pos; i += blockDim.x)
        out[(uint64_t)t * stride + K * ratio + i - tail] = i;
    if (!threadIdx.x) count[t] = K * ratio + pos + 1 - tail;
}

template<unsigned D>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale) {
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    float qv[D / 32], acc[D / 32] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < D / 32; i++) qv[i] = q[((uint64_t)t * H + h) * D + lane + 32 * i] * scale;
    for (unsigned j = split * per; j < min(n, (split + 1) * per); j++) {
        const unsigned p = sparse ? (unsigned)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        float score = 0;
        for (unsigned i = 0; i < D / 32; i++) score += qv[i] * __half2float(kc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        score = sum(score);
        const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
        denom = denom * correction + w;
        for (unsigned i = 0; i < D / 32; i++) acc[i] = acc[i] * correction + w * __half2float(vc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        m = nm;
    }
    if (splits == 1) {
        for (unsigned i = 0; i < D / 32; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + lane + 32 * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < D / 32; i++) dst[2 + lane + 32 * i] = acc[i];
    }
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

/* The heads in a KV group share the same selected keys. Keep their output
 * accumulators in registers and use tensor cores for both products. Scaled
 * residual components retain the fine part of the FP32 queries/probabilities;
 * K and V are already half, so neither requires further rounding. */
__global__ void attention_group(float *out, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
#if __CUDA_ARCH__ >= 800
    const unsigned D = 256, tid = threadIdx.x, lane = tid&31, warp = tid/32;
    const unsigned t = blockIdx.y, kh = blockIdx.x, group = H/Hkv;
    const unsigned qr = tid/16, col = tid%16, h = kh*group+qr;
    const unsigned n = sparse ? counts[t] : pos0+t+1;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16)
        if (qr < group) mx = fmaxf(mx,fabsf(q[((uint64_t)t*H+h)*D+d]*scale));
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,16));
    const int exp = mx > 0 ? max(-120,min(120,(int)((__float_as_uint(mx)>>23)&255)-127)) : 0;
    const float inv = ldexpf(1,-exp);
    if (!col) { qs[qr] = ldexpf(1,exp); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float v = qr < group ? q[((uint64_t)t*H+h)*D+d]*scale*inv : 0;
        qh[qr][d] = __float2half_rn(v);
        ql[qr][d] = __float2half_rn((v-__half2float(qh[qr][d]))*4096);
    }
    float result[4][4] = {};
    __syncthreads();
    for (unsigned j0 = 0; j0 < n; j0 += 32) {
        if (tid < 32) {
            const unsigned j = j0+tid;
            const unsigned p = j < n ? (sparse ? (unsigned)sel[(uint64_t)t*stride+j] : j) : UINT_MAX;
            positions[tid] = p <= pos0+t ? p : UINT_MAX;
        }
        __syncthreads();
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],kc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp/4, key0 = (warp%4)*8;
        for (unsigned k = split*128; k < (split+1)*128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            tt_ldmatrix_x4(ah,&qh[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x4(al,&ql[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x2(b,&kv[key0+lane%8][k+((lane%16)/8)*8]);
            tt_mma_m16n8k16_f16_f32(hi,ah,b);
            tt_mma_m16n8k16_f16_f32(lo,al,b);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++)
            scores.part[split][tt_mma_c_i(lane,i)][key0+tt_mma_c_j(lane,i)] = hi[i]+lo[i]*0x1p-12f;
        __syncthreads();
        float prob[2], peak = max_score[qr];
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col+i*16;
            prob[i] = positions[key] != UINT_MAX ?
                (scores.part[0][qr][key]+scores.part[1][qr][key])*qs[qr] : -3e38f;
            peak = fmaxf(peak,prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off,16));
        const float old = expf(max_score[qr]-peak);
        float total = 0;
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col+i*16] != UINT_MAX ? expf(prob[i]-peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff,total,off,16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr] = peak;
            denom[qr] = denom[qr]*old+total;
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col+i*16] = p;
            scores.prob[1][qr][col+i*16] = __float2half_rn((prob[i]-__half2float(p))*4096);
        }
        for (unsigned i = tid*8; i < 32*D; i += 256*8) {
            const unsigned r = i/D, d = i%D, p = positions[r];
            tt_cp_async_16B(&kv[r][d],vc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        #pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi[4] = {}, lo[4] = {};
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                tt_ldmatrix_x4(ah,&scores.prob[0][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x4(al,&scores.prob[1][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x2_trans(b,&kv[k+lane%16][warp*32+tile*8]);
                tt_mma_m16n8k16_f16_f32(hi,ah,b);
                tt_mma_m16n8k16_f16_f32(lo,al,b);
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; i++) result[tile][i] =
                result[tile][i]*correction[tt_mma_c_i(lane,i)]+(hi[i]+lo[i]*0x1p-12f);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = tt_mma_c_i(lane,i), d = warp*32+tile*8+tt_mma_c_j(lane,i);
            if (r < group) {
                const uint64_t dst = ((uint64_t)t*H+kh*group+r)*D+d;
                out[dst] = (denom[r] > 0 ? result[tile][i]/denom[r] : 0)*sigmoid(gate[dst]);
            }
        }
    }
#endif
}

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 2: return n % 32 ? 0 : n / 32 * 18;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    case 39: return n % 32 ? 0 : n / 32 * 17;
    case 10: return n % 256 ? 0 : n / 256 * 84;
    case 12: return n % 256 ? 0 : n / 256 * 144;
    case 16: return n % 256 ? 0 : n / 256 * 66;
    default: return 0;
    }
}

static bool tiered_cache(const void *map) {
    return g_qwen4_tiered_cache &&
           map == g_model_host_base && g_n_gpus == 1;
}

static const char *expert_weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes);
static uint64_t dense_reserved;
/* Primitive tests have no model directory; real graphs supply exact sizes. */
static uint64_t dense_required = 4ull * 1073741824ull;
static uint64_t expert_min_bytes, expert_max_bytes;
static uint32_t cache_ctx, cache_prefill;
static bool cache_budget_logged;

static const char *mapped_weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    return cuda_resolve_weight_ptr(map, off, bytes, 0, "Qwen weights");
}

static const char *weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    /* Registration makes host weights addressable, not resident in VRAM.
     * Test dense residency independently of routed-expert placement; copy
     * the original GGUF bytes through the existing bounded CUDA arena. */
    if ((tiered_cache(map) || ds4_env_flag_enabled("DS4_CUDA_QWEN4_DENSE_CACHE", 0)) && g_n_gpus <= 1 &&
        cuda_model_cache_limit_bytes() != UINT64_MAX &&
        map == g_model_host_base && (g_model_registered || tiered_cache(map)) && g_model_fd >= 0) {
        auto exact = g_model_range_by_offset.find(off);
        if (exact == g_model_range_by_offset.end()) {
            const char *cached = cuda_model_range_ptr_from_fd(map, off, bytes, "Qwen dense weights");
            auto installed = g_model_range_by_offset.find(off);
            if (!cached || installed == g_model_range_by_offset.end() ||
                !g_model_ranges[installed->second].arena_allocated ||
                g_model_ranges[installed->second].host_base != map ||
                g_model_ranges[installed->second].bytes < bytes) {
                size_t free_bytes = 0, total_bytes = 0;
                (void)cudaMemGetInfo(&free_bytes, &total_bytes);
                fprintf(stderr, "ds4: Qwen dense admission failed bytes=%llu cached=%.2f/%.2f GiB fixed=%.2f/%.2f GiB free=%.2f GiB\n",
                        (unsigned long long)bytes, (double)g_model_range_bytes / 1073741824.0,
                        (double)cuda_model_cache_limit_bytes() / 1073741824.0,
                        (double)dense_reserved / 1073741824.0, (double)dense_required / 1073741824.0,
                        (double)free_bytes / 1073741824.0);
                return NULL;
            }
            dense_reserved += bytes;
        }
    }
    if (tiered_cache(map)) {
        auto hit = g_model_range_by_offset.find(off);
        if (hit == g_model_range_by_offset.end() || g_model_ranges[hit->second].bytes < bytes ||
            !g_model_ranges[hit->second].arena_allocated) {
            fprintf(stderr, "ds4: Qwen dense VRAM budget exhausted; refusing an unpinned host pointer\n");
            return NULL;
        }
        return g_model_ranges[hit->second].device_ptr;
    }
    return mapped_weight(map, size, off, bytes);
}

static const char *expert_weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    /* In tiered mode this address is only a range descriptor. Every routed
     * compute path below supplies staged device bytes before launching. */
    return tiered_cache(map) ? (const char *)map + off : mapped_weight(map, size, off, bytes);
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* These readers preserve the GGUF values, including padded Q2_K down rows.
 * Templates remove unused formats from each matrix kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i,
        const uint64_t *grid_table = NULL, const uint8_t *sign_table = NULL) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return __bfloat162float(((const __nv_bfloat16 *)row)[i]);
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    if (TYPE == 2) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return __half2float(*(const __half *)b) *
            (float)((int)((b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15) - 8);
    }
    if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 17;
        const unsigned q = (b[1 + i % 16] >> (4 * (i % 32 / 16))) & 15;
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0] << 23);
        return (q & 8 ? -levels[q & 7] : levels[q & 7]) * scale;
    }
    if (TYPE == 10) {
        const cuda_block_q2_K *b = (const cuda_block_q2_K *)row + i / 256;
        const unsigned j = i % 256, sc = b->scales[j / 16];
        const unsigned q = (b->qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        return dev_f16_to_f32(b->d) * (sc & 15) * q - dev_f16_to_f32(b->dmin) * (sc >> 4);
    }
    if (TYPE == 12) {
        const cuda_block_q4_K *b = (const cuda_block_q4_K *)row + i / 256;
        const unsigned j = i % 256, group = j / 32;
        unsigned sc, mn;
        if (group < 4) { sc = b->scales[group] & 63; mn = b->scales[group + 4] & 63; }
        else {
            sc = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
            mn = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        }
        const unsigned q = (b->qs[j / 64 * 32 + j % 32] >> (4 * (group & 1))) & 15;
        return dev_f16_to_f32(b->d) * sc * q - dev_f16_to_f32(b->dmin) * mn;
    }
    if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row + i / 256;
        const unsigned j = i % 256, group = j / 32, sub = j % 32 / 8;
        const uint16_t *p = b->qs + group * 4;
        const unsigned grid_ids = (unsigned)p[0] | ((unsigned)p[1] << 16);
        const unsigned signs_scale = (unsigned)p[2] | ((unsigned)p[3] << 16);
        const unsigned gi = (grid_ids >> (8 * sub)) & 255, si = (signs_scale >> (7 * sub)) & 127;
        const uint64_t grid = grid_table ? grid_table[gi] : cuda_iq2xxs_grid[gi];
        const unsigned signs = sign_table ? sign_table[si] : cuda_ksigns_iq2xs[si];
        float v = (float)((grid >> (8 * (j & 7))) & 255);
        if (signs & (1u << (j & 7))) v = -v;
        return dev_f16_to_f32(b->d) * (.5f + (signs_scale >> 28)) * .25f * v;
    }
    return 0;
}

__device__ __forceinline__ float scalar(const char *row, unsigned i, unsigned type) {
    switch (type) {
    case 0: return value<0>(row, i);
    case 1: return value<1>(row, i);
    case 2: return value<2>(row, i);
    case 8: return value<8>(row, i);
    case 30: return value<30>(row, i);
    case 39: return value<39>(row, i);
    default: return 0;
    }
}

template<unsigned TYPE>
__device__ __forceinline__ float4 value4(const char *row, unsigned i,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float4 out;
    float *v = (float *)&out;
    if (TYPE == 1) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
        out = make_float4(a.x,a.y,b.x,b.y);
    } else if (TYPE == 0) {
        out = *(const float4 *)(row+i*4);
    } else if (TYPE == 8) {
        const char *b = row+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        out = make_float4((float)(int8_t)q[0]*scale,(float)(int8_t)(q[0]>>8)*scale,
                         (float)(int8_t)q[1]*scale,(float)(int8_t)(q[1]>>8)*scale);
    } else if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row+(i/32)*17;
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0]<<23);
        unsigned packed;
        memcpy(&packed,b+1+i%16,4);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned q = (packed>>(k*8+(i%32/16)*4))&15, mag = q&7;
            const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            v[k] = (q&8 ? -level : level)*scale;
        }
    } else if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row+i/256;
        const unsigned j = i%256, sub = (j%32)/8;
        const uint16_t *p = b->qs+(j/32)*4;
        const unsigned ids = (unsigned)p[0]|((unsigned)p[1]<<16);
        const unsigned ss = (unsigned)p[2]|((unsigned)p[3]<<16);
        const uint64_t grid = grid_table[(ids>>(8*sub))&255];
        const unsigned signs = sign_table[(ss>>(7*sub))&127];
        const float scale = dev_f16_to_f32(b->d)*(.5f+(ss>>28))*.25f;
        /* Apply four signs before conversion. IQ2's nonzero magnitudes keep
         * the packed negations from carrying into neighboring bytes. */
        const unsigned offset = j&7;
        const unsigned bits = (((signs>>offset)&15)*0x00204081u)&0x01010101u;
        const unsigned packed = (((unsigned)(grid>>(8*offset)))^(bits*255u))+bits;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            v[k] = scale*(float)(int8_t)(packed>>(8*k));
        }
    } else if (TYPE == 10 || TYPE == 12) {
        const unsigned j = i%256;
        unsigned sc, mn, qs, shift;
        float d, dm;
        if (TYPE == 10) {
            const cuda_block_q2_K *b = (const cuda_block_q2_K *)row+i/256;
            const unsigned s = b->scales[j/16];
            sc = s&15; mn = s>>4;
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/128)*32+j%32);
            shift = 2*((j%128)/32);
        } else {
            const cuda_block_q4_K *b = (const cuda_block_q4_K *)row+i/256;
            const unsigned group = j/32;
            if (group < 4) { sc = b->scales[group]&63; mn = b->scales[group+4]&63; }
            else {
                sc = (b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
                mn = (b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
            }
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/64)*32+j%32);
            shift = 4*(group&1);
        }
        const float scale = d*sc, offset = dm*mn;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = scale*((qs>>(8*k+shift))&(TYPE == 10 ? 3 : 15))-offset;
    } else {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = value<TYPE>(row,i+k,grid_table,sign_table);
    }
    return out;
}

static uint64_t expert_row_bytes(unsigned type, unsigned K) {
    if (type == 10 && K % 32) return 0;
    return row_bytes(type, type == 10 ? ((uint64_t)K + 255) / 256 * 256 : K);
}

__global__ void router(int *selected, float *weights, const float *logits,
        const float *x, const char *gate, float *shared_gate,
        unsigned NE, unsigned NS, unsigned K, unsigned type) {
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    __shared__ float p[512], maxima[256], red[32];
    __shared__ unsigned ids[256];
    float mx = -FLT_MAX;
    for (unsigned e = tid; e < NE; e += 256) mx = fmaxf(mx, logits[(uint64_t)t * NE + e]);
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    mx = maxima[0];
    float ps = 0;
    for (unsigned e = tid; e < NE; e += 256) { p[e] = expf(logits[(uint64_t)t * NE + e] - mx); ps += p[e]; }
    const float total = block_sum(ps, red);
    for (unsigned e = tid; e < NE; e += 256) p[e] /= total;
    if (K) {
        float v = 0;
        for (unsigned i = tid; i < K; i += 256) v += scalar(gate, i, type) * x[(uint64_t)t * K + i];
        v = block_sum(v, red);
        if (!tid) shared_gate[t] = v;
    }
    __syncthreads();
    for (unsigned s = 0; s < NS; s++) {
        float best = -1;
        unsigned id = UINT_MAX;
        for (unsigned e = tid; e < NE; e += 256) if (p[e] > best) { best = p[e]; id = e; }
        maxima[tid] = best; ids[tid] = id;
        __syncthreads();
        for (unsigned stride = 128; stride; stride /= 2) {
            if (tid < stride && (maxima[tid + stride] > maxima[tid] ||
                (maxima[tid + stride] == maxima[tid] && ids[tid + stride] < ids[tid]))) {
                maxima[tid] = maxima[tid + stride]; ids[tid] = ids[tid + stride];
            }
            __syncthreads();
        }
        if (!tid) {
            selected[(uint64_t)t * NS + s] = ids[0];
            weights[(uint64_t)t * NS + s] = maxima[0];
            p[ids[0]] = -1;
        }
        __syncthreads();
    }
    if (tid < NS) {
        float denom = 0;
        for (unsigned s = 0; s < NS; s++) denom += weights[(uint64_t)t * NS + s];
        red[tid] = denom;
    }
    __syncthreads();
    if (tid < NS) weights[(uint64_t)t * NS + tid] /= red[tid];
}

template<unsigned TYPE, bool DOWN>
__global__ void moe_mv(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned NE, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb, const char *const *routed0, const char *const *routed1) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    if (row >= M) return;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    if (shared) {
        for (unsigned i = threadIdx.x & 31; i < K; i += 32) {
            a += scalar(sh0 + row * srb, i, shared_type) * xt[i];
            if (!DOWN) b += scalar(sh1 + row * srb, i, shared_type) * xt[i];
        }
        a = sum(a); b = sum(b);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t route = (uint64_t)t * NS + slot;
            const char *expert0 = routed0 ? routed0[route] : w0 + (uint64_t)e * M * rb;
            const char *expert1 = routed1 ? routed1[route] : (w1 ? w1 + (uint64_t)e * M * rb : NULL);
            const uint64_t off = (uint64_t)row * rb;
            if ((TYPE == 16 || TYPE == 10 || TYPE == 12 || TYPE == 39) && !((uintptr_t)xt&15)) {
                for (unsigned i = (threadIdx.x&31)*4; i < K; i += 128) {
                    const float4 xv = *(const float4 *)(xt+i);
                    const float4 av = value4<TYPE>(expert0+off,i,grid_table,sign_table);
                    a += av.x*xv.x; a += av.y*xv.y; a += av.z*xv.z; a += av.w*xv.w;
                    if (!DOWN) {
                        const float4 bv = value4<TYPE>(expert1+off,i,grid_table,sign_table);
                        b += bv.x*xv.x; b += bv.y*xv.y; b += bv.z*xv.z; b += bv.w*xv.w;
                    }
                }
                a = sum(a); b = sum(b);
            } else {
                a = dot<TYPE>(expert0+off,xt,K,grid_table,sign_table);
                if (!DOWN) b = dot<TYPE>(expert1+off,xt,K,grid_table,sign_table);
            }
        }
    }
    if (!(threadIdx.x & 31)) out[pair * M + row] = DOWN ? a : silu(a) * b;
}

/* The CUDA arena owns storage. This table owns only expert replacement
 * metadata; dense tensors are never candidates. Model-map release resets it. */
static std::vector<ds4_qwen4_expert_slot> expert_slots;
static std::unordered_map<uint64_t, size_t> expert_slot_by_offset;
static std::unordered_map<uint64_t, uint64_t> expert_frequency;
static uint64_t expert_epoch, expert_reserved, expert_hits, expert_misses, expert_evictions;
static std::vector<ds4_qwen4_expert_slot> ram_slots;
static std::vector<void *> ram_raw; /* slot starts, owned by ram_slabs */
static std::vector<void *> ram_slabs;
static uint64_t ram_allocated, ram_slab_used, ram_slab_capacity, ram_budget;
static bool ram_budget_set;
static std::vector<uint64_t> ram_bytes;
static std::unordered_map<uint64_t, size_t> ram_slot_by_offset;
static uint64_t ram_reserved, ram_hits, ram_misses, ram_evictions, expert_disk_bytes, expert_demotions, expert_h2d_bytes;
static FILE *expert_trace;
static bool expert_trace_opened;
/* The list builder gives a new generation even when a session reuses the same
 * tensor addresses. Read T*NS selected IDs once; gate/up/down share the plan. */
static const int *route_selected, *route_lists, *route_counts;
static unsigned route_T, route_NS, route_NE, route_cap;
static int *route_host;
static uint64_t route_host_capacity, route_readback_bytes, route_readbacks;
static bool route_ready;
static cudaStream_t expert_copy_stream;
static cudaEvent_t expert_copy_ready;
static cudaEvent_t expert_group_consumed[2];
static int *expert_group_counts;
static char *expert_group_host;
static uint64_t expert_group_capacity, expert_group_counts_capacity;
static uint64_t expert_copy_groups;
static double expert_read_seconds;
static std::vector<size_t> gpu_victims, ram_gpu_victims, ram_other_victims;
static uint64_t gpu_victim_epoch = UINT64_MAX, ram_victim_epoch = UINT64_MAX;
static uint64_t gpu_victim_bytes, ram_victim_bytes;
static double victim_cpu_seconds;

/* Exact-router read-ahead. Only the inference thread reserves/publishes RAM
 * slots. Workers own duplicated descriptors and write reserved, epoch-pinned
 * storage; they do not touch CUDA, cache maps, ranking, or counters. */
struct expert_read_request {
    ds4_read_task task;
    uint64_t offset, bytes, capacity;
    size_t slot;
    char *destination;
    const char *payload;
    double elapsed;
    int error;
    bool active;
};
static ds4_read_queue expert_reads;
static expert_read_request expert_read_requests[DS4_READ_QUEUE_DEPTH];
static int expert_read_fd[DS4_READ_QUEUE_WORKERS], expert_read_direct_fd[DS4_READ_QUEUE_WORKERS];
static uint64_t expert_read_align, expert_read_file_size;
static std::vector<uint64_t> expert_read_offsets;
static size_t expert_read_cursor;
static uint64_t expert_read_bytes, expert_async_reads;
static double expert_read_wait_seconds;
static unsigned expert_read_workers, expert_read_active;

static void expert_read_run(void *, unsigned worker, void *payload) {
    auto *r = (expert_read_request *)payload;
    const double start = cuda_wall_sec();
    errno = 0;
    const int ok = cuda_model_stage_read_from(expert_read_fd[worker], &expert_read_direct_fd[worker],
        expert_read_align, expert_read_file_size, r->destination, r->capacity,
        r->offset, r->bytes, &r->payload);
    r->error = ok ? 0 : (errno ? errno : EIO);
    r->elapsed = cuda_wall_sec() - start;
}

static int expert_read_publish(expert_read_request *r) {
    const double start = cuda_wall_sec();
    ds4_read_queue_wait(&expert_reads, &r->task);
    expert_read_wait_seconds += cuda_wall_sec() - start;
    expert_read_seconds += r->elapsed;
    r->active = false; expert_read_active--;
    if (r->error) {
        ram_slot_by_offset.erase(r->offset);
        fprintf(stderr, "ds4: Qwen async expert read failed offset=%llu: %s\n",
                (unsigned long long)r->offset, strerror(r->error));
        return 0;
    }
    ram_slots[r->slot].device_ptr = (char *)r->payload;
    expert_disk_bytes += r->bytes; expert_async_reads++;
    return 1;
}

static int expert_reads_finish(void) {
    int ok = 1;
    for (auto &r : expert_read_requests) if (r.active && !expert_read_publish(&r)) ok = 0;
    expert_read_offsets.clear(); expert_read_cursor = 0;
    return ok;
}

static void expert_reads_reset(void) {
    if (ds4_read_queue_workers(&expert_reads)) {
        (void)expert_reads_finish();
        const unsigned workers = ds4_read_queue_workers(&expert_reads);
        ds4_read_queue_stop(&expert_reads);
        for (unsigned i = 0; i < workers; i++) {
            close(expert_read_fd[i]);
            if (expert_read_direct_fd[i] >= 0) close(expert_read_direct_fd[i]);
        }
    }
}

static int expert_reads_start(void) {
    if (ds4_read_queue_workers(&expert_reads)) return 1;
    if (g_model_fd < 0 || ds4_env_flag_enabled("DS4_CUDA_QWEN4_SYNC_READS", 0)) return 0;
    unsigned workers = 4;
    const char *env = getenv("DS4_CUDA_QWEN4_IO_WORKERS");
    if (env && *env) {
        char *end = NULL; const unsigned long n = strtoul(env, &end, 10);
        if (end != env && !*end && n >= 1 && n <= DS4_READ_QUEUE_WORKERS) workers = (unsigned)n;
    }
    for (unsigned i = 0; i < workers; i++) {
        expert_read_fd[i] = dup(g_model_fd);
        expert_read_direct_fd[i] = g_model_direct_fd >= 0 ? dup(g_model_direct_fd) : -1;
        if (expert_read_fd[i] < 0) {
            for (unsigned j = 0; j <= i; j++) {
                if (expert_read_fd[j] >= 0) close(expert_read_fd[j]);
                if (expert_read_direct_fd[j] >= 0) close(expert_read_direct_fd[j]);
            }
            return 0;
        }
    }
    expert_read_align = g_model_direct_align; expert_read_file_size = g_model_file_size;
    if (ds4_read_queue_start(&expert_reads, workers, expert_read_run, NULL)) {
        expert_read_workers = workers; return 1;
    }
    for (unsigned i = 0; i < workers; i++) {
        close(expert_read_fd[i]);
        if (expert_read_direct_fd[i] >= 0) close(expert_read_direct_fd[i]);
    }
    return 0; /* thread/descriptor pressure preserves the synchronous path */
}


/* Ranking is immutable during one projection: observe and pin all current
 * hits precede admission. New/reused slots are pinned and cannot be victims.
 * Build once instead of rescanning tens of thousands of slots per miss. */
static size_t expert_gpu_victim(uint64_t bytes) {
    const double start = cuda_wall_sec();
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_LINEAR_VICTIMS", 0)) {
        const size_t i = ds4_qwen4_expert_victim(expert_slots.data(), expert_slots.size(), bytes, expert_epoch);
        victim_cpu_seconds += cuda_wall_sec() - start;
        return i;
    }
    auto worse = [](size_t a, size_t b) {
        return ds4_qwen4_expert_rank_before(expert_slots.data(), b, a);
    };
    if (gpu_victim_epoch != expert_epoch || gpu_victim_bytes != bytes) {
        gpu_victims.clear();
        for (size_t i = 0; i < expert_slots.size(); i++)
            if (expert_slots[i].capacity >= bytes && expert_slots[i].pinned_epoch != expert_epoch)
                gpu_victims.push_back(i);
        std::make_heap(gpu_victims.begin(), gpu_victims.end(), worse);
        gpu_victim_epoch = expert_epoch; gpu_victim_bytes = bytes;
    }
    while (!gpu_victims.empty() && expert_slots[gpu_victims.front()].pinned_epoch == expert_epoch) {
        std::pop_heap(gpu_victims.begin(), gpu_victims.end(), worse); gpu_victims.pop_back();
    }
    const size_t i = gpu_victims.empty() ? SIZE_MAX : gpu_victims.front();
    victim_cpu_seconds += cuda_wall_sec() - start;
    return i; /* a low-frequency admission may reject it without consuming it */
}

static size_t expert_ram_victim(uint64_t capacity) {
    const double start = cuda_wall_sec();
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_LINEAR_VICTIMS", 0)) {
        size_t index = SIZE_MAX;
        for (unsigned pass = 0; pass < 2 && index == SIZE_MAX; pass++)
            for (size_t i = 0; i < ram_slots.size(); i++) {
                const auto &s = ram_slots[i];
                if (s.capacity < capacity || s.pinned_epoch == expert_epoch ||
                    (!pass && !expert_slot_by_offset.count(s.offset))) continue;
                if (index == SIZE_MAX || ds4_qwen4_expert_rank_before(ram_slots.data(), i, index)) index = i;
            }
        victim_cpu_seconds += cuda_wall_sec() - start;
        return index;
    }
    auto worse = [](size_t a, size_t b) {
        return ds4_qwen4_expert_rank_before(ram_slots.data(), b, a);
    };
    if (ram_victim_epoch != expert_epoch || ram_victim_bytes != capacity) {
        ram_gpu_victims.clear(); ram_other_victims.clear();
        for (size_t i = 0; i < ram_slots.size(); i++) {
            const auto &s = ram_slots[i];
            if (s.capacity < capacity || s.pinned_epoch == expert_epoch) continue;
            auto &v = expert_slot_by_offset.count(s.offset) ? ram_gpu_victims : ram_other_victims;
            v.push_back(i);
        }
        std::make_heap(ram_gpu_victims.begin(), ram_gpu_victims.end(), worse);
        std::make_heap(ram_other_victims.begin(), ram_other_victims.end(), worse);
        ram_victim_epoch = expert_epoch; ram_victim_bytes = capacity;
    }
    /* Demoting a GPU victim can change its RAM preference during a projection.
     * Reclassify lazily before selecting; the frequency/recency order remains. */
    for (unsigned pass = 0; pass < 2; pass++) {
        auto &v = pass ? ram_other_victims : ram_gpu_victims;
        while (!v.empty()) {
            const size_t i = v.front();
            std::pop_heap(v.begin(), v.end(), worse); v.pop_back();
            const auto &s = ram_slots[i];
            if (s.pinned_epoch == expert_epoch) continue;
            if (!pass && !expert_slot_by_offset.count(s.offset)) {
                ram_other_victims.push_back(i);
                std::push_heap(ram_other_victims.begin(), ram_other_victims.end(), worse);
                continue;
            }
            victim_cpu_seconds += cuda_wall_sec() - start;
            return i;
        }
    }
    victim_cpu_seconds += cuda_wall_sec() - start;
    return SIZE_MAX;
}


static void route_reset(void) {
    route_selected = route_lists = route_counts = NULL;
    route_ready = false;
}

static void route_invalidate(const void *ptr, uint64_t bytes) {
    const uintptr_t first = (uintptr_t)ptr;
    const uintptr_t selected = (uintptr_t)route_selected;
    const uintptr_t lists = (uintptr_t)route_lists, counts = (uintptr_t)route_counts;
    if ((selected >= first && selected - first < bytes) ||
        (lists >= first && lists - first < bytes) || (counts >= first && counts - first < bytes)) {
        route_reset();
    }
}

/* Metadata lives beyond an async launch. The next route readback drains all
 * preceding groups before this storage or a cache victim can be reused. */
static int expert_group_ensure(unsigned NE, unsigned groups) {
    const uint64_t count_bytes = (uint64_t)NE * groups * sizeof(int);
    const uint64_t pointer_offset = (count_bytes + sizeof(char *) - 1u) & ~(uint64_t)(sizeof(char *) - 1u);
    const uint64_t bytes = pointer_offset + (uint64_t)NE * 2u * sizeof(char *);
    if (!expert_copy_stream && !cuda_ok(cudaStreamCreateWithFlags(&expert_copy_stream,
                     cudaStreamNonBlocking), "Qwen expert copy stream")) return 0;
    if (!expert_copy_ready && !cuda_ok(cudaEventCreateWithFlags(&expert_copy_ready,
                     cudaEventDisableTiming), "Qwen expert copy ready event")) return 0;
    for (unsigned i = 0; i < 2; i++)
        if (!expert_group_consumed[i] && !cuda_ok(cudaEventCreateWithFlags(&expert_group_consumed[i],
                     cudaEventDisableTiming), "Qwen group consumed event")) return 0;
    if (bytes <= expert_group_capacity && count_bytes <= expert_group_counts_capacity) return 1;
    if (!cuda_ok(cudaDeviceSynchronize(), "Qwen group buffer resize")) return 0;
    if (expert_group_host) (void)cudaFreeHost(expert_group_host);
    if (expert_group_counts) (void)cudaFree(expert_group_counts);
    expert_group_host = NULL; expert_group_counts = NULL;
    expert_group_capacity = expert_group_counts_capacity = 0;
    if (!cuda_ok(cudaMallocHost((void **)&expert_group_host, bytes), "Qwen group host metadata") ||
        !cuda_ok(cudaMalloc((void **)&expert_group_counts, count_bytes), "Qwen group device counts")) return 0;
    expert_group_capacity = bytes; expert_group_counts_capacity = count_bytes;
    return 1;
}


static int route_read(const int *selected, unsigned T, unsigned NS, unsigned NE,
                      cudaStream_t stream) {
    const uint64_t n = (uint64_t)T * NS, bytes = (n + NE) * sizeof(int);
    if (bytes > route_host_capacity) {
        if (!cuda_ok(cudaStreamSynchronize(stream), "Qwen routing buffer resize")) return 0;
        if (route_host) { (void)cudaFreeHost(route_host); route_host = NULL; route_host_capacity = 0; }
        if (!cuda_ok(cudaMallocHost((void **)&route_host, bytes), "Qwen routing host buffer")) return 0;
        route_host_capacity = bytes;
    }
    if (!cuda_ok(cudaMemcpyAsync(route_host, selected, n * sizeof(int),
                      cudaMemcpyDeviceToHost, stream), "Qwen compact routing") ||
        !cuda_ok(cudaStreamSynchronize(stream), "Qwen compact routing ready")) return 0;
    int *uses = route_host + n;
    memset(uses, 0, NE * sizeof(int));
    for (uint64_t i = 0; i < n; i++) if (route_host[i] >= 0 && (unsigned)route_host[i] < NE) uses[route_host[i]]++;
    route_readback_bytes += n * sizeof(int); route_readbacks++;
    return 1;
}

struct expert_usage_window {
    unsigned experts, selected, cursor, rows;
    uint64_t bytes;
    std::vector<int> history;
    std::vector<unsigned> counts;
};
static std::unordered_map<uint64_t, expert_usage_window> expert_windows;


static void expert_cache_reset(void) {
    expert_reads_reset();
    if (!ram_slots.empty() || route_host || expert_copy_stream) (void)cudaDeviceSynchronize();
    if (expert_copy_ready) (void)cudaEventDestroy(expert_copy_ready);
    for (unsigned i = 0; i < 2; i++) {
        if (expert_group_consumed[i]) (void)cudaEventDestroy(expert_group_consumed[i]);
        expert_group_consumed[i] = NULL;
    }
    if (expert_copy_stream) (void)cudaStreamDestroy(expert_copy_stream);
    if (expert_group_host) (void)cudaFreeHost(expert_group_host);
    if (expert_group_counts) (void)cudaFree(expert_group_counts);
    expert_copy_stream = NULL; expert_copy_ready = NULL;
    expert_group_host = NULL; expert_group_counts = NULL;
    expert_group_capacity = expert_group_counts_capacity = 0;
    if (route_host) (void)cudaFreeHost(route_host);
    route_host = NULL; route_host_capacity = 0;
    route_reset();
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0))
        fprintf(stderr, "ds4: Qwen routing readbacks=%llu bytes=%llu copy_groups=%llu expert_read_ms=%.3f victim_select_ms=%.3f\n",
                (unsigned long long)route_readbacks, (unsigned long long)route_readback_bytes,
                (unsigned long long)expert_copy_groups, expert_read_seconds * 1000.0, victim_cpu_seconds * 1000.0);
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0))
        fprintf(stderr, "ds4: Qwen async reads=%llu host_wait_ms=%.3f workers=%u\n",
                (unsigned long long)expert_async_reads, expert_read_wait_seconds * 1000.0, expert_read_workers);
    expert_async_reads = 0; expert_read_wait_seconds = 0; expert_read_workers = 0;
    route_readbacks = route_readback_bytes = expert_copy_groups = 0; expert_read_seconds = victim_cpu_seconds = 0;
    gpu_victims.clear(); ram_gpu_victims.clear(); ram_other_victims.clear();
    gpu_victim_epoch = ram_victim_epoch = UINT64_MAX;
    if (expert_trace) fclose(expert_trace);
    expert_trace = NULL; expert_trace_opened = false;
    if (!ram_slots.empty() && (ds4_env_flag_enabled("DS4_CUDA_QWEN4_EXPERT_CACHE_VERBOSE", 0) || ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0)))
        fprintf(stderr, "ds4: Qwen RAM cache slots=%zu reserved=%llu allocated=%llu hits=%llu misses=%llu evictions=%llu disk_bytes=%llu demotions=%llu h2d_bytes=%llu direct_io=%d\n",
            ram_slots.size(), (unsigned long long)ram_reserved, (unsigned long long)ram_allocated, (unsigned long long)ram_hits,
            (unsigned long long)ram_misses, (unsigned long long)ram_evictions,
            (unsigned long long)expert_disk_bytes, (unsigned long long)expert_demotions, (unsigned long long)expert_h2d_bytes, g_model_direct_fd >= 0);
    for (void *raw : ram_slabs) (void)cudaFreeHost(raw);
    ram_slabs.clear(); ram_allocated = ram_slab_used = ram_slab_capacity = ram_budget = 0;
    ram_budget_set = false;
    ram_slots.clear(); ram_raw.clear(); ram_bytes.clear(); ram_slot_by_offset.clear();
    ram_reserved = ram_hits = ram_misses = ram_evictions = expert_disk_bytes = expert_demotions = expert_h2d_bytes = 0;
    if (!expert_slots.empty() && (ds4_env_flag_enabled("DS4_CUDA_QWEN4_EXPERT_CACHE_VERBOSE", 0) || ds4_env_flag_enabled("DS4_CUDA_QWEN4_CACHE_STATS", 0)))
        fprintf(stderr, "ds4: Qwen expert cache slots=%zu reserved=%llu hits=%llu misses=%llu evictions=%llu\n",
                expert_slots.size(), (unsigned long long)expert_reserved,
                (unsigned long long)expert_hits, (unsigned long long)expert_misses,
                (unsigned long long)expert_evictions);
    expert_slots.clear(); expert_slot_by_offset.clear(); expert_frequency.clear(); expert_windows.clear();
    expert_epoch = expert_reserved = expert_hits = expert_misses = expert_evictions = dense_reserved = 0;
    dense_required = 4ull * 1073741824ull;
    expert_min_bytes = expert_max_bytes = 0;
    cache_ctx = cache_prefill = 0;
    cache_budget_logged = false;
}

static bool expert_cache_enabled(const void *map) {
    return (tiered_cache(map) || ds4_env_flag_enabled("DS4_CUDA_QWEN4_EXPERT_CACHE", 0)) &&
           map == g_model_host_base && (g_model_registered || tiered_cache(map)) && g_n_gpus <= 1 &&
           cuda_model_cache_limit_bytes() != UINT64_MAX;
}

static void expert_cache_next_epoch(void) {
    if (++expert_epoch == 0) {
        expert_epoch = 1;
        for (auto &slot : expert_slots) slot.last_used = slot.pinned_epoch = 0;
        for (auto &slot : ram_slots) slot.last_used = slot.pinned_epoch = 0;
    }

}

static void expert_cache_touch(uint64_t off, uint64_t uses) {
    (void)uses;
    const uint64_t frequency = expert_frequency[off];
    auto ram = ram_slot_by_offset.find(off);
    if (ram != ram_slot_by_offset.end()) {
        auto &slot = ram_slots[ram->second];
        slot.frequency = frequency; slot.last_used = slot.pinned_epoch = expert_epoch;
    }
    auto hit = expert_slot_by_offset.find(off);
    if (hit != expert_slot_by_offset.end()) {
        auto &slot = expert_slots[hit->second];
        slot.frequency = frequency; slot.last_used = slot.pinned_epoch = expert_epoch;
    }
}

static void expert_cache_observe(uint64_t base, uint64_t bytes, unsigned NE,
        unsigned T, unsigned NS, const int *ids) {
    if (!expert_trace_opened) {
        expert_trace_opened = true;
        const char *path = getenv("DS4_CUDA_QWEN4_EXPERT_TRACE");
        if (path && *path) expert_trace = fopen(path, "w");
    }
    if (expert_trace) for (unsigned t = 0; t < T; t++) {
        fprintf(expert_trace, "%llu,%llu,%u,%u,%u", (unsigned long long)base,
                (unsigned long long)bytes, NE, NS, T);
        for (unsigned k = 0; k < NS; k++) fprintf(expert_trace, ",%d", ids[(uint64_t)t * NS + k]);
        fputc('\n', expert_trace);
    }
    auto &window = expert_windows[base];
    if (window.experts != NE || window.selected != NS || T >= 128) {
        window = {NE, NS, 0, 0, bytes, std::vector<int>(128u * NS, -1), std::vector<unsigned>(3u * NE, 0)};
    }
    const unsigned skip = T > 128 ? T - 128 : 0;
    ds4_qwen4_expert_observe(window.history.data(), window.counts.data(), NE, NS,
                            &window.cursor, &window.rows, T - skip, ids + (uint64_t)skip * NS);
    /* Measured pressure workloads favor recent LFU over 128 rows. The ranking
     * still adapts on every route; no older context contributes. Shorter and
     * change-detected windows remain diagnostic comparisons, not the default. */
    unsigned width = 128;
    const char *fixed = getenv("DS4_CUDA_QWEN4_EXPERT_WINDOW");
    if (fixed && !strcmp(fixed, "adaptive"))
        width = ds4_qwen4_expert_window(window.counts.data(), window.counts.data() + NE,
                                       NE, window.rows, NS);
    else if (fixed && (atoi(fixed) == 32 || atoi(fixed) == 64 || atoi(fixed) == 128)) width = (unsigned)atoi(fixed);
    const unsigned index = width == 32 ? 0 : width == 64 ? 1 : 2;
    const unsigned divisor = std::min(width, window.rows);
    const char *policy = getenv("DS4_CUDA_QWEN4_EXPERT_POLICY");
    const bool lru = policy && !strcmp(policy, "lru");
    for (unsigned e = 0; e < NE; e++) {
        const uint64_t off = base + (uint64_t)e * bytes;
        const uint64_t frequency = lru ? 0 : ((uint64_t)window.counts[index * NE + e] << 20) / divisor;
        expert_frequency[off] = frequency;
        auto hit = expert_slot_by_offset.find(off);
        if (hit != expert_slot_by_offset.end()) expert_slots[hit->second].frequency = frequency;
        auto ram = ram_slot_by_offset.find(off);
        if (ram != ram_slot_by_offset.end()) ram_slots[ram->second].frequency = frequency;
    }
}

/* RAM contains unchanged quantized bytes. GPU copies are cheap RAM victims:
 * prefer retaining the middle tier for experts currently absent from VRAM. */
static size_t expert_ram_slot(uint64_t off, uint64_t bytes) {
    auto hit = ram_slot_by_offset.find(off);
    if (hit != ram_slot_by_offset.end()) return hit->second;
    const uint64_t align = std::max<uint64_t>(g_model_direct_align, 1);
    const uint64_t capacity = bytes + 3 * align;
    if (!ram_budget_set) {
        ram_budget = 16ull * 1073741824ull; /* fallback when host memory is unknown */
        uint64_t available = 0;
        /* A fixed 16 GiB ceiling thrashes on Flash even when host RAM is free.
         * Allocate slabs lazily within the available-memory reserve, so smaller
         * hosts remain bounded without stranding RAM on larger machines. */
        if (ds4_linux_nonmovable_memory(&available))
            ram_budget = ds4_qwen4_ram_auto_budget(available, UINT64_MAX);
        const char *env = getenv("DS4_CUDA_QWEN4_RAM_CACHE_MB");
        if (env && *env && strcmp(env, "auto")) {
            char *end = NULL; const uint64_t mb = strtoull(env, &end, 10);
            if (end != env && !*end && *env != '-' && mb <= UINT64_MAX / 1048576ull)
                ram_budget = mb * 1048576ull;
            else fprintf(stderr, "ds4: invalid DS4_CUDA_QWEN4_RAM_CACHE_MB; using auto budget\n");
        }
        ram_budget_set = true;
        fprintf(stderr, "ds4: Qwen pinned RAM budget %.2f GiB (available %.2f GiB; 64 MiB slabs)\n",
                (double)ram_budget / 1073741824.0, (double)available / 1073741824.0);
    }
    void *raw = NULL;
    size_t index = SIZE_MAX;
    /* cudaMallocHost can synchronize the device. Amortize it over many slots;
     * charge whole slabs, including padding and the unused tail, to the budget. */
    const uint64_t allocation = (capacity + align - 1u) / align * align;
    if (allocation <= ram_slab_capacity - ram_slab_used) {
        raw = (char *)ram_slabs.back() + ram_slab_used;
        ram_slab_used += allocation;
    } else if (ram_allocated <= ram_budget && allocation <= ram_budget - ram_allocated) {
        const uint64_t slab = std::min<uint64_t>(std::max<uint64_t>(allocation, 64ull * 1048576ull), ram_budget - ram_allocated);
        void *storage = NULL;
        if (cudaMallocHost(&storage, slab) == cudaSuccess) {
            ram_slabs.push_back(storage); ram_allocated += slab;
            ram_slab_capacity = slab; ram_slab_used = allocation; raw = storage;
        } else (void)cudaGetLastError();
    }
    if (raw) {
        index = ram_slots.size();
        ram_slots.push_back({off, capacity, expert_frequency[off], expert_epoch, expert_epoch, NULL});
        ram_raw.push_back(raw); ram_bytes.push_back(bytes); ram_reserved += capacity;
    } else {
        index = expert_ram_victim(capacity);
        if (index == SIZE_MAX) return index;
        ram_slot_by_offset.erase(ram_slots[index].offset); ram_evictions++;
        ram_slots[index].offset = off; ram_bytes[index] = bytes;
    }
    auto &slot = ram_slots[index];
    slot.frequency = expert_frequency[off]; slot.last_used = slot.pinned_epoch = expert_epoch;
    slot.device_ptr = (char *)cuda_align_ptr(ram_raw[index], align);
    ram_slot_by_offset[off] = index;
    return index;
}

/* Keep at most sixteen reads reserved ahead, in the same projection/group
 * order as copies. RAM=0 or pinned pressure leaves an ordinary sync miss. */
static void expert_reads_pump(void) {
    for (auto &r : expert_read_requests) {
        if (r.active) continue;
        while (expert_read_cursor < expert_read_offsets.size()) {
            const uint64_t off = expert_read_offsets[expert_read_cursor++];
            if (expert_slot_by_offset.count(off) || ram_slot_by_offset.count(off)) continue;
            const size_t slot = expert_ram_slot(off, expert_read_bytes);
            if (slot == SIZE_MAX) continue;
            r.offset = off; r.bytes = expert_read_bytes; r.slot = slot;
            r.destination = ram_slots[slot].device_ptr;
            r.capacity = ram_slots[slot].capacity - g_model_direct_align;
            r.payload = NULL; r.elapsed = 0; r.error = 0;
            if (!ds4_read_queue_submit(&expert_reads, &r.task, &r)) {
                ram_slot_by_offset.erase(off); break;
            }
            r.active = true; expert_read_active++;
            break;
        }
    }
}

static void expert_reads_begin(uint64_t o0, uint64_t o1, uint64_t bytes,
                               unsigned NE, unsigned width, const int *uses, bool down) {
    if (!expert_reads_start()) return;
    expert_read_bytes = bytes; expert_read_cursor = 0;
    expert_read_offsets.clear();
    for (unsigned first = 0; first < NE; first += width)
        for (unsigned w = 0; w < (down ? 1u : 2u); w++)
            for (unsigned e = first; e < std::min(first + width, NE); e++)
                if (uses[e] > 0) expert_read_offsets.push_back((w ? o1 : o0) + e * bytes);
    expert_reads_pump();
}

static int expert_ram_source(uint64_t off, uint64_t bytes, const char **ptr) {
    *ptr = NULL;
    auto hit = ram_slot_by_offset.find(off);
    if (hit != ram_slot_by_offset.end()) {
        if (expert_read_active) for (auto &r : expert_read_requests) if (r.active && r.offset == off) {
            ram_misses++;
            if (!expert_read_publish(&r)) return 0;
            *ptr = ram_slots[r.slot].device_ptr;
            expert_reads_pump();
            return 1;
        }
        ram_hits++; *ptr = ram_slots[hit->second].device_ptr; return 1;
    }
    ram_misses++;
    const size_t index = expert_ram_slot(off, bytes);
    if (index == SIZE_MAX) return 1; /* bounded transient disk staging below */
    auto &slot = ram_slots[index];
    const char *payload = NULL;
    const double read_start = cuda_wall_sec();
    const int read_ok = cuda_model_stage_read(slot.device_ptr, slot.capacity - g_model_direct_align,
                                             off, bytes, &payload);
    expert_read_seconds += cuda_wall_sec() - read_start;
    if (!read_ok) {
        ram_slot_by_offset.erase(off);
        fprintf(stderr, "ds4: Qwen expert disk read failed offset=%llu: %s\n", (unsigned long long)off, strerror(errno));
        return 0;
    }
    slot.device_ptr = (char *)payload; expert_disk_bytes += bytes;
    *ptr = payload; return 1;
}

static int expert_transfer(const void *map, uint64_t off, uint64_t bytes,
                           char *dst, cudaStream_t stream) {
    expert_h2d_bytes += bytes;
    if (!tiered_cache(map)) return cuda_ok(cudaMemcpyAsync(dst, (const char *)map + off,
                                            bytes, cudaMemcpyHostToDevice, stream), "Qwen expert transfer");
    const char *src = NULL;
    if (!expert_ram_source(off, bytes, &src)) return 0;
    if (src) return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "Qwen RAM expert transfer");
    /* No RAM victim is safe (or RAM disabled). A synchronized transient source
     * handles the miss without dropping experts or stealing an active source. */
    if (!cuda_ok(cudaStreamSynchronize(stream), "Qwen transient source drain") ||
        !cuda_model_stage_pool_alloc(bytes + 3 * g_model_direct_align)) return 0;
    const double read_start = cuda_wall_sec();
    const int read_ok = cuda_model_stage_read(g_model_stage[0], g_model_stage_bytes - g_model_direct_align, off, bytes, &src);
    expert_read_seconds += cuda_wall_sec() - read_start;
    if (!read_ok) return 0;
    expert_disk_bytes += bytes;
    return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "Qwen disk expert transfer") &&
           cuda_ok(cudaStreamSynchronize(stream), "Qwen transient source ready");
}

static int expert_demote(const void *map, const ds4_qwen4_expert_slot &slot,
                         uint64_t bytes, cudaStream_t stream) {
    if (!tiered_cache(map) || ram_slot_by_offset.find(slot.offset) != ram_slot_by_offset.end()) return 1;
    const size_t index = expert_ram_slot(slot.offset, bytes);
    if (index == SIZE_MAX) return 1; /* capacity pressure leaves the expert on disk */
    expert_demotions++;
    return cuda_ok(cudaMemcpyAsync(ram_slots[index].device_ptr, slot.device_ptr, bytes,
                                  cudaMemcpyDeviceToHost, stream), "Qwen expert demotion") &&
           cuda_ok(cudaStreamSynchronize(stream), "Qwen expert demotion ready");
}

static int expert_cache_get(const void *map, uint64_t off, uint64_t bytes,
                            cudaStream_t stream, const char **ptr) {
    *ptr = NULL;
    auto hit = expert_slot_by_offset.find(off);
    if (hit != expert_slot_by_offset.end()) {
        *ptr = expert_slots[hit->second].device_ptr;
        expert_hits++;
        return 1;
    }
    expert_misses++;
    const uint64_t limit = cuda_model_cache_limit_bytes(), reserve = dense_required;
    uint64_t budget = limit > reserve ? limit - reserve : 0;
    const char *env = getenv("DS4_CUDA_QWEN4_EXPERT_CACHE_MB");
    if (env && env[0]) {
        char *end = NULL;
        const uint64_t mb = strtoull(env, &end, 10);
        if (end != env && mb && mb <= UINT64_MAX / 1048576ull)
            budget = std::min<uint64_t>(budget, mb * 1048576ull);
    }
    if (!cache_budget_logged && cache_ctx) {
        cache_budget_logged = true;
        /* The total-cache bound also charges auxiliary weights such as the
         * vision projector. Report the effective quota, not just the separate
         * expert bound, which could otherwise count that space twice. */
        const uint64_t fixed_cached = g_model_range_bytes >= expert_reserved ?
            g_model_range_bytes - expert_reserved : 0;
        const uint64_t available = std::min(budget, limit > fixed_cached ? limit - fixed_cached : 0);
        fprintf(stderr, "ds4: Qwen VRAM cache ctx=%u prefill=%u fixed_weights=%.2f GiB other_weights=%.2f GiB expert_budget=%.2f GiB complete_expert_capacity=%llu..%llu (gate/up/down; variable quant sizes)\n",
                cache_ctx, cache_prefill, (double)reserve / 1073741824.0,
                (double)(fixed_cached > dense_reserved ? fixed_cached - dense_reserved : 0) / 1073741824.0,
                (double)available / 1073741824.0,
                (unsigned long long)(expert_max_bytes ? available / expert_max_bytes : 0),
                (unsigned long long)(expert_min_bytes ? available / expert_min_bytes : 0));
    }
    char *dst = NULL;
    size_t slot_index = SIZE_MAX, range_index = SIZE_MAX;
    /* expert_reserved excludes dense tensors. Reserve them once in the
     * expert quota; charging the reserve again against total resident bytes
     * would strand several GiB even after all dense weights are loaded. */
    if (bytes <= budget && expert_reserved <= budget - bytes &&
        g_model_range_bytes <= limit && bytes <= limit - g_model_range_bytes) {
        /* Dense projections of later layers and MTP have not necessarily
         * been touched yet. Reserve their outstanding physical bytes as well
         * as logical quota, otherwise a full cold prefill can cache experts
         * before the dense projections fit. The reserve shrinks as they load. */
        dst = cuda_model_arena_alloc(bytes, "Qwen routed expert",
                                    reserve > dense_reserved ? reserve - dense_reserved : 0);
    }
    if (!dst) {
        slot_index = expert_gpu_victim(bytes);
        if (slot_index == SIZE_MAX) return 1; /* all eligible slots active: stage the miss */
        const auto &slot = expert_slots[slot_index];
        /* A single cold request is still executed from transient VRAM, but
         * cannot evict a more valuable recent working set. */
        if (expert_frequency[off] < slot.frequency) return 1;
        auto range = g_model_range_by_offset.find(slot.offset);
        if (range == g_model_range_by_offset.end()) return 0;
        range_index = range->second;
        const auto &r = g_model_ranges[range_index];
        if (r.host_base != map || r.device_ptr != slot.device_ptr || !r.arena_allocated) return 0;
        if (!expert_demote(map, slot, r.bytes, stream)) return 0;
        dst = slot.device_ptr;
    }
    if (!expert_transfer(map, off, bytes, dst, stream)) return 0;
    if (slot_index == SIZE_MAX) {
        range_index = g_model_ranges.size();
        g_model_ranges.push_back({map, off, bytes, dst, NULL, NULL, 0, 0, 1});
        g_model_range_bytes += bytes;
        expert_reserved += bytes;
        slot_index = expert_slots.size();
        expert_slots.push_back({off, bytes, expert_frequency[off], expert_epoch, expert_epoch, dst});
    } else {
        /* Remove the victim while its old ranking key is still intact.
         * Updating a pinned node inside the heap would break its ordering. */
        if (!gpu_victims.empty() && gpu_victims.front() == slot_index &&
            !ds4_env_flag_enabled("DS4_CUDA_QWEN4_LINEAR_VICTIMS", 0)) {
            auto worse = [](size_t a, size_t b) {
                return ds4_qwen4_expert_rank_before(expert_slots.data(), b, a);
            };
            std::pop_heap(gpu_victims.begin(), gpu_victims.end(), worse); gpu_victims.pop_back();
        }
        auto &slot = expert_slots[slot_index];
        if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_EXPERT_CACHE_VERBOSE", 0))
            fprintf(stderr, "ds4: Qwen expert eviction old=%llu new=%llu old_frequency=%llu bytes=%llu\n",
                    (unsigned long long)slot.offset, (unsigned long long)off,
                    (unsigned long long)slot.frequency, (unsigned long long)bytes);
        expert_slot_by_offset.erase(slot.offset);
        g_model_range_by_offset.erase(slot.offset);
        slot.offset = off; slot.frequency = expert_frequency[off];
        slot.last_used = slot.pinned_epoch = expert_epoch;
        auto &r = g_model_ranges[range_index];
        r.offset = off; r.bytes = bytes;
        /* Reserved capacity remains charged: shrinking a live range cannot
         * create imaginary free VRAM in the arena's budget. */
        expert_evictions++;
    }
    g_model_range_by_offset[off] = range_index;
    expert_slot_by_offset[off] = slot_index;
    *ptr = dst;
    return 1;
}

static uint64_t expert_state_bytes(void) {
    if (expert_windows.empty()) return 0;
    uint64_t n = 32 + 24ull * (expert_slots.size() + ram_slots.size());
    for (const auto &entry : expert_windows) n += 32 + 4ull * entry.second.rows * entry.second.selected;
    return n;
}

static void cache_put32(char **p, uint32_t v) {
    for (unsigned i = 0; i < 4; i++) *(*p)++ = (char)(v >> (8 * i));
}
static uint32_t cache_get32(const char **p) {
    uint32_t v = 0; for (unsigned i = 0; i < 4; i++) v |= (uint32_t)(uint8_t)*(*p)++ << (8 * i); return v;
}
static void cache_put64(char **p, uint64_t v) { cache_put32(p, (uint32_t)v); cache_put32(p, (uint32_t)(v >> 32)); }
static uint64_t cache_get64(const char **p) { uint64_t v = cache_get32(p); return v | ((uint64_t)cache_get32(p) << 32); }

static int expert_state_export(void *buffer, uint64_t size) {
    if (!buffer || size != expert_state_bytes()) return 0;
    char *p = (char *)buffer;
    cache_put32(&p, 0x31434551u); cache_put32(&p, 2);
    cache_put64(&p, g_model_registered_size);
    cache_put32(&p, expert_windows.size()); cache_put32(&p, expert_slots.size());
    cache_put32(&p, ram_slots.size()); cache_put32(&p, 0);
    /* Sorting makes snapshots deterministic; the ring is stored chronologically. */
    std::vector<uint64_t> bases;
    for (const auto &entry : expert_windows) bases.push_back(entry.first);
    std::sort(bases.begin(), bases.end());
    for (uint64_t base : bases) {
        const auto &w = expert_windows[base];
        cache_put64(&p, base); cache_put64(&p, w.bytes);
        cache_put32(&p, w.experts); cache_put32(&p, w.selected); cache_put32(&p, w.rows); cache_put32(&p, 0);
        for (unsigned t = 0; t < w.rows; t++) for (unsigned k = 0; k < w.selected; k++)
            cache_put32(&p, w.history[((w.cursor + 128 - w.rows + t) % 128) * w.selected + k]);
    }
    for (unsigned tier = 0; tier < 2; tier++) {
        const auto &slots = tier ? ram_slots : expert_slots;
        for (const auto &slot : slots) {
            cache_put64(&p, slot.offset); cache_put64(&p, expert_epoch - slot.last_used);
            cache_put32(&p, tier); cache_put32(&p, 0);
        }
    }
    return (uint64_t)(p - (char *)buffer) == size;
}

static int expert_state_import(const void *buffer, uint64_t size) {
    if (!buffer || size < 32 || size > 16ull * 1048576ull) return 0;
    const char *p = (const char *)buffer, *end = p + size;
    if (cache_get32(&p) != 0x31434551u) return 0;
    const unsigned version = cache_get32(&p);
    if ((version != 1 && version != 2) || cache_get64(&p) != g_model_registered_size) return 0;
    const unsigned count = cache_get32(&p), gpu = cache_get32(&p), ram = cache_get32(&p);
    if (cache_get32(&p) || count > 4096 || gpu > 262144 || ram > 262144) return 0;
    struct saved_window { uint64_t base, bytes; unsigned NE, NS, rows; std::vector<int> ids; };
    std::vector<saved_window> saved;
    for (unsigned i = 0; i < count; i++) {
        if (end - p < 32) return 0;
        saved_window w; w.base = cache_get64(&p); w.bytes = cache_get64(&p);
        w.NE = cache_get32(&p); w.NS = cache_get32(&p); w.rows = cache_get32(&p);
        if (cache_get32(&p) || !w.NE || w.NE > 512 || !w.NS || w.NS > 512 || !w.rows || w.rows > 128 ||
            !w.bytes || w.base > g_model_registered_size || w.bytes > (g_model_registered_size - w.base) / w.NE ||
            (uint64_t)(end - p) < 4ull * w.rows * w.NS) return 0;
        for (const auto &old : saved) if (old.base == w.base) return 0;
        w.ids.resize(w.rows * w.NS);
        for (auto &id : w.ids) { id = (int)cache_get32(&p); if (id < -1 || (id >= 0 && (unsigned)id >= w.NE)) return 0; }
        saved.push_back(w);
    }
    if ((uint64_t)(end - p) != 24ull * ((uint64_t)gpu + ram)) return 0;
    const char *hints = p;
    uint64_t max_age = 0;
    for (unsigned i = 0; i < gpu + ram; i++) {
        const uint64_t off = cache_get64(&p), age = cache_get64(&p);
        if (version == 2) max_age = std::max<uint64_t>(max_age, std::min<uint64_t>(age, UINT32_MAX));
        const unsigned tier = cache_get32(&p);
        if (cache_get32(&p) || tier != (i < gpu ? 0u : 1u)) return 0;
        bool found = false;
        for (const auto &w : saved) if (off >= w.base && off < w.base + w.bytes * w.NE && (off - w.base) % w.bytes == 0) found = true;
        if (!found) return 0;
    }
    /* Complete validation precedes mutation. Physical residency is rebuilt on
     * demand under CURRENT budgets; child/cache pointers cannot enter a parent. */
    expert_windows.clear(); expert_frequency.clear();
    gpu_victim_epoch = ram_victim_epoch = UINT64_MAX;
    route_reset();
    expert_epoch = max_age + 1;
    for (auto &slot : expert_slots) { slot.frequency = 0; slot.last_used = slot.pinned_epoch = 0; }
    for (auto &slot : ram_slots) { slot.frequency = 0; slot.last_used = slot.pinned_epoch = 0; }
    p = hints;
    for (unsigned i = 0; i < gpu + ram; i++) {
        const uint64_t off = cache_get64(&p), age = cache_get64(&p);
        (void)cache_get32(&p); (void)cache_get32(&p);
        const uint64_t last = version == 2 ? expert_epoch - std::min<uint64_t>(age, UINT32_MAX) : 0;
        auto g = expert_slot_by_offset.find(off); if (g != expert_slot_by_offset.end()) expert_slots[g->second].last_used = last;
        auto r = ram_slot_by_offset.find(off); if (r != ram_slot_by_offset.end()) ram_slots[r->second].last_used = last;
    }
    FILE *trace = expert_trace; expert_trace = NULL; const bool opened = expert_trace_opened; expert_trace_opened = true;
    for (const auto &w : saved) expert_cache_observe(w.base, w.bytes, w.NE, w.rows, w.NS, w.ids.data());
    expert_trace = trace; expert_trace_opened = opened;
    if (ds4_env_flag_enabled("DS4_CUDA_QWEN4_EXPERT_CACHE_VERBOSE", 0))
        fprintf(stderr, "ds4: Qwen expert checkpoint restored windows=%u GPU_hints=%u RAM_hints=%u (lazy residency, current budgets)\n", count, gpu, ram);
    return 1;
}

/* Compact only the selected experts into the existing temporary slab.
 * The kernels keep their original slot ordering and dot products. Copying
 * contiguous GGUF bytes avoids issuing every quantized load across PCIe. */
static int stage_experts(const void *map, uint64_t size, uint64_t o0, uint64_t o1,
        uint64_t bytes, const int *selected, unsigned NE, unsigned T, unsigned NS,
        bool pair, const char *const **p0, const char *const **p1) {
    *p0 = NULL; *p1 = NULL;
    if ((!tiered_cache(map) && (!ds4_env_flag_enabled("DS4_CUDA_QWEN4_STAGE_EXPERTS", 0) || T > 8)) || g_n_gpus > 1) return 1;
    const uint64_t n = (uint64_t)T * NS;
    if (!bytes || !n || (!tiered_cache(map) && n > 256) || o0 > size || bytes * NE > size - o0 ||
        (pair && (o1 > size || bytes * NE > size - o1))) return 0;
    std::vector<int> ids(n);
    const cudaStream_t stream = cuda_decode_stream();
    if (!cuda_ok(cudaMemcpyAsync(ids.data(), selected, n * sizeof(int),
                                cudaMemcpyDeviceToHost, stream), "Qwen selected expert IDs") ||
        !cuda_ok(cudaStreamSynchronize(stream), "Qwen selected expert IDs ready")) return 0;
    const uint64_t data_bytes = n * bytes * (pair ? 2u : 1u);
    const uint64_t table_off = (data_bytes + 255u) & ~255ull;
    char *tmp = (char *)cuda_tmp_alloc_on(0, table_off + n * sizeof(char *) * 2u,
                                        "Qwen selected expert bytes");
    if (!tmp) return 0;
    std::vector<const char *> ptrs(n * 2u, NULL);
    const bool cache_enabled = expert_cache_enabled(map);
    if (cache_enabled) {
        expert_cache_next_epoch();
        expert_cache_observe(o0, bytes, NE, T, NS, ids.data());
        if (pair) expert_cache_observe(o1, bytes, NE, T, NS, ids.data());
        /* Pin all hits first, including a later slot and duplicate IDs.
         * The selected-ID readback above also drains prior expert kernels,
         * so no previous launch can still consume a victim's allocation. */
        for (uint64_t i = 0; i < n; i++) {
            if (ids[i] < 0 || (unsigned)ids[i] >= NE) continue;
            for (unsigned w = 0; w < (pair ? 2u : 1u); w++)
                expert_cache_touch((w ? o1 : o0) + (uint64_t)ids[i] * bytes, 1);
        }
    }
    for (uint64_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (unsigned)ids[i] >= NE) continue;
        for (unsigned w = 0; w < (pair ? 2u : 1u); w++) {
            const uint64_t off = (w ? o1 : o0) + (uint64_t)ids[i] * bytes;
            const char *cached = NULL;
            if (cache_enabled && !expert_cache_get(map, off, bytes, stream, &cached)) return 0;
            char *dst = tmp + (w * n + i) * bytes;
            if (!cached && !expert_transfer(map, off, bytes, dst, stream)) return 0;
            ptrs[w * n + i] = cached ? cached : dst;
        }
    }
    if (!cuda_ok(cudaMemcpyAsync(tmp + table_off, ptrs.data(), n * sizeof(char *) * 2u,
                                cudaMemcpyHostToDevice, stream), "Qwen expert pointer table") ||
        !cuda_ok(cudaStreamSynchronize(stream), "Qwen selected experts ready")) return 0;
    *p0 = (const char *const *)(tmp + table_off);
    *p1 = pair ? *p0 + n : NULL;
    return 1;
}

static int moe_mv_dispatch(float *out, const float *x, const int *sel,
        const char *w0, const char *w1, const char *s0, const char *s1,
        unsigned type, unsigned st, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M, bool down,
        const char *const *routed0 = NULL, const char *const *routed1 = NULL) {
    const uint64_t rb = expert_row_bytes(type, K), srb = row_bytes(st, K);
    const dim3 grid((M + 3) / 4, NS + (st != UINT_MAX), T);
#define QWEN_MOE(TYPE) case TYPE: \
    if (down) moe_mv<TYPE, true><<<grid,128,0,cuda_decode_stream()>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,routed0,routed1); \
    else moe_mv<TYPE, false><<<grid,128,0,cuda_decode_stream()>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,routed0,routed1); break
    switch (type) {
    QWEN_MOE(0); QWEN_MOE(1); QWEN_MOE(2); QWEN_MOE(8); QWEN_MOE(10);
    QWEN_MOE(12); QWEN_MOE(16); QWEN_MOE(30); QWEN_MOE(39);
    default: return 0;
    }
#undef QWEN_MOE
    return launched();
}

__global__ void moe_reduce(float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        unsigned NS, unsigned stride, unsigned D, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    __shared__ float inject[4];
    if (threadIdx.x < hc) inject[threadIdx.x] = injection(inj + (uint64_t)t * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= D) return;
    float v = 0;
    for (unsigned s = 0; s < NS; s++) v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += sigmoid(gate[t]) * (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)t * hc + s) * D + d] += inject[s] * v;
}

__global__ void expert_lists(int *lists, int *counts, const int *selected,
        unsigned pairs, unsigned NE, unsigned cap) {
    __shared__ unsigned counters[512];
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counters[e] = 0;
    __syncthreads();
    for (unsigned p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = (unsigned)selected[p];
        if (e < NE) {
            const unsigned i = atomicAdd(counters + e, 1u);
            if (i < cap) lists[(uint64_t)e * cap + i] = p;
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counts[e] = min(counters[e], cap);
}

/* Tiles share dequantized weights across 16 tokens without requantizing the
 * activations. Expert counts determine the work; no expert list is truncated. */
template<unsigned TYPE, bool DOWN, bool EXPERT>
__global__ void matrix(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned T, unsigned NS, unsigned NO,
        unsigned K, unsigned M, unsigned cap, uint64_t rb,
        const char *const *routed0, const char *const *routed1) {
    const unsigned r = threadIdx.x % 16, c = threadIdx.x / 16, e = blockIdx.y;
    const unsigned row = blockIdx.x * 16 + r;
    const unsigned count = EXPERT ? (unsigned)counts[e] : T;
    if (EXPERT && !count) return;
    if (EXPERT && routed0) { w0 = routed0[e]; if (!DOWN) w1 = routed1[e]; }
    __shared__ float a[16][33], b[16][33], v[16][33];
    for (unsigned t0 = blockIdx.z * 16; t0 < count; t0 += gridDim.z * 16) {
        const unsigned item = t0 + c;
        const int pair = EXPERT && item < count ? lists[(uint64_t)e * cap + item] : (int)item;
        const unsigned tok = EXPERT ? pair / NS : item, slot = EXPERT ? pair % NS : 0;
        float acc = 0, up = 0;
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = threadIdx.x; j < 16 * 32; j += 256) {
                const unsigned rr = j / 32, kk = j % 32, global_row = blockIdx.x * 16 + rr;
                const unsigned logical = k0 + kk;
                const char *wr = w0 + ((uint64_t)(EXPERT && !routed0 ? e : 0) * M + global_row) * rb;
                a[rr][kk] = global_row < M && logical < K ? value<TYPE>(wr, logical) : 0;
                if (EXPERT && !DOWN) {
                    const char *ur = w1 + ((uint64_t)(routed1 ? 0 : e) * M + global_row) * rb;
                    b[rr][kk] = global_row < M && logical < K ? value<TYPE>(ur, logical) : 0;
                }
                const unsigned ti = t0 + rr;
                const int pp = EXPERT && ti < count ? lists[(uint64_t)e * cap + ti] : (int)ti;
                const unsigned tt = EXPERT ? pp / NS : ti, ss = EXPERT ? pp % NS : 0;
                const uint64_t xx = DOWN && EXPERT ? (uint64_t)tt * NO + ss : tt;
                v[rr][kk] = ti < count && logical < K ? x[xx * K + logical] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k++) {
                acc += a[r][k] * v[c][k];
                if (EXPERT && !DOWN) up += b[r][k] * v[c][k];
            }
            __syncthreads();
        }
        if (row < M && item < count) {
            const uint64_t op = EXPERT ? (uint64_t)tok * NO + slot : item;
            out[op * M + row] = EXPERT && !DOWN ? silu(acc) * up : acc;
        }
    }
}

__global__ void expert_tiles(unsigned *prefix, const int *counts, unsigned NE, unsigned nt) {
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        n += ((unsigned)counts[e]+nt-1)/nt;
        prefix[e+1] = n;
    }
}

/* Two TF32 components retain the fine part of each FP32 operand. Keep the
 * correction products in their own accumulator: repeatedly adding them to
 * the much larger leading product loses their precision on tensor cores. */
template<unsigned TYPE>
__global__ void matrix_tc(float *out, const float *x, const char *w0,
        unsigned T, unsigned K, unsigned M, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, r0 = blockIdx.x*64;
    /* Padding separates the banks of adjacent rows in the TF32 fragment
     * loads. The leading planes also hold the completed output tiles. */
    __shared__ __align__(32) float ah[64][36], al[64][36];
    __shared__ __align__(32) float bh[16][36], bl[16][36];
    for (unsigned t0 = blockIdx.z*16; t0 < T; t0 += gridDim.z*16) {
        wm::fragment<wm::accumulator,16,16,8,float> acc, corr, total;
        wm::fill_fragment(acc,0);
        wm::fill_fragment(corr,0);
        wm::fill_fragment(total,0);
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = tid; j < 64*32; j += 128) {
                const unsigned r = j/32, kk = j%32, row = r0+r, col = k0+kk;
                const uint64_t off = (uint64_t)row*rb;
                const float a = row < M && col < K ? value<TYPE>(w0+off,col) : 0;
                ah[r][kk] = wm::__float_to_tf32(a);
                al[r][kk] = wm::__float_to_tf32(a-ah[r][kk]);
            }
            for (unsigned j = tid; j < 16*32; j += 128) {
                const unsigned item = t0+j/32, kk = j%32, col = k0+kk;
                const float b = item < T && col < K ? x[(uint64_t)item*K+col] : 0;
                bh[j/32][kk] = wm::__float_to_tf32(b);
                bl[j/32][kk] = wm::__float_to_tf32(b-bh[j/32][kk]);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 8) {
                wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> a, alo;
                wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> b, blo;
                wm::load_matrix_sync(a,&ah[warp*16][k],36);
                wm::load_matrix_sync(alo,&al[warp*16][k],36);
                wm::load_matrix_sync(b,&bh[0][k],36);
                wm::load_matrix_sync(blo,&bl[0][k],36);
                wm::mma_sync(corr,alo,blo,corr);
                wm::mma_sync(corr,alo,b,corr);
                wm::mma_sync(corr,a,blo,corr);
                wm::mma_sync(acc,a,b,acc);
            }
            __syncthreads();
            /* Bound tensor-core accumulation depth. CUDA FP32 adds the
             * partial sums, avoiding drift on long HC projection rows. */
            if ((k0+32)%256 == 0 || k0+32 >= K) {
                for (unsigned i = 0; i < acc.num_elements; i++) {
                    total.x[i] += acc.x[i] + corr.x[i];
                }
                wm::fill_fragment(acc,0);
                wm::fill_fragment(corr,0);
            }
        }
        wm::store_matrix_sync(&ah[warp*16][0],total,36,wm::mem_row_major);
        __syncthreads();
        for (unsigned j = tid; j < 64*16; j += 128) {
            const unsigned row = r0+j/16, item = t0+j%16;
            if (row < M && item < T) out[(uint64_t)item*M+row] = ah[j/16][j%16];
        }
        __syncthreads();
    }
#endif
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template<unsigned TYPE, bool DOWN, unsigned NC>
__global__ void matrix_reg(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb,
        const char *const *routed0, const char *const *routed1) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x&31, warp = threadIdx.x/32;
    const unsigned row_tiles = (M+31)/32, job = blockIdx.x/row_tiles;
    if (job >= tiles[NE]) return;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1;
        else end = mid;
    }
    if (routed0) { w0 = routed0[e]; if (!DOWN) w1 = routed1[e]; }
    const unsigned count = counts[e], t0 = (job-tiles[e])*16*NC;
    const unsigned r0 = (blockIdx.x%row_tiles)*32+(warp%2)*16, c0 = t0+(warp/2)*8*NC;
    if (r0 >= M || c0 >= count) return;
    float acc[NC][4] = {}, corr[NC][4] = {}, total[NC][4] = {};
    float uacc[NC][4] = {}, ucorr[NC][4] = {}, utotal[NC][4] = {};
    const float *xp[NC];
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        const unsigned item = c0+nc*8+lane/4;
        const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
        xp[nc] = x+(DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS)*K;
    }
    for (unsigned k0 = 0; k0 < K; k0 += 8) {
        unsigned a[4], al[4], u[4], ul[4], b[2], bl[2];
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = r0+lane/4+(i%2)*8, k = k0+lane%4+(i/2)*4;
            const uint64_t off = ((uint64_t)(routed0 ? 0 : e)*M+r)*rb;
            const float v = r < M && k < K ? value<TYPE>(w0+off,k,grid_table,sign_table) : 0;
            const float hi = nvcuda::wmma::__float_to_tf32(v);
            a[i] = __float_as_uint(hi);
            al[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            if (!DOWN) {
                const float v = r < M && k < K ? value<TYPE>(w1+off,k,grid_table,sign_table) : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                u[i] = __float_as_uint(hi);
                ul[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
        }
        #pragma unroll
        for (unsigned nc = 0; nc < NC; nc++) {
            if (c0+nc*8 >= count) continue;
            const unsigned item = c0+nc*8+lane/4;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                const unsigned k = k0+lane%4+i*4;
                const float v = item < count && k < K ? xp[nc][k] : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                b[i] = __float_as_uint(hi);
                bl[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
            mma_tf32(corr[nc],al,bl);
            mma_tf32(corr[nc],al,b);
            mma_tf32(corr[nc],a,bl);
            mma_tf32(acc[nc],a,b);
            if (!DOWN) {
                mma_tf32(ucorr[nc],ul,bl);
                mma_tf32(ucorr[nc],ul,b);
                mma_tf32(ucorr[nc],u,bl);
                mma_tf32(uacc[nc],u,b);
            }
            if ((k0+8)%256 == 0 || k0+8 >= K) {
                #pragma unroll
                for (unsigned i = 0; i < 4; i++) {
                    total[nc][i] += acc[nc][i]+corr[nc][i];
                    acc[nc][i] = corr[nc][i] = 0;
                    if (!DOWN) { utotal[nc][i] += uacc[nc][i]+ucorr[nc][i]; uacc[nc][i] = ucorr[nc][i] = 0; }
                }
            }
        }
    }
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned row = r0+lane/4+(i/2)*8, item = c0+nc*8+(lane%4)*2+(i%2);
            if (row < M && item < count) {
                const unsigned pair = lists[(uint64_t)e*cap+item];
                out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = DOWN ? total[nc][i] : silu(total[nc][i])*utotal[nc][i];
            }
        }
    }
#endif
}

/* Match the half operands of the production Metal expert tiles. Products
 * accumulate in FP32; only the matrix operands are rounded to half. */
template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64>
__global__ void matrix_half_tile(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb,
        const char *const *routed0, const char *const *routed1) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, nr = (M+NR-1)/NR, job = blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1; else end = mid;
    }
    if (routed0) { w0 = routed0[e]; if (!DOWN) w1 = routed1[e]; }
    const unsigned count = counts[e], t0 = (job-tiles[e])*NT, r0 = (blockIdx.x%nr)*NR;
    const unsigned wr = warp%(NR/16), wc = warp/(NR/16);
    constexpr unsigned NC = NT/(8/(NR/16));
    union Tile {
        __half ab[(NR+(DOWN ? 0 : NR)+NT)*72];
        float c[NR][NT];
    };
    __shared__ __align__(32) Tile tile;
    __half (*a)[72] = (__half (*)[72])tile.ab;
    __half (*u)[72] = a+NR;
    __half (*b)[72] = a+NR+(DOWN ? 0 : NR);
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = tid; i < 256; i += 256) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = tid; i < 128; i += 256) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    wm::fragment<wm::accumulator,16,16,16,float> acc[NC/16], up[NC/16];
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        wm::fill_fragment(acc[j],0);
        if (!DOWN) wm::fill_fragment(up[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*4; i < NR*64; i += 256*4) {
            const unsigned row = r0+i/64, k = k0+i%64;
            const uint64_t off = ((uint64_t)(routed0 ? 0 : e)*M+row)*rb;
            const float4 av = row < M && k < K ? value4<TYPE>(w0+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
            *(__half2 *)&a[i/64][i%64] = __floats2half2_rn(av.x,av.y);
            *(__half2 *)&a[i/64][i%64+2] = __floats2half2_rn(av.z,av.w);
            if (!DOWN) {
                const float4 uv = row < M && k < K ? value4<TYPE>(w1+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
                *(__half2 *)&u[i/64][i%64] = __floats2half2_rn(uv.x,uv.y);
                *(__half2 *)&u[i/64][i%64+2] = __floats2half2_rn(uv.z,uv.w);
            }
        }
        for (unsigned i = tid; i < NT*64; i += 256) {
            const unsigned item = t0+i/64, k = k0+i%64;
            const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row = DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            b[i/64][i%64] = __float2half_rn(item < count && k < K ? x[row*K+k] : 0);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> af, uf;
            wm::load_matrix_sync(af,&a[wr*16][k],72);
            if (!DOWN) wm::load_matrix_sync(uf,&u[wr*16][k],72);
            #pragma unroll
            for (unsigned j = 0; j < NC/16; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> bf;
                wm::load_matrix_sync(bf,&b[wc*NC+j*16][k],72);
                wm::mma_sync(acc[j],af,bf,acc[j]);
                if (!DOWN) wm::mma_sync(up[j],uf,bf,up[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        if (!DOWN) for (unsigned i = 0; i < acc[j].num_elements; i++)
            acc[j].x[i] = silu(acc[j].x[i])*up[j].x[i];
        wm::store_matrix_sync(&tile.c[wr*16][wc*NC+j*16],acc[j],NT,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < NR*NT; i += 256) {
        const unsigned row = r0+i/NT, item = t0+i%NT;
        if (row < M && item < count) {
            const unsigned pair = lists[(uint64_t)e*cap+item];
            out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = tile.c[i/NT][i%NT];
        }
    }
#endif
}

static int matrix_dispatch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned type, unsigned NE, unsigned T,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, bool down,
        const char *const *routed0 = NULL, const char *const *routed1 = NULL,
        unsigned active_pairs = 0, unsigned active_experts = 0) {
    const uint64_t pairs = active_pairs ? active_pairs : (uint64_t)T * NS;
    const unsigned padding = active_experts ? active_experts : NE;
    const dim3 grid((M + 15) / 16, lists ? NE : 1, std::min(8u, (T + 15) / 16));
    const uint64_t rb = expert_row_bytes(type, K);
    if (ds4_cuda_attn_tokentile_arch_ok() && !getenv("DS4_CUDA_NO_TF32")) {
        if (lists && !g_quality_mode && (type == 16 || type == 10 || type == 12 || type == 39)) {
            const unsigned nt = T >= 2048 ? 64 : 32;
            unsigned *tiles = (unsigned *)cuda_tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tiles");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,nt);
            if (!launched()) return 0;
            /* Wider rows reuse activations; Q2_K down is faster at 64 rows. */
            const unsigned nr = nt == 64 && type != 10 ? 128 : 64;
            const uint64_t blocks = ((pairs+nt-1)/nt+padding)*((M+nr-1)/nr);
            if (blocks > INT_MAX) return 0;
#define QWEN_HALF(TYPE, DOWN) \
            if (nt == 64) matrix_half_tile<TYPE,DOWN,64,(TYPE == 10 ? 64 : 128)><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb,routed0,routed1); \
            else matrix_half_tile<TYPE,DOWN,32><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb,routed0,routed1)
#define QWEN_HALF_TYPE(TYPE) case TYPE: if (down) { QWEN_HALF(TYPE,true); } else { QWEN_HALF(TYPE,false); } break
            switch (type) { QWEN_HALF_TYPE(16); QWEN_HALF_TYPE(10); QWEN_HALF_TYPE(12); QWEN_HALF_TYPE(39); }
#undef QWEN_HALF_TYPE
#undef QWEN_HALF
            return launched();
        }
        unsigned *tiles = NULL;
        dim3 tcgrid((M+63)/64,1,std::min(8u,(T+15)/16));
        if (lists) {
            tiles = (unsigned *)cuda_tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tile schedule");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,32);
            if (!launched()) return 0;
            const uint64_t jobs = (pairs+31)/32+padding;
            const uint64_t blocks = jobs*((M+31)/32);
            if (blocks > INT_MAX) return 0;
            tcgrid = dim3((unsigned)blocks,1,1);
        }
#define QWEN_TC(TYPE) case TYPE: \
        if (!lists) matrix_tc<TYPE><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,T,K,M,rb); \
        else if (down) matrix_reg<TYPE,true,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb,routed0,routed1); \
        else matrix_reg<TYPE,false,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb,routed0,routed1); break
        switch (type) {
        QWEN_TC(0); QWEN_TC(1); QWEN_TC(2); QWEN_TC(8); QWEN_TC(10);
        QWEN_TC(12); QWEN_TC(16); QWEN_TC(30); QWEN_TC(39);
        default: return 0;
        }
#undef QWEN_TC
        return launched();
    }
#define QWEN_MM(TYPE) case TYPE: \
    if (!lists) matrix<TYPE,false,false><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb,routed0,routed1); \
    else if (down) matrix<TYPE,true,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb,routed0,routed1); \
    else matrix<TYPE,false,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb,routed0,routed1); break
    switch (type) {
    QWEN_MM(0); QWEN_MM(1); QWEN_MM(2); QWEN_MM(8); QWEN_MM(10);
    QWEN_MM(12); QWEN_MM(16); QWEN_MM(30); QWEN_MM(39);
    default: return 0;
    }
#undef QWEN_MM
    return launched();
}

template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid, const uint8_t *signs) {
    float acc = 0;
    if (TYPE == 1 && !(n%4) && !((uintptr_t)x&15) && !((uintptr_t)row&7)) {
        for (unsigned i = (threadIdx.x&31)*4; i < n; i += 128) {
            const float4 w = value4<TYPE>(row,i,grid,signs), v = *(const float4 *)(x+i);
            acc += w.x*v.x; acc += w.y*v.y; acc += w.z*v.z; acc += w.w*v.w;
        }
    } else {
        for (unsigned i = threadIdx.x & 31; i < n; i += 32) acc += value<TYPE>(row, i, grid, signs) * x[i];
    }
    return sum(acc);
}

template<unsigned TYPE>
__global__ void matvec(float *out, const char *w, const float *x,
                       unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    if (row >= M) return;
    const float v = dot<TYPE>(w + row * stride, x + (uint64_t)tok * K, K);
    if (!(threadIdx.x & 31)) out[(uint64_t)tok * M + row] = v;
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_rows(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    float a[ROWS] = {};
    const char *wr = w+(uint64_t)row*stride;
    if (TYPE == 1 && !(K%4) && !((uintptr_t)x&15) && !((uintptr_t)wr&7)) {
        for (unsigned i = lane*4; i < K; i += 128) {
            const float4 v = value4<TYPE>(wr,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x+(uint64_t)t*K+i);
                a[t] += v.x*xv.x; a[t] += v.y*xv.y; a[t] += v.z*xv.z; a[t] += v.w*xv.w;
            }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float v = value<TYPE>(wr,i);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) a[t] += v*x[(uint64_t)t*K+i];
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(a[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS>
__global__ void matvec_q8(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    const char *wr = w+(uint64_t)row*stride;
    float acc[ROWS] = {};
    for (unsigned i = lane*4; i < K; i += 128) {
        const char *b = wr+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        const unsigned q01 = q[0], q23 = q[1];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 v = *(const float4 *)(x+(uint64_t)t*K+i);
            float part = (float)(int8_t)q01*v.x;
            part += (float)(int8_t)(q01>>8)*v.y;
            part += (float)(int8_t)q23*v.z;
            part += (float)(int8_t)(q23>>8)*v.w;
            acc[t] += part*scale;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

static int matvec_dispatch(float *out, const char *w, const float *x,
                           unsigned type, unsigned T, unsigned K, unsigned M) {
    const dim3 grid((M + 3) / 4, T);
    const uint64_t stride = row_bytes(type, K);
    if (!stride) return 0;
    if (type == 8 && T <= 8 && !((uintptr_t)x&15)) {
#define QWEN_Q8_ROWS(N) matvec_q8<N><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_Q8_ROWS(1); }
        else if (T == 2) { QWEN_Q8_ROWS(2); }
        else if (T <= 4) { QWEN_Q8_ROWS(4); }
        else { QWEN_Q8_ROWS(8); }
#undef QWEN_Q8_ROWS
        return launched();
    }
#define QWEN_MV(TYPE) case TYPE: \
    if (T == 2) matvec_rows<TYPE,2><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 2 && T <= 4) matvec_rows<TYPE,4><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 4 && T <= 8) matvec_rows<TYPE,8><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else matvec<TYPE><<<grid,128,0,cuda_decode_stream()>>>(out,w,x,K,M,stride); break
    switch (type) {
    QWEN_MV(0); QWEN_MV(1); QWEN_MV(2); QWEN_MV(8); QWEN_MV(10);
    QWEN_MV(12); QWEN_MV(16); QWEN_MV(30); QWEN_MV(39);
    default: return 0;
    }
#undef QWEN_MV
    return launched();
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range. Preserve the residual too,
 * so dense projections retain substantially more than half precision. */
template<unsigned TYPE>
__global__ void pack_half_components(float *scales, __half *hi, __half *lo,
        const char *x, unsigned K, uint64_t rb) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
        lo[(uint64_t)row*K+k] = __float2half_rn((v-__half2float(h))*4096);
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

/* Short projections reuse the weights for both activation components.
 * Keep separate FP32 sums and the same scaled residual as the cuBLAS path. */
__global__ void dense_f16_components(float *out, const __half *xh, const __half *xl,
        const __half *w, const float *scales, unsigned T, unsigned K, unsigned M) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32;
    const unsigned m0 = blockIdx.x*64, t0 = blockIdx.y*64;
    union Tile { __half data[3][64][72]; float result[64][64]; };
    __shared__ __align__(32) Tile tile;
    wm::fragment<wm::accumulator,16,16,16,float> hi[2], lo[2];
    for (unsigned j = 0; j < 2; j++) {
        wm::fill_fragment(hi[j],0);
        wm::fill_fragment(lo[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*8; i < 4096; i += 2048) {
            const unsigned r = i/64, k = i%64;
            const bool valid_w = m0+r < M, valid_x = t0+r < T;
            tt_cp_async_16B(&tile.data[0][r][k],w+((uint64_t)(valid_w ? m0+r : 0))*K+k0+k,valid_w);
            tt_cp_async_16B(&tile.data[1][r][k],xh+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
            tt_cp_async_16B(&tile.data[2][r][k],xl+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;
            wm::load_matrix_sync(a,&tile.data[0][(warp%4)*16][k],72);
            for (unsigned j = 0; j < 2; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> b, c;
                wm::load_matrix_sync(b,&tile.data[1][(warp/4)*32+j*16][k],72);
                wm::load_matrix_sync(c,&tile.data[2][(warp/4)*32+j*16][k],72);
                wm::mma_sync(hi[j],a,b,hi[j]);
                wm::mma_sync(lo[j],a,c,lo[j]);
            }
        }
        __syncthreads();
    }
    for (unsigned j = 0; j < 2; j++) {
        for (unsigned i = 0; i < hi[j].num_elements; i++) hi[j].x[i] += lo[j].x[i]*0x1p-12f;
        wm::store_matrix_sync(&tile.result[(warp%4)*16][(warp/4)*32+j*16],hi[j],64,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < 4096; i += 256) {
        const unsigned m = m0+i%64, t = t0+i/64;
        if (m < M && t < T) out[(uint64_t)t*M+m] = tile.result[i%64][i/64]*scales[t];
    }
#endif
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    __half *hi = (__half *)cuda_tmp_alloc((xn+T)*4, "Qwen F16 projection scratch");
    if (!hi) return 0;
    __half *lo = hi+xn;
    float *scales = (float *)(lo+xn);
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(scales,hi,lo,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    if (K <= 512 && K%64 == 0 && T >= 32 && ds4_cuda_attn_tokentile_arch_ok()) {
        dense_f16_components<<<dim3((M+63)/64,(T+63)/64),256,0,cuda_decode_stream()>>>(out,hi,lo,w,scales,T,K,M);
        return launched();
    }
    const float zero = 0, one = 1, low = 0x1p-12f;
    if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&low,w,CUDA_R_16F,K,lo,CUDA_R_16F,K,&zero,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 residual projection") ||
        !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&one,w,CUDA_R_16F,K,hi,CUDA_R_16F,K,&one,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 leading projection")) return 0;
    dense_rescale<<<((uint64_t)M*T+255)/256,256,0,cuda_decode_stream()>>>(out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

static int dense_q8_blas(float *out, const float *x, const char *w,
        unsigned T, unsigned K, unsigned M) {
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(32)<<20)/((uint64_t)K*4));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    __half *xh = (__half *)cuda_tmp_alloc((xn+wn+T+tile)*4,"Qwen Q8 projection scratch");
    if (!xh) return 0;
    __half *xl = xh+xn, *wh = xl+xn, *wl = wh+wn;
    float *xs = (float *)(wl+wn), *ws = xs+T;
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(xs,xh,xl,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1, low = 0x1p-12f;
    const uint64_t rb = row_bytes(8,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        pack_half_components<8><<<n,256,0,cuda_decode_stream()>>>(ws,wh,wl,w+(uint64_t)r*rb,K,rb);
        if (!launched()) return 0;
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wl,CUDA_R_16F,K,xh,CUDA_R_16F,K,&zero,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 weight residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wh,CUDA_R_16F,K,xl,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 input residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&one,wh,CUDA_R_16F,K,xh,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 leading product")) return 0;
        dense_rescale2<<<((uint64_t)n*T+255)/256,256,0,cuda_decode_stream()>>>(out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. cuBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)cuda_tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: unpack<TYPE><<<((uint64_t)n*K+255)/256,256,0,cuda_decode_stream()>>>(scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_UNPACK(1); QWEN_UNPACK(2); QWEN_UNPACK(8); QWEN_UNPACK(10);
            QWEN_UNPACK(12); QWEN_UNPACK(16); QWEN_UNPACK(30); QWEN_UNPACK(39);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void mtp_stage(float *cat, const float *e, const float *R,
        const float *ge, const float *gh, unsigned E, unsigned hc, float eps) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const bool emb = row == 0;
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e : R;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < n; i += blockDim.x) ss += rs[i] * rs[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? e : R + (uint64_t)(row - 1) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float *o = cat + (uint64_t)row * 2 * E;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        o[(emb ? 0 : E) + i] = src[i] * inv * g[i];
        o[(emb ? E : 0) + i] = 0;
    }
}

__global__ void mtp_combine(float *out, const float *proj, unsigned E, unsigned hc) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) out[i] = proj[i % E] + proj[E + i];
}

struct max_pair { float value; int index; };
template<bool FINISH>
__global__ void argmax(int *out, max_pair *scratch, const float *logits, unsigned n) {
    const unsigned tid = threadIdx.x;
    __shared__ max_pair best[256];
    max_pair v = {-1e30f, 0};
    for (unsigned i = (FINISH ? 0 : blockIdx.x * 4096) + tid;
         i < (FINISH ? n : min(n, (blockIdx.x + 1) * 4096)); i += 256) {
        const max_pair p = FINISH ? scratch[i] : max_pair{logits[i], (int)i};
        if (p.value > v.value || (p.value == v.value && p.index < v.index)) v = p;
    }
    best[tid] = v;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (tid < s) {
            const max_pair p = best[tid + s], q = best[tid];
            if (p.value > q.value || (p.value == q.value && p.index < q.index)) best[tid] = p;
        }
        __syncthreads();
    }
    if (!tid) { if (FINISH) *out = best[0].index; else scratch[blockIdx.x] = best[0]; }
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    const unsigned row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, E = H*D;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p = 0; p < N; p++) {
        const uint64_t kb = (uint64_t)p*E + h*D;
        float score = 0;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*k[kb+lane+j*32];
        score = sum(score) * rsqrtf((float)D);
        const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
        denom = denom*old + prob;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) acc[j] = acc[j]*old + prob*v[kb+lane+j*32];
        mx = nm;
    }
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

__global__ void hc_norm(float *xn, float *inj, const float *R, const float *gamma,
                        const char *wi, unsigned type, unsigned E, unsigned hc,
                        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    float acc[4] = {};
    const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
    for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
        const float v = R[base + i] * inv * gamma[stream * E + i];
        xn[base + i] = v;
        for (unsigned j = 0; j < ni; j++) acc[j] += scalar(wi, j * dim + stream * E + i, type) * v;
    }
    for (unsigned j = 0; j < ni; j++) {
        const float v = block_sum(acc[j], red);
        if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
    }
}

/* Prefill has enough tokens to share one normalization across all eight
 * injection chunks. Each warp preserves the four partial sums of the
 * decode kernel, including their order, without rereading R eight times. */
template<unsigned TYPE>
__global__ void hc_norm_prefill(float *xn, float *inj, const float *R,
        const float *gamma, const char *wi, unsigned E, unsigned hc,
        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, lane = tid & 31, chunk = tid / 32;
    const unsigned dim = E * hc, per = (E + 7) / 8;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    if (tid < 128) for (unsigned i = tid; i < E; i += 128)
        ss += R[base+i] * R[base+i];
    const float inv = rsqrtf(block_sum(ss,red) / E + eps);
    float acc[4][4] = {};
    const unsigned end = min(E,(chunk+1)*per);
    #pragma unroll
    for (unsigned part = 0; part < 4; part++) {
        for (unsigned i = chunk*per+lane+part*32; i < end; i += 128) {
            const float v = R[base+i] * inv * gamma[stream*E+i];
            xn[base+i] = v;
            #pragma unroll
            for (unsigned j = 0; j < 4; j++) if (j < ni)
                acc[j][part] += value<TYPE>(wi,j*dim+stream*E+i) * v;
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) if (j < ni) {
        float v = 0;
        #pragma unroll
        for (unsigned part = 0; part < 4; part++) v += sum(acc[j][part]);
        if (!lane) inj[((uint64_t)tok*hc*8+stream*8+chunk)*ni+j] = v;
    }
}

template<unsigned TYPE>
__global__ void hc_mix(float *out, const float *xn, const float *lo,
                       const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    __shared__ __align__(16) float activated[512];
    if (rank <= 512) {
        for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
            activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
        __syncthreads();
    }
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const char *row = up+((uint64_t)stream*E+d)*rb;
        if (!(rank%4) && rank <= 512 &&
            !(TYPE == 0 ? (uintptr_t)row&15 : TYPE == 1 ? (uintptr_t)row&7 : 0)) {
            for (unsigned r = l*4; r < rank; r += 32) {
                const float4 w = value4<TYPE>(row,r,NULL,NULL), v = *(const float4 *)(activated+r);
                a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
            }
        } else {
            for (unsigned r = l; r < rank; r += 8)
                a += value<TYPE>(row,r) *
                    (rank <= 512 ? activated[r] : silu(lo[(uint64_t)tok * rank + r] / hc));
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor_sync(0xffffffff, a, off);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 16);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

__device__ float injection(const float *inj, unsigned hc, unsigned stream) {
    float v = 0;
    for (unsigned i = 0; i < hc * 8; i++) v += inj[i * hc + stream];
    return 2 * sigmoid(v / hc);
}

__global__ void hc_combine(float *R, const float *out, const float *inj, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj + (uint64_t)tok * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= E) return;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)tok * hc + s) * E + d] += weights[s] * out[(uint64_t)tok * E + d];
}

__global__ void hc_lo(float *dst, const float *src, uint64_t n, unsigned hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = silu(src[i] / hc);
}

__global__ void hc_rows(float *dst, const float *up, const float *xn, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= E) return;
    float v = 0;
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t i = ((uint64_t)t * hc + s) * E + d;
        v += sigmoid(up[i]) * xn[i];
    }
    dst[(uint64_t)t * E + d] = v / hc;
}

} // namespace qwen4_cuda

namespace qwen4_cuda {

__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps) {
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) o[idx + i] = o[idx + i] * r * w[k0 + i] * sigmoid(z[idx + i]);
}

__global__ void ngram_gate(float *gated, float *normed, const float *R, const float *key,
                           const float *val, const float *gk, const float *gq, const float *gc,
                           unsigned E, unsigned hc, float eps) {
    const unsigned s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    __shared__ float red[32];
    float sk = 0, sq = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { sk += key[base + i] * key[base + i]; sq += R[base + i] * R[base + i]; }
    const float ik = rsqrtf(block_sum(sk, red) / E + eps);
    const float iq = rsqrtf(block_sum(sq, red) / E + eps);
    float dot = 0;
    for (unsigned i = tid; i < E; i += blockDim.x)
        dot += (key[base + i] * ik * gk[s * E + i]) * (R[base + i] * iq * gq[s * E + i]);
    const float a = block_sum(dot, red) * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = sigmoid(a > 0 ? mag : a < 0 ? -mag : 0);
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float v = gate * val[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ void ngram_conv(float *R, const float *gated, const float *normed, float *history,
                           const char *w, unsigned type, unsigned T, unsigned C, unsigned K,
                           unsigned dilation, float *snap, unsigned st, float *snap2, unsigned st2) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned H = (K - 1) * dilation;
    float hist[9], taps[4];
    for (unsigned i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = scalar(w, c * K + i, type);
    for (unsigned t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1] * cur;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * hist[i * dilation];
        for (unsigned i = 0; i + 1 < H; i++) hist[i] = hist[i + 1];
        hist[H - 1] = cur;
        R[p] += gated[p] + silu(v);
        if (snap && t == st) for (unsigned i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (unsigned i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (unsigned i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen4_cuda;
    if (!T || !C || K < 2 || K > 4 || !tensor(x, (uint64_t)T * C * 4) || !tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, NULL, UINT_MAX, NULL, UINT_MAX);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen4_cuda;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = weight(map, size, ao, (uint64_t)Hv * 4), *bias = weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    gdn_prep<<<dim3(Hk, T), 32, 0, cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) { (void)on; }

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !tensor(out, (uint64_t)T * Hv * D * 4) || !tensor(state, bytes) ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !tensor(snap, bytes)) || (snap2 && !tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) gdn_scan<ROWS,DIM><<<dim3((DIM+4*ROWS-1)/(4*ROWS),Hv),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2)
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !tensor(out, n * 4) || !tensor(z, n * 4)) return 0;
    const char *w = weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)z->ptr, (const float *)w, H, D, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(ds4_gpu_tensor *gated, ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *R, const ds4_gpu_tensor *key, const ds4_gpu_tensor *val,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)T * E * hc * 4, wb = (uint64_t)E * hc * 4;
    if (!T || !E || !hc || hc > 4 || !tensor(gated, bytes) || !tensor(normed, bytes) ||
        !tensor(R, bytes) || !tensor(key, bytes) || !tensor(val, (uint64_t)T * E * 4)) return 0;
    const char *gk = weight(map, size, ko, wb), *gq = weight(map, size, qo, wb), *gc = weight(map, size, co, wb);
    if (!gk || !gq || !gc) return 0;
    ngram_gate<<<dim3(hc, T), 128, 0, cuda_decode_stream()>>>((float *)gated->ptr, (float *)normed->ptr,
        (const float *)R->ptr, (const float *)key->ptr, (const float *)val->ptr,
        (const float *)gk, (const float *)gq, (const float *)gc, E, hc, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *normed, ds4_gpu_tensor *history, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * C * 4, hb = (uint64_t)(K - 1) * dilation * C * 4;
    if (!T || !C || K < 2 || K > 4 || !dilation || dilation > 3 ||
        (type != 0 && type != 1) || !tensor(R, n) || !tensor(gated, n) || !tensor(normed, n) ||
        !tensor(history, hb) || (snap && !tensor(snap, hb)) || (snap2 && !tensor(snap2, hb))) return 0;
    const char *w = weight(map, size, off, row_bytes(type, (uint64_t)C * K));
    if (!w) return 0;
    ngram_conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)R->ptr,
        (const float *)gated->ptr, (const float *)normed->ptr, (float *)history->ptr, w, type, T, C, K,
        dilation, snap ? (float *)snap->ptr : NULL, st, snap2 ? (float *)snap2->ptr : NULL, st2);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * E * hc;
    if (!T || !E || !hc || hc > 8 || ni > 4 ||
        !tensor(xn, n * 4) || !tensor(R, n * 4) ||
        (ni && !tensor(inj, (uint64_t)T * hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, (uint64_t)E * hc * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(type, (uint64_t)E * hc) * ni) : gamma;
    if (!gamma || !wi || (type != 0 && type != 1 && type != 8)) return 0;
    if (T > 8) {
#define QWEN_HC_NORM(TYPE) hc_norm_prefill<TYPE><<<dim3(hc,T),256,0,cuda_decode_stream()>>>((float *)xn->ptr, \
        ni ? (float *)inj->ptr : NULL,(const float *)R->ptr,(const float *)gamma,wi,E,hc,ni,eps)
        if (type == 0) { QWEN_HC_NORM(0); }
        else if (type == 1) { QWEN_HC_NORM(1); }
        else { QWEN_HC_NORM(8); }
#undef QWEN_HC_NORM
        return launched();
    }
    hc_norm<<<dim3(hc * 8, T), 128, 0, cuda_decode_stream()>>>((float *)xn->ptr,
        ni ? (float *)inj->ptr : NULL, (const float *)R->ptr, (const float *)gamma, wi, type, E, hc, ni, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !rank || !tensor(out, (uint64_t)T * E * 4) ||
        !tensor(xn, (uint64_t)T * E * hc * 4) || !tensor(lo, (uint64_t)T * rank * 4)) return 0;
    const char *w = weight(map, size, offset, row_bytes(type, rank) * E * hc);
    if (!w || (type != 0 && type != 1 && type != 8)) return 0;
#define QWEN_HC(TYPE) hc_mix<TYPE><<<dim3((E + 3) / 4, T),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,E,hc,rank,row_bytes(TYPE,rank))
    if (type == 0) { QWEN_HC(0); }
    else if (type == 1) { QWEN_HC(1); }
    else { QWEN_HC(8); }
#undef QWEN_HC
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(R, (uint64_t)T * E * hc * 4) ||
        !tensor(out, (uint64_t)T * E * 4) || !tensor(inj, (uint64_t)T * hc * hc * 8 * 4)) return 0;
    hc_combine<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)R->ptr, (const float *)out->ptr, (const float *)inj->ptr, E, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src, uint32_t T, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * rank;
    if (!n || !hc || !tensor(dst, n * 4) || !tensor(src, n * 4)) return 0;
    hc_lo<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)dst->ptr, (const float *)src->ptr, n, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *xn, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(dst, (uint64_t)T * E * 4) ||
        !tensor(up, (uint64_t)T * hc * E * 4) || !tensor(xn, (uint64_t)T * hc * E * 4)) return 0;
    hc_rows<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)dst->ptr, (const float *)up->ptr, (const float *)xn->ptr, E, hc);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n, float scale) {
    qwen4_cuda::rope_set = freq != NULL;
    qwen4_cuda::rope_scale = freq ? scale : 1;
    memset(qwen4_cuda::rope_freq, 0, sizeof(qwen4_cuda::rope_freq));
    if (freq) memcpy(qwen4_cuda::rope_freq, freq, std::min(n, 32u) * sizeof(float));
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4, ib = (uint64_t)T * Hi * Di * 4;
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32 || D > 256 || D % 32 ||
        Di < 32 || Di > 128 || Di % 32 || nrot > 64 || nrot > D || nrot > Di || nrot % 2 ||
        (uint64_t)pos0 + T > cap || !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) || !tensor(iq, ib) || !tensor(iqout, ib) ||
        !tensor(ik, (uint64_t)T * Di * 4) || !tensor(ikc, (uint64_t)cap * Di * 4) ||
        !tensor(kc, (uint64_t)cap * Hkv * D * 2) || !tensor(vc, (uint64_t)cap * Hkv * D * 2) ||
        !tensor(pos3, (uint64_t)cap * 16)) return 0;
    const char *gq = weight(map, size, qo, D * 4), *gk = weight(map, size, ko, D * 4), *giq = weight(map, size, io, Di * 4);
    if (!gq || !gk || !giq) return 0;
    attn_prep<<<dim3(H + Hkv + Hi + 1, T), 32, 0, cuda_decode_stream()>>>((float *)q->ptr,
        (float *)gate->ptr, (__half *)kc->ptr, (__half *)vc->ptr, (float *)iqout->ptr, (float *)ikc->ptr,
        (const float *)qg->ptr, (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr,
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, (const float *)gq, (const float *)gk, (const float *)giq,
        H, Hkv, D, Hi, Di, pos0, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *map, uint64_t size,
        uint64_t off, uint32_t b0, uint32_t N, uint32_t ratio, uint32_t D, uint32_t nrot,
        float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t rows = (uint64_t)b0 + N;
    if (!N || !ratio || D < 32 || D > 128 || D % 32 || nrot > D || nrot > 64 || nrot % 2 ||
        !tensor(out, rows * D * 2) || !tensor(ik, rows * ratio * D * 4) || !tensor(pos3, rows * ratio * 16)) return 0;
    const char *w = weight(map, size, off, D * 4);
    if (!w) return 0;
    block_key<<<N, 32, 0, cuda_decode_stream()>>>((__half *)out->ptr, (const float *)ik->ptr,
        (const uint32_t *)pos3->ptr, (const float *)w, b0, ratio, D, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *tiles,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *key, uint32_t T, uint32_t N,
        uint32_t H, uint32_t D, uint32_t pos0, uint32_t ratio) {
    using namespace qwen4_cuda;
    const unsigned nt = (N + 7) / 8;
    if (!T || !N || !H || !D || !ratio || !tensor(out, (uint64_t)T * N * 4) ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(key, (uint64_t)N * D * 2) ||
        (tiles && !tensor(tiles, (uint64_t)T * nt * 4))) return 0;
    idx_score<<<dim3((N + 3) / 4, T), 128, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)q->ptr, (const __half *)key->ptr, N, H, D, pos0, ratio);
    if (!launched()) return 0;
    if (tiles) tile_max<<<dim3((nt + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (unsigned *)tiles->ptr, (const float *)out->ptr, N, nt);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tiles, uint32_t N, uint32_t T, uint32_t K) {
    using namespace qwen4_cuda;
    (void)tiles;
    if (!T || !K || K > N || !tensor(out, (uint64_t)T * K * 4) || !tensor(score, (uint64_t)T * N * 4)) return 0;
    idx_select<<<T, 256, 0, cuda_decode_stream()>>>((int *)out->ptr, (const float *)score->ptr, N, K);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *count,
        const ds4_gpu_tensor *blocks, uint32_t T, uint32_t K, uint32_t ratio, uint32_t pos0, uint32_t stride) {
    using namespace qwen4_cuda;
    if (!T || !K || !ratio || (uint64_t)K * ratio + ratio - 1 > stride ||
        !tensor(out, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4) || !tensor(blocks, (uint64_t)T * K * 4)) return 0;
    idx_expand<<<T, 256, 0, cuda_decode_stream()>>>((int *)out->ptr, (unsigned *)count->ptr,
        (const int *)blocks->ptr, K, ratio, pos0, stride);
    return launched();
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * 64 * (D + 2);
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    using namespace qwen4_cuda;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(map, size, off, rb*M);
    if (!w) return 0;
    if (T <= 8) return matvec_dispatch((float *)out->ptr, w, (const float *)x->ptr, type, T, K, M);
    if (type == 1 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type == 8 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_q8_blas((float *)out->ptr,(const float *)x->ptr,w,T,K,M);
    if (g_cublas_ready && T >= 32 && K <= INT_MAX && M <= INT_MAX && T <= INT_MAX)
        return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    return matrix_dispatch((float *)out->ptr, (const float *)x->ptr, w, NULL, NULL, NULL,
                           type, 1, T, 1, 1, K, M, 0, false);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t off, uint64_t K, uint64_t M, const ds4_gpu_tensor *x, uint64_t T) {
    if (K > UINT_MAX || M > UINT_MAX || T > UINT_MAX) return 0;
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, off, 8, T, K, M);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
        uint32_t K, uint32_t M, const ds4_gpu_tensor *x) {
    using namespace qwen4_cuda;
    const uint64_t rb = row_bytes(8, K);
    if (!K || !M || !rb || !tensor(w, rb*M) || !tensor(x, (uint64_t)K*4) || !tensor(out, (uint64_t)M*4)) return 0;
    return matvec_dispatch((float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, 8, 1, K, M);
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t T,
        uint32_t K, uint32_t N, ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    if (!N || N > 4 || !outs || !offsets || !types || !rows) return 0;
    for (unsigned i = 0; i < N; i++)
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], T, K, rows[i])) return 0;
    return 1;
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *o0, ds4_gpu_tensor *o1,
        const void *map, uint64_t size, uint64_t w0, uint64_t w1, uint64_t K,
        uint64_t M0, uint64_t M1, const ds4_gpu_tensor *x, uint64_t T) {
    if ((M0 & 1) || (M1 & 1)) return 0;
    return ds4_gpu_qwen4_matmul_q8_0_tensor(o0, map, size, w0, K, M0, x, T) &&
           ds4_gpu_qwen4_matmul_q8_0_tensor(o1, map, size, w1, K, M1, x, T);
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    using namespace qwen4_cuda;
    if (!T || !NE || NE > 512 || !NS || NS > NE || NS > 32 ||
        !tensor(sel, (uint64_t)T*NS*4) || !tensor(weights, (uint64_t)T*NS*4) || !tensor(logits, (uint64_t)T*NE*4)) return 0;
    const char *wg = K ? weight(map, size, off, row_bytes(type, K)) : NULL;
    if (K && (!wg || !tensor(x, (uint64_t)T*K*4) || !tensor(sg, (uint64_t)T*4))) return 0;
    router<<<T,256,0,cuda_decode_stream()>>>((int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        K ? (const float *)x->ptr : NULL, wg, K ? (float *)sg->ptr : NULL, NE, NS, K, type);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(mid, (uint64_t)T*NO*M*4) ||
        !tensor(x, (uint64_t)T*K*4) || !tensor(sel, (uint64_t)T*NS*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type, K)*M*NE, sb = row_bytes(st, K)*M;
    const char *g = expert_weight(map,size,go,bytes), *u = expert_weight(map,size,uo,bytes);
    const char *sg = st != UINT_MAX ? weight(map,size,sgo,sb) : NULL;
    const char *su = st != UINT_MAX ? weight(map,size,suo,sb) : NULL;
    if (!g || !u || (st != UINT_MAX && (!sg || !su))) return 0;
    const char *const *p0 = NULL, *const *p1 = NULL;
    if (!stage_experts(map,size,go,uo,bytes/NE,(const int *)sel->ptr,NE,T,NS,true,&p0,&p1)) return 0;
    return moe_mv_dispatch((float *)mid->ptr,(const float *)x->ptr,(const int *)sel->ptr,
                           g,u,sg,su,type,st,NE,T,NS,K,M,false,p0,p1);
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(part,(uint64_t)T*NO*M*4) ||
        !tensor(mid,(uint64_t)T*NO*K*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    const char *w = expert_weight(map,size,off,expert_row_bytes(type,K)*M*NE);
    const char *sw = st != UINT_MAX ? weight(map,size,so,row_bytes(st,K)*M) : NULL;
    if (!w || (st != UINT_MAX && !sw)) return 0;
    const char *const *p0 = NULL, *const *p1 = NULL;
    if (!stage_experts(map,size,off,0,expert_row_bytes(type,K)*M,(const int *)sel->ptr,
                       NE,T,NS,false,&p0,&p1)) return 0;
    return moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,(const int *)sel->ptr,
                           w,NULL,sw,NULL,type,st,NE,T,NS,K,M,true,p0,p1);
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh,
        ds4_gpu_tensor *R, const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4 ||
        !tensor(out,(uint64_t)T*D*4) || !tensor(part,(uint64_t)T*stride*D*4) ||
        !tensor(weights,(uint64_t)T*NS*4) || (sg && !tensor(sg,(uint64_t)T*4)) ||
        (sh && !tensor(sh,(uint64_t)T*D*4)) ||
        (hc && (!tensor(R,(uint64_t)T*hc*D*4) || !tensor(inj,(uint64_t)T*hc*hc*8*4)))) return 0;
    moe_reduce<<<dim3((D+255)/256,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
        hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL, (const float *)part->ptr,
        (const float *)weights->ptr, sg ? (const float *)sg->ptr : NULL, sh ? (const float *)sh->ptr : NULL,
        NS,stride,D,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *sel, uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    using namespace qwen4_cuda;
    if (!T || !NS || NS > NE || !NE || NE > 512 || cap < T || (uint64_t)T*NS > INT_MAX ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    route_selected = (const int *)sel->ptr; route_lists = (const int *)lists->ptr;
    route_counts = (const int *)counts->ptr;
    route_T = T; route_NS = NS; route_NE = NE; route_cap = cap; route_ready = false;
    expert_lists<<<1,256,0,cuda_decode_stream()>>>((int *)lists->ptr,(int *)counts->ptr,(const int *)sel->ptr,T*NS,NE,cap);
    return launched();
}

static int qwen4_moe_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap, bool down) {
    using namespace qwen4_cuda;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T ||
        !tensor(out,(uint64_t)T*NO*M*4) || !tensor(x,(uint64_t)T*(down ? NO : 1)*K*4) ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type,K)*M*NE;
    std::vector<int> fallback_uses, fallback_ids;
    const int *uses = NULL, *ids = NULL;
    const char *const *routed0 = NULL, *const *routed1 = NULL;
    if (expert_cache_enabled(map)) {
        const cudaStream_t stream = cuda_decode_stream();
        if ((!route_ready || down) && route_lists == lists->ptr && route_counts == counts->ptr &&
            route_T == T && route_NS == NS && route_NE == NE && route_cap == cap) {
            if (!route_ready && !route_read(route_selected, T, NS, NE, stream)) return 0;
            route_ready = true;
            ids = route_host; uses = route_host + (uint64_t)T * NS;
        } else {
            /* External callers may supply prebuilt lists. Preserve that API;
             * only the engine/list-builder path can reuse compact routing. */
            fallback_uses.resize(NE); fallback_ids.resize((uint64_t)T * NS, -1);
            std::vector<int> routes((uint64_t)NE * cap);
            if (!cuda_ok(cudaMemcpyAsync(fallback_uses.data(), counts->ptr, NE * sizeof(int),
                         cudaMemcpyDeviceToHost, stream), "Qwen prefill expert usage") ||
                !cuda_ok(cudaMemcpyAsync(routes.data(), lists->ptr, routes.size() * sizeof(int),
                         cudaMemcpyDeviceToHost, stream), "Qwen prefill routes") ||
                !cuda_ok(cudaStreamSynchronize(stream), "Qwen prefill routes ready")) return 0;
            for (unsigned e = 0; e < NE; e++) for (int j = 0; j < fallback_uses[e]; j++) {
                const int r = routes[(uint64_t)e * cap + j];
                if (r >= 0 && (uint64_t)r < (uint64_t)T * NS) fallback_ids[r] = e;
            }
            ids = fallback_ids.data(); uses = fallback_uses.data();
        }
        /* Even when down reuses the host plan, drain the preceding GEMM before
         * eviction can overwrite an allocation it consumes. */
        if (down && !cuda_ok(cudaStreamSynchronize(stream), "Qwen prior experts consumed")) return 0;
        expert_cache_next_epoch();
        expert_cache_observe(o0, bytes / NE, NE, T, NS, ids);
        if (!down) expert_cache_observe(o1, bytes / NE, NE, T, NS, ids);
        for (unsigned e = 0; e < NE; e++) if (uses[e] > 0) {
            expert_cache_touch(o0 + e * (bytes / NE), 1);
            if (!down) expert_cache_touch(o1 + e * (bytes / NE), 1);
        }
    }
    const char *w0 = expert_weight(map,size,o0,bytes), *w1 = down ? NULL : expert_weight(map,size,o1,bytes);
    if (!w0 || (!down && !w1)) return 0;
    if ((tiered_cache(map) || (ds4_env_flag_enabled("DS4_CUDA_QWEN4_PREFILL_STAGE", 0) && T >= 64)) && g_n_gpus <= 1) {
        /* The legacy staging path retains its full-expert layout. Tiered
         * staging uses two bounded groups with explicit consumption events;
         * its storage is separate from the kernel's tile-schedule scratch. */
        cuda_stream_selected_cache &cache = g_stream_selected_cache;
        cache.logical_tier = 0;
        const cudaStream_t stream = cuda_decode_stream();
        if (tiered_cache(map)) {
            /* All current hits were pinned above. Reuse the same immutable
             * expert slots as decode, without copying hits into a second
             * full-layer buffer. Cold experts that cannot enter the cache
             * still execute from transient VRAM; no route is omitted. */
            unsigned width = 64;
            const char *group_env = getenv("DS4_CUDA_QWEN4_COPY_GROUP");
            if (group_env && *group_env) {
                char *end = NULL; const unsigned long v = strtoul(group_env, &end, 10);
                if (end != group_env && !*end && v >= 1 && v <= NE) width = (unsigned)v;
            }
            const bool overlap = !ds4_env_flag_enabled("DS4_CUDA_QWEN4_NO_OVERLAP", 0);
            if (!overlap) width = NE;
            width = std::min(width, NE);
            const unsigned groups = (NE + width - 1u) / width;
            if (!expert_group_ensure(NE, groups)) return 0;
            const unsigned buffers = overlap ? 2u : 1u;
            const uint64_t projection_bytes = (uint64_t)width * (bytes / NE);
            const uint64_t group_bytes = projection_bytes * (down ? 1u : 2u);
            const uint64_t table_offset = (group_bytes * buffers + 255u) & ~255ull;
            if (!cuda_stream_selected_ensure_bytes(&cache.prefill_ptr, &cache.prefill_capacity,
                         table_offset + (uint64_t)NE * 2u * sizeof(char *), "Qwen grouped prefill weights")) return 0;
            const uint64_t count_bytes = (uint64_t)NE * groups * sizeof(int);
            const uint64_t pointer_offset = (count_bytes + sizeof(char *) - 1u) & ~(uint64_t)(sizeof(char *) - 1u);
            int *host_counts = (int *)expert_group_host;
            const char **ptrs = (const char **)(expert_group_host + pointer_offset);
            memset(host_counts, 0, count_bytes);
            memset(ptrs, 0, NE * 2u * sizeof(char *));
            char *table = cache.prefill_ptr + table_offset;
            const cudaStream_t copy = overlap ? expert_copy_stream : stream;
            expert_reads_begin(o0, o1, bytes / NE, NE, width, uses, down);
            int ok = 0;
            for (unsigned group = 0; group < groups; group++) {
                const unsigned first = group * width, last = std::min(first + width, NE);
                unsigned pairs = 0, active = 0;
                for (unsigned e = first; e < last; e++) if (uses[e] > 0) {
                    host_counts[(uint64_t)group * NE + e] = std::min(uses[e], (int)cap);
                    pairs += host_counts[(uint64_t)group * NE + e]; active++;
                }
                if (!pairs) continue;
                const unsigned buffer = group % buffers;
                /* An empty previous group has no writes/consumer. Waiting on
                 * a prior recorded event is harmless; an unrecorded event is
                 * complete by CUDA's contract. */
                if (overlap && group >= buffers && !cuda_ok(cudaStreamWaitEvent(copy,
                        expert_group_consumed[buffer], 0), "Qwen staging reuse")) goto prefill_reads_done;
                char *group_weights = cache.prefill_ptr + buffer * group_bytes;
                for (unsigned w = 0; w < (down ? 1u : 2u); w++) for (unsigned e = first; e < last; e++) {
                    if (uses[e] <= 0) continue;
                    const uint64_t off = (w ? o1 : o0) + e * (bytes / NE);
                    const char *cached = NULL;
                    if (!expert_cache_get(map, off, bytes / NE, copy, &cached)) goto prefill_reads_done;
                    char *dst = group_weights + w * projection_bytes + (e - first) * (bytes / NE);
                    if (!cached && !expert_transfer(map, off, bytes / NE, dst, copy)) goto prefill_reads_done;
                    ptrs[w * NE + e] = cached ? cached : dst;
                }
                int *group_counts = expert_group_counts + (uint64_t)group * NE;
                if (!cuda_ok(cudaMemcpyAsync(group_counts, host_counts + (uint64_t)group * NE,
                             NE * sizeof(int), cudaMemcpyHostToDevice, copy), "Qwen group counts")) goto prefill_reads_done;
                for (unsigned w = 0; w < (down ? 1u : 2u); w++)
                    if (!cuda_ok(cudaMemcpyAsync(table + (w * NE + first) * sizeof(char *),
                                 ptrs + w * NE + first, (last - first) * sizeof(char *),
                                 cudaMemcpyHostToDevice, copy), "Qwen group pointers")) goto prefill_reads_done;
                /* Groups write disjoint (token, slot) outputs. Their reductions
                 * remain in the original final kernel and slot order. The host
                 * wait is replaced by the precise copy -> compute dependency. */
                if (overlap && (!cuda_ok(cudaEventRecord(expert_copy_ready, copy), "Qwen group ready") ||
                    !cuda_ok(cudaStreamWaitEvent(stream, expert_copy_ready, 0), "Qwen group dependency"))) goto prefill_reads_done;
                expert_copy_groups++;
                if (!matrix_dispatch((float *)out->ptr, (const float *)x->ptr,
                        group_weights, down ? NULL : group_weights + projection_bytes,
                        (const int *)lists->ptr, group_counts, type, NE, T, NS, NO, K, M, cap,
                        down, (const char *const *)table, down ? NULL : (const char *const *)table + NE,
                        pairs, active)) goto prefill_reads_done;
                if (overlap && !cuda_ok(cudaEventRecord(expert_group_consumed[buffer], stream),
                                       "Qwen staging consumed")) goto prefill_reads_done;
            }
            ok = 1;
prefill_reads_done:
            const int read_ok = expert_reads_finish();
            return ok && read_ok;
        } else {
            if (!cuda_stream_selected_ensure_bytes(&cache.prefill_ptr, &cache.prefill_capacity,
                             bytes * (down ? 1u : 2u), "Qwen prefill weights")) return 0;
            if (!cuda_ok(cudaMemcpyAsync(cache.prefill_ptr, (const char *)map + o0, bytes,
                                        cudaMemcpyHostToDevice, stream), "Qwen prefill expert transfer")) return 0;
            if (!down && !cuda_ok(cudaMemcpyAsync(cache.prefill_ptr + bytes, (const char *)map + o1, bytes,
                                                  cudaMemcpyHostToDevice, stream), "Qwen prefill up transfer")) return 0;
        }
        w0 = cache.prefill_ptr;
        if (!down) w1 = cache.prefill_ptr + bytes;
    }
    return matrix_dispatch((float *)out->ptr,(const float *)x->ptr,w0,w1,(const int *)lists->ptr,
        (const int *)counts->ptr,type,NE,T,NS,NO,K,M,cap,down,routed0,routed1);
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,go,uo,type,NE,T,NS,NO,K,M,cap,false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    const int ok = qwen4_moe_mm(out,x,lists,counts,map,size,off,0,type,NE,T,NS,NO,K,M,cap,true);
    /* The compact plan belongs to one mid/down pair. A later external call
     * with the same list addresses must read back and drain its predecessor. */
    qwen4_cuda::route_reset();
    return ok;
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed, ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *map, uint64_t size, uint64_t co, uint64_t ao, uint64_t bo, uint64_t so, uint64_t dto,
        uint32_t type, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, hb = (CK-1)*C*4;
    if (!T || !Hk || !Hv || !D || !K || C > UINT_MAX || CK < 2 || CK > 4 ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(state,hb) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *w = weight(map,size,co,C*CK*4);
    if (!w || !ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,ao,type,T,K,Hv) ||
              !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,bo,type,T,K,Hv)) return 0;
    conv<<<(C+255)/256,256,0,cuda_decode_stream()>>>((float *)qkv->ptr,(float *)state->ptr,(const float *)w,
        T,C,CK,true,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2);
    return launched() && ds4_gpu_qwen4_gdn_prep_tensor(qkv,ga,gb,map,size,so,dto,T,Hk,Hv,D);
}

extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) { return 1; }

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    const uint64_t bytes = (uint64_t)T*E*hc*4;
    if (!qwen4_cuda::tensor(next,bytes) || !qwen4_cuda::tensor(R,bytes) || next->ptr == R->ptr ||
        !inj || !oldinj || inj->ptr == oldinj->ptr) return 0;
    if (!cuda_ok(cudaMemcpyAsync(next->ptr,R->ptr,bytes,cudaMemcpyDeviceToDevice,cuda_decode_stream()),"Qwen HC copy")) return 0;
    return ds4_gpu_qwen4_hc_combine_tensor(next,blk,oldinj,T,E,hc) &&
           ds4_gpu_qwen4_hc_norm_tensor(xn,inj,next,map,size,go,io,type,T,E,hc,ni,eps);
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(cat,(uint64_t)(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)E*4) || !tensor(R,(uint64_t)hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    mtp_stage<<<hc+1,256,0,cuda_decode_stream()>>>((float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *proj,
        uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(out,(uint64_t)hc*E*4) || !tensor(proj,(uint64_t)(hc+1)*E*4)) return 0;
    mtp_combine<<<((uint64_t)E*hc+255)/256,256,0,cuda_decode_stream()>>>((float *)out->ptr,(const float *)proj->ptr,E,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N) {
    using namespace qwen4_cuda;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !tensor(out,4) || !tensor(logits,(uint64_t)N*4) || !tensor(scratch,blocks*sizeof(max_pair))) return 0;
    argmax<false><<<blocks,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N);
    argmax<true><<<1,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,NULL,blocks);
    return launched();
}

extern "C" int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const void *map, uint64_t size, const ds4_qwen4_vision_weights *w) {
    using namespace qwen4_cuda;
    if (!out || !patches || !pos_embed || !w || !map || !N || N > 65535 || ds4_gpu_commands_active()) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    std::vector<ds4_gpu_tensor *> buffers;
    auto alloc = [&](uint64_t n) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n*4);
        buffers.push_back(t);
        return t;
    };
    ds4_gpu_tensor *patch = alloc((uint64_t)N*IP), *a0 = alloc((uint64_t)N*E), *a1 = alloc((uint64_t)N*E);
    ds4_gpu_tensor *pos = alloc((uint64_t)N*E), *x = alloc((uint64_t)N*E), *tmp = alloc((uint64_t)N*E);
    ds4_gpu_tensor *qkv = alloc((uint64_t)N*3*E), *q = alloc((uint64_t)N*E), *k = alloc((uint64_t)N*E);
    ds4_gpu_tensor *v = alloc((uint64_t)N*E), *attn = alloc((uint64_t)N*E), *ffn = alloc((uint64_t)N*FF);
    ds4_gpu_tensor *m0 = alloc((uint64_t)merged*ME), *res = alloc((uint64_t)merged*O);
    bool ok = true, active = false;
    for (const auto b : buffers) if (!b) ok = false;
    auto ptr = [](ds4_gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(map,size,off,(uint64_t)n*4); };
    auto mm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return ds4_gpu_qwen4_dense_mm_tensor(dst,src,map,size,off,type,rows,K,M);
    };
    auto norm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        vis_norm<<<N,256,0,cuda_decode_stream()>>>(ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        vis_add<<<((uint64_t)rows*width+255)/256,256,0,cuda_decode_stream()>>>(ptr(dst),
            src ? (const float *)src->ptr : NULL,bias,rows,width,mode);
        return launched();
    };
    do {
        if (!ok) break;
        ok = ds4_gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             ds4_gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok || !(active = ds4_gpu_begin_commands())) { ok = false; break; }
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        vis_patch<<<((uint64_t)N*E+255)/256,256,0,cuda_decode_stream()>>>(ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < DS4_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            vis_qkv<<<N,256,0,cuda_decode_stream()>>>(ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            vis_attention<<<dim3(N,H),32,0,cuda_decode_stream()>>>(ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (active && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) ds4_gpu_tensor_free(*it);
    return ok;
}

/* The grouped tensor-core attention kernel faults on the audited sm_86
 * CUDA 12.4 path. Keep the existing scalar/split kernels for this device;
 * their independent double-precision gates cover the same model shapes. */
static bool qwen4_group_attention_arch_ok(void) {
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
    return !(prop.major == 8 && prop.minor == 6);
}

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * Hkv * D * 2;
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out, n) || !tensor(q, n) || !tensor(gate, n) || !tensor(kc, cb) || !tensor(vc, cb) ||
        (sparse && (!stride || !tensor(sel, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4)))) return 0;
    const unsigned keys = sparse ? stride : pos0 + T;
    if (!partial && T >= 32 && D == 256 && H/Hkv <= 16 &&
        !((uintptr_t)kc->ptr&15) && !((uintptr_t)vc->ptr&15) &&
        ds4_cuda_attn_tokentile_arch_ok() && qwen4_group_attention_arch_ok() &&
        !g_quality_mode && !getenv("DS4_QWEN4_NO_ATTN_MM")) {
        attention_group<<<dim3(Hkv,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
            (const float *)q->ptr,(const float *)gate->ptr,(const __half *)kc->ptr,(const __half *)vc->ptr,
            sparse ? (const int *)sel->ptr : NULL,sparse ? (const unsigned *)count->ptr : NULL,
            H,Hkv,pos0,stride,sparse,scale);
        return launched();
    }
    const unsigned splits = partial ? std::min(64u, (keys + 31) / 32) : 1;
    if (partial && !tensor(partial, (uint64_t)T * H * splits * (D + 2) * 4)) return 0;
    const dim3 grid((H + 3) / 4, T, splits);
#define QWEN_ATTN(DIM) attention<DIM><<<grid, 128, 0, cuda_decode_stream()>>>((float *)out->ptr, \
        partial ? (float *)partial->ptr : NULL, (const float *)q->ptr, (const float *)gate->ptr, \
        (const __half *)kc->ptr, (const __half *)vc->ptr, sparse ? (const int *)sel->ptr : NULL, \
        sparse ? (const unsigned *)count->ptr : NULL, H, Hkv, pos0, stride, sparse, splits, (keys + splits - 1) / splits, scale)
    if (D == 32) { QWEN_ATTN(32); }
    else if (D == 128) { QWEN_ATTN(128); }
    else { QWEN_ATTN(256); }
#undef QWEN_ATTN
    if (!launched()) return 0;
    if (splits > 1) attn_merge<<<dim3(H, T), D, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)partial->ptr, (const float *)gate->ptr, H, D, splits);
    return launched();
}

extern "C" int ds4_gpu_qwen4_cache_prepare(uint32_t ctx, uint32_t prefill,
        const void *map, uint64_t size,
        const uint64_t *offsets, const uint64_t *sizes, uint32_t count,
        uint64_t min_bytes, uint64_t max_bytes) {
    using namespace qwen4_cuda;
    if (!ctx || !prefill || !tiered_cache(map) || !offsets || !sizes || !count) return 0;
    uint64_t dense_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!sizes[i] || offsets[i] > size || sizes[i] > size - offsets[i] ||
            dense_bytes > UINT64_MAX - sizes[i]) return 0;
        dense_bytes += sizes[i];
    }
    if (cache_ctx != ctx || cache_prefill != prefill || dense_required != dense_bytes) {
        /* Free VRAM already reflects all graphs and the auxiliary model.
         * Recompute for a newly allocated context; conversation length alone
         * cannot release a KV allocation made for a larger --ctx. */
        g_model_auto_cache_limit = 0;
        cache_budget_logged = false;
    }
    cache_ctx = ctx; cache_prefill = prefill; dense_required = dense_bytes;
    expert_min_bytes = min_bytes; expert_max_bytes = max_bytes;
    for (uint32_t i = 0; i < count; i++)
        if (!weight(map, size, offsets[i], sizes[i])) return 0;
    return 1;
}

extern "C" uint64_t ds4_gpu_qwen4_expert_state_bytes(void) { return qwen4_cuda::expert_state_bytes(); }
extern "C" int ds4_gpu_qwen4_expert_state_export(void *buffer, uint64_t bytes) { return qwen4_cuda::expert_state_export(buffer, bytes); }
extern "C" int ds4_gpu_qwen4_expert_state_import(const void *buffer, uint64_t bytes) { return qwen4_cuda::expert_state_import(buffer, bytes); }
