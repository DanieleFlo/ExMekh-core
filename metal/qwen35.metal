// Qwen27 hybrid layers. Compared with Strata 46c152a/metal/kernels/gdn*.metal:
// shard the 128x128 state across four row blocks to bound register pressure.
// DS4 keeps head-major FP32 state, four convolution slots, L2(sum+eps), and
// SiLU(z). Strata Flash's interleaved state and sigmoid(z) are not compatible.
struct ds4_qwen35_args {
    uint pos, rows, capacity, spatial, snapshots;
};

static inline float ds4_qwen35_sum(float x, threadgroup float *partial, ushort tid, ushort count) {
    float s = simd_sum(x);
    if ((tid & 31) == 0)
        partial[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = 0;
    for (ushort i = 0; i < count / 32; ++i)
        s += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// Q and K are normalized in place, with 64 rotary dimensions out of 256.
// Spatial positions use the pinned interleaving [11,11,10] across 32 pairs.
kernel void ds4_qwen35_prepare_q(constant ds4_qwen35_args &a [[buffer(0)]],
                                 device float *q [[buffer(1)]],
                                 device const float *norm [[buffer(2)]],
                                 device const uint *positions [[buffer(3)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 ushort d [[thread_index_in_threadgroup]]) {
    threadgroup float part[8], values[256];
    const ulong at = ulong(group.y) * 12288 + group.x * 512 + d;
    float v = q[at];
    float sum = ds4_qwen35_sum(v * v, part, d, 256);
    values[d] = v * rsqrt(sum / 256.0f + 1.e-6f) * norm[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = values[d];
    if (d < 64) {
        uint pair = d & 31;
        uint p = a.spatial ? positions[group.y * 3 + pair % 3] : a.pos + group.y;
        float angle = float(p) * pow(1.e7f, -float(pair) / 32.0f);
        float sn = sin(angle), cs = cos(angle);
        v = d < 32 ? values[pair] * cs - values[pair + 32] * sn
                   : values[pair] * sn + values[pair + 32] * cs;
    }
    q[at] = v;
}

kernel void ds4_qwen35_prepare_kv(constant ds4_qwen35_args &a [[buffer(0)]],
                                  device float *k [[buffer(1)]],
                                  device const float *v [[buffer(2)]],
                                  device float *kc [[buffer(3)]], device float *vc [[buffer(4)]],
                                  device const float *norm [[buffer(5)]],
                                  device const uint *positions [[buffer(6)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  ushort d [[thread_index_in_threadgroup]]) {
    threadgroup float part[8], values[256];
    const ulong at = ulong(group.y) * 1024 + group.x * 256 + d;
    float x = k[at];
    float sum = ds4_qwen35_sum(x * x, part, d, 256);
    values[d] = x * rsqrt(sum / 256.0f + 1.e-6f) * norm[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x = values[d];
    if (d < 64) {
        uint pair = d & 31;
        uint p = a.spatial ? positions[group.y * 3 + pair % 3] : a.pos + group.y;
        float angle = float(p) * pow(1.e7f, -float(pair) / 32.0f);
        float sn = sin(angle), cs = cos(angle);
        x = d < 32 ? values[pair] * cs - values[pair + 32] * sn
                   : values[pair] * sn + values[pair + 32] * cs;
    }
    k[at] = x;
    ulong dst = ulong(a.pos + group.y) * 1024 + group.x * 256 + d;
    kc[dst] = x;
    vc[dst] = v[at];
}

// Online softmax bounds scratch independently of context length. Each lane
// owns a value dimension; all 256 lanes cooperate on the QK dot product.
kernel void ds4_qwen35_attention(constant ds4_qwen35_args &a [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 device const float *q [[buffer(2)]],
                                 device const float *kc [[buffer(3)]],
                                 device const float *vc [[buffer(4)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 ushort d [[thread_index_in_threadgroup]]) {
    threadgroup float part[8];
    ulong qi = ulong(group.y) * 12288 + group.x * 512 + d;
    float query = q[qi], acc = 0, total = 0, maximum = -MAXFLOAT;
    for (uint key = 0; key <= a.pos + group.y; ++key) {
        ulong at = ulong(key) * 1024 + (group.x / 6) * 256 + d;
        float score = ds4_qwen35_sum(query * kc[at], part, d, 256) / 16.0f;
        float next = max(maximum, score);
        float old = exp(maximum - next), weight = exp(score - next);
        acc = acc * old + vc[at] * weight;
        total = total * old + weight;
        maximum = next;
    }
    out[(ulong(group.y) * 24 + group.x) * 256 + d] = acc / total / (1.0f + exp(-q[qi + 256]));
}

kernel void ds4_qwen35_conv(constant ds4_qwen35_args &a [[buffer(0)]],
                            device float *qkv [[buffer(1)]], device float *history [[buffer(2)]],
                            device const float *weights [[buffer(3)]],
                            device float *snapshots [[buffer(4)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= 10240)
        return;
    float4 h = float4(history[c * 4], history[c * 4 + 1], history[c * 4 + 2], history[c * 4 + 3]);
    float4 w = float4(weights[c * 4], weights[c * 4 + 1], weights[c * 4 + 2], weights[c * 4 + 3]);
    for (uint t = 0; t < a.rows; ++t) {
        ulong at = ulong(t) * 10240 + c;
        h = float4(h.y, h.z, h.w, qkv[at]);
        float y = h.x * w.x + h.y * w.y + h.z * w.z + h.w * w.w;
        qkv[at] = y / (1.0f + exp(-y));
        if (a.snapshots && t + 1 < a.rows)
            for (uint j = 0; j < 4; ++j)
                snapshots[(ulong(t) * 10240 + c) * 4 + j] = h[j];
    }
    for (uint j = 0; j < 4; ++j)
        history[c * 4 + j] = h[j];
}

[[max_total_threads_per_threadgroup(512)]]
kernel void
ds4_qwen35_gdn(constant ds4_qwen35_args &a [[buffer(0)]], device float *out [[buffer(1)]],
               device const float *qkv [[buffer(2)]], device const float *z [[buffer(3)]],
               device const float *alpha [[buffer(4)]], device const float *beta [[buffer(5)]],
               device float *state [[buffer(6)]], device const float *decay_weights [[buffer(7)]],
               device const float *dt [[buffer(8)]], device const float *norm [[buffer(9)]],
               device float *snapshots [[buffer(10)]], uint head [[threadgroup_position_in_grid]],
               ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float q[128], k[128], partial[16], shards[4][128];
    const uint d = tid % 128, block = tid / 128, first = block * 32;
    const ulong base = ulong(head) * 128 * 128;
    float st[32];
    for (uint r = 0; r < 32; ++r)
        st[r] = state[base + (first + r) * 128 + d];
    for (uint t = 0; t < a.rows; ++t) {
        ulong at = ulong(t) * 10240;
        float qv = qkv[at + (head % 16) * 128 + d];
        float kv = qkv[at + 2048 + (head % 16) * 128 + d];
        float qs = ds4_qwen35_sum(block == 0 ? qv * qv : 0, partial, tid, 512);
        float ks = ds4_qwen35_sum(block == 0 ? kv * kv : 0, partial, tid, 512);
        if (block == 0) {
            q[d] = qv * rsqrt(qs + 1.e-6f);
            k[d] = kv * rsqrt(ks + 1.e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x = alpha[ulong(t) * 48 + head] + dt[head];
        float e = exp(min(x, 20.0f)), u = 1.0f + e;
        float sp = x > 20 ? x : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
        float decay = exp(decay_weights[head] * sp);
        float be = 1.0f / (1.0f + exp(-beta[ulong(t) * 48 + head]));
        float pred = 0;
        for (uint r = 0; r < 32; ++r)
            pred += decay * st[r] * k[first + r];
        shards[block][d] = pred;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pred = (shards[0][d] + shards[1][d]) + (shards[2][d] + shards[3][d]);
        float delta = be * (qkv[at + 4096 + head * 128 + d] - pred), y = 0;
        for (uint r = 0; r < 32; ++r) {
            st[r] = decay * st[r] + k[first + r] * delta;
            y += st[r] * q[first + r] * 0.08838834764831845f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shards[block][d] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        y = (shards[0][d] + shards[1][d]) + (shards[2][d] + shards[3][d]);
        float square = ds4_qwen35_sum(block == 0 ? y * y : 0, partial, tid, 512);
        if (block == 0) {
            ulong oi = (ulong(t) * 48 + head) * 128 + d;
            float gate = z[oi];
            out[oi] = y * rsqrt(square / 128.0f + 1.e-6f) * norm[d] * gate / (1.0f + exp(-gate));
        }
        if (a.snapshots && t + 1 < a.rows)
            for (uint r = 0; r < 32; ++r)
                snapshots[ulong(t) * 48 * 128 * 128 + base + (first + r) * 128 + d] = st[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 32; ++r)
        state[base + (first + r) * 128 + d] = st[r];
}
