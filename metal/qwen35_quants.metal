// IQ unit decoders adapted from Niko1221/Strata 46c152a1310f77cd2f0ce2c7cd213962df14fe83,
// metal/kernels/quant.metal. MIT: see third_party/strata/LICENSE.
// Reuse the per-threadgroup codebook staging and decode 16/32 weights per lane.
// DS4 consumes original FP32 activations: no prepared/int8 activation variant.
namespace ds4_qwen_ud {
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_Q3_K 11
template <int F> struct fmt;
template <> struct fmt<17> {
    enum { QK = 256, BB = 74, UNIT = 16, TAB = 1056 };
};
template <> struct fmt<18> {
    enum { QK = 256, BB = 98, UNIT = 16, TAB = 288 };
};
template <> struct fmt<20> {
    enum { QK = 32, BB = 18, UNIT = 32, TAB = 256 };
};
template <> struct fmt<21> {
    enum { QK = 256, BB = 110, UNIT = 16, TAB = 544 };
};
template <> struct fmt<22> {
    enum { QK = 256, BB = 82, UNIT = 16, TAB = 2080 };
};
template <> struct fmt<23> {
    enum { QK = 256, BB = 136, UNIT = 32, TAB = 256 };
};
template <> struct fmt<11> {
    enum { QK = 256, BB = 110, UNIT = 32, TAB = 1 };
};
template <> struct fmt<13> {
    enum { QK = 256, BB = 176, UNIT = 32, TAB = 1 };
};
template <> struct fmt<14> {
    enum { QK = 256, BB = 210, UNIT = 32, TAB = 1 };
};
static inline uint rd32(device const uchar *p) {
    device const ushort *s = (device const ushort *)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(device const uchar *p) { return float(*(device const half *)p); }
static inline uint ksg(threadgroup const uint *tab, int at, uint i) {
    return ((threadgroup const uchar *)(tab + at))[i];
}
static inline uint g_iq3xxs(threadgroup const uint *t, uint i) { return t[i]; }
static inline uint g_iq3s(threadgroup const uint *t, uint i) { return t[i]; }
static inline uint g_iq2xs(threadgroup const uint *t, uint i, int w) { return t[2 * i + w]; }
static inline uint g_iq2s(threadgroup const uint *t, uint i, int w) { return t[2 * i + w]; }
template <int F> static inline void load_tab(threadgroup uint *tab, ushort tid, ushort nth) {
    if (F == 17 || F == 22) {
        const uint n = F == 17 ? 512 : 1024;
        for (uint i = tid; i < n; i += nth) {
            ulong g = F == 17 ? cuda_iq2xs_grid[i] : cuda_iq2s_grid[i];
            tab[2 * i] = uint(g);
            tab[2 * i + 1] = uint(g >> 32);
        }
    } else if (F == 18 || F == 21) {
        uint n = F == 18 ? 256 : 512;
        for (uint i = tid; i < n; i += nth)
            tab[i] = F == 18 ? cuda_iq3xxs_grid[i] : cuda_iq3s_grid[i];
    } else if (F == 20 || F == 23) {
        for (uint i = tid; i < 256; i += nth)
            tab[i] = as_type<uint>(
                half2(half(cuda_kvalues_iq4nl[i & 15]), half(cuda_kvalues_iq4nl[i >> 4])));
    }
    if (F == 17 || F == 18) {
        uint at = F == 17 ? 1024 : 256;
        threadgroup uchar *signs = (threadgroup uchar *)(tab + at);
        for (uint i = tid; i < 128; i += nth)
            signs[i] = uchar(i | ((popcount(i) & 1) << 7));
    }
}
template <int F>
static inline float deq(device const uchar *b, int ui, thread float *q,
                        threadgroup const uint *tab);
static inline void put_signed4(thread float *q, uint g, uint s) {
    const uchar4 v = as_type<uchar4>(g);
    q[0] = (s & 1) ? -float(v[0]) : float(v[0]);
    q[1] = (s & 2) ? -float(v[1]) : float(v[1]);
    q[2] = (s & 4) ? -float(v[2]) : float(v[2]);
    q[3] = (s & 8) ? -float(v[3]) : float(v[3]);
}
template <>
inline float deq<T_IQ2_XS>(device const uchar *b, int ui, thread float *q,
                           threadgroup const uint *tab) {
    const int ib = ui >> 1, h = ui & 1;
    device const ushort *qs = (device const ushort *)(b + 2);
    for (int l2 = 0; l2 < 2; ++l2) {
        const uint v = qs[4 * ib + 2 * h + l2];
        const uint idx = v & 511, s = ksg(tab, 1024, v >> 9);
        put_signed4(q + 8 * l2, g_iq2xs(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2xs(tab, idx, 1), s >> 4);
    }
    const uint sc = b[66 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}
template <>
inline float deq<T_IQ3_XXS>(device const uchar *b, int ui, thread float *q,
                            threadgroup const uint *tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint aux = rd32(b + 2 + 64 + 4 * ib);
    device const uchar *qs = b + 2 + 8 * ib + 4 * h;
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint s = ksg(tab, 256, (aux >> (7 * l)) & 127);
        put_signed4(q + 8 * l2, g_iq3xxs(tab, qs[2 * l2]), s);
        put_signed4(q + 8 * l2 + 4, g_iq3xxs(tab, qs[2 * l2 + 1]), s >> 4);
    }
    return rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
}
template <>
inline float deq<T_IQ4_NL>(device const uchar *b, int, thread float *q,
                           threadgroup const uint *tab) {
    device const ushort *p = (device const ushort *)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    return rdh(b);
}
template <>
inline float deq<T_IQ3_S>(device const uchar *b, int ui, thread float *q,
                          threadgroup const uint *tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint i1 = uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
        const uint s = b[74 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq3s(tab, i1), s);
        put_signed4(q + 8 * l2 + 4, g_iq3s(tab, i2), s >> 4);
    }
    const uint sc = b[106 + (ib >> 1)];
    return rdh(b) * float(1 + 2 * ((sc >> (4 * (ib & 1))) & 15));
}
template <>
inline float deq<T_IQ2_S>(device const uchar *b, int ui, thread float *q,
                          threadgroup const uint *tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = uint(b[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint s = b[2 + 32 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq2s(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2s(tab, idx, 1), s >> 4);
    }
    const uint sc = b[74 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}
template <>
inline float deq<T_IQ4_XS>(device const uchar *b, int ui, thread float *q,
                           threadgroup const uint *tab) {
    device const ushort *p = (device const ushort *)(b + 8 + 16 * ui);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    const uint sh = *(device const ushort *)(b + 2);
    const int ls =
        int((b[4 + (ui >> 1)] >> (4 * (ui & 1))) & 15) | int(((sh >> (2 * ui)) & 3) << 4);
    return rdh(b) * float(ls - 32);
}
template <>
inline float deq<11>(device const uchar *b, int ui, thread float *q, threadgroup const uint *) {
    device const uchar *sc = b + 96;
    float d = rdh(b + 108);
    for (uint j = 0; j < 32; j++) {
        uint i = uint(ui) * 32 + j, n = i / 128, l = i % 32, group = (i % 128) / 32,
             is = 8 * n + 2 * group + l / 16;
        uint scale = (is < 8 ? sc[is] & 15 : sc[is - 8] >> 4) |
                     (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
        int code = int((b[32 + 32 * n + l] >> (2 * group)) & 3) -
                   ((b[l] & (1 << (4 * n + group))) ? 0 : 4);
        q[j] = d * float(int(scale) - 32) * float(code);
    }
    return 1;
}
template <>
inline float deq<13>(device const uchar *b, int ui, thread float *q, threadgroup const uint *) {
    uint group = uint(ui);
    device const uchar *s = b + 4;
    uint sc = group < 4 ? s[group] & 63 : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
    uint mn = group < 4 ? s[group + 4] & 63 : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
    float d = rdh(b) * float(sc), m = rdh(b + 2) * float(mn);
    for (uint j = 0; j < 32; j++) {
        uint code = ((b[48 + (group / 2) * 32 + j] >> (4 * (group & 1))) & 15) |
                    (((b[16 + j] >> group) & 1) << 4);
        q[j] = d * float(code) - m;
    }
    return 1;
}
template <>
inline float deq<14>(device const uchar *b, int ui, thread float *q, threadgroup const uint *) {
    float d = rdh(b + 208);
    device const char *sc = (device const char *)(b + 192);
    for (uint j = 0; j < 32; j++) {
        uint i = uint(ui) * 32 + j, n = i / 128, l = i % 32, quarter = (i % 128) / 32;
        uint lo = (b[n * 64 + (quarter & 1) * 32 + l] >> (quarter >= 2 ? 4 : 0)) & 15;
        uint hi = (b[128 + n * 32 + l] >> (2 * quarter)) & 3;
        q[j] = d * float(sc[n * 8 + 2 * quarter + l / 16]) * float(int(lo | (hi << 4)) - 32);
    }
    return 1;
}
struct Args {
    uint in_dim, out_dim, rows, row_bytes, vocab, token, gather;
};
/* Prefill shares each decoded unit across four rows; the depth-four verifier
 * uses five. Both retain the same lane/unit/FMA order for each row. */
template <int F, int ROWS>
kernel void
project(constant Args &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = group.x * 4 + sg;
    if (r >= a.out_dim)
        return;
    float acc[ROWS] = {};
    for (uint u = lane; u < a.in_dim / fmt<F>::UNIT; u += 32) {
        float q[fmt<F>::UNIT];
        float scale =
            deq<F>(w + ulong(r) * a.row_bytes + ulong(u / (fmt<F>::QK / fmt<F>::UNIT)) * fmt<F>::BB,
                   u % (fmt<F>::QK / fmt<F>::UNIT), q, tab);
        for (uint t = 0; t < ROWS; t++)
            if (group.y * ROWS + t < a.rows) {
                float sum = 0;
                ulong at = ulong(group.y * ROWS + t) * a.in_dim + u * fmt<F>::UNIT;
                for (uint j = 0; j < fmt<F>::UNIT; j++)
                    sum = fma(q[j], x[at + j], sum);
                acc[t] += scale * sum;
            }
    }
    for (uint t = 0; t < ROWS; t++) {
        float sum = simd_sum(acc[t]);
        if (lane == 0 && group.y * ROWS + t < a.rows)
            y[ulong(group.y * ROWS + t) * a.out_dim + r] = sum;
    }
}
/* Each SIMD32 owns two output rows and reuses their activation loads. Keep
 * the original lane/unit assignment, serial FMA and SIMD sum for each row:
 * changing that reduction changes ordinary versus speculative logits. */
template <int F>
kernel void project_decode_pair(constant Args &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
                             device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = group.x * 8 + 2 * (tid / 32);
    if (row >= a.out_dim)
        return;
    uint next = min(row + 1, a.out_dim - 1);
    float acc0 = 0, acc1 = 0;
    for (uint u = tid % 32; u < a.in_dim / fmt<F>::UNIT; u += 32) {
        float q0[fmt<F>::UNIT], q1[fmt<F>::UNIT];
        float scale0 = deq<F>(
            w + ulong(row) * a.row_bytes + ulong(u / (fmt<F>::QK / fmt<F>::UNIT)) * fmt<F>::BB,
            u % (fmt<F>::QK / fmt<F>::UNIT), q0, tab);
        float scale1 = deq<F>(
            w + ulong(next) * a.row_bytes + ulong(u / (fmt<F>::QK / fmt<F>::UNIT)) * fmt<F>::BB,
            u % (fmt<F>::QK / fmt<F>::UNIT), q1, tab);
        float sum0 = 0, sum1 = 0;
        for (uint j = 0; j < fmt<F>::UNIT; j++) {
            float value = x[u * fmt<F>::UNIT + j];
            sum0 = fma(q0[j], value, sum0);
            sum1 = fma(q1[j], value, sum1);
        }
        acc0 += scale0 * sum0;
        acc1 += scale1 * sum1;
    }
    float result0 = simd_sum(acc0), result1 = simd_sum(acc1);
    if (tid % 32 == 0) {
        y[row] = result0;
        if (row + 1 < a.out_dim)
            y[row + 1] = result1;
    }
}
template <int F>
kernel void gather(constant Args &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
                   device const int *ids [[buffer(2)]], device float *y [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint u = group.x * 64 + tid;
    if (u >= a.in_dim / fmt<F>::UNIT)
        return;
    int token = a.gather ? ids[group.y] : int(a.token);
    if (token < 0 || uint(token) >= a.vocab)
        return;
    float q[fmt<F>::UNIT];
    float scale =
        deq<F>(w + ulong(token) * a.row_bytes + ulong(u / (fmt<F>::QK / fmt<F>::UNIT)) * fmt<F>::BB,
               u % (fmt<F>::QK / fmt<F>::UNIT), q, tab);
    for (uint j = 0; j < fmt<F>::UNIT; j++)
        y[ulong(group.y) * a.in_dim + u * fmt<F>::UNIT + j] = scale * q[j];
}
#define DS4_UD(F)                                                                                  \
    template [[host_name("ds4_qwen_ud_project_" #F)]] kernel decltype(project<F, 4>) project<F, 4>; \
    template [[host_name("ds4_qwen_ud_project5_" #F)]] kernel decltype(project<F, 5>) project<F, 5>; \
    template [[host_name("ds4_qwen_ud_project_pair_" #F)]] kernel decltype(project_decode_pair<F>) project_decode_pair<F>; \
    template [[host_name("ds4_qwen_ud_gather_" #F)]] kernel decltype(gather<F>) gather<F>;
DS4_UD(11) DS4_UD(13) DS4_UD(14) DS4_UD(17) DS4_UD(18) DS4_UD(20) DS4_UD(21) DS4_UD(22) DS4_UD(23)
#undef DS4_UD
} // namespace ds4_qwen_ud
