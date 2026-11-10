#include <metal_stdlib>
#include "../ds4_numeric_metal_args.h"
using namespace metal;
namespace ds4_nm {
inline float rd(device const uchar *p, ulong i, uint t) {
    return t == 0 ? ((device const float *)p)[i]
                  : as_type<float>(uint(((device const ushort *)p)[i]) << 16);
}
inline float rnd(float f, uint t) {
    if (!t)
        return f;
    uint u = as_type<uint>(f);
    u = (u & 0x7fffffff) > 0x7f800000 ? (u | 0x400000) : u + 0x7fff + ((u >> 16) & 1);
    return as_type<float>(u & 0xffff0000);
}
// Metal's tanh implementation can overflow for large positive arguments.
// Beyond +/-9 the correctly rounded F32 result is already +/-1.
inline float stable_tanh(float x) { return abs(x) >= 9 ? copysign(1.0f, x) : tanh(x); }
inline void wr(device uchar *p, ulong i, uint t, float f) {
    if (!t)
        ((device float *)p)[i] = f;
    else
        ((device ushort *)p)[i] = ushort(as_type<uint>(rnd(f, t)) >> 16);
}
inline float reduce(float x, threadgroup float *s, uint tid, bool maximum = false) {
    uint lane = tid & 31, simd = tid >> 5;
    x = maximum ? simd_max(x) : simd_sum(x);
    if (!lane)
        s[simd] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x = tid < 8 ? s[tid] : (maximum ? -INFINITY : 0.0f);
    x = maximum ? simd_max(x) : simd_sum(x);
    if (!tid)
        s[0] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s[0];
}
} // namespace ds4_nm
using namespace ds4_nm;
#include "numeric_ordered.metal"
// erfc approximation (max absolute error below 1.3e-7); Metal has no erf.
// Evaluate the small negative tail directly to avoid 1-erf cancellation.
inline float nm_gelu(float z) {
    float v = abs(z) * 0.7071067811865475f, t = 1 / (1 + 0.5f * v);
    float c =
        t * exp(-v * v - 1.26551223f +
                t * (1.00002368f +
                     t * (0.37409196f +
                          t * (0.09678418f +
                               t * (-0.18628806f +
                                    t * (0.27886807f +
                                         t * (-1.13520398f +
                                              t * (1.48851587f +
                                                   t * (-0.82215223f + t * 0.17087277f)))))))));
    return 0.5f * z * (z >= 0 ? 2 - c : c);
}
kernel void nm_point(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                     device const uchar *x [[buffer(2)]], device const uchar *b [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0])
        return;
    if(a.t[1]==30 && a.t[2]==30 && (a.u[0]==2 || a.u[0]==3 || a.u[0]==4) &&
       nm_bf16_tiny_activation(y,x,i))
        return;
    float z = rd(x, i, a.t[2]);
    switch (a.u[0]) {
    case 0:
        z += rd(b, i % a.n[1], a.t[3]);
        break;
    case 1:
        z *= rd(b, i % a.n[1], a.t[3]);
        break;
    case 2:
        z = z / (1 + exp(-z));
        break;
    case 3:
        z = a.t[1]==30 && a.t[2]==30 ? nm_bf16_gelu_erf(z) : nm_gelu(z);
        break;
    case 4:
        z = a.t[1]==30 && a.t[2]==30 ? nm_bf16_gelu_tanh(z) :
            0.5f * z * (1 + stable_tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
        break;
    case 5:
        z = 1 / (1 + exp(-z));
        break;
    case 6:
        z *= a.f[0];
        break;
    case 8:
        z = max(z, 0.0f);
        break;
    case 9:
        z = z > 20 ? z : log(1 + exp(z));
        break;
    case 10:
        z = (as_type<uint>(z)&0x7fffffff)<0x00800000 ? z : nm_sleef_tanh(z);
        break;
    case 11:
        z = min(z, rd(b, i % a.n[1], a.t[3]));
        break;
    case 12:
        z = max(z, rd(b, i % a.n[1], a.t[3]));
        break;
    case 13:
        z = nm_sleef_sin(z);
        break;
    case 14:
        z = nm_sleef_cos(z);
        break;
    }
    wr(y, i, a.t[1], z);
}
/* General norms retain the fixed SIMD reduction. BF16 RMS uses its separate
 * ordered reduction without changing this F32/LayerNorm path. */
kernel void nm_norm(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                    device const uchar *x [[buffer(2)]], device const uchar *w [[buffer(3)]],
                    device const uchar *b [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[8];
    uint width = a.u[0], kind = a.u[1], mode = a.u[2];
    float sum = 0;
    if (kind == 1)
        for (uint j = tid; j < width; j += 256)
            sum += rd(x, ulong(r) * width + j, a.t[2]);
    float mean = reduce(sum, s, tid) / width;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0;
    for (uint j = tid; j < width; j += 256) {
        float v = rd(x, ulong(r) * width + j, a.t[2]) - mean;
        ss += v * v;
    }
    float var = reduce(ss, s, tid),
          div = kind == 2 ? max(sqrt(var), a.f[0]) : sqrt(var / width + a.f[0]);
    for (uint j = tid; j < width; j += 256) {
        float z = (rd(x, ulong(r) * width + j, a.t[2]) - mean) / div;
        if (a.u[3])
            z = rnd(z, a.t[2]);
        if (mode)
            z *= rd(w, j, a.t[3]) + (mode == 2 ? 1.0f : 0.0f);
        if (a.u[4])
            z += rd(b, j, a.t[4]);
        wr(y, ulong(r) * width + j, a.t[1], z);
    }
}
kernel void nm_channel(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                       device const uchar *x [[buffer(2)]], device const uchar *w [[buffer(3)]],
                       uint p [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[8];
    uint channels = a.u[0], spatial = a.u[1];
    float sum = 0;
    for (uint c = tid; c < channels; c += 256) {
        float v = rd(x, ulong(c) * spatial + p, a.t[2]);
        sum += v * v;
    }
    sum = reduce(sum, s, tid);
    float div = a.u[2] ? sqrt(sum / channels + a.f[0]) : max(sqrt(sum), a.f[0]);
    for (uint c = tid; c < channels; c += 256) {
        float z = rd(x, ulong(c) * spatial + p, a.t[2]) / div;
        if (a.u[2])
            z *= rd(w, c, a.t[3]);
        wr(y, ulong(p) * channels + c, a.t[1], z);
    }
}
kernel void nm_rope(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                    device const uchar *x [[buffer(2)]], device const uchar *co [[buffer(3)]],
                    device const uchar *si [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0] / 2)
        return;
    uint dim = a.u[1], d = i % (dim / 2);
    ulong h = i / (dim / 2), ia = h * dim + (a.u[2] ? 2 * d : d), ib = ia + (a.u[2] ? 1 : dim / 2),
          p = h / a.u[0] * (dim / 2) + d;
    float c = rd(co, p, a.t[3]), sn = rd(si, p, a.t[4]), u = rd(x, ia, a.t[2]),
          v = rd(x, ib, a.t[2]);
    float uc = u * c, vs = v * sn, vc = v * c, us = u * sn;
    if (a.u[3] == 1) {
        uc = rnd(uc, a.t[2]);
        vs = rnd(vs, a.t[2]);
        vc = rnd(vc, a.t[2]);
        us = rnd(us, a.t[2]);
    }
    wr(y, ia, a.t[1], a.u[3] == 2 ? fma(u, c, -vs) : uc - vs);
    wr(y, ib, a.t[1], a.u[3] == 2 ? fma(u, sn, vc) : vc + us);
}
kernel void nm_reorder(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                       device const uchar *x [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0])
        return;
    ulong rem = i, index = 0;
    for (uint d = a.u[0]; d > 0; d--) {
        index += (rem % a.shape[d - 1]) * a.stride[d - 1];
        rem /= a.shape[d - 1];
    }
    wr(y, i, a.t[1], rd(x, index, a.t[2]));
}
kernel void nm_pool(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                    device const uchar *x [[buffer(2)]], device const uint *indices [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0])
        return;
    uint r = i / a.u[1], d = i % a.u[1];
    float sum = 0;
    for (uint j = 0; j < a.u[0]; j++) {
        uint ix = indices[ulong(r) * a.u[0] + j];
        if (ix != UINT_MAX)
            sum += rd(x, ulong(ix) * a.u[1] + d, a.t[2]);
    }
    wr(y, i, a.t[1], sum / a.f[0]);
}
kernel void nm_nearest(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                       device const uchar *x [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0])
        return;
    ulong rem = i, index = 0, step = 1;
    for (int d = 2; d >= 0; d--) {
        ulong p = rem % a.u[3 + d];
        rem /= a.u[3 + d];
        index += ((2 * p + 1) * a.u[d] / (2ul * a.u[3 + d])) * step;
        step *= a.u[d];
    }
    wr(y, i, a.t[1], rd(x, rem * step + index, a.t[2]));
}
/* Eight keys are reduced concurrently, one SIMD group per key. There is no
 * quadratic score allocation. A second pass preserves normalized BF16
 * probability rounding; Strata's one-pass FP32 recipe cannot retain it. */
kernel void nm_attention(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                         device const uchar *q [[buffer(2)]], device const uchar *k [[buffer(3)]],
                         device const uchar *v [[buffer(4)]], device const uchar *rel [[buffer(5)]],
                         device const uint *bounds [[buffer(6)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scores[8], prob[8], stats[2];
    uint dim = a.u[5], heads = a.u[3], kvh = a.u[4], queries = a.u[1], keys = a.u[2], h = g % heads,
         row = g / heads, bt = row / queries, r = row % queries, lo = bounds[2 * row],
         hi = bounds[2 * row + 1], warp = tid / 32, lane = tid % 32;
    ulong qi = ulong(g) * dim;
    float maximum = -INFINITY, total = 0, acc0 = 0, acc1 = 0;
    for (uint pass = 0; pass < 2; pass++) {
        for (uint base = lo; base < hi; base += 8) {
            uint key = base + warp;
            float z = -INFINITY;
            if (key < hi) {
                ulong ki = ((ulong(bt) * keys + key) * kvh + h / (heads / kvh)) * dim;
                float dot = 0, relative = 0;
                long ri = long(key) - long(r) + as_type<int>(a.u[9]);
                for (uint d = lane; d < dim; d += 32) {
                    float qv = rd(q, qi + d, a.t[2]);
                    dot += qv * rd(k, ki + d, a.t[3]);
                    if (a.u[8] && ri >= 0 && ri < a.u[8])
                        relative += qv * rd(rel, (ulong(ri) * heads + h) * dim + d, a.t[5]);
                }
                dot = simd_sum(dot);
                relative = simd_sum(relative);
                dot += relative;
                if (a.u[6])
                    dot = rnd(dot, a.t[2]);
                z = dot * a.f[0];
                if (a.u[6])
                    z = rnd(z, a.t[2]);
                if (a.f[1] > 0)
                    z = a.f[1] * stable_tanh(z / a.f[1]);
            }
            if (!lane)
                scores[warp] = z;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!pass) {
                if (!tid) {
                    float m = -INFINITY, s = 0;
                    for (uint j = 0; j < 8; j++)
                        m = max(m, scores[j]);
                    for (uint j = 0; j < 8 && base + j < hi; j++)
                        s += exp(scores[j] - m);
                    float next = max(maximum, m);
                    stats[0] = next;
                    stats[1] = total * exp(maximum - next) + s * exp(m - next);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maximum = stats[0];
                total = stats[1];
            } else {
                if (tid < 8) {
                    float p = base + tid < hi ? exp(scores[tid] - maximum) / total : 0;
                    if (a.u[7])
                        p = rnd(p, a.t[2]);
                    prob[tid] = p;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0; j < 8 && base + j < hi; j++) {
                    ulong vi = ((ulong(bt) * keys + base + j) * kvh + h / (heads / kvh)) * dim;
                    if (tid < dim)
                        acc0 += prob[j] * rd(v, vi + tid, a.t[4]);
                    if (tid + 256 < dim)
                        acc1 += prob[j] * rd(v, vi + tid + 256, a.t[4]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < dim)
        wr(y, qi + tid, a.t[1], acc0);
    if (tid + 256 < dim)
        wr(y, qi + tid + 256, a.t[1], acc1);
}
kernel void nm_epilogue(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                        device const float *x [[buffer(2)]], device const uchar *bias [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i < a.n[0])
        wr(y, i, a.t[1], x[i] + (a.u[1] ? rd(bias, i % a.u[0], a.t[3]) : 0.0f));
}
/* Direct grouped NCDHW convolution keeps the workspace independent of image
 * size. The 1x1 case is routed to matrix multiplication by the host. */
kernel void nm_conv(constant ds4_nm_args &a [[buffer(0)]], device uchar *y [[buffer(1)]],
                    device const uchar *x [[buffer(2)]], device const uchar *w [[buffer(3)]],
                    device const uchar *bias [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    ulong i = gid;
    if (i >= a.n[0])
        return;
    uint icg = a.u[0] / a.u[2], ocg = a.u[1] / a.u[2], out[3] = {a.u[21], a.u[22], a.u[23]}, p[3];
    ulong spatial = ulong(out[0]) * out[1] * out[2], ins = ulong(a.u[3]) * a.u[4] * a.u[5],
          ks = ulong(a.u[6]) * a.u[7] * a.u[8], rem = i % spatial;
    uint oc = (i / spatial) % a.u[1], bt = i / (spatial * a.u[1]);
    for (int d = 2; d >= 0; d--) {
        p[d] = rem % out[d];
        rem /= out[d];
    }
    float sum = 0;
    for (uint c = 0; c < icg; c++)
        for (ulong j = 0; j < ks; j++) {
            ulong r = j, xi = 0;
            bool ok = true;
            long coord[3];
            for (int d = 2; d >= 0; d--) {
                coord[d] = long(p[d]) * a.u[9 + d] + long(r % a.u[6 + d]) * a.u[12 + d] -
                           long(a.u[15 + d]);
                r /= a.u[6 + d];
                if (coord[d] < 0 || coord[d] >= a.u[3 + d])
                    ok = false;
            }
            if (!ok)
                continue;
            for (uint d = 0; d < 3; d++)
                xi = xi * a.u[3 + d] + coord[d];
            uint channel = oc / ocg * icg + c;
            sum += rd(x, (ulong(bt) * a.u[0] + channel) * ins + xi, a.t[2]) *
                   rd(w, (ulong(oc) * icg + c) * ks + j, a.t[3]);
        }
    wr(y, i, a.t[1], sum + (a.u[24] ? rd(bias, oc, a.t[4]) : 0.0f));
}
