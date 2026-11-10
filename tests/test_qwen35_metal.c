/* Scalar packing and recurrence oracles retained from qwen_numerics_probe.c. */
#include "ds4_gpu.h"
#include "qwen_half_reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}
enum {
    GDN_KEY_HEADS = 16,
    GDN_VALUE_HEADS = 48,
    GDN_DIM = 128,
    GDN_QKV = 10240,
    GDN_HEADS = 6144,
    GDN_CONV = 4,
    GDN_STEPS = 4,
};

typedef struct {
    double mae;
    double rmse;
    double max_abs;
    double max_ref;
    double cosine;
} probe_metrics;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} probe_block_q4_k;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} probe_block_q5_k;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} probe_block_q6_k;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} probe_block_q3_k;

typedef struct {
    uint16_t d;
    uint16_t qs[32];
    uint8_t scales[8];
} probe_block_iq2_xs;

typedef struct {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t scales[8];
} probe_block_iq2_s;

typedef struct {
    uint16_t d;
    uint8_t qs[96];
} probe_block_iq3_xxs;

typedef struct {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
} probe_block_iq3_s;

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} probe_block_iq4_nl;

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
} probe_block_iq4_xs;

_Static_assert(sizeof(probe_block_q3_k) == 110, "Q3_K block layout");
_Static_assert(sizeof(probe_block_iq2_xs) == 74, "IQ2_XS block layout");
_Static_assert(sizeof(probe_block_iq2_s) == 82, "IQ2_S block layout");
_Static_assert(sizeof(probe_block_iq3_xxs) == 98, "IQ3_XXS block layout");
_Static_assert(sizeof(probe_block_iq3_s) == 110, "IQ3_S block layout");
_Static_assert(sizeof(probe_block_iq4_nl) == 18, "IQ4_NL block layout");
_Static_assert(sizeof(probe_block_iq4_xs) == 136, "IQ4_XS block layout");

/* Reuse only the immutable GGML codebooks.  The host dequantizers below are
 * separate scalar implementations, so the test still catches packing,
 * indexing, sign, scale, dispatch, and CUDA-reduction mistakes. */
#define DS4_IQ2_TABLE_STORAGE static const __attribute__((unused))
#include "ds4_iq2_tables_cuda.inc"
#undef DS4_IQ2_TABLE_STORAGE
#define DS4_IQ_UD_TABLE_STORAGE static const
#include "ds4_iq_ud_tables_cuda.inc"
#undef DS4_IQ_UD_TABLE_STORAGE


static void q4_k_scale_min(uint32_t group, const uint8_t *packed, uint8_t *scale,
                           uint8_t *minimum) {
    if (group < 4) {
        *scale = packed[group] & 63u;
        *minimum = packed[group + 4] & 63u;
    } else {
        *scale = (packed[group + 4] & 15u) | ((packed[group - 4] >> 6) << 4);
        *minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

static float q5_k_value(const probe_block_q5_k *block, uint32_t i) {
    const uint32_t group = i / 32u;
    const uint32_t byte = (group >> 1) * 32u + (i & 31u);
    const uint32_t shift = (group & 1u) * 4u;
    uint8_t scale, minimum;
    q4_k_scale_min(group, block->scales, &scale, &minimum);
    const uint32_t high = (block->qh[i & 31u] >> group) & 1u;
    const uint32_t q = ((block->qs[byte] >> shift) & 15u) | (high << 4u);
    return qwen_reference_f16_to_f32(block->d) * (float)scale * (float)q -
           qwen_reference_f16_to_f32(block->dmin) * (float)minimum;
}

static float q6_k_value(const probe_block_q6_k *block, uint32_t i) {
    const uint32_t half = i >> 7u;
    const uint32_t within = i & 127u;
    const uint32_t lane = within & 31u;
    const uint8_t high = block->qh[half * 32u + lane];
    uint8_t q;
    uint32_t scale;
    if (within < 32u) {
        q = (block->ql[half * 64u + lane] & 15u) | ((high & 3u) << 4u);
        scale = half * 8u + (lane >> 4u);
    } else if (within < 64u) {
        q = (block->ql[half * 64u + lane + 32u] & 15u) | (((high >> 2u) & 3u) << 4u);
        scale = half * 8u + 2u + (lane >> 4u);
    } else if (within < 96u) {
        q = (block->ql[half * 64u + lane] >> 4u) | (((high >> 4u) & 3u) << 4u);
        scale = half * 8u + 4u + (lane >> 4u);
    } else {
        q = (block->ql[half * 64u + lane + 32u] >> 4u) | (((high >> 6u) & 3u) << 4u);
        scale = half * 8u + 6u + (lane >> 4u);
    }
    return qwen_reference_f16_to_f32(block->d) * (float)block->scales[scale] * (float)((int)q - 32);
}

static uint8_t u64_byte(uint64_t value, uint32_t index) { return (uint8_t)(value >> (8u * index)); }

static uint16_t load_u16_le(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u);
}

static float q3_k_value(const probe_block_q3_k *block, uint32_t i) {
    const uint32_t n = i >> 7u;
    const uint32_t within = i & 127u;
    const uint32_t j = within >> 5u;
    const uint32_t l = within & 31u;
    const uint32_t is0 = l >> 4u;
    const uint32_t is = 8u * n + 2u * j + is0;
    uint8_t scale;
    if (is < 4u) {
        scale = (block->scales[is] & 0x0fu) | (((block->scales[is + 8u] >> 0u) & 3u) << 4u);
    } else if (is < 8u) {
        scale = (block->scales[is] & 0x0fu) | (((block->scales[is + 4u] >> 2u) & 3u) << 4u);
    } else if (is < 12u) {
        scale = (block->scales[is - 8u] >> 4u) | (((block->scales[is] >> 4u) & 3u) << 4u);
    } else {
        scale = (block->scales[is - 8u] >> 4u) | (((block->scales[is - 4u] >> 6u) & 3u) << 4u);
    }
    const uint32_t mask = 1u << (4u * n + j);
    const int q =
        (int)((block->qs[32u * n + l] >> (2u * j)) & 3u) - ((block->hmask[l] & mask) ? 0 : 4);
    return qwen_reference_f16_to_f32(block->d) * (float)((int)scale - 32) * (float)q;
}

static float iq2_xs_value(const probe_block_iq2_xs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint16_t q = block->qs[4u * ib + il];
    const uint8_t grid = u64_byte(cuda_iq2xs_grid[q & 511u], j);
    const uint8_t signs = cuda_ksigns_iq2xs[q >> 9u];
    const uint32_t scale = (block->scales[ib] >> (4u * (il >> 1u))) & 0x0fu;
    const float d = qwen_reference_f16_to_f32(block->d) * (0.5f + (float)scale) * 0.25f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq2_s_value(const probe_block_iq2_s *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint32_t grid_index =
        (uint32_t)block->qs[4u * ib + il] | (((uint32_t)block->qh[ib] << (8u - 2u * il)) & 0x300u);
    const uint8_t grid = u64_byte(cuda_iq2s_grid[grid_index], j);
    const uint8_t signs = block->qs[32u + 4u * ib + il];
    const uint32_t scale = (block->scales[ib] >> (4u * (il >> 1u))) & 0x0fu;
    const float d = qwen_reference_f16_to_f32(block->d) * (0.5f + (float)scale) * 0.25f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq3_xxs_value(const probe_block_iq3_xxs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint8_t *q3 = block->qs + 8u * ib;
    const uint8_t *gas = block->qs + 64u + 4u * ib;
    const uint32_t aux = (uint32_t)load_u16_le(gas) | ((uint32_t)load_u16_le(gas + 2u) << 16u);
    const uint32_t code = q3[2u * il + (j >> 2u)];
    const uint8_t grid = (uint8_t)(cuda_iq3xxs_grid[code] >> (8u * (j & 3u)));
    const uint8_t signs = cuda_ksigns_iq2xs[(aux >> (7u * il)) & 127u];
    const float d = qwen_reference_f16_to_f32(block->d) * (0.5f + (float)(aux >> 28u)) * 0.5f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq3_s_value(const probe_block_iq3_s *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint8_t *qs = block->qs + 8u * ib;
    const uint32_t code =
        j < 4u
            ? ((uint32_t)qs[2u * il] | (((uint32_t)block->qh[ib] << (8u - 2u * il)) & 0x100u))
            : ((uint32_t)qs[2u * il + 1u] | (((uint32_t)block->qh[ib] << (7u - 2u * il)) & 0x100u));
    const uint8_t grid = (uint8_t)(cuda_iq3s_grid[code] >> (8u * (j & 3u)));
    const uint8_t signs = block->signs[4u * ib + il];
    const uint32_t scale = (block->scales[ib >> 1u] >> (4u * (ib & 1u))) & 0x0fu;
    const float d = qwen_reference_f16_to_f32(block->d) * (float)(1u + 2u * scale);
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq4_nl_value(const probe_block_iq4_nl *block, uint32_t i) {
    const uint8_t q = block->qs[i & 15u];
    const uint8_t code = i < 16u ? (q & 0x0fu) : (q >> 4u);
    return qwen_reference_f16_to_f32(block->d) * (float)cuda_kvalues_iq4nl[code];
}

static float iq4_xs_value(const probe_block_iq4_xs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t j = i & 31u;
    const uint8_t q = block->qs[16u * ib + (j & 15u)];
    const uint8_t code = j < 16u ? (q & 0x0fu) : (q >> 4u);
    const uint32_t scale = ((block->scales_l[ib >> 1u] >> (4u * (ib & 1u))) & 0x0fu) |
                           (((block->scales_h >> (2u * ib)) & 3u) << 4u);
    return qwen_reference_f16_to_f32(block->d) * (float)((int)scale - 32) * (float)cuda_kvalues_iq4nl[code];
}

static float pattern(uint32_t index, uint32_t salt, float scale) {
    uint32_t x = index ^ (salt * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return ((int32_t)(x % 2001u) - 1000) * scale;
}

static float tree_sum_128(const float *values) {
    float work[GDN_DIM];
    memcpy(work, values, sizeof(work));
    for (uint32_t stride = GDN_DIM / 2; stride; stride >>= 1) {
        for (uint32_t i = 0; i < stride; i++)
            work[i] += work[i + stride];
    }
    return work[0];
}

static probe_metrics measure(const float *got, const float *ref, size_t n) {
    long double abs_sum = 0.0L, sq_sum = 0.0L;
    long double dot = 0.0L, got_sq = 0.0L, ref_sq = 0.0L;
    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], r = ref[i], d = g - r;
        const double ad = fabs(d), ar = fabs(r);
        abs_sum += ad;
        sq_sum += d * d;
        dot += g * r;
        got_sq += g * g;
        ref_sq += r * r;
        if (ad > max_abs)
            max_abs = ad;
        if (ar > max_ref)
            max_ref = ar;
    }
    probe_metrics m = {
        .mae = (double)(abs_sum / (long double)n),
        .rmse = sqrt((double)(sq_sum / (long double)n)),
        .max_abs = max_abs,
        .max_ref = max_ref,
        .cosine = (got_sq && ref_sq) ? (double)(dot / sqrtl(got_sq * ref_sq))
                                     : (max_abs == 0.0 ? 1.0 : 0.0),
    };
    return m;
}

static const char *classify(probe_metrics m, double abs_floor, double rel) {
    if (m.max_abs == 0.0)
        return "exact";
    if (m.max_abs <= abs_floor + rel * m.max_ref && m.cosine >= 0.999999)
        return "roundoff";
    return "outside_roundoff_envelope";
}

static int emit_metric(const char *tensor, uint32_t step, const float *got, const float *ref,
                       size_t n, double abs_floor, double rel) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i]) || !isfinite(ref[i])) {
            fprintf(stderr, "FAIL %s step=%u non-finite value at %zu\n", tensor, step, i);
            return 1;
        }
    }
    const probe_metrics m = measure(got, ref, n);
    const char *kind = classify(m, abs_floor, rel);
    printf("{\"probe\":\"gdn\",\"step\":%u,\"tensor\":\"%s\","
           "\"values\":%zu,\"mae\":%.9g,\"rmse\":%.9g,"
           "\"max_abs\":%.9g,\"max_ref\":%.9g,\"cosine\":%.12g,"
           "\"classification\":\"%s\"}\n",
           step, tensor, n, m.mae, m.rmse, m.max_abs, m.max_ref, m.cosine, kind);
    return strcmp(kind, "outside_roundoff_envelope") == 0;
}

static void cpu_gdn_step(float *heads, float *qkv, const float *z, const float *alpha,
                         const float *beta_in, float *conv_state, float *state, const float *a,
                         const float *conv, const float *dt, const float *norm) {
    for (uint32_t c = 0; c < GDN_QKV; c++) {
        float *s = conv_state + (uint64_t)c * GDN_CONV;
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = qkv[c];
        float y = 0.0f;
        for (uint32_t i = 0; i < GDN_CONV; i++)
            y += s[i] * conv[(uint64_t)c * GDN_CONV + i];
        qkv[c] = y / (1.0f + expf(-y));
    }

    for (uint32_t vh = 0; vh < GDN_VALUE_HEADS; vh++) {
        /* The production GGUF stores value-side tensors in tiled head order,
         * so physical value head vh is paired with key head vh % Hk. */
        const uint32_t kh = vh % GDN_KEY_HEADS;
        float q[GDN_DIM], k[GDN_DIM], squares[GDN_DIM], out[GDN_DIM];
        for (uint32_t j = 0; j < GDN_DIM; j++) {
            q[j] = qkv[(uint64_t)kh * GDN_DIM + j];
            k[j] = qkv[2048u + (uint64_t)kh * GDN_DIM + j];
            squares[j] = q[j] * q[j];
        }
        const float qscale = 1.0f / sqrtf(tree_sum_128(squares) + 1.0e-6f);
        for (uint32_t j = 0; j < GDN_DIM; j++) {
            q[j] *= qscale;
            squares[j] = k[j] * k[j];
        }
        const float kscale = 1.0f / sqrtf(tree_sum_128(squares) + 1.0e-6f);
        for (uint32_t j = 0; j < GDN_DIM; j++)
            k[j] *= kscale;

        const float beta = 1.0f / (1.0f + expf(-beta_in[vh]));
        const float x = alpha[vh] + dt[vh];
        const float softplus = x > 20.0f ? x : log1pf(expf(x));
        const float decay = expf(a[vh] * softplus);
        for (uint32_t d = 0; d < GDN_DIM; d++) {
            const float value = qkv[4096u + (uint64_t)vh * GDN_DIM + d];
            float pred = 0.0f;
            for (uint32_t j = 0; j < GDN_DIM; j++)
                pred += decay * state[((uint64_t)vh * GDN_DIM + j) * GDN_DIM + d] * k[j];
            float y = 0.0f;
            for (uint32_t j = 0; j < GDN_DIM; j++) {
                float *cell = state + ((uint64_t)vh * GDN_DIM + j) * GDN_DIM + d;
                *cell = decay * *cell + beta * k[j] * (value - pred);
                y += *cell * q[j] * 0.08838834764831845f;
            }
            out[d] = y;
            squares[d] = y * y;
        }
        const float rms = 1.0f / sqrtf(tree_sum_128(squares) / GDN_DIM + 1.0e-6f);
        for (uint32_t d = 0; d < GDN_DIM; d++) {
            const float gate = z[(uint64_t)vh * GDN_DIM + d];
            heads[(uint64_t)vh * GDN_DIM + d] =
                out[d] * rms * norm[d] * (gate / (1.0f + expf(-gate)));
        }
    }
}

static uint32_t ud_block_size(uint32_t type) {
    switch (type) {
    case 2u:
        return 18u;
    case 8u:
        return 34u;
    case 12u:
        return sizeof(probe_block_q4_k);
    case 13u:
        return sizeof(probe_block_q5_k);
    case 14u:
        return sizeof(probe_block_q6_k);
    case 11u:
        return sizeof(probe_block_q3_k);
    case 17u:
        return sizeof(probe_block_iq2_xs);
    case 18u:
        return sizeof(probe_block_iq3_xxs);
    case 20u:
        return sizeof(probe_block_iq4_nl);
    case 21u:
        return sizeof(probe_block_iq3_s);
    case 22u:
        return sizeof(probe_block_iq2_s);
    case 23u:
        return sizeof(probe_block_iq4_xs);
    default:
        return 0u;
    }
}

static uint32_t ud_quant_size(uint32_t type) {
    return type == 2u || type == 8u || type == 20u ? 32u : 256u;
}

static float ud_value(uint32_t type, const uint8_t *row, uint32_t i) {
    const uint32_t qk = ud_quant_size(type);
    const uint32_t block_size = ud_block_size(type);
    const uint8_t *block = row + (i / qk) * block_size;
    const uint32_t qi = i % qk;
    switch (type) {
    case 2u: {
        unsigned q = (block[2u + qi % 16u] >> (qi < 16u ? 0u : 4u)) & 15u;
        return qwen_reference_f16_to_f32(load_u16_le(block)) * (float)((int)q - 8);
    }
    case 8u:
        return qwen_reference_f16_to_f32(load_u16_le(block)) * (float)(int8_t)block[2u + qi];
    case 12u: {
        const probe_block_q4_k *b = (const probe_block_q4_k *)block;
        uint8_t scale, minimum;
        q4_k_scale_min(qi / 32u, b->scales, &scale, &minimum);
        unsigned q = (b->qs[(qi / 64u) * 32u + qi % 32u] >> ((qi % 64u) / 32u * 4u)) & 15u;
        return qwen_reference_f16_to_f32(b->d) * scale * q - qwen_reference_f16_to_f32(b->dmin) * minimum;
    }
    case 13u:
        return q5_k_value((const probe_block_q5_k *)block, qi);
    case 14u:
        return q6_k_value((const probe_block_q6_k *)block, qi);
    case 11u:
        return q3_k_value((const probe_block_q3_k *)block, qi);
    case 17u:
        return iq2_xs_value((const probe_block_iq2_xs *)block, qi);
    case 18u:
        return iq3_xxs_value((const probe_block_iq3_xxs *)block, qi);
    case 20u:
        return iq4_nl_value((const probe_block_iq4_nl *)block, qi);
    case 21u:
        return iq3_s_value((const probe_block_iq3_s *)block, qi);
    case 22u:
        return iq2_s_value((const probe_block_iq2_s *)block, qi);
    case 23u:
        return iq4_xs_value((const probe_block_iq4_xs *)block, qi);
    default:
        return 0.0f;
    }
}

static void ud_fill_weights(uint32_t type, uint8_t *weights, uint32_t rows, uint32_t inputs) {
    const uint32_t qk = ud_quant_size(type);
    const uint32_t block_size = ud_block_size(type);
    const uint32_t blocks_per_row = inputs / qk;
    const uint32_t row_bytes = blocks_per_row * block_size;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < row_bytes; i++) {
            weights[(size_t)row * row_bytes + i] =
                (uint8_t)(37u * i + 53u * row + 19u * type + 11u);
        }
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            uint8_t *ptr = weights + (size_t)row * row_bytes + (size_t)block * block_size;
            /* Small, exactly representable positive per-block scales keep the
             * oracle sensitive without overflowing adversarial byte patterns. */
            const uint16_t d = (uint16_t)(0x1800u + ((row + 3u * block + type) & 3u) * 0x0100u);
            switch (type) {
            case 2u:
            case 8u:
                memcpy(ptr, &d, sizeof(d));
                break;
            case 12u:
                ((probe_block_q4_k *)ptr)->d = d;
                ((probe_block_q4_k *)ptr)->dmin = 0x1400;
                break;
            case 11u:
                ((probe_block_q3_k *)ptr)->d = d;
                break;
            case 13u:
                ((probe_block_q5_k *)ptr)->d = d;
                ((probe_block_q5_k *)ptr)->dmin = 0x1400;
                break;
            case 14u:
                ((probe_block_q6_k *)ptr)->d = d;
                break;
            case 17u:
                ((probe_block_iq2_xs *)ptr)->d = d;
                break;
            case 18u:
                ((probe_block_iq3_xxs *)ptr)->d = d;
                break;
            case 20u:
                ((probe_block_iq4_nl *)ptr)->d = d;
                break;
            case 21u:
                ((probe_block_iq3_s *)ptr)->d = d;
                break;
            case 22u:
                ((probe_block_iq2_s *)ptr)->d = d;
                break;
            case 23u:
                ((probe_block_iq4_xs *)ptr)->d = d;
                break;
            default:
                break;
            }
        }
    }
}
static int run_gdn_probe(void) {
    const size_t qkv_n = GDN_QKV;
    const size_t heads_n = GDN_HEADS;
    const size_t conv_n = (size_t)GDN_QKV * GDN_CONV;
    const size_t state_n = (size_t)GDN_VALUE_HEADS * GDN_DIM * GDN_DIM;
    const size_t map_n = (GDN_VALUE_HEADS + conv_n + GDN_VALUE_HEADS + GDN_DIM);
    size_t mapped = (map_n * 4 + getpagesize() - 1) / getpagesize() * getpagesize();
    float *map = mmap(NULL, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED)
        return 1;
    float *qkv = malloc(qkv_n * sizeof(float));
    float *qkv_ref = malloc(qkv_n * sizeof(float));
    float *z = malloc(heads_n * sizeof(float));
    float *alpha = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *beta = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *conv_state = malloc(conv_n * sizeof(float));
    float *conv_ref = malloc(conv_n * sizeof(float));
    float *state = malloc(state_n * sizeof(float));
    float *state_ref = malloc(state_n * sizeof(float));
    float *heads = malloc(heads_n * sizeof(float));
    float *heads_ref = malloc(heads_n * sizeof(float));
    if (!qkv || !qkv_ref || !z || !alpha || !beta || !conv_state || !conv_ref || !state ||
        !state_ref || !heads || !heads_ref)
        return 1;

    const uint64_t a_off = 0;
    const uint64_t conv_off = GDN_VALUE_HEADS * sizeof(float);
    const uint64_t dt_off = conv_off + conv_n * sizeof(float);
    const uint64_t norm_off = dt_off + GDN_VALUE_HEADS * sizeof(float);
    float *a = map + a_off / sizeof(float);
    float *conv = map + conv_off / sizeof(float);
    float *dt = map + dt_off / sizeof(float);
    float *norm = map + norm_off / sizeof(float);
    for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
        a[i] = -0.04f - 0.0007f * (float)i;
        dt[i] = -2.4f + 0.013f * (float)i;
    }
    for (size_t i = 0; i < conv_n; i++)
        conv[i] = pattern((uint32_t)i, 41, 0.00035f);
    for (uint32_t i = 0; i < GDN_DIM; i++)
        norm[i] = 0.85f + 0.002f * (float)i;
    for (size_t i = 0; i < conv_n; i++)
        conv_state[i] = pattern((uint32_t)i, 71, 0.00002f);
    for (size_t i = 0; i < state_n; i++)
        state[i] = pattern((uint32_t)i, 97, 0.0000005f);
    memcpy(conv_ref, conv_state, conv_n * sizeof(float));
    memcpy(state_ref, state, state_n * sizeof(float));

    if (!ds4_gpu_set_model_map(map, map_n * 4))
        return 1;
    ds4_gpu_tensor *t_qkv = ds4_gpu_tensor_alloc(GDN_STEPS * qkv_n * sizeof(float));
    ds4_gpu_tensor *t_z = ds4_gpu_tensor_alloc(GDN_STEPS * heads_n * sizeof(float));
    ds4_gpu_tensor *t_alpha = ds4_gpu_tensor_alloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *t_beta = ds4_gpu_tensor_alloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *t_conv = ds4_gpu_tensor_alloc(conv_n * sizeof(float));
    ds4_gpu_tensor *t_state = ds4_gpu_tensor_alloc(state_n * sizeof(float));
    ds4_gpu_tensor *t_heads = ds4_gpu_tensor_alloc(GDN_STEPS * heads_n * sizeof(float));
    int failed = !t_qkv || !t_z || !t_alpha || !t_beta || !t_conv || !t_state || !t_heads;
    if (failed) {
        fprintf(stderr, "qwen numerics probe: GPU tensor allocation failed\n");
        goto cleanup;
    }
    failed |= !ds4_gpu_set_model_map(map, map_n * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_conv, 0, conv_state, conv_n * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_state, 0, state, state_n * sizeof(float));

    for (uint32_t step = 0; step < GDN_STEPS && !failed; step++) {
        for (size_t i = 0; i < qkv_n; i++)
            qkv[i] = pattern((uint32_t)i, 101 + step, 0.00008f);
        for (size_t i = 0; i < heads_n; i++)
            z[i] = pattern((uint32_t)i, 151 + step, 0.0004f);
        for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
            alpha[i] = pattern(i, 181 + step, 0.003f);
            beta[i] = pattern(i, 211 + step, 0.003f);
        }
        memcpy(qkv_ref, qkv, qkv_n * sizeof(float));
        cpu_gdn_step(heads_ref, qkv_ref, z, alpha, beta, conv_ref, state_ref, a, conv, dt, norm);

        failed |= !ds4_gpu_tensor_write(t_qkv, 0, qkv, qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_z, 0, z, heads_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_alpha, 0, alpha, GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_beta, 0, beta, GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_qwen35_gated_delta_net_tensor(
            t_heads, t_qkv, t_z, t_alpha, t_beta, t_conv, t_state, map, map_n * sizeof(float),
            a_off, conv_off, dt_off, norm_off);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(t_qkv, 0, qkv, qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_conv, 0, conv_state, conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_state, 0, state, state_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_heads, 0, heads, heads_n * sizeof(float));
        if (!failed) {
            failed |= emit_metric("conv_silu", step, qkv, qkv_ref, qkv_n, 2e-6, 2e-5);
            failed |= emit_metric("conv_state", step, conv_state, conv_ref, conv_n, 0.0, 0.0);
            failed |= emit_metric("recurrent_state", step, state, state_ref, state_n, 2e-6, 3e-5);
            failed |= emit_metric("heads", step, heads, heads_ref, heads_n, 5e-6, 5e-5);
        }
    }

    /* Exercise the optimized rows kernel in the way prefill uses it: all
     * tokens in one launch while convolution and recurrent state stay local
     * across the token loop.  Repeating the one-row API above cannot catch a
     * missing or misordered state update inside that loop. */
    float *qkv_rows = malloc(GDN_STEPS * qkv_n * sizeof(float));
    float *qkv_rows_ref = malloc(GDN_STEPS * qkv_n * sizeof(float));
    float *z_rows = malloc(GDN_STEPS * heads_n * sizeof(float));
    float *alpha_rows = malloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    float *beta_rows = malloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    float *heads_rows = malloc(GDN_STEPS * heads_n * sizeof(float));
    float *heads_rows_ref = malloc(GDN_STEPS * heads_n * sizeof(float));
    if (!qkv_rows || !qkv_rows_ref || !z_rows || !alpha_rows || !beta_rows || !heads_rows ||
        !heads_rows_ref) {
        failed = 1;
    }
    if (!failed) {
        for (size_t i = 0; i < conv_n; i++)
            conv_state[i] = pattern((uint32_t)i, 71, 0.00002f);
        for (size_t i = 0; i < state_n; i++)
            state[i] = pattern((uint32_t)i, 97, 0.0000005f);
        memcpy(conv_ref, conv_state, conv_n * sizeof(float));
        memcpy(state_ref, state, state_n * sizeof(float));
        for (uint32_t step = 0; step < GDN_STEPS; step++) {
            float *qr = qkv_rows + (size_t)step * qkv_n;
            float *qref = qkv_rows_ref + (size_t)step * qkv_n;
            float *zr = z_rows + (size_t)step * heads_n;
            float *ar = alpha_rows + (size_t)step * GDN_VALUE_HEADS;
            float *br = beta_rows + (size_t)step * GDN_VALUE_HEADS;
            for (size_t i = 0; i < qkv_n; i++)
                qr[i] = pattern((uint32_t)i, 101 + step, 0.00008f);
            for (size_t i = 0; i < heads_n; i++)
                zr[i] = pattern((uint32_t)i, 151 + step, 0.0004f);
            for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
                ar[i] = pattern(i, 181 + step, 0.003f);
                br[i] = pattern(i, 211 + step, 0.003f);
            }
            memcpy(qref, qr, qkv_n * sizeof(float));
            cpu_gdn_step(heads_rows_ref + (size_t)step * heads_n, qref, zr, ar, br, conv_ref,
                         state_ref, a, conv, dt, norm);
        }
        failed |= !ds4_gpu_tensor_write(t_qkv, 0, qkv_rows, GDN_STEPS * qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_z, 0, z_rows, GDN_STEPS * heads_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_alpha, 0, alpha_rows,
                                        GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_beta, 0, beta_rows,
                                        GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_conv, 0, conv_state, conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_state, 0, state, state_n * sizeof(float));
        failed |= !ds4_gpu_qwen35_gated_delta_net_rows_tensor(
            t_heads, t_qkv, t_z, t_alpha, t_beta, t_conv, t_state, NULL, NULL, map,
            map_n * sizeof(float), a_off, conv_off, dt_off, norm_off, GDN_STEPS);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(t_qkv, 0, qkv_rows, GDN_STEPS * qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_conv, 0, conv_state, conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_state, 0, state, state_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_heads, 0, heads_rows, GDN_STEPS * heads_n * sizeof(float));
        if (!failed) {
            failed |= emit_metric("rows_conv_silu", GDN_STEPS, qkv_rows, qkv_rows_ref,
                                  GDN_STEPS * qkv_n, 2e-6, 2e-5);
            failed |=
                emit_metric("rows_conv_state", GDN_STEPS, conv_state, conv_ref, conv_n, 0.0, 0.0);
            failed |= emit_metric("rows_recurrent_state", GDN_STEPS, state, state_ref, state_n,
                                  2e-6, 3e-5);
            failed |= emit_metric("rows_heads", GDN_STEPS, heads_rows, heads_rows_ref,
                                  GDN_STEPS * heads_n, 5e-6, 5e-5);
        }
    }
    free(heads_rows_ref);
    free(heads_rows);
    free(beta_rows);
    free(alpha_rows);
    free(z_rows);
    free(qkv_rows_ref);
    free(qkv_rows);

cleanup:
    if (t_heads)
        ds4_gpu_tensor_free(t_heads);
    if (t_state)
        ds4_gpu_tensor_free(t_state);
    if (t_conv)
        ds4_gpu_tensor_free(t_conv);
    if (t_beta)
        ds4_gpu_tensor_free(t_beta);
    if (t_alpha)
        ds4_gpu_tensor_free(t_alpha);
    if (t_z)
        ds4_gpu_tensor_free(t_z);
    if (t_qkv)
        ds4_gpu_tensor_free(t_qkv);
    free(heads_ref);
    free(heads);
    free(state_ref);
    free(state);
    free(conv_ref);
    free(conv_state);
    free(beta);
    free(alpha);
    free(z);
    free(qkv_ref);
    free(qkv);
    munmap(map, mapped);
    return failed;
}

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static int test_half_reference(void) {
    /* All finite half encodings, using an arithmetic oracle rather than the
     * decoder's bit construction; preserve negative zero by comparing bytes. */
    for (uint32_t half = 0; half <= UINT16_MAX; half++) {
        uint32_t exponent = (half >> 10) & 31u, fraction = half & 1023u;
        float actual = qwen_reference_f16_to_f32((uint16_t)half);
        if (exponent == 31) {
            CHECK(fraction ? isnan(actual) : isinf(actual));
            continue;
        }
        double magnitude = exponent ? ldexp(1.0 + (double)fraction / 1024.0, (int)exponent - 15)
                                    : ldexp((double)fraction, -24);
        float expected = (float)copysign(magnitude, half & 0x8000u ? -1.0 : 1.0);
        CHECK(!memcmp(&actual, &expected, sizeof(actual)));
    }
    puts("PASS scalar FP16 reference: all 65536 storage encodings, including subnormals");
    return 0;
}
/* Borrow the quant oracle's buffers; only test_stream_gather owns the file. */
typedef struct {
    unsigned type, cols;
    const unsigned char *weights;
    size_t bytes, row_bytes;
    ds4_gpu_tensor *out;
    float *actual;
    int fd;
} quant_gather_fixture;
enum { STREAM_GATHER_VOCAB = 17 };

static int check_gather_rows(const quant_gather_fixture *f, const int32_t *ids, unsigned count) {
    CHECK(ds4_gpu_tensor_read(f->out, 0, f->actual, (uint64_t)count * f->cols * sizeof(float)));
    for (unsigned t = 0; t < count; t++)
        for (unsigned i = 0; i < f->cols; i++)
            CHECK(f->actual[t * f->cols + i] ==
                  ud_value(f->type, f->weights + ids[t] * f->row_bytes, i));
    return 0;
}
static int check_gather_arguments(const quant_gather_fixture *f) {
    uint64_t before = 0, after = 0;
    ds4_gpu_qwen4_stream_memory(NULL, &before);
    CHECK(!ds4_gpu_embed_tokens_quant_tensor(f->out, NULL, f->weights, f->bytes, 0,
                                             f->type, STREAM_GATHER_VOCAB, 2, f->cols));
    CHECK(!ds4_gpu_embed_tokens_quant_tensor(NULL, f->out, f->weights, f->bytes, 0,
                                             f->type, STREAM_GATHER_VOCAB, 2, f->cols));
    CHECK(!ds4_gpu_embed_tokens_quant_tensor(f->out, f->out, NULL, f->bytes, 0,
                                             f->type, STREAM_GATHER_VOCAB, 2, f->cols));
    CHECK(!ds4_gpu_embed_tokens_quant_tensor(f->out, f->out, f->weights, f->bytes, 0,
                                             f->type, 0, 2, f->cols));
    CHECK(!ds4_gpu_embed_token_quant_tensor(NULL, f->weights, f->bytes, 0,
                                            f->type, STREAM_GATHER_VOCAB, 0, f->cols));
    CHECK(!ds4_gpu_embed_token_quant_tensor(f->out, NULL, f->bytes, 0,
                                            f->type, STREAM_GATHER_VOCAB, 0, f->cols));
    ds4_gpu_qwen4_stream_memory(NULL, &after);
    CHECK(before == after); /* Reject incomplete requests before reading weights. */
    const int32_t previous = 16;
    return check_gather_rows(f, &previous, 1);
}
static int check_stream_batch(const quant_gather_fixture *f) {
    const int32_t batch[] = {16, 0, 16, 8};
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(sizeof(batch));
    CHECK(ids && ds4_gpu_tensor_write(ids, 0, batch, sizeof(batch)));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_embed_tokens_quant_tensor(f->out, ids, f->weights, f->bytes, 0, f->type,
                                            STREAM_GATHER_VOCAB, 4, f->cols));
    CHECK(ds4_gpu_commands_active());
    CHECK(!check_gather_rows(f, batch, 4));
    CHECK(ds4_gpu_qwen4_stream_layer(f->weights, f->bytes));
    const int32_t invalid[] = {0, STREAM_GATHER_VOCAB};
    CHECK(ds4_gpu_tensor_write(ids, 0, invalid, sizeof(invalid)));
    CHECK(!ds4_gpu_embed_tokens_quant_tensor(f->out, ids, f->weights, f->bytes, 0, f->type,
                                             STREAM_GATHER_VOCAB, 2, f->cols));
    CHECK(!check_gather_rows(f, batch, 4));
    CHECK(ds4_gpu_end_commands());
    ds4_gpu_tensor_free(ids);
    return 0;
}
static int check_late_retained_row(const quant_gather_fixture *f) {
    /* Transient misses must not spend an as-yet unloaded head's quota. */
    CHECK(ds4_gpu_qwen4_stream_open(f->weights, f->bytes, f->fd, 3 * f->row_bytes));
    const uint64_t retained_offset = 16 * f->row_bytes, retained_bytes = f->row_bytes;
    CHECK(ds4_gpu_qwen4_stream_retain(f->weights, f->bytes, &retained_offset,
                                      &retained_bytes, 1, 2 * f->row_bytes));
    const int32_t late_head[] = {0, 1, 2, 16, 3, 16};
    for (unsigned i = 0; i < sizeof(late_head) / sizeof(late_head[0]); i++) {
        if (i == 3) {
            CHECK(!ftruncate(f->fd, 0));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_tensor_fill_f32(f->out, 0, f->cols));
            for (unsigned attempt = 0; attempt < 2; attempt++) {
                CHECK(!ds4_gpu_embed_token_quant_tensor(f->out, f->weights, f->bytes, 0,
                                                        f->type, STREAM_GATHER_VOCAB,
                                                        late_head[i], f->cols));
                uint64_t owned = 0;
                ds4_gpu_qwen4_stream_memory(&owned, NULL);
                CHECK(owned == 3 * f->row_bytes); /* Failed storage stays owned. */
            }
            CHECK(ds4_gpu_commands_active());
            CHECK(pwrite(f->fd, f->weights, f->bytes, 0) == (ssize_t)f->bytes);
        }
        CHECK(ds4_gpu_embed_token_quant_tensor(f->out, f->weights, f->bytes, 0,
                                                f->type, STREAM_GATHER_VOCAB,
                                                late_head[i], f->cols));
        CHECK(!check_gather_rows(f, late_head + i, 1));
        if (i == 3)
            CHECK(ds4_gpu_end_commands());
        uint64_t owned = 0;
        ds4_gpu_qwen4_stream_memory(&owned, NULL);
        CHECK(owned <= 3 * f->row_bytes);
    }
    uint64_t reads = 0;
    ds4_gpu_qwen4_stream_memory(NULL, &reads);
    CHECK(reads == 5 * f->row_bytes);
    return 0;
}
static int test_stream_gather(unsigned type, unsigned cols, const unsigned char *weights,
                               size_t bytes, size_t row_bytes, ds4_gpu_tensor *out, float *actual) {
    FILE *file = tmpfile();
    CHECK(file && fwrite(weights, 1, bytes, file) == bytes && !fflush(file));
    quant_gather_fixture f = {.type = type, .cols = cols, .weights = weights,
                               .bytes = bytes, .row_bytes = row_bytes,
                               .out = out, .actual = actual, .fd = fileno(file)};
    ds4_gpu_set_ssd_streaming(true);
    /* Exactly one row fits: a full-table binding must not be needed. */
    CHECK(ds4_gpu_qwen4_stream_open(weights, bytes, f.fd, row_bytes));
    const int32_t ids[] = {0, 8, 16};
    for (unsigned t = 0; t < sizeof(ids) / sizeof(ids[0]); t++) {
        CHECK(ds4_gpu_embed_token_quant_tensor(out, weights, bytes, 0, type,
                                                STREAM_GATHER_VOCAB, ids[t], cols));
        CHECK(!check_gather_rows(&f, ids + t, 1));
        CHECK(ds4_gpu_qwen4_stream_layer(weights, bytes));
    }
    uint64_t read_bytes = 0;
    ds4_gpu_qwen4_stream_memory(NULL, &read_bytes);
    CHECK(read_bytes == 3 * row_bytes);
    CHECK(!check_gather_arguments(&f));
    const bool ud = type == 11 || type == 13 || type == 14 || type == 17 || type == 18 ||
                    (type >= 20 && type <= 23);
    if (ud) {
        CHECK(!check_stream_batch(&f));
        CHECK(!check_late_retained_row(&f));
    }
    CHECK(!ds4_gpu_embed_token_quant_tensor(out, weights, bytes, 0, type,
                                            STREAM_GATHER_VOCAB, STREAM_GATHER_VOCAB, cols));
    CHECK(!ds4_gpu_embed_token_quant_tensor(out, weights, bytes - 1, 0, type,
                                            STREAM_GATHER_VOCAB, 0, cols));
    CHECK(ds4_gpu_qwen4_stream_close());
    ds4_gpu_set_ssd_streaming(false);
    CHECK(!fclose(file));
    return 0;
}
static int test_quant(unsigned type, unsigned cols) {
    const unsigned rows = 17, T = 17;
    size_t rb = cols / ud_quant_size(type) * ud_block_size(type), n = rows * rb;
    size_t mapped = (n + getpagesize() - 1) / getpagesize() * getpagesize();
    unsigned char *w = mmap(NULL, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(w != MAP_FAILED);
    ud_fill_weights(type, w, rows, cols);
    CHECK(ds4_gpu_set_model_map(w, n));
    float *x = malloc(T * cols * 4), *got = malloc(T * cols * 4);
    CHECK(x && got);
    for (unsigned i = 0; i < T * cols; i++)
        x[i] = pattern(i, 131, 0.001f);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(T * cols * 4),
                   *yt = ds4_gpu_tensor_alloc(T * cols * 4);
    CHECK(xt && yt);
    CHECK(ds4_gpu_tensor_write(xt, 0, x, T * cols * 4));
    CHECK(ds4_gpu_embed_token_quant_tensor(yt, w, n, 0, type, rows, 16, cols));
    CHECK(ds4_gpu_tensor_read(yt, 0, got, cols * 4));
    for (unsigned i = 0; i < cols; i++)
        CHECK(got[i] == ud_value(type, w + 16 * rb, i));
    quant_gather_fixture arguments = {.type = type, .cols = cols, .weights = w,
                                       .bytes = n, .row_bytes = rb,
                                       .out = yt, .actual = got, .fd = -1};
    CHECK(!check_gather_arguments(&arguments));
    CHECK(!test_stream_gather(type, cols, w, n, rb, yt, got));
    int ids[3] = {0, 5, 16};
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(it);
    CHECK(ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_embed_tokens_quant_tensor(yt, it, w, n, 0, type, rows, 3, cols));
    CHECK(ds4_gpu_tensor_read(yt, 0, got, 3 * cols * 4));
    for (unsigned t = 0; t < 3; t++)
        for (unsigned i = 0; i < cols; i++)
            CHECK(got[t * cols + i] == ud_value(type, w + ids[t] * rb, i));
    /* UD stays FP32 across these tile boundaries. Generic formats retain
     * their original small-batch cases; their half GEMM is qualified separately. */
    const unsigned small_counts[] = {1, 3, 5};
    const unsigned tiled_counts[] = {1, 3, 4, 5, 7, 8, 9, 16, T};
    const bool ud = type == 11 || type == 13 || type == 14 || type == 17 || type == 18 ||
                    (type >= 20 && type <= 23);
    const unsigned *token_counts = ud ? tiled_counts : small_counts;
    const unsigned cases = ud ? sizeof(tiled_counts) / sizeof(tiled_counts[0])
                               : sizeof(small_counts) / sizeof(small_counts[0]);
    for (unsigned case_id = 0; case_id < cases; case_id++) {
        unsigned tokens = token_counts[case_id];
        CHECK(ds4_gpu_matmul_quant_tensor(yt, w, n, 0, type, cols, rows, xt, tokens));
        CHECK(ds4_gpu_tensor_read(yt, 0, got, tokens * rows * 4));
        double worst = 0;
        for (unsigned t = 0; t < tokens; t++)
            for (unsigned r = 0; r < rows; r++) {
                double ref = 0;
                for (unsigned i = 0; i < cols; i++)
                    ref += (double)ud_value(type, w + r * rb, i) * x[t * cols + i];
                double d = fabs(got[t * rows + r] - ref);
                if (d > worst)
                    worst = d;
                CHECK(isfinite(got[t * rows + r]) && d <= 2e-4 + 2e-5 * fabs(ref));
            }
        printf("PASS UD type=%u cols=%u rows=%u tokens=%u max_abs=%.9g\n", type, cols, rows, tokens,
               worst);
    }
    CHECK(!ds4_gpu_matmul_quant_tensor(yt, w, n, 1, type, cols, rows, xt, T));
    ds4_gpu_tensor_free(it);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(yt);
    free(x);
    free(got);
    ds4_gpu_cleanup();
    munmap(w, mapped);
    CHECK(ds4_gpu_init());
    return 0;
}
static int test_resource_views(void) {
    ds4_resources *r = ds4_gpu_resources();
    ds4_resource_totals before, after;
    ds4_resources_totals(r, 0, &before);
    ds4_gpu_tensor *base = ds4_gpu_tensor_alloc(1024), *out = ds4_gpu_tensor_alloc(256);
    CHECK(base && out);
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(base, 128, 512);
    ds4_gpu_tensor *nested = ds4_gpu_tensor_view(view, 64, 256);
    CHECK(view && nested);
    CHECK(ds4_gpu_tensor_fill_f32(base, 1, 256));
    ds4_resources_totals(r, 0, &after);
    CHECK(after.allocated[DS4_MEMORY_DEVICE] == before.allocated[DS4_MEMORY_DEVICE] + 1280);
    CHECK(after.views == before.views + 2);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_add_tensor(out, nested, nested, 64));
    ds4_gpu_tensor_free(base);
    ds4_gpu_tensor_free(view);
    ds4_gpu_tensor_free(nested);
    ds4_resources_totals(r, 0, &after);
    CHECK(after.allocated[DS4_MEMORY_DEVICE] == before.allocated[DS4_MEMORY_DEVICE] + 1280);
    CHECK(ds4_gpu_end_commands());
    ds4_resources_totals(r, 0, &after);
    CHECK(after.allocated[DS4_MEMORY_DEVICE] == before.allocated[DS4_MEMORY_DEVICE] + 256);
    CHECK(after.views == before.views);
    float values[64];
    CHECK(ds4_gpu_tensor_read(out, 0, values, sizeof(values)));
    for (unsigned i = 0; i < 64; i++)
        CHECK(values[i] == 2);
    ds4_gpu_tensor_free(out);
    CHECK(ds4_gpu_synchronize());
    ds4_resources_totals(r, 0, &after);
    CHECK(after.allocated[DS4_MEMORY_DEVICE] == before.allocated[DS4_MEMORY_DEVICE]);
    CHECK(after.roots == before.roots && after.views == before.views);
    puts("PASS Metal aliases counted once and retired after GPU completion");
    return 0;
}
int main(void) {
    CHECK(!test_half_reference());
    CHECK(ds4_gpu_init());
    unsigned types[] = {2, 8, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23};
    for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        CHECK(!test_quant(types[i], 256));
        CHECK(!test_quant(types[i], 512));
    }
    CHECK(!test_resource_views());
    int failed = run_gdn_probe();
    ds4_gpu_cleanup();
    return failed;
}
