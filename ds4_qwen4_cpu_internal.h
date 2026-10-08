/* Private scalar/SIMD arithmetic; included only by the CPU implementation
 * and its independent primitive tests. CUDA callers use the public API. */
#ifndef DS4_QWEN4_CPU_INTERNAL_H
#define DS4_QWEN4_CPU_INTERNAL_H

#include <math.h>
#include "ds4_qwen4_cpu.h"
#include <string.h>
#if defined(__AVX2__) && !defined(__CUDACC__)
#include <immintrin.h>
#include "ds4_qwen4_cpu_sign_masks.inc"
#endif

#define DS4_IQ2_TABLE_STORAGE static const
#define cuda_ksigns_iq2xs ds4_qwen4_cpu_signs
#define cuda_iq2xxs_grid ds4_qwen4_cpu_grid
#include "ds4_iq2_tables_cuda.inc"
#undef cuda_iq2xxs_grid
#undef cuda_ksigns_iq2xs
#undef DS4_IQ2_TABLE_STORAGE

static inline float ds4_qwen4_cpu_half(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    unsigned e = (h >> 10) & 31u, m = h & 1023u;
    if (!e && m) return (sign ? -1.0f : 1.0f) * ldexpf((float)m, -24);
    uint32_t bits = sign | (e == 31 ? 0x7f800000u | (m << 13) :
                           e ? ((e + 112u) << 23) | (m << 13) : 0u);
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

static inline void ds4_qwen4_cpu_block(unsigned type, const uint8_t *b, float *v) {
    if (type == 16) {
        uint16_t dh; memcpy(&dh, b, 2);
        const float d = ds4_qwen4_cpu_half(dh);
        for (unsigned g = 0; g < 8; g++) {
            uint32_t ids, ss; memcpy(&ids, b + 2 + g * 8, 4); memcpy(&ss, b + 6 + g * 8, 4);
            const float scale = d * (0.5f + (float)(ss >> 28)) * 0.25f;
            for (unsigned s = 0; s < 4; s++) {
                const uint64_t grid = ds4_qwen4_cpu_grid[(ids >> (s * 8)) & 255u];
                const unsigned sign_index = (ss >> (s * 7)) & 127u;
#if defined(__AVX2__) && !defined(__CUDACC__)
                const uint64_t mask = ds4_qwen4_cpu_sign_masks[sign_index];
                const uint64_t signed_grid = (grid ^ mask) + (mask & 0x0101010101010101ull);
                const __m256 weights = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128((long long)signed_grid)));
                _mm256_storeu_ps(v + g * 32 + s * 8, _mm256_mul_ps(weights, _mm256_set1_ps(scale)));
#else
                const unsigned signs = ds4_qwen4_cpu_signs[sign_index];
                for (unsigned j = 0; j < 8; j++)
                    v[g * 32 + s * 8 + j] = scale * (float)((grid >> (j * 8)) & 255u) *
                                            ((signs & (1u << j)) ? -1.0f : 1.0f);
#endif
            }
        }
    } else {
        uint16_t dh, mh; memcpy(&dh, b + 80, 2); memcpy(&mh, b + 82, 2);
        const float d = ds4_qwen4_cpu_half(dh), dm = ds4_qwen4_cpu_half(mh);
#if defined(__AVX2__) && !defined(__CUDACC__)
        for (unsigned j = 0; j < 256; j += 8) {
            const unsigned sc = b[j / 16];
            uint64_t packed; memcpy(&packed, b + 16 + (j / 128) * 32 + j % 32, 8);
            const __m256i q = _mm256_and_si256(_mm256_srlv_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)packed)),
                _mm256_set1_epi32(2 * ((j % 128) / 32))), _mm256_set1_epi32(3));
            const __m256 weights = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q), _mm256_set1_ps(d * (float)(sc & 15u))),
                                                _mm256_set1_ps(dm * (float)(sc >> 4)));
            _mm256_storeu_ps(v + j, weights);
        }
#else
        for (unsigned j = 0; j < 256; j++) {
            const unsigned sc = b[j / 16];
            const unsigned q = (b[16 + (j / 128) * 32 + j % 32] >> (2 * ((j % 128) / 32))) & 3u;
            v[j] = (d * (float)(sc & 15u)) * (float)q - dm * (float)(sc >> 4);
        }
#endif
    }
}

static inline float ds4_qwen4_cpu_dot(unsigned type, const uint8_t *row,
                                     const float *x, unsigned n) {
    float values[256];
    float a[32] = {0};
    for (unsigned b = 0; b < (n + 255u) / 256u; b++) {
        ds4_qwen4_cpu_block(type, row + (uint64_t)b * (type == 16 ? 66u : 84u), values);
        const unsigned width = n - b * 256u < 256u ? n - b * 256u : 256u;
        /* CUDA moe_mv gives each lane four consecutive elements, then
         * advances by 128; each product is fused into that lane's sum. */
        for (unsigned j = 0; j < width; j += 128u)
            for (unsigned lane = 0; lane < 32u && j + lane * 4u < width; ++lane)
                for (unsigned k = 0; k < 4u; ++k)
                    a[lane] = fmaf(values[j + lane * 4u + k], x[b * 256u + j + lane * 4u + k], a[lane]);
    }
    for (unsigned d = 16; d; d >>= 1)
        for (unsigned lane = 0; lane < d; ++lane) a[lane] += a[lane + d];
    return a[0];
}

static inline float ds4_qwen4_cpu_silu(float g) {
    const float e = expf(-fabsf(g));
    return g * (g >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

static inline void ds4_qwen4_cpu_expert(const uint8_t *gate, const uint8_t *up,
        const uint8_t *down, unsigned gu_type, unsigned down_type,
        const float *x, unsigned E, unsigned F, float *mid, float *out) {
    const uint64_t gr = ds4_qwen4_cpu_row_bytes(gu_type, E);
    const uint64_t dr = ds4_qwen4_cpu_row_bytes(down_type, F);
    for (unsigned r = 0; r < F; r++) {
        const float g = ds4_qwen4_cpu_dot(gu_type, gate + r * gr, x, E);
        const float u = ds4_qwen4_cpu_dot(gu_type, up + r * gr, x, E);
        mid[r] = ds4_qwen4_cpu_silu(g) * u;
    }
    for (unsigned r = 0; r < E; r++)
        out[r] = ds4_qwen4_cpu_dot(down_type, down + r * dr, mid, F);
}
#endif
