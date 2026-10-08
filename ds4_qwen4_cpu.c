#include "ds4_qwen4_cpu_internal.h"

void ds4_qwen4_cpu_expert_run(const uint8_t *gate, const uint8_t *up,
        const uint8_t *down, unsigned gu_type, unsigned down_type,
        const float *x, unsigned E, unsigned F, float *mid, float *out) {
    ds4_qwen4_cpu_expert(gate, up, down, gu_type, down_type, x, E, F, mid, out);
}

/* Dequantize once for shared routed rows; retain CUDA's four values per
 * warp lane and fused FP32 accumulation order. */
static void dot_rows(unsigned type, const uint8_t *row, const float *const *x,
                     unsigned rows, unsigned n, float *out) {
    float values[256];
#if defined(__AVX2__) && defined(__FMA__)
    __m256 packed[4][4];
    const __m256i indices = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned s = 0; s < 4u; ++s) packed[r][s] = _mm256_setzero_ps();
#else
    float acc[4][32] = {{0}};
#endif
    for (unsigned b = 0; b < (n + 255u) / 256u; b++) {
        ds4_qwen4_cpu_block(type, row + (uint64_t)b * (type == 16 ? 66u : 84u), values);
        const unsigned width = n - b * 256u < 256u ? n - b * 256u : 256u;
        for (unsigned j = 0; j < width; j += 128u) {
#if defined(__AVX2__) && defined(__FMA__)
            for (unsigned s = 0; s < 4u && j + s * 32u < width; ++s)
                for (unsigned k = 0; k < 4u; ++k) {
                    const unsigned i = j + s * 32u + k;
                    const __m256 w = _mm256_i32gather_ps(values + i, indices, 4);
                    for (unsigned r = 0; r < rows; ++r)
                        packed[r][s] = _mm256_fmadd_ps(w,
                            _mm256_i32gather_ps(x[r] + b * 256u + i, indices, 4), packed[r][s]);
                }
#else
            for (unsigned lane = 0; lane < 32u && j + lane * 4u < width; ++lane)
                for (unsigned k = 0; k < 4u; ++k) {
                    const unsigned i = j + lane * 4u + k;
                    for (unsigned r = 0; r < rows; ++r)
                        acc[r][lane] = fmaf(values[i], x[r][b * 256u + i], acc[r][lane]);
                }
#endif
        }
    }
    for (unsigned r = 0; r < rows; r++) {
#if defined(__AVX2__) && defined(__FMA__)
        float acc[32];
        for (unsigned s = 0; s < 4u; ++s) _mm256_storeu_ps(acc + s * 8u, packed[r][s]);
        for (unsigned d = 16; d; d >>= 1)
            for (unsigned lane = 0; lane < d; ++lane) acc[lane] += acc[lane + d];
        out[r] = acc[0];
#else
        for (unsigned d = 16; d; d >>= 1)
            for (unsigned lane = 0; lane < d; ++lane) acc[r][lane] += acc[r][lane + d];
        out[r] = acc[r][0];
#endif
    }
}

void ds4_qwen4_cpu_expert_rows_run(const uint8_t *gate, const uint8_t *up,
        const uint8_t *down, unsigned gu_type, unsigned down_type,
        const float *const *x, unsigned rows, unsigned E, unsigned F,
        float *mid, float *const *out) {
    if (!rows || rows > 4) return;
    const uint64_t gr = ds4_qwen4_cpu_row_bytes(gu_type, E), dr = ds4_qwen4_cpu_row_bytes(down_type, F);
    const float *mx[4];
    for (unsigned r = 0; r < rows; r++) mx[r] = mid + (uint64_t)r * F;
    for (unsigned f = 0; f < F; f++) {
        float g[4], u[4];
        dot_rows(gu_type, gate + f * gr, x, rows, E, g);
        dot_rows(gu_type, up + f * gr, x, rows, E, u);
        for (unsigned r = 0; r < rows; r++) mid[(uint64_t)r * F + f] = ds4_qwen4_cpu_silu(g[r]) * u[r];
    }
    for (unsigned d = 0; d < E; d++) {
        float values[4]; dot_rows(down_type, down + d * dr, mx, rows, F, values);
        for (unsigned r = 0; r < rows; r++) out[r][d] = values[r];
    }
}

void ds4_qwen4_cpu_gate_up_rows_run(const uint8_t *gate, const uint8_t *up,
        unsigned type, const float *const *x, unsigned rows, unsigned E, unsigned F,
        float *const *gate_out, float *const *up_out) {
    if (!rows || rows > 4u) return;
    const uint64_t bytes = ds4_qwen4_cpu_row_bytes(type, E);
    for (unsigned f = 0; f < F; ++f) {
        float g[4], u[4];
        dot_rows(type, gate + f * bytes, x, rows, E, g);
        dot_rows(type, up + f * bytes, x, rows, E, u);
        for (unsigned r = 0; r < rows; ++r) { gate_out[r][f] = g[r]; up_out[r][f] = u[r]; }
    }
}

void ds4_qwen4_cpu_down_rows_run(const uint8_t *down, unsigned type,
        const float *const *mid, unsigned rows, unsigned E, unsigned F, float *const *out) {
    if (!rows || rows > 4u) return;
    const uint64_t bytes = ds4_qwen4_cpu_row_bytes(type, F);
    for (unsigned d = 0; d < E; ++d) {
        float values[4]; dot_rows(type, down + d * bytes, mid, rows, F, values);
        for (unsigned r = 0; r < rows; ++r) out[r][d] = values[r];
    }
}
