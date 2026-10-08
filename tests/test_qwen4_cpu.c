#include "../ds4_qwen4_cpu_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t random_state = 17;
static uint32_t next(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}
static float half_ref(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 31u, m = h & 1023u, bits;
    if (!e && m) {
        e = 113;
        while (!(m & 1024u)) { m <<= 1; e--; }
        bits = sign | e << 23 | (m & 1023u) << 13;
    } else bits = sign | (e == 31 ? 0x7f800000u | m << 13 : e ? (e + 112u) << 23 | m << 13 : 0);
    float value; memcpy(&value, &bits, 4); return value;
}
static double value_ref(unsigned type, const uint8_t *row, unsigned j) {
    const uint8_t *b = row + (uint64_t)(j / 256u) * (type == 16 ? 66u : 84u);
    j %= 256;
    if (type == 16) {
        uint16_t h; uint32_t ids, aux;
        memcpy(&h, b, 2); memcpy(&ids, b + 2 + (j / 32u) * 8u, 4);
        memcpy(&aux, b + 6 + (j / 32u) * 8u, 4);
        const unsigned sub = (j % 32u) / 8u;
        const uint64_t grid = ds4_qwen4_cpu_grid[(ids >> (sub * 8u)) & 255u];
        const unsigned raw = (aux >> (sub * 7u)) & 127u;
        unsigned parity = raw;
        parity ^= parity >> 4; parity ^= parity >> 2; parity ^= parity >> 1;
        const unsigned signs = raw | ((parity & 1u) << 7);
        return half_ref(h) * ((aux >> 28) + 0.5) * 0.25 *
            ((grid >> (8u * (j % 8u))) & 255u) * ((signs & (1u << (j % 8u))) ? -1 : 1);
    }
    uint16_t h, mh; memcpy(&h, b + 80, 2); memcpy(&mh, b + 82, 2);
    const unsigned scale = b[j / 16u];
    const unsigned q = (b[16u + (j / 128u) * 32u + j % 32u] >> (2u * ((j % 128u) / 32u))) & 3u;
    return (double)half_ref(h) * (scale & 15u) * q - (double)half_ref(mh) * (scale >> 4);
}
static double dot_ref(unsigned type, const uint8_t *row, const float *x, unsigned n) {
    double sum = 0;
    for (unsigned j = 0; j < n; j++) sum += value_ref(type, row, j) * x[j];
    return sum;
}
static uint8_t *weights(unsigned type, unsigned rows, unsigned cols) {
    const uint64_t bytes = ds4_qwen4_cpu_row_bytes(type, cols) * rows;
    uint8_t *w = malloc(bytes); assert(w);
    for (uint64_t i = 0; i < bytes; i++) w[i] = (uint8_t)(next() >> 24);
    for (uint64_t off = 0; off < bytes; off += type == 16 ? 66u : 84u) {
        const uint16_t h = 0x1801, mh = 0x1401;
        memcpy(w + off + (type == 16 ? 0 : 80), &h, 2);
        if (type == 10) memcpy(w + off + 82, &mh, 2);
    }
    return w;
}
int main(void) {
    for (unsigned h = 0; h < 65536; h++) {
        float a = half_ref((uint16_t)h), b = ds4_qwen4_cpu_half((uint16_t)h);
        assert(!memcmp(&a, &b, 4));
    }
    const unsigned E = 256, F = 672;
    uint8_t *g = weights(16, F, E), *u = weights(16, F, E), *d = weights(10, E, F);
    float x[4][256], out[4][256], single[256], mid[4 * 672], one[672];
    const float *inputs[4]; float *outputs[4];
    for (unsigned r = 0; r < 4; r++) {
        inputs[r] = x[r]; outputs[r] = out[r];
        for (unsigned j = 0; j < E; j++) x[r][j] = ((int)(next() >> 16) - 32768) / 32768.0f;
    }
    for (unsigned rows = 1; rows <= 4; rows++) {
        ds4_qwen4_cpu_expert_rows_run(g, u, d, 16, 10, inputs, rows, E, F, mid, outputs);
        for (unsigned r = 0; r < rows; r++) {
            ds4_qwen4_cpu_expert_run(g, u, d, 16, 10, inputs[r], E, F, one, single);
            assert(!memcmp(single, out[r], sizeof(single)));
            float oracle_mid[672];
            for (unsigned j = 0; j < F; j++) {
                double a = dot_ref(16, g + j * 66u, inputs[r], E), b = dot_ref(16, u + j * 66u, inputs[r], E);
                oracle_mid[j] = (float)(a / (1.0 + exp(-a)) * b);
                assert(fabs(mid[r * F + j] - oracle_mid[j]) <= 2e-5 * fmax(1, fabs(oracle_mid[j])));
            }
            for (unsigned j = 0; j < E; j++) {
                double ref = dot_ref(10, d + j * ds4_qwen4_cpu_row_bytes(10, F), oracle_mid, F);
                assert(fabs(out[r][j] - ref) <= 2e-5 * fmax(1, fabs(ref)));
            }
        }
    }
    free(g); free(u); free(d);
    puts("PASS: all FP16 encodings, FP64 MoE oracle, padded down and grouped CPU rows");
    return 0;
}
