#include "../ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <float.h>

/* Independent float64 oracle, with the HF [11,11,10] frequency assignment.
 * This checks Q/K normalization, spatial rotation, cache-only MTP and exact
 * legacy text dispatch; it does not duplicate the attention implementation. */
static const unsigned axis[32] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                  1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1};
static void oracle_head(double y[256], const float *x, const float *norm, const unsigned pos[3]) {
    double square = 0;
    for (unsigned d = 0; d < 256; d++)
        square += (double)x[d] * x[d];
    for (unsigned d = 0; d < 256; d++)
        y[d] = x[d] / sqrt(square / 256 + 1e-6) * norm[d];
    for (unsigned d = 0; d < 32; d++) {
        double angle = pos[axis[d]] / pow(1e7, 2.0 * d / 64), a = y[d], b = y[d + 32];
        y[d] = a * cos(angle) - b * sin(angle);
        y[d + 32] = a * sin(angle) + b * cos(angle);
    }
}
/* Materialize the complete causal score row in double precision. This oracle
 * shares neither the online softmax nor the kernel's prepared Q/K values. */
static double attention_error(const float *raw_q, const float *raw_k, const float *v,
                              const float *actual, const float *norm, const unsigned pos[9]) {
    double worst = 0;
    for (unsigned t = 0; t < 3; t++)
        for (unsigned h = 0; h < 24; h++) {
            double q[256], keys[3][256], scores[3] = {0}, total = 0, maximum = -DBL_MAX;
            oracle_head(q, raw_q + t * 12288 + h * 512, norm, pos + t * 3);
            for (unsigned j = 0; j <= t; j++) {
                oracle_head(keys[j], raw_k + j * 1024 + (h / 6) * 256, norm + 256, pos + j * 3);
                for (unsigned d = 0; d < 256; d++)
                    scores[j] += q[d] * keys[j][d] / 16;
                if (scores[j] > maximum)
                    maximum = scores[j];
            }
            for (unsigned j = 0; j <= t; j++) {
                scores[j] = exp(scores[j] - maximum);
                total += scores[j];
            }
            for (unsigned d = 0; d < 256; d++) {
                double value = 0;
                for (unsigned j = 0; j <= t; j++)
                    value += scores[j] / total * v[j * 1024 + (h / 6) * 256 + d];
                value /= 1 + exp(-(double)raw_q[t * 12288 + h * 512 + 256 + d]);
                double error = fabs(value - actual[(t * 24 + h) * 256 + d]);
                if (error > worst)
                    worst = error;
            }
        }
    return worst;
}
static double oracle_error(const float *raw, const float *actual, unsigned stride, unsigned heads,
                           const unsigned pos[9], const float *norm) {
    double worst = 0;
    for (unsigned t = 0; t < 3; t++)
        for (unsigned h = 0; h < heads; h++) {
            const float *x = raw + t * stride + h * (heads == 24 ? 512 : 256);
            const float *y = actual + t * stride + h * (heads == 24 ? 512 : 256);
            double sum = 0, v[256];
            for (unsigned d = 0; d < 256; d++)
                sum += (double)x[d] * x[d];
            for (unsigned d = 0; d < 256; d++)
                v[d] = x[d] / sqrt(sum / 256 + 1e-6) * norm[d];
            for (unsigned d = 0; d < 32; d++) {
                double angle = pos[t * 3 + axis[d]] / pow(1e7, 2.0 * d / 64);
                double a = v[d], b = v[d + 32];
                v[d] = a * cos(angle) - b * sin(angle);
                v[d + 32] = a * sin(angle) + b * cos(angle);
            }
            for (unsigned d = 0; d < 256; d++)
                if (fabs(v[d] - y[d]) > worst)
                    worst = fabs(v[d] - y[d]);
        }
    return worst;
}
#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c);                                   \
            rc = 1;                                                                                \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
int main(void) {
    int rc = 0;
    const size_t weight_bytes = 512 * sizeof(float);
    size_t map_bytes =
        (weight_bytes + (size_t)getpagesize() - 1) / (size_t)getpagesize() * (size_t)getpagesize();
    float *weights = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (weights == MAP_FAILED)
        return 1;
    float q[3 * 12288], k[3 * 1024], v[3 * 1024];
    float legacy[3 * 6144], actual[3 * 6144], qr[3 * 12288], kr[3 * 1024], cached[3 * 1024];
    unsigned text[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2},
             spatial[9] = {10, 20, 30, 10, 20, 31, 40, 40, 40};
    ds4_gpu_tensor *qt = NULL, *kt = NULL, *vt = NULL, *kc = NULL, *vc = NULL, *out = NULL,
                   *pos = NULL;
    for (unsigned i = 0; i < 512; i++)
        weights[i] = .8f + (i % 17) * .01f;
    for (unsigned i = 0; i < 3 * 12288; i++)
        q[i] = sinf(i * .013f);
    for (unsigned i = 0; i < 3 * 1024; i++) {
        k[i] = cosf(i * .017f);
        v[i] = sinf(i * .021f);
    }
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(weights, weight_bytes));
    qt = ds4_gpu_tensor_alloc(sizeof(q));
    kt = ds4_gpu_tensor_alloc(sizeof(k));
    vt = ds4_gpu_tensor_alloc(sizeof(v));
    kc = ds4_gpu_tensor_alloc(sizeof(k));
    vc = ds4_gpu_tensor_alloc(sizeof(v));
    out = ds4_gpu_tensor_alloc(sizeof(legacy));
    pos = ds4_gpu_tensor_alloc(sizeof(spatial));
    CHECK(qt && kt && vt && kc && vc && out && pos);
    CHECK(ds4_gpu_tensor_write(qt, 0, q, sizeof(q)) && ds4_gpu_tensor_write(kt, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(vt, 0, v, sizeof(v)));
    CHECK(ds4_gpu_qwen35_full_attention_rows_tensor(out, qt, kt, vt, kc, vc, weights, weight_bytes,
                                                    0, 256 * 4, 0, 3, 3));
    CHECK(ds4_gpu_tensor_read(out, 0, legacy, sizeof(legacy)));
    CHECK(ds4_gpu_tensor_write(qt, 0, q, sizeof(q)) && ds4_gpu_tensor_write(kt, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(pos, 0, text, sizeof(text)));
    CHECK(ds4_gpu_qwen35_full_attention_mrope_rows_tensor(out, qt, kt, vt, kc, vc, weights,
                                                          weight_bytes, 0, 256 * 4, 0, 3, 3, pos));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    CHECK(!memcmp(legacy, actual, sizeof(actual)));
    CHECK(ds4_gpu_tensor_write(qt, 0, q, sizeof(q)) && ds4_gpu_tensor_write(kt, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(pos, 0, spatial, sizeof(spatial)));
    CHECK(ds4_gpu_qwen35_full_attention_mrope_rows_tensor(out, qt, kt, vt, kc, vc, weights,
                                                          weight_bytes, 0, 256 * 4, 0, 3, 3, pos));
    CHECK(ds4_gpu_tensor_read(qt, 0, qr, sizeof(qr)) &&
          ds4_gpu_tensor_read(kt, 0, kr, sizeof(kr)) &&
          ds4_gpu_tensor_read(kc, 0, cached, sizeof(cached)));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    double qe = oracle_error(q, qr, 12288, 24, spatial, weights),
           ke = oracle_error(k, kr, 1024, 4, spatial, weights + 256);
    double ae = attention_error(q, k, v, actual, weights, spatial);
    CHECK(ae < 2e-5);
    CHECK(qe < 2e-5 && ke < 2e-5);
    CHECK(!memcmp(kr, cached, sizeof(kr)));
    CHECK(ds4_gpu_tensor_write(kt, 0, k, sizeof(k)));
    CHECK(ds4_gpu_qwen35_store_kv_mrope_rows_tensor(kt, vt, kc, vc, weights, weight_bytes, 256 * 4,
                                                    0, 3, 3, pos));
    CHECK(ds4_gpu_tensor_read(kc, 0, cached, sizeof(cached)));
    CHECK(!memcmp(kr, cached, sizeof(kr)));
    printf("PASS mRoPE: text bit-exact; CPU float64 Q max_abs=%.9g K max_abs=%.9g attention "
           "max_abs=%.9g; MTP cache bit-exact\n",
           qe, ke, ae);
done:
    ds4_gpu_tensor_free(qt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kc);
    ds4_gpu_tensor_free(vc);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(pos);
    ds4_gpu_cleanup();
    munmap(weights, map_bytes);
    return rc;
}
