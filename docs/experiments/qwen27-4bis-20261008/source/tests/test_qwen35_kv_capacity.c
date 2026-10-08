#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CTX = 65536, ROW = 1024, MAX_T = 512 };
static float pattern(size_t i, unsigned salt) {
    return (float)((int)((i * 131u + salt * 73u) % 997u) - 498) / 1024.0f;
}
static ds4_gpu_tensor *upload(const void *p, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes); assert(t);
    if (p) assert(ds4_gpu_tensor_write(t, 0, p, bytes));
    return t;
}
static int attention(ds4_gpu_tensor *out, ds4_gpu_tensor *q, ds4_gpu_tensor *k,
        ds4_gpu_tensor *v, ds4_gpu_tensor **cache, const float *norm,
        unsigned position, unsigned tokens, unsigned multimodal, ds4_gpu_tensor *positions) {
    if (tokens > 5u && tokens <= 16u)
        return ds4_gpu_qwen35_full_attention_verify_rows_tensor(out, q, k, v,
            cache[0], cache[1], norm, 512u * 4u, 0, 256u * 4u, position, tokens, CTX,
            multimodal ? positions : NULL);
    if (multimodal)
        return ds4_gpu_qwen35_full_attention_mrope_rows_tensor(out, q, k, v,
            cache[0], cache[1], norm, 512u * 4u, 0, 256u * 4u, position, tokens, CTX, positions);
    return ds4_gpu_qwen35_full_attention_rows_tensor(out, q, k, v,
        cache[0], cache[1], norm, 512u * 4u, 0, 256u * 4u, position, tokens, CTX);
}
int main(void) {
    assert(ds4_gpu_init());
    float norm[512]; for (unsigned i = 0; i < 512; ++i) norm[i] = 1.0f + (i % 7u) / 64.0f;
    assert(ds4_gpu_set_model_map(norm, sizeof(norm)));
    float *past[2], *q = malloc((size_t)MAX_T * 12288u * 4u);
    float *rows[2], *result = malloc((size_t)MAX_T * 6144u * 4u), *reference = malloc((size_t)MAX_T * 6144u * 4u);
    float *kv_reference[2], *kv_result = malloc((size_t)CTX * ROW * 4u);
    uint32_t pos3[MAX_T * 3];
    assert(q && result && reference && kv_result);
    for (unsigned i = 0; i < 2; ++i) {
        past[i] = malloc((size_t)CTX * ROW * 4u); rows[i] = malloc((size_t)MAX_T * ROW * 4u);
        kv_reference[i] = malloc((size_t)CTX * ROW * 4u); assert(past[i] && rows[i] && kv_reference[i]);
        for (size_t j = 0; j < (size_t)CTX * ROW; ++j) past[i][j] = pattern(j, 11u + i);
        for (size_t j = 0; j < (size_t)MAX_T * ROW; ++j) rows[i][j] = pattern(j, 17u + i);
    }
    for (size_t j = 0; j < (size_t)MAX_T * 12288u; ++j) q[j] = pattern(j, 23u);
    const unsigned cases[][3] = {{63,3,0}, {127,1,0}, {127,5,1}, {511,512,1},
        {4093,5,1}, {8190,2,0}, {32765,3,0}, {65531,5,1}, {65534,2,0},
        {65520,8,0}, {65520,15,1}, {65520,16,0}};
    const char *modes[] = {"off", "grow", "hybrid-attention", "hybrid-attention", "hybrid-attention"};
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        const unsigned position = cases[c][0], tokens = cases[c][1], multimodal = cases[c][2];
        const size_t output_bytes = (size_t)tokens * 6144u * 4u;
        const size_t prefix_bytes = (size_t)(position + tokens) * ROW * 4u;
        for (unsigned j = 0; j < tokens; ++j) {
            pos3[j * 3u] = position + j; pos3[j * 3u + 1] = j % 11u; pos3[j * 3u + 2] = j % 13u;
        }
        for (unsigned m = 0; m < 5; ++m) {
            setenv("DS4_CUDA_QWEN_KV_CAPACITY_MODE", modes[m], 1);
            setenv("DS4_CUDA_QWEN_KV_RESIDENT", "64", 1);
            setenv("DS4_CUDA_QWEN_MTP_KV_MODE", m == 3 ? "resident" : m == 4 ? "auto" : "shared", 1);
            /* The target keeps its small physical budget while NextN grows
             * independently. Both must coexist and preserve logical IO. */
            ds4_gpu_tensor *control[2] = {0}, *cache[2] = {0};
            if (m >= 3) assert(ds4_gpu_qwen35_kv_alloc(&control[0], &control[1], CTX));
            assert(m >= 3 ? ds4_gpu_qwen35_mtp_kv_alloc(&cache[0], &cache[1], CTX) :
                           ds4_gpu_qwen35_kv_alloc(&cache[0], &cache[1], CTX));
            for (unsigned i = 0; i < 2; ++i) {
                assert(ds4_gpu_tensor_fill_f32(cache[i], 0.0f, (uint64_t)CTX * ROW));
                assert(ds4_gpu_tensor_write(cache[i], 0, past[i], (size_t)position * ROW * 4u));
            }
            ds4_gpu_tensor *tq = upload(q, (size_t)tokens * 12288u * 4u);
            ds4_gpu_tensor *tk = upload(rows[0], (size_t)tokens * ROW * 4u);
            ds4_gpu_tensor *tv = upload(rows[1], (size_t)tokens * ROW * 4u);
            ds4_gpu_tensor *out = upload(NULL, output_bytes), *positions = upload(pos3, tokens * 3u * 4u);
            int ok = attention(out, tq, tk, tv, cache, norm, position, tokens, multimodal, positions);
            assert(ok && ds4_gpu_tensor_read(out, 0, result, output_bytes));
            if (!m) memcpy(reference, result, output_bytes);
            else assert(!memcmp(reference, result, output_bytes));
            if (m == 4) for (unsigned repeat = 0; repeat < 2; ++repeat) {
                if (repeat) {
                    /* A different target prefix consumes the shared prefill
                     * workspace. NextN must reload its own logical history. */
                    for (unsigned i = 0; i < 2; ++i) {
                        assert(ds4_gpu_tensor_fill_f32(control[i], 0.0f, (uint64_t)CTX * ROW));
                        assert(ds4_gpu_tensor_write(control[i], 0, past[1u-i], (size_t)position * ROW * 4u));
                    }
                    assert(ds4_gpu_tensor_write(tq, 0, q, (size_t)tokens * 12288u * 4u));
                    assert(ds4_gpu_tensor_write(tk, 0, rows[0], (size_t)tokens * ROW * 4u));
                    setenv("DS4_CUDA_QWEN_KV_REFERENCE_ATTENTION", "1", 1);
                    assert(attention(out, tq, tk, tv, control, norm, position, tokens, multimodal, positions));
                    assert(ds4_gpu_synchronize());
                    unsetenv("DS4_CUDA_QWEN_KV_REFERENCE_ATTENTION");
                }
                /* Query preparation is in-place; restore the independent
                 * original input before exercising cached KV again. */
                assert(ds4_gpu_tensor_write(tq, 0, q, (size_t)tokens * 12288u * 4u));
                assert(ds4_gpu_tensor_write(tk, 0, rows[0], (size_t)tokens * ROW * 4u));
                assert(attention(out, tq, tk, tv, cache, norm, position, tokens, multimodal, positions));
                assert(ds4_gpu_tensor_read(out, 0, result, output_bytes));
                assert(!memcmp(reference, result, output_bytes));
            }
            for (unsigned i = 0; i < 2; ++i) {
                assert(ds4_gpu_tensor_read(cache[i], 0, kv_result, prefix_bytes));
                if (!m) memcpy(kv_reference[i], kv_result, prefix_bytes);
                else assert(!memcmp(kv_reference[i], kv_result, prefix_bytes));
                /* Logical restore into a new session with a different physical budget. */
                ds4_gpu_tensor *restored[2] = {0};
                if (m == 2) {
                    setenv("DS4_CUDA_QWEN_KV_RESIDENT", "128", 1);
                    setenv("DS4_CUDA_QWEN_MTP_KV_MODE", "resident", 1);
                }
                /* Cross-placement restore: resident NextN -> offloaded
                 * target and conversely, without persisting physical slots. */
                assert(m == 2 ? ds4_gpu_qwen35_mtp_kv_alloc(&restored[0], &restored[1], CTX) :
                                ds4_gpu_qwen35_kv_alloc(&restored[0], &restored[1], CTX));
                setenv("DS4_CUDA_QWEN_KV_RESIDENT", "64", 1);
                assert(ds4_gpu_tensor_write(restored[i], 0, kv_result, prefix_bytes));
                memset(kv_result, 0, prefix_bytes);
                assert(ds4_gpu_tensor_read(restored[i], 0, kv_result, prefix_bytes));
                assert(!memcmp(kv_reference[i], kv_result, prefix_bytes));
                ds4_gpu_tensor_free(restored[0]); ds4_gpu_tensor_free(restored[1]);
            }
            ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tv);
            ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(positions);
            ds4_gpu_tensor_free(cache[0]); ds4_gpu_tensor_free(cache[1]);
            ds4_gpu_tensor_free(control[0]); ds4_gpu_tensor_free(control[1]);
        }
        printf("PASS: dense KV position=%u rows=%u mRoPE=%u; resident/grow/offload/NextN bit-exact\n", position, tokens, multimodal);
    }
    for (unsigned i = 0; i < 2; ++i) { free(past[i]); free(rows[i]); free(kv_reference[i]); }
    free(q); free(result); free(reference); free(kv_result);
    ds4_gpu_cleanup(); return 0;
}
