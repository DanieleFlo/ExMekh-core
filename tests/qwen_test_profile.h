#ifndef DS4_TEST_QWEN_PROFILE_H
#define DS4_TEST_QWEN_PROFILE_H

#include "../ds4.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Model-backed tests are explicitly opt-in. On Apple Silicon their default
 * budget is 8k/64; never silently switch a refused GPU test to CPU inference.
 */
static inline int qwen_test_positive_env(const char *name, int fallback, int *out) {
    const char *value = getenv(name);
    if (!value || !value[0]) {
        *out = fallback;
        return 1;
    }
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed <= 0 || parsed > INT_MAX) {
        fprintf(stderr, "%s must be a positive integer\n", name);
        return 0;
    }
    *out = (int)parsed;
    return 1;
}
static inline int qwen_test_profile(ds4_engine_options *options, int cuda_context) {
    ds4_engine_options candidate = *options;
    const char *backend = getenv("DS4_TEST_BACKEND");
    if (!backend || !backend[0]) {
#ifdef __APPLE__
        backend = "metal";
#else
        backend = "cuda";
#endif
    }
    if (!strcmp(backend, "metal"))
        candidate.backend = DS4_BACKEND_METAL;
    else if (!strcmp(backend, "cuda"))
        candidate.backend = DS4_BACKEND_CUDA;
    else {
        fprintf(stderr, "DS4_TEST_BACKEND must be metal or cuda\n");
        return 0;
    }
    const bool metal = candidate.backend == DS4_BACKEND_METAL;
    int prefill = 0;
    if (!qwen_test_positive_env("DS4_TEST_CTX", metal ? 8192 : cuda_context,
                                &candidate.context_size))
        return 0;
    int default_prefill = metal ? 64 : 2048;
    if (default_prefill > candidate.context_size)
        default_prefill = candidate.context_size;
    if (!qwen_test_positive_env("DS4_TEST_PREFILL", default_prefill, &prefill))
        return 0;
    if (prefill > candidate.context_size) {
        fprintf(stderr, "DS4_TEST_PREFILL exceeds DS4_TEST_CTX\n");
        return 0;
    }
    candidate.prefill_chunk = (uint32_t)prefill;
    const char *streamed = getenv("DS4_TEST_SSD_STREAMING");
    if (streamed && streamed[0]) {
        if (!metal || (strcmp(streamed, "0") && strcmp(streamed, "1"))) {
            fprintf(stderr, "DS4_TEST_SSD_STREAMING requires Metal and 0 or 1\n");
            return 0;
        }
        candidate.ssd_streaming = !strcmp(streamed, "1");
        candidate.ssd_streaming_cold = candidate.ssd_streaming;
        if (candidate.ssd_streaming && !candidate.ssd_streaming_cache_bytes)
            candidate.ssd_streaming_cache_bytes = 2ull << 30;
    }
    *options = candidate;
    return 1;
}
#endif
