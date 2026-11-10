#include "qwen_test_profile.h"
/* Keep test actions active even when the caller builds with NDEBUG. */
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    ds4_engine_options options = {0};
    unsetenv("DS4_TEST_SSD_STREAMING");
    unsetenv("DS4_TEST_CTX");
    unsetenv("DS4_TEST_PREFILL");
    setenv("DS4_TEST_BACKEND", "metal", 1);
    CHECK(qwen_test_profile(&options, 16384));
    CHECK(options.backend == DS4_BACKEND_METAL && options.context_size == 8192 &&
          options.prefill_chunk == 64);
    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    CHECK(qwen_test_profile(&options, 16384) && options.ssd_streaming &&
          options.ssd_streaming_cold && options.ssd_streaming_cache_bytes == (2ull << 30));
    setenv("DS4_TEST_SSD_STREAMING", "invalid", 1);
    ds4_engine_options streamed = options;
    CHECK(!qwen_test_profile(&options, 16384) && !memcmp(&options, &streamed, sizeof(options)));
    setenv("DS4_TEST_SSD_STREAMING", "0", 1);
    CHECK(qwen_test_profile(&options, 16384) && !options.ssd_streaming);
    unsetenv("DS4_TEST_SSD_STREAMING");
    setenv("DS4_TEST_BACKEND", "cuda", 1);
    CHECK(qwen_test_profile(&options, 16384));
    CHECK(options.backend == DS4_BACKEND_CUDA && options.context_size == 16384 &&
          options.prefill_chunk == 2048);
    setenv("DS4_TEST_CTX", "128", 1);
    setenv("DS4_TEST_PREFILL", "8", 1);
    CHECK(qwen_test_profile(&options, 16384));
    CHECK(options.context_size == 128 && options.prefill_chunk == 8);
    ds4_engine_options valid = options;
    const char *bad[] = {"-1", "0", "99999999999999999", "64x"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        setenv("DS4_TEST_CTX", bad[i], 1);
        CHECK(!qwen_test_profile(&options, 16384));
        CHECK(!memcmp(&options, &valid, sizeof(options)));
    }
    setenv("DS4_TEST_CTX", "4", 1);
    CHECK(!qwen_test_profile(&options, 16384));
    setenv("DS4_TEST_BACKEND", "cpu", 1);
    CHECK(!qwen_test_profile(&options, 16384));
    CHECK(!memcmp(&options, &valid, sizeof(options)));
    puts("PASS Qwen live-test backend and explicit context/prefill budgets");
}
