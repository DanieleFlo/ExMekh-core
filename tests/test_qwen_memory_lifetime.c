/* CUDA/Qwen27 live audit: repeated draft windows and session teardown must
 * stabilize after warm-up. File-backed GGUF RSS is distinct from anonymous RAM. */
#include "../ds4_gpu.h"
#include <stddef.h>

static int allocations_until_failure = -1;
static ds4_gpu_tensor *memory_test_alloc(uint64_t bytes) {
    if (allocations_until_failure == 0) return NULL;
    if (allocations_until_failure > 0) allocations_until_failure--;
    return ds4_gpu_tensor_alloc(bytes);
}

/* Fault injection affects this test translation unit, not the GPU backend. */
#define ds4_gpu_tensor_alloc memory_test_alloc
#include "../ds4.c"
#undef ds4_gpu_tensor_alloc
#include <malloc.h>

/* Test-only CUDA measurement; the production session API stays backend-neutral. */
extern int cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes);

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s (%s)\n", __LINE__, #c, err); exit(1); } } while (0)

typedef struct { uint64_t gpu_used, rss_kib, anonymous_kib; } memory_sample;

static memory_sample measure(const char *stage, int cycle, ds4_session *session) {
    memory_sample sample = {0};
    size_t free_bytes = 0, total_bytes = 0;
    if (!ds4_gpu_synchronize() || cudaMemGetInfo(&free_bytes, &total_bytes)) exit(1);
    sample.gpu_used = total_bytes - free_bytes;
    FILE *fp = fopen("/proc/self/smaps_rollup", "r");
    if (!fp) exit(1);
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long value;
        if (sscanf(line, "Rss: %llu kB", &value) == 1) sample.rss_kib = value;
        if (sscanf(line, "Anonymous: %llu kB", &value) == 1) sample.anonymous_kib = value;
    }
    fclose(fp);
    uint64_t draft_bytes = 0;
    if (session) {
        ds4_qwen_gpu_graph *g = &session->qwen_graph;
        draft_bytes = ds4_gpu_tensor_bytes(g->draft_hidden_rows) + ds4_gpu_tensor_bytes(g->draft_logits);
        for (unsigned layer = 0; layer < DS4_N_LAYER; layer++)
            draft_bytes += ds4_gpu_tensor_bytes(g->draft_conv_snapshot[layer]) +
                ds4_gpu_tensor_bytes(g->draft_recurrent_snapshot[layer]);
    }
    printf("{\"stage\":\"%s\",\"cycle\":%d,\"gpu_global_used_bytes\":%" PRIu64
           ",\"rss_kib\":%" PRIu64 ",\"anonymous_kib\":%" PRIu64
           ",\"draft_workspace_bytes\":%" PRIu64 "}\n",
           stage, cycle, sample.gpu_used, sample.rss_kib, sample.anonymous_kib, draft_bytes);
    fflush(stdout);
    return sample;
}

static int accept_proposal(void *opaque, const float *logits, int vocab,
                           const int *prefix, int count, uint64_t *rng) {
    (void)logits; (void)vocab; (void)prefix; (void)rng;
    return ((const int *)opaque)[count];
}

static int decline_proposal(void *opaque, const float *logits, int vocab,
                            const int *prefix, int count, uint64_t *rng) {
    (void)opaque; (void)logits; (void)vocab; (void)prefix; (void)count;
    *rng += 7;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) return 2;
    char err[256] = {0};
    ds4_engine *engine = NULL;
    ds4_engine_options options = {.model_path=argv[1], .mtp_path=argc == 3 ? argv[2] : NULL,
        .backend=DS4_BACKEND_CUDA, .context_size=4096, .prefill_chunk=2048,
        .mtp_draft_tokens=argc == 3 ? 2 : 0};
    CHECK(ds4_engine_open(&engine, &options) == 0);
    measure("engine_ready", 0, NULL);
    ds4_tokens pattern = {0}, prompt = {0}, draft = {0};
    ds4_tokenize_text(engine, "A known source file is copied exactly.\n", &pattern);
    CHECK(pattern.len > 0);
    for (int i = 0; i < 2053; i++) ds4_tokens_push(&prompt, pattern.v[i % pattern.len]);
    for (int i = 0; i < 20; i++) ds4_tokenize_text(engine, "unique source α中🙂 <p>hello &amp; world</p>\r\n", &draft);
    CHECK(draft.len > 64);
    memory_sample free_baseline = {0};
    for (int lifecycle = 0; lifecycle < 4; lifecycle++) {
        ds4_session *session = NULL;
        CHECK(ds4_session_create(&session, engine, options.context_size) == 0);
        CHECK(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        measure("prefill_done", lifecycle, session);
        FILE *base = tmpfile(); CHECK(base);
        uint64_t base_bytes = ds4_session_payload_bytes(session);
        CHECK(ds4_session_save_payload(session, base, err, sizeof(err)) == 0);
        CHECK(fflush(base) == 0);
        if (!lifecycle) {
            float *expected = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
            float *actual = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
            CHECK(ds4_session_eval(session, draft.v[0], err, sizeof(err)) == 0);
            CHECK(ds4_session_copy_logits(session, expected, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
            const int failures[] = {0, 1, 10};
            for (size_t i = 0; i < sizeof(failures)/sizeof(failures[0]); i++) {
                rewind(base);
                CHECK(ds4_session_load_payload(session, base, base_bytes, err, sizeof(err)) == 0);
                allocations_until_failure = failures[i];
                int accepted[64], next = -1; uint64_t rng = 42;
                int kept = ds4_session_eval_draft_tokens(session, draft.v, 64, accepted, &next,
                    &rng, accept_proposal, draft.v, err, sizeof(err));
                allocations_until_failure = -1;
                CHECK(kept == 1 && accepted[0] == draft.v[0] && next == -1 && rng == 42);
                CHECK(ds4_session_pos(session) == prompt.len + 1);
                CHECK(ds4_session_copy_logits(session, actual, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
                CHECK(!memcmp(expected, actual, (size_t)DS4_N_VOCAB * sizeof(float)));
                CHECK(!session->qwen_graph.draft_hidden_rows && !session->qwen_graph.draft_logits);
                for (unsigned layer = 0; layer < DS4_N_LAYER; layer++)
                    CHECK(!session->qwen_graph.draft_conv_snapshot[layer] &&
                        !session->qwen_graph.draft_recurrent_snapshot[layer]);
                measure("allocation_failure_released", failures[i], session);
            }
            rewind(base);
            CHECK(ds4_session_load_payload(session, base, base_bytes, err, sizeof(err)) == 0);
            int accepted[64], next = -1; uint64_t rng = 42;
            CHECK(ds4_session_eval_draft_tokens(session, draft.v, 64, accepted, &next,
                &rng, decline_proposal, NULL, err, sizeof(err)) == 1);
            CHECK(next == -1 && rng == 42);
            CHECK(ds4_session_copy_logits(session, actual, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
            CHECK(!memcmp(expected, actual, (size_t)DS4_N_VOCAB * sizeof(float)));
            CHECK(!session->qwen_graph.draft_hidden_rows && !session->qwen_graph.draft_logits);
            for (unsigned layer = 0; layer < DS4_N_LAYER; layer++)
                CHECK(!session->qwen_graph.draft_conv_snapshot[layer] &&
                    !session->qwen_graph.draft_recurrent_snapshot[layer]);
            measure("declined_window_released", 0, session);
            free(expected); free(actual);
        }
        memory_sample steady = {0};
        for (int repeat = 0; repeat < 8; repeat++) {
            rewind(base);
            CHECK(ds4_session_load_payload(session, base, base_bytes, err, sizeof(err)) == 0);
            int accepted[64], next = -1; uint64_t rng = 42;
            CHECK(ds4_session_eval_draft_tokens(session, draft.v, 64, accepted, &next,
                &rng, accept_proposal, draft.v, err, sizeof(err)) == 64);
            CHECK(next == -1 && rng == 42);
            memory_sample current = measure("draft_done", lifecycle * 8 + repeat, session);
            if (repeat == 2) steady = current;
            /* Runtime/cuBLAS/allocator warm-up is bounded; subsequent windows
             * must not grow one workspace allocation per call. */
            if (repeat > 2) {
                /* cudaMemGetInfo includes other WSL/Windows GPU clients. */
                CHECK(current.gpu_used <= steady.gpu_used + 128ull * 1048576);
                CHECK(current.anonymous_kib <= steady.anonymous_kib + 32ull * 1024);
            }
        }
        fclose(base);
        ds4_session_free(session);
        memory_sample current = measure("session_freed", lifecycle, NULL);
        if (!lifecycle) free_baseline = current;
        else {
            CHECK(current.gpu_used <= free_baseline.gpu_used + 128ull * 1048576);
            CHECK(current.anonymous_kib <= free_baseline.anonymous_kib + 32ull * 1024);
        }
    }
    ds4_tokens_free(&pattern); ds4_tokens_free(&prompt); ds4_tokens_free(&draft);
    ds4_engine_close(engine);
    measure("engine_closed", 0, NULL);
    /* Diagnostic only: distinguish free glibc arena pages from live data. */
    malloc_trim(0);
    measure("allocator_trimmed", 0, NULL);
    fprintf(stderr, "PASS: repeated OLD draft windows and session teardown stabilize memory\n");
    return 0;
}
