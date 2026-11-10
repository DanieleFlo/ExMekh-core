/* Opt-in real-model diagnostic: restore parity and mixed prefill/decode
 * segmentation are separate comparisons. No server internals or sampling. */
#include "../ds4.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { CONTEXT_TOKENS = 8192, PREFILL_CHUNK = 64, PROMPT_LIMIT_BYTES = 1048576 };

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #condition, error); \
        goto done; \
    } \
} while (0)

static char *read_prompt(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    struct stat info;
    char *text = NULL;
    if (!fstat(fileno(file), &info) && info.st_size > 0 && info.st_size <= PROMPT_LIMIT_BYTES) {
        text = calloc((size_t)info.st_size + 1, 1);
        if (text && (fread(text, 1, (size_t)info.st_size, file) != (size_t)info.st_size ||
                     memchr(text, '\0', (size_t)info.st_size))) {
            free(text);
            text = NULL;
        }
    }
    if (fclose(file)) { free(text); text = NULL; }
    return text;
}

static int parse_position(const char *value) {
    char *end;
    errno = 0;
    long n = strtol(value, &end, 10);
    return !errno && *value && !*end && n > 0 && n < CONTEXT_TOKENS ? (int)n : 0;
}

static int restore_payload(ds4_session *session, FILE *file, uint64_t bytes,
                            char *error, size_t error_bytes) {
    if (fseek(file, 0, SEEK_SET)) return 1;
    return ds4_session_load_payload(session, file, bytes, error, error_bytes);
}

static int save_logits(const char *directory, const char *name, const float *data, int count) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s.f32", directory, name);
    if (length < 0 || length >= (int)sizeof(path)) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(data, sizeof(*data), (size_t)count, file) == (size_t)count;
    return !fclose(file) && ok;
}

static float *capture_logits(ds4_session *session, const char *directory,
                              const char *name, int vocab) {
    float *values = malloc((size_t)vocab * sizeof(float));
    if (!values) return NULL;
    if (ds4_session_copy_logits(session, values, vocab) != vocab ||
        !save_logits(directory, name, values, vocab)) {
        free(values);
        return NULL;
    }
    return values;
}

typedef struct {
    int direct_top, restored_top, cold_top;
    double restore_max, cold_max;
    bool restore_exact;
} logit_comparison;

static bool compare_logits(const float *direct, const float *restored, const float *cold,
                            int vocab, logit_comparison *out) {
    *out = (logit_comparison){0};
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(direct[i]) || !isfinite(restored[i]) || !isfinite(cold[i])) return false;
        if (direct[i] > direct[out->direct_top]) out->direct_top = i;
        if (restored[i] > restored[out->restored_top]) out->restored_top = i;
        if (cold[i] > cold[out->cold_top]) out->cold_top = i;
        out->restore_max = fmax(out->restore_max, fabs((double)direct[i] - restored[i]));
        out->cold_max = fmax(out->cold_max, fabs((double)direct[i] - cold[i]));
    }
    out->restore_exact = !memcmp(direct, restored, (size_t)vocab * sizeof(float));
    return true;
}

static bool write_comparison(const char *directory, const logit_comparison *c,
                              int vocab, int target, int inner, int start, int frontier, int anchor) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/result.json", directory);
    if (length < 0 || length >= (int)sizeof(path)) return false;
    FILE *report = fopen(path, "w");
    if (!report) return false;
    int written = fprintf(report, "{\"restore_exact\":%s,\"restore_max_abs\":%.9g,"
        "\"cold_max_abs\":%.9g,\"direct_top\":%d,\"restored_top\":%d,"
        "\"cold_top\":%d,\"vocab\":%d,\"target_tokens\":%d,\"inner_tokens\":%d,"
        "\"prefill_tokens\":%d,\"frontier_tokens\":%d,\"anchor_tokens\":%d}\n",
        c->restore_exact ? "true" : "false", c->restore_max, c->cold_max,
        c->direct_top, c->restored_top, c->cold_top, vocab, target, inner, start, frontier, anchor);
    return !fclose(report) && written >= 0;
}

int main(int argc, char **argv) {
    char error[256] = {0};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens target = {0}, inner = {0};
    FILE *checkpoint = NULL;
    char *target_text = NULL, *inner_text = NULL;
    float *direct = NULL, *restored = NULL, *cold = NULL;
    int result = 1;
    REQUIRE(argc == 7 || argc == 8);
    int start = parse_position(argv[5]), frontier = parse_position(argv[6]);
    int anchor = argc == 8 ? parse_position(argv[7]) : start;
    REQUIRE(anchor && anchor <= start && frontier > start);
    REQUIRE(!mkdir(argv[4], 0700));
    target_text = read_prompt(argv[2]);
    inner_text = read_prompt(argv[3]);
    REQUIRE(target_text && inner_text);
    ds4_engine_options options = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = CONTEXT_TOKENS, .prefill_chunk = PREFILL_CHUNK,
        .ssd_streaming = true, .ssd_streaming_cold = true,
        .ssd_streaming_cache_bytes = 2ull << 30,
    };
    REQUIRE(!ds4_engine_open(&engine, &options));
    ds4_tokenize_rendered_chat(engine, target_text, &target);
    ds4_tokenize_rendered_chat(engine, inner_text, &inner);
    REQUIRE(target.len > frontier && target.len <= CONTEXT_TOKENS &&
            inner.len > frontier && inner.len <= CONTEXT_TOKENS);
    REQUIRE(!memcmp(target.v, inner.v, (size_t)frontier * sizeof(int)));
    int vocab = ds4_engine_vocab_size(engine);
    REQUIRE(vocab > 0);
    REQUIRE(!ds4_session_create(&session, engine, CONTEXT_TOKENS));
    ds4_tokens prefix = target;
    prefix.len = anchor;
    REQUIRE(!ds4_session_sync(session, &prefix, error, sizeof(error)));
    prefix.len = start;
    REQUIRE(!ds4_session_sync(session, &prefix, error, sizeof(error)));
    for (int i = start; i < frontier; i++)
        REQUIRE(!ds4_session_eval(session, target.v[i], error, sizeof(error)));
    checkpoint = tmpfile();
    REQUIRE(checkpoint);
    REQUIRE(!ds4_session_save_payload(session, checkpoint, error, sizeof(error)));
    long bytes = ftell(checkpoint);
    REQUIRE(bytes > 0);
    REQUIRE(!ds4_session_sync(session, &target, error, sizeof(error)));
    direct = capture_logits(session, argv[4], "direct", vocab);
    REQUIRE(direct);
    puts("direct continuation complete"); fflush(stdout);
    REQUIRE(!restore_payload(session, checkpoint, (uint64_t)bytes, error, sizeof(error)));
    REQUIRE(!ds4_session_sync(session, &inner, error, sizeof(error)));
    REQUIRE(!restore_payload(session, checkpoint, (uint64_t)bytes, error, sizeof(error)));
    REQUIRE(!ds4_session_sync(session, &target, error, sizeof(error)));
    restored = capture_logits(session, argv[4], "restored", vocab);
    REQUIRE(restored);
    puts("restore after inner forward complete"); fflush(stdout);
    ds4_session_free(session); session = NULL;
    REQUIRE(!ds4_session_create(&session, engine, CONTEXT_TOKENS));
    prefix.len = anchor;
    REQUIRE(!ds4_session_sync(session, &prefix, error, sizeof(error)));
    REQUIRE(!ds4_session_sync(session, &target, error, sizeof(error)));
    cold = capture_logits(session, argv[4], "cold", vocab);
    REQUIRE(cold);
    logit_comparison comparison;
    REQUIRE(compare_logits(direct, restored, cold, vocab, &comparison));
    REQUIRE(write_comparison(argv[4], &comparison, vocab, target.len, inner.len,
                              start, frontier, anchor));
    REQUIRE(comparison.restore_exact);
    result = 0;
done:
    if (checkpoint) fclose(checkpoint);
    ds4_session_free(session);
    ds4_engine_close(engine);
    ds4_tokens_free(&target); ds4_tokens_free(&inner);
    free(target_text); free(inner_text);
    free(direct); free(restored); free(cold);
    return result;
}
