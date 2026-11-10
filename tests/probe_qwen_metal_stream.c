/* Opt-in real-model probe. Public session APIs exercise SSD replay and lifecycle;
 * fixture graph tests separately compare streamed and resident arithmetic. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include "../ds4_sha256.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef DS4_PROBE_SUITE
#include "../ds4_embedding_gemma2.h"
#include "../ds4_qwen_image_pipeline.h"
/* Real standalone adapter, sharing the LLM runtime and physical ledger. The
 * result remains private until the caller has restored the conversation. */
static ds4_status alternate_embedding(ds4_engine *engine, ds4_gemma_result **result) {
    ds4_numeric *numeric = NULL;
    ds4_bundle *bundle = NULL;
    ds4_embedding_gemma2 *embedding = NULL;
    ds4_gemma_input *input = NULL;
    ds4_resources *resources = ds4_engine_resources(engine);
    ds4_status status = ds4_numeric_metal(&numeric, resources, 500);
    if (!status)
        status = ds4_numeric_metal_budget(numeric, ds4_metal_host_reserve_bytes());
    if (!status)
        status = ds4_bundle_open(&bundle, "build/metal-models/gemma.ds4b/bundle.ds4b",
                                 "build/metal-models", resources, 501);
    if (!status)
        status = ds4_embedding_gemma2_open(&embedding, bundle, numeric, ds4_engine_runtime(engine),
                                           "build/metal-models/gemma.ds4g");
    const ds4_gemma_text texts[] = {{"Il gatto dorme.", 0}, {"A cat is sleeping.", 0}};
    if (!status)
        status = ds4_gemma_prepare(embedding, texts, 2, &input);
    if (!status)
        status = ds4_gemma_embed(embedding, input, 128, NULL, result);
    ds4_gemma_input_free(input);
    ds4_status cleanup = ds4_embedding_gemma2_close(embedding);
    if (!status)
        status = cleanup;
    cleanup = ds4_bundle_close(bundle);
    if (!status)
        status = cleanup;
    cleanup = ds4_numeric_close(numeric);
    return status ? status : cleanup;
}
static ds4_status alternate_image(ds4_engine *engine, ds4_qwen_image_rgba **result) {
    ds4_numeric *numeric = NULL;
    ds4_bundle *bundle = NULL;
    ds4_qwen_image_pipeline *pipeline = NULL;
    ds4_qwen_image_job *job = NULL;
    ds4_resources *resources = ds4_engine_resources(engine);
    ds4_status status = ds4_numeric_metal(&numeric, resources, 500);
    if (!status)
        status = ds4_numeric_metal_budget(numeric, ds4_metal_host_reserve_bytes());
    if (!status)
        status = ds4_bundle_open(&bundle, "build/metal-models/image-bf16.ds4b/bundle.ds4b",
                                 "build/metal-models", resources, 501);
    if (!status)
        status = ds4_qwen_image_pipeline_open(&pipeline, bundle, numeric,
            ds4_engine_runtime(engine), "build/metal-models/image-tokenizer.ds4q");
    const ds4_qwen_image_request request = {
        .prompt = "A red square beside a blue circle, clean white background.",
        .width = 256, .height = 256, .seed = 20261020, .steps = 8, .guidance = 1,
    };
    if (!status)
        status = ds4_qwen_image_job_create(&job, pipeline, &request, NULL, 0, NULL);
    if (!status)
        status = ds4_qwen_image_job_run(job);
    if (!status)
        status = ds4_qwen_image_job_take(job, result);
    if (!status) {
        const ds4_qwen_image_report *report = ds4_qwen_image_job_report(job);
        printf("IMAGE recipe=%u width=%u height=%u steps=%u seed=%llu seconds=%.6f\n",
            report->recipe_version, report->width, report->height, report->steps,
            (unsigned long long)report->seed, report->total_seconds);
    }
    ds4_status cleanup = ds4_qwen_image_job_close(job);
    if (!status) status = cleanup;
    cleanup = ds4_qwen_image_pipeline_close(pipeline);
    if (!status) status = cleanup;
    cleanup = ds4_bundle_close(bundle);
    if (!status) status = cleanup;
    cleanup = ds4_numeric_close(numeric);
    return status ? status : cleanup;
}
#endif

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s (%s)\n", __LINE__, #x, error);                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
static int payload(ds4_session *session, uint8_t digest[32], char *error) {
    FILE *file = tmpfile();
    if (!file)
        return 0;
    int ok = !ds4_session_save_payload(session, file, error, 256);
    rewind(file);
    ds4_sha256 sha;
    ds4_sha256_init(&sha);
    uint8_t bytes[65536];
    size_t n;
    while (ok && (n = fread(bytes, 1, sizeof(bytes), file)))
        ds4_sha256_update(&sha, bytes, n);
    ok = ok && !ferror(file);
    if (fclose(file))
        ok = 0;
    ds4_sha256_final(&sha, digest);
    return ok;
}
static int save_logits(const char *directory, const char *name, const float *values, int vocab) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.f32", directory, name);
    FILE *file = fopen(path, "wb");
    if (!file)
        return 0;
    int ok = fwrite(values, sizeof(float), vocab, file) == (size_t)vocab;
    return fclose(file) == 0 && ok;
}
int main(int argc, char **argv) {
    char error[256] = {0};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_checkpoint *checkpoint = NULL;
    ds4_tokens tokens = {0};
    float *logits = NULL, *expected = NULL;
    int result = 1, generated[8] = {0};
#ifdef DS4_PROBE_SUITE
    ds4_gemma_result *pending_embedding = NULL;
    ds4_qwen_image_rgba *pending_image = NULL;
    const char *image_mode = getenv("DS4_PROBE_IMAGE");
    bool image_probe = image_mode && !strcmp(image_mode, "1");
#endif
    CHECK(argc >= 3 && argc <= 5);
    CHECK(!mkdir(argv[2], 0700));
    ds4_engine_options options = {.model_path = argv[1],
                                  .backend = DS4_BACKEND_METAL,
                                  .context_size = 8192,
                                  .prefill_chunk = 64,
                                  .ssd_streaming = true,
                                  .ssd_streaming_cold = true};
    const char *cache_bytes = getenv("DS4_PROBE_SSD_CACHE_BYTES");
    if (cache_bytes) {
        char *end = NULL;
        errno = 0;
        unsigned long long bytes = strtoull(cache_bytes, &end, 10);
        CHECK(cache_bytes[0] >= '0' && cache_bytes[0] <= '9' && !errno && end &&
              !*end && bytes > 0);
        options.ssd_streaming_cache_bytes = bytes;
    }
    const char *mtp = getenv("DS4_PROBE_MTP");
    const char *sidecar = getenv("DS4_PROBE_MTP_MODEL");
    if (sidecar && sidecar[0]) options.mtp_path = sidecar;
    else options.glm_mtp = mtp && !strcmp(mtp, "1");
    if (options.mtp_path || options.glm_mtp) options.mtp_draft_tokens = 2;
    const char *draft_tokens = getenv("DS4_PROBE_MTP_DRAFT");
    if (draft_tokens) {
        char *end = NULL;
        errno = 0;
        unsigned long depth = strtoul(draft_tokens, &end, 10);
        CHECK(options.mtp_draft_tokens && draft_tokens[0] >= '2' && draft_tokens[0] <= '4' &&
              !errno && end && !*end && depth >= 2 && depth <= 4);
        options.mtp_draft_tokens = (int)depth;
    }
    CHECK(!ds4_engine_open(&engine, &options));
    int vocab = ds4_engine_vocab_size(engine);
    CHECK(vocab > 0);
    logits = malloc((size_t)vocab * sizeof(float));
    expected = malloc((size_t)vocab * sizeof(float));
    CHECK(logits && expected);
    CHECK(!ds4_session_create(&session, engine, 8192));
    ds4_tokenize_text(
        engine,
        "The following short sentence is written in English. Italy has a capital city called Rome. "
        "Italy is a country in Europe. Please complete the final sentence. The capital of France "
        "is",
        &tokens);
    if (argc >= 4) {
        int target = atoi(argv[3]), original = tokens.len;
        CHECK(target > 0 && target <= 8176 && original > 0);
        while (tokens.len < target)
            ds4_tokens_push(&tokens, tokens.v[tokens.len % original]);
        tokens.len = target;
    }
    CHECK(tokens.len >= 32 && tokens.len <= 8176);
    if (argc == 5) {
        FILE *saved = fopen(argv[4], "rb");
        struct stat info;
        CHECK(saved && !fstat(fileno(saved), &info) && info.st_size > 0);
        int status =
            ds4_session_load_payload(session, saved, (uint64_t)info.st_size, error, sizeof(error));
        CHECK(!fclose(saved) && !status);
        const ds4_tokens *prefix = ds4_session_tokens(session);
        CHECK(prefix->len < tokens.len && !memcmp(prefix->v, tokens.v, prefix->len * sizeof(int)));
    }
    CHECK(!ds4_session_sync(session, &tokens, error, sizeof(error)));
    CHECK(ds4_session_copy_logits(session, logits, vocab) == vocab);
    CHECK(save_logits(argv[2], "prefill", logits, vocab));
    char state_path[4096];
    snprintf(state_path, sizeof(state_path), "%s/prefill.state", argv[2]);
    FILE *state = fopen(state_path, "wb");
    CHECK(state);
    int saved = ds4_session_save_payload(session, state, error, sizeof(error));
    CHECK(!fclose(state) && !saved);
#ifdef __APPLE__
    uint64_t prefill_owned, prefill_reads;
    ds4_gpu_qwen4_stream_memory(&prefill_owned, &prefill_reads);
    printf("PREFILL occupied=%d SSD_owned=%llu read_bytes=%llu\n", tokens.len,
           (unsigned long long)prefill_owned, (unsigned long long)prefill_reads);
    fflush(stdout);
#endif
    uint8_t before[32], after[32];
    CHECK(payload(session, before, error));
    CHECK(!ds4_session_sync(session, &tokens, error, sizeof(error)));
    CHECK(payload(session, after, error) && !memcmp(before, after, 32));
    uint64_t metadata = 20261031, restored = 0;
    CHECK(!ds4_session_checkpoint_capture(session, argv[2], &metadata, sizeof(metadata),
                                          &checkpoint));
    /* Optional cache retirement must preserve the live frontier; the next
     * forward regrows storage and must match the frozen continuation logits. */
    CHECK(!ds4_engine_trim_cache(engine));
    CHECK(payload(session, after, error) && !memcmp(before, after, 32));
    CHECK(ds4_session_copy_logits(session, expected, vocab) == vocab);
    CHECK(!memcmp(logits, expected, (size_t)vocab * sizeof(float)));
    int next = 0;
    for (int i = 0; i < vocab; i++) {
        CHECK(isfinite(logits[i]));
        if (logits[i] > logits[next])
            next = i;
    }
    generated[0] = next;
    CHECK(!ds4_session_eval(session, next, error, sizeof(error)));
    CHECK(ds4_session_copy_logits(session, expected, vocab) == vocab);
    CHECK(save_logits(argv[2], "continuation", expected, vocab));
    CHECK(!ds4_session_park(session, argv[2]));
    CHECK(!ds4_engine_weights_release(engine));
    CHECK(ds4_session_resume(session) == DS4_STATUS_NOT_RESIDENT);
#ifdef DS4_PROBE_SUITE
    if (image_probe) CHECK(!alternate_image(engine, &pending_image));
    else CHECK(!alternate_embedding(engine, &pending_embedding));
#endif
    CHECK(!ds4_engine_weights_load(engine));
    CHECK(!ds4_session_resume(session));
    size_t bytes = sizeof(restored);
    CHECK(!ds4_session_checkpoint_restore(session, checkpoint, &restored, &bytes));
    CHECK(bytes == sizeof(metadata) && restored == metadata);
    CHECK(payload(session, after, error) && !memcmp(before, after, 32));
    CHECK(!ds4_session_eval(session, next, error, sizeof(error)));
    CHECK(ds4_session_copy_logits(session, logits, vocab) == vocab);
    CHECK(!memcmp(logits, expected, (size_t)vocab * sizeof(float)));
#ifdef DS4_PROBE_SUITE
    if (image_probe) {
        ds4_qwen_image_rgba_view image = ds4_qwen_image_rgba_data(pending_image);
        CHECK(image.width == 256 && image.height == 256);
        for (size_t i = 0; i < (size_t)image.width * image.height * 4; i++)
            CHECK(isfinite(image.pixels[i]) && image.pixels[i] >= 0 && image.pixels[i] <= 1);
        char path[4096];
        snprintf(path, sizeof(path), "%s/image-after-restore.png", argv[2]);
        CHECK(!ds4_qwen_image_rgba_save_png(pending_image, path));
        ds4_qwen_image_rgba_free(pending_image);
        pending_image = NULL;
    } else {
        ds4_gemma_result_view embedding = ds4_gemma_result_data(pending_embedding);
        CHECK(embedding.count == 2 && embedding.dimensions == 128);
        CHECK(save_logits(argv[2], "embedding-after-restore", embedding.vectors, 256));
        ds4_gemma_result_free(pending_embedding);
        pending_embedding = NULL;
    }
#endif
    for (unsigned step = 1; step < 8; step++) {
        next = 0;
        for (int i = 0; i < vocab; i++) {
            CHECK(isfinite(logits[i]));
            if (logits[i] > logits[next])
                next = i;
        }
        generated[step] = next;
        CHECK(!ds4_session_eval(session, next, error, sizeof(error)));
        CHECK(ds4_session_copy_logits(session, logits, vocab) == vocab);
        char name[32];
        snprintf(name, sizeof(name), "position-%02u", step + 1);
        CHECK(save_logits(argv[2], name, logits, vocab));
    }
    if (options.mtp_path || options.glm_mtp) {
        CHECK(ds4_engine_has_mtp(engine));
        memcpy(expected, logits, (size_t)vocab * sizeof(float));
        uint8_t speculative[32];
        for (unsigned replay = 0; replay < 2; replay++) {
            bytes = sizeof(restored);
            CHECK(!ds4_session_checkpoint_restore(session, checkpoint, &restored, &bytes));
            unsigned committed = 0;
            while (committed < 8) {
                int accepted[8];
                uint64_t rng = 20261031;
                int count = ds4_session_eval_speculative(session,
                    ds4_session_argmax(session), 8 - committed, -1,
                    0.0f, 0, 1.0f, 0.0f, &rng, accepted, 8, error, sizeof(error));
                CHECK(count > 0 && count <= (int)(8 - committed));
                CHECK(!memcmp(accepted, generated + committed, count * sizeof(int)));
                committed += (unsigned)count;
            }
            CHECK(ds4_session_copy_logits(session, logits, vocab) == vocab);
            char replay_name[64];
            snprintf(replay_name, sizeof(replay_name), "mtp-replay-%u", replay);
            CHECK(save_logits(argv[2], replay_name, logits, vocab));
            float maximum = 0;
            for (int i = 0; i < vocab; i++) {
                CHECK(isfinite(logits[i]));
                float delta = fabsf(logits[i] - expected[i]);
                float bound = 2e-5f + 5e-5f * fabsf(expected[i]);
                if (delta > bound)
                    fprintf(stderr, "MTP mismatch replay=%u index=%d got=%.9g expected=%.9g "
                                    "delta=%.9g bound=%.9g\n", replay, i, logits[i], expected[i],
                            delta, bound);
                CHECK(delta <= bound);
                maximum = fmaxf(maximum, delta);
            }
            CHECK(payload(session, after, error));
            if (replay) CHECK(!memcmp(speculative, after, 32));
            else memcpy(speculative, after, 32);
            printf("MTP replay=%u tokens=8 greedy_exact=true max_logit_delta=%.9g\n",
                replay, maximum);
        }
    }
#ifdef __APPLE__
    uint64_t owned, reads;
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    printf("SSD owned=%llu read_bytes=%llu\n", (unsigned long long)owned,
           (unsigned long long)reads);
#endif
    printf("TOKEN_IDS");
    for (unsigned i = 0; i < 8; i++)
        printf(" %d", generated[i]);
    putchar('\n');
    printf(
        "PASS: ctx=8192 prefill=64 occupied=%d generated=8 prefix/payload/restore/logits exact\n",
        tokens.len);
    result = 0;
done:
#ifdef DS4_PROBE_SUITE
    ds4_gemma_result_free(pending_embedding);
    ds4_qwen_image_rgba_free(pending_image);
#endif
    ds4_session_checkpoint_free(checkpoint);
    ds4_session_free(session);
    ds4_engine_close(engine);
    ds4_tokens_free(&tokens);
    free(expected);
    free(logits);
    return result;
}
