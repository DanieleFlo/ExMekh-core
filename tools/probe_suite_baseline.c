/* Small, deterministic full-vocabulary/payload baseline for lifecycle work.
 * Run model processes serially. Output files belong to the caller's fresh dir. */
#include "ds4.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    ds4_engine_options options = {.model_path=argv[1], .backend=DS4_BACKEND_CUDA,
        .context_size=2048, .prefill_chunk=128,
        .mtp_path=strcmp(argv[2], "-") ? argv[2] : NULL,
        .vision_path=strcmp(argv[3], "-") ? argv[3] : NULL};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_vision_embedding embedding = {0};
    ds4_vision_span span = {0};
    char error[256] = {0};
    FILE *logits = NULL, *payload = NULL;
    float *values = NULL;
    int rc = 1;
    char path[4096];
    double started = now();
    if (ds4_engine_open(&engine, &options)) goto done;
    if (ds4_session_create(&session, engine, 2048)) goto done;
    ds4_tokenize_text(engine, "A red square beside a blue circle. Describe the colors.\n", &prompt);
    if (options.vision_path) {
        if (!ds4_engine_vision_encode_file(engine, argv[4], &embedding, error, sizeof(error))) goto done;
        if (!ds4_prompt_append_vision(engine, &prompt, &span, &embedding, error, sizeof(error))) goto done;
        if (ds4_session_sync_multimodal(session, &prompt, &span, 1, error, sizeof(error))) goto done;
    } else if (ds4_session_sync(session, &prompt, error, sizeof(error))) goto done;
    const int vocab = ds4_engine_vocab_size(engine);
    values = malloc((size_t)vocab * sizeof(float));
    if (!values) goto done;
    snprintf(path, sizeof(path), "%s/logits.f32", argv[5]);
    logits = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/session.payload", argv[5]);
    payload = fopen(path, "wb");
    if (!logits || !payload) goto done;
    if (ds4_session_save_payload(session, payload, error, sizeof(error))) goto done;
    printf("{\"vocab\":%d,\"prefix_tokens\":%d,\"mtp\":%s,\"vision\":%s,\"tokens\":[",
           vocab, prompt.len, options.mtp_path ? "true" : "false", options.vision_path ? "true" : "false");
    for (int i = 0; i < 9; i++) {
        if (ds4_session_copy_logits(session, values, vocab) != vocab ||
            fwrite(values, sizeof(float), vocab, logits) != (size_t)vocab) goto done;
        if (i == 8) break;
        int token = ds4_session_argmax(session);
        printf("%s%d", i ? "," : "", token);
        if (ds4_session_eval(session, token, error, sizeof(error))) goto done;
    }
    printf("],\"elapsed_s\":%.6f}\n", now()-started);
    rc = 0;
done:
    if (rc) fprintf(stderr, "baseline failed: %s\n", error);
    if (logits) fclose(logits);
    if (payload) fclose(payload);
    free(values);
    ds4_vision_embedding_free(&embedding);
    ds4_vision_embedding_free(&span.embedding);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
