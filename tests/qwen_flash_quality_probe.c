/* Exercise the production session API at an explicit prefill geometry.
 * Input: comma-separated prompt IDs|teacher target IDs<TAB>output path.
 * Output: one complete FP32 vocabulary row for every teacher target. */
#include "../ds4.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_ids(char *text, ds4_tokens *tokens, int vocab) {
    char *p = text, *end = NULL;
    while (*p) {
        errno = 0;
        long value = strtol(p, &end, 10);
        if (errno || end == p || value < 0 || value >= vocab || (*end && *end != ',')) return -1;
        ds4_tokens_push(tokens, (int)value);
        p = *end ? end + 1 : end;
    }
    return tokens->len ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s model ctx chunk inputs.tsv embedded-mtp(0|1)\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    FILE *manifest = NULL, *output = NULL;
    float *logits = NULL;
    char *line = NULL, err[256] = {0};
    size_t capacity = 0;
    int rc = 1;
    ds4_tokens prompt = {0}, targets = {0};
    ds4_engine_options options = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .context_size = atoi(argv[2]), .prefill_chunk = (uint32_t)atoi(argv[3]),
        .glm_mtp = atoi(argv[5]) != 0, .dspark_exact_sampling = true};
    if (options.context_size <= 0 || options.prefill_chunk == 0 || ds4_engine_open(&engine, &options)) goto done;
    if (!ds4_engine_is_qwen4(engine)) { snprintf(err, sizeof(err), "Flash model required"); goto done; }
    const int vocab = ds4_engine_vocab_size(engine);
    logits = malloc((size_t)vocab * sizeof(float));
    manifest = fopen(argv[4], "r");
    if (!logits || !manifest) goto done;
    while (getline(&line, &capacity, manifest) >= 0) {
        line[strcspn(line, "\r\n")] = 0;
        char *bar = strchr(line, '|'), *tab = strchr(line, '\t');
        if (!bar || !tab || bar >= tab) goto done;
        *bar++ = *tab++ = 0;
        if (parse_ids(line, &prompt, vocab) || parse_ids(bar, &targets, vocab) ||
            prompt.len + targets.len >= options.context_size ||
            ds4_session_create(&session, engine, options.context_size) ||
            ds4_session_sync(session, &prompt, err, sizeof(err))) goto done;
        output = fopen(tab, "wb");
        if (!output) goto done;
        for (int row = 0; row < targets.len; row++) {
            if (ds4_session_copy_logits(session, logits, vocab) != vocab ||
                fwrite(logits, sizeof(float), (size_t)vocab, output) != (size_t)vocab ||
                (row + 1 < targets.len && ds4_session_eval(session, targets.v[row], err, sizeof(err)))) goto done;
        }
        if (fclose(output)) { output = NULL; goto done; }
        output = NULL;
        fprintf(stderr, "PASS dump: prompt=%d target=%d vocab=%d ctx=%d chunk=%u path=%s\n",
            prompt.len, targets.len, vocab, options.context_size, options.prefill_chunk, tab);
        ds4_session_free(session); session = NULL;
        ds4_tokens_free(&prompt); ds4_tokens_free(&targets);
    }
    if (ferror(manifest)) goto done;
    rc = 0;
done:
    if (rc) fprintf(stderr, "FAIL Flash quality probe: %s (%s)\n", err, strerror(errno));
    if (output) fclose(output);
    if (manifest) fclose(manifest);
    free(logits); free(line);
    ds4_tokens_free(&prompt); ds4_tokens_free(&targets);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
