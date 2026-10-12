/* Live external-draft qualification: teacher-forced full-vocabulary logits,
 * recurrent/attention/MTP cache, mismatch/cancellation, and decode throughput. */
#include "../ds4.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s (%s)\n", __LINE__, #c, err); exit(1); } } while (0)

static ds4_session_snapshot save(ds4_session *s, char *err, size_t errlen) {
    ds4_session_snapshot snap = {0};
    FILE *fp = tmpfile();
    if (!fp || ds4_session_save_payload(s, fp, err, errlen)) exit(1);
    snap.len = ds4_session_payload_bytes(s);
    snap.ptr = xmalloc(snap.len);
    rewind(fp);
    if (fread(snap.ptr, 1, snap.len, fp) != snap.len) exit(1);
    fclose(fp);
    return snap;
}

static void restore(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    FILE *fp = fmemopen(snap->ptr, snap->len, "rb");
    if (!fp || ds4_session_load_payload(s, fp, snap->len, err, errlen)) exit(1);
    fclose(fp);
}

typedef struct {
    const int *drafts;
    const float *reference;
    int mismatch, decline, invalid, calls, failures;
    double max_error, min_cosine;
} verifier;

static int verify(void *opaque, const float *logits, int vocab,
                  const int *prefix, int len, uint64_t *rng) {
    verifier *v = opaque;
    (void)prefix;
    if (!v->reference) { v->calls++; *rng += 7; return v->drafts[len]; }
    const float *expected = v->reference + (size_t)(len - 1) * vocab;
    double dot = 0, a = 0, b = 0, maximum = 0;
    int top_a[20], top_b[20];
    for (int k = 0; k < 20; k++) top_a[k] = top_b[k] = -1;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i]) || !isfinite(expected[i])) v->failures++;
        dot += (double)logits[i] * expected[i]; a += (double)logits[i] * logits[i]; b += (double)expected[i] * expected[i];
        double error = fabs((double)logits[i] - expected[i]); if (error > maximum) maximum = error;
        for (int which = 0; which < 2; which++) {
            int *top = which ? top_b : top_a;
            const float *values = which ? expected : logits;
            for (int k = 0; k < 20; k++) if (top[k] < 0 || values[i] > values[top[k]]) {
                for (int j = 19; j > k; j--) top[j] = top[j - 1];
                top[k] = i; break;
            }
        }
    }
    double cosine = dot / sqrt(a * b);
    int overlap = 0;
    for (int k = 0; k < 20; k++) for (int j = 0; j < 20; j++) if (top_a[k] == top_b[j]) overlap++;
    if (top_a[0] != top_b[0] || overlap < 19 || cosine < .999) {
        fprintf(stderr, "NUMERIC FAIL row=%d argmax=%d/%d top20=%d cosine=%.9f max=%.6g\n", len, top_a[0], top_b[0], overlap, cosine, maximum);
        v->failures++;
    }
    if (maximum > v->max_error) v->max_error = maximum;
    if (cosine < v->min_cosine) v->min_cosine = cosine;
    v->calls++;
    *rng += 7;
    if (len == v->decline) return -1;
    if (len == v->invalid) return vocab;
    if (len == v->mismatch) return (v->drafts[len] + 1) % vocab;
    return v->drafts[len];
}

static float *cache(ds4_session *s, size_t *length) {
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    size_t capacity = 0;
    for (uint32_t l = 0; l < DS4_N_LAYER; l++) {
        capacity += qwen_layer_is_recurrent(l) ? ds4_gpu_tensor_bytes(g->conv_state[l]) + ds4_gpu_tensor_bytes(g->recurrent_state[l]) :
            2ull * s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(float);
    }
    if (g->mtp_ready) capacity += (2ull * s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM + DS4_N_EMBD) * sizeof(float);
    float *data = xmalloc(capacity); size_t offset = 0;
#define READ(t, bytes) do { size_t n = (bytes); if (!ds4_gpu_tensor_read((t), 0, (char *)data + offset, n)) exit(1); offset += n; } while (0)
    for (uint32_t l = 0; l < DS4_N_LAYER; l++) {
        if (qwen_layer_is_recurrent(l)) {
            READ(g->conv_state[l], ds4_gpu_tensor_bytes(g->conv_state[l]));
            READ(g->recurrent_state[l], ds4_gpu_tensor_bytes(g->recurrent_state[l]));
        } else {
            READ(g->key_cache[l], (size_t)s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(float));
            READ(g->value_cache[l], (size_t)s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(float));
        }
    }
    if (g->mtp_ready) {
        READ(g->mtp_key_cache, (size_t)s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(float));
        READ(g->mtp_value_cache, (size_t)s->checkpoint.len * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(float));
        READ(g->mtp_pending_h, (size_t)DS4_N_EMBD * sizeof(float));
    }
#undef READ
    *length = offset / sizeof(float); return data;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) return 2;
    char err[256] = {0}; ds4_engine *e = NULL; ds4_session *s = NULL;
    int context = getenv("DS4_DRAFT_TEST_CONTEXT") ? atoi(getenv("DS4_DRAFT_TEST_CONTEXT")) : 2053;
    ds4_engine_options options = {.model_path=argv[1], .mtp_path=argc == 3 ? argv[2] : NULL,
        .backend=DS4_BACKEND_CUDA, .context_size=context > 4032 ? context + 512 : 4096,
        .prefill_chunk=2048, .mtp_draft_tokens=argc == 3 ? 2 : 0};
    CHECK(ds4_engine_open(&e, &options) == 0);
    CHECK(ds4_session_create(&s, e, options.context_size) == 0);
    ds4_tokens pattern = {0}, prompt = {0}, draft = {0};
    ds4_tokenize_text(e, "A known source file is being copied exactly.\n", &pattern);
    CHECK(context > 0 && context + 64 <= options.context_size);
    for (int i = 0; i < context; i++) ds4_tokens_push(&prompt, pattern.v[i % pattern.len]);
    for (int i = 0; i < 20; i++) ds4_tokenize_text(e, "unique source α中🙂 <p>hello &amp; world</p>\r\n", &draft);
    CHECK(draft.len > 64);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    ds4_session_snapshot base = save(s, err, sizeof(err));
    float *reference = xmalloc(64ull * DS4_N_VOCAB * sizeof(float));
    for (int i = 0; i < 64; i++) {
        CHECK(ds4_session_eval(s, draft.v[i], err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(s, reference + (size_t)i * DS4_N_VOCAB, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
    }
    const struct { int count, mismatch, decline, invalid; } cases[] = {
        {32,0,0,0}, {64,0,0,0}, {64,1,0,0}, {64,2,0,0}, {64,31,0,0}, {64,32,0,0}, {64,63,0,0},
        {32,0,1,0}, {64,0,31,0}, {3,0,0,0}, {2,1,0,0}, {1,0,0,0},
        {32,0,0,1}, {64,0,0,31},
    };
    for (size_t c = 0; !getenv("DS4_DRAFT_TEST_PERF_ONLY") && c < sizeof(cases)/sizeof(cases[0]); c++) {
        bool declined = cases[c].decline || cases[c].invalid;
        int kept = declined ? 1 : cases[c].mismatch ? cases[c].mismatch : cases[c].count;
        restore(s, &base, err, sizeof(err));
        for (int i = 0; i < kept; i++) CHECK(ds4_session_eval(s, draft.v[i], err, sizeof(err)) == 0);
        size_t expected_length = 0; float *expected = cache(s, &expected_length);
        restore(s, &base, err, sizeof(err));
        verifier v = {.drafts=draft.v, .reference=reference, .mismatch=cases[c].mismatch,
            .decline=cases[c].decline, .invalid=cases[c].invalid, .min_cosine=1};
        uint64_t rng = 42; int accepted[64], next = -1;
        int n = ds4_session_eval_draft_tokens(s, draft.v, cases[c].count, accepted, &next, &rng, verify, &v, err, sizeof(err));
        CHECK(n == kept && !memcmp(accepted, draft.v, (size_t)n * sizeof(int)));
        CHECK(ds4_session_pos(s) == prompt.len + kept);
        CHECK(rng == (declined ? 42 : 42 + 7ull * v.calls));
        CHECK(next == (cases[c].mismatch ? (draft.v[kept] + 1) % (int)DS4_N_VOCAB : -1));
        CHECK(!s->qwen_graph.mtp_ready || s->qwen_graph.mtp_cache_len == (uint32_t)ds4_session_pos(s));
        verifier frontier = {.drafts=draft.v, .reference=reference, .min_cosine=1};
        uint64_t unused_rng = 42;
        (void)verify(&frontier, s->logits, DS4_N_VOCAB, draft.v, kept, &unused_rng);
        CHECK(!frontier.failures);
        if (frontier.max_error > v.max_error) v.max_error = frontier.max_error;
        if (frontier.min_cosine < v.min_cosine) v.min_cosine = frontier.min_cosine;
        size_t actual_length = 0; float *actual = cache(s, &actual_length); CHECK(actual_length == expected_length);
        double error = 0, norm = 0, maximum = 0;
        for (size_t i = 0; i < actual_length; i++) {
            CHECK(isfinite(actual[i]));
            double d = actual[i] - expected[i]; error += d*d; norm += (double)expected[i]*expected[i];
            if (fabs(d) > maximum) maximum = fabs(d);
        }
        double relative = sqrt(error / norm);
        printf("{\"case\":%zu,\"rows\":%d,\"kept\":%d,\"calls\":%d,\"numeric_failures\":%d,\"logits_max_error\":%.9g,\"min_cosine\":%.12g,\"cache_relative_l2\":%.9g,\"cache_max_error\":%.9g}\n",
            c,cases[c].count,kept,v.calls,v.failures,v.max_error,v.min_cosine,relative,maximum); fflush(stdout);
        CHECK(!v.failures && relative < .001);
        free(expected); free(actual);
    }
    /* Performance excludes checkpoint restore, with a warm-up then five runs.
     * The same teacher-forced token stream tests target verification cost. */
    for (int batch = 0; batch <= 64; batch = batch ? 64 : 32) {
        for (int repeat = 0; repeat < 6; repeat++) {
            restore(s, &base, err, sizeof(err));
            verifier v = {.drafts=draft.v, .min_cosine=1};
            uint64_t rng = 42; int accepted[64], next = -1;
            double start = now_sec();
            int n = batch ? ds4_session_eval_draft_tokens(s, draft.v, batch, accepted, &next, &rng, verify, &v, err, sizeof(err)) : 64;
            if (!batch) for (int i = 0; i < 64; i++) CHECK(ds4_session_eval(s, draft.v[i], err, sizeof(err)) == 0);
            double seconds = now_sec() - start;
            CHECK(n == (batch ? batch : 64) && !v.failures);
            printf("{\"benchmark\":true,\"batch\":%d,\"repeat\":%d,\"tokens\":%d,\"seconds\":%.9g,\"tokens_per_second\":%.9g}\n",batch,repeat,n,seconds,n/seconds); fflush(stdout);
        }
        if (batch == 64) break;
    }
    /* The last available context rows are a truncated batch. Reaching EOF of
     * context is different from declining a speculative row. */
    restore(s, &base, err, sizeof(err));
    for (int i = prompt.len; i < options.context_size - 3; i++)
        ds4_tokens_push(&prompt, pattern.v[i % pattern.len]);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_draft_capacity(s) == 3);
    verifier tail = {.drafts=draft.v, .min_cosine=1};
    int accepted[64], next = -1; uint64_t rng = 42;
    CHECK(ds4_session_eval_draft_tokens(s, draft.v, 3, accepted, &next, &rng, verify, &tail, err, sizeof(err)) == 3);
    CHECK(ds4_session_draft_capacity(s) == 0);
    CHECK(ds4_session_pos(s) == options.context_size);
    /* Rewind invalidates a hybrid frontier; normal sync must rebuild it before
     * external proposals become available again. */
    ds4_session_rewind(s, prompt.len);
    CHECK(ds4_session_draft_capacity(s) == 0);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_draft_capacity(s) == 3);
    fprintf(stderr, "PASS: context end and rewind/sync\n");
    free(reference); ds4_session_snapshot_free(&base);
    ds4_tokens_free(&pattern); ds4_tokens_free(&prompt); ds4_tokens_free(&draft);
    ds4_session_free(s); ds4_engine_close(e);
    fprintf(stderr,"PASS: external snapshot drafts\n"); return 0;
}
