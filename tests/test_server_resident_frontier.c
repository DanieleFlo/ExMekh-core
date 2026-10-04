#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

typedef struct {
    int last_position;
    int evaluated_tokens;
    int restarted_chunks;
} prefill_work;

static void count_prefill_work(void *ud, const char *event, int current, int total) {
    (void)total;
    prefill_work *work = ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current < work->last_position) {
        work->restarted_chunks++;
        work->evaluated_tokens += current;
    } else {
        work->evaluated_tokens += current - work->last_position;
    }
    work->last_position = current;
}

/* A new client can reuse a system frontier after a failed request reset KV. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_engine_options options = {
        .model_path = argv[1],
        .mtp_path = getenv("DS4_TEST_MTP"),
        .mtp_draft_tokens = getenv("DS4_TEST_MTP") ? 4 : 0,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .prefill_chunk = getenv("DS4_TEST_PREFILL_CHUNK") ?
            parse_int_arg(getenv("DS4_TEST_PREFILL_CHUNK"), "DS4_TEST_PREFILL_CHUNK") : 2048,
    };
    resident_kv_frontier frontier = {0}, local = {0};
    char skill_dir[] = "/tmp/qs-skill-regression.XXXXXX";
    server srv = {0};
    server_slot slot = {0};
    pthread_mutex_init(&srv.tool_mu, NULL);
    pthread_mutex_init(&srv.inference_mu, NULL);
    bool skill_dir_created = false;
    ds4_tokens prefix = {0}, tail = {0}, prompt = {0}, unrelated = {0};
    float *expected = NULL, *actual = NULL;
    char *rendered_prompt = NULL;
    char err[256] = {0};
    int rc = 1;
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #c, err); \
    goto done; } } while (0)
    REQUIRE(ds4_engine_open(&engine, &options) == 0);
    REQUIRE(ds4_session_create(&session, engine, 1024) == 0);
    buf system = {0};
    buf_puts(&system, "<|im_start|>system\n");
    for (int i = 0; i < 12; i++)
        buf_puts(&system, "The registered tool is mock-tool. Invoke it when asked. ");
    buf_puts(&system, "<|im_end|>\n");
    ds4_tokenize_rendered_chat(engine, system.ptr, &prefix);
    buf_puts(&system, "<|im_start|>user\nTest mock-tool.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n");
    rendered_prompt = xstrdup(system.ptr);
    buf_free(&system);
    ds4_tokenize_text(engine,
        "<|im_start|>user\nTest mock-tool.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n", &tail);
    ds4_tokenize_rendered_chat(engine,
        "<|im_start|>system\nYou translate poems."
        "<|im_end|>\n<|im_start|>user\nTranslate hello."
        "<|im_end|>\n<|im_start|>assistant\n", &unrelated);
    /* Exercise the production minimum system boundary, not just a tiny prompt. */
    REQUIRE(prefix.len >= RESIDENT_SYSTEM_MIN_TOKENS);
    ds4_tokens_copy(&prompt, &prefix);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&prompt, tail.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(resident_kv_frontier_capture(session, &frontier, false,
                                         err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    int vocab = ds4_engine_vocab_size(engine);
    expected = xmalloc((size_t)vocab * sizeof(float));
    actual = xmalloc((size_t)vocab * sizeof(float));
    REQUIRE(ds4_session_copy_logits(session, expected, vocab) == vocab);
    /* Exercise the production cold path, including resident-system capture and
     * the disk target chosen afterward. An aligned shorter target used to reset
     * Flash's recurrent state and evaluate the same system tokens twice. */
    srv.engine = engine;
    srv.kv.enabled = true;
    srv.kv.opt = kv_cache_default_options();
    srv.kv.opt.min_tokens = RESIDENT_SYSTEM_MIN_TOKENS;
    srv.kv.opt.boundary_trim_tokens = 16;
    srv.kv.opt.boundary_align_tokens = 64;
    REQUIRE(kv_cache_store_len(&srv.kv, prompt.len) < prefix.len);
    const ds4_model_descriptor *model = ds4_engine_model_descriptor(engine);
    srv.qwen_session_cache = model && model->managed_cache_subdir != NULL;
    slot.session = session;
    ds4_tokens_copy(&slot.active_system_prefix, &prefix);
    request req = {.kind = REQ_CHAT, .prompt_text = rendered_prompt,
                   .model_syntax = server_model_syntax_for_engine(engine)};
    bool system_anchor = false;
    /* Ordinary API requests have no active agent-context prefix. Retain the
     * rendered-prefix fallback used by the existing Qwen27 cold cache. */
    REQUIRE(kv_cache_cold_store_target_request(
        &srv, &req, &prompt, NULL, &system_anchor) == prefix.len);
    REQUIRE(system_anchor);
    prefill_work work = {0};
    ds4_session_invalidate(session);
    ds4_session_set_progress(session, count_prefill_work, &work);
    REQUIRE(ensure_resident_system_frontier(&srv, &slot, &prompt, err, sizeof(err)));
    const int cold_len = kv_cache_cold_store_target_request(
        &srv, &req, &prompt, &slot.active_system_prefix, &system_anchor);
    REQUIRE(cold_len == prefix.len && system_anchor);
    ds4_tokens cold = prompt;
    cold.len = cold_len;
    REQUIRE(server_session_sync(&srv, &slot, &cold, err, sizeof(err)) == 0);
    REQUIRE(server_session_sync(&srv, &slot, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);
    REQUIRE(work.restarted_chunks == 0 && work.evaluated_tokens == prompt.len);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    fprintf(stderr, "PASS cold prefill: evaluated=%d prompt=%d, no replay, logits bit-exact\n",
            work.evaluated_tokens, prompt.len);
    /* The remaining restore checks use their private temporary checkpoint dir. */
    srv.kv.enabled = false;
    srv.qwen_session_cache = false;
    for (int attempt = 0; attempt < 3; attempt++) {
        ds4_session_invalidate(session);
        if (attempt == 1)
            REQUIRE(ds4_session_sync(session, &unrelated, err, sizeof(err)) == 0);
        REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
        REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
        REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    }
    /* Qwen27 can use compact rollback; Flash uses the complete portable payload. */
    REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
    const bool compact_rollback = ds4_session_skill_state_bytes(session) != 0;
    REQUIRE(resident_kv_frontier_capture(session, &local, true, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(resident_kv_frontier_restore(session, &local, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    /* A skill is saved before post-response canonicalization. Rewriting the
     * sampled response destroys attention rows that a compact frame omits. */
    ds4_session_invalidate(session);
    if (compact_rollback) {
        REQUIRE(!resident_kv_frontier_restore(session, &local, err, sizeof(err)));
        REQUIRE(ds4_session_pos(session) == 0);
    } else {
        REQUIRE(resident_kv_frontier_restore(session, &local, err, sizeof(err)));
        REQUIRE(ds4_session_pos(session) == prefix.len);
        REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
        REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    }
    err[0] = '\0';
    REQUIRE(mkdtemp(skill_dir) != NULL);
    skill_dir_created = true;
    srv.skill_dir = skill_dir;
    srv.slot_count = 1;
    srv.slots = &slot;
    slot.session = session;
    REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
    tool_call call = {.id = "call_regression", .name = "mock-skill"};
    REQUIRE(skill_frame_save(&srv, &slot, &call, NULL, err, sizeof(err)));
    /* Corrupt bytes must be rejected before any mutation; repairing the file
     * must then allow the same frame to be restored. */
    FILE *checkpoint = fopen(slot.skills->checkpoint_path, "r+b");
    REQUIRE(checkpoint != NULL);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    int last_byte = fgetc(checkpoint);
    REQUIRE(last_byte != EOF);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    REQUIRE(fputc(last_byte ^ 1, checkpoint) != EOF);
    REQUIRE(fclose(checkpoint) == 0);
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(!skill_frame_restore(&slot, slot.skills, NULL, err, sizeof(err)));
    REQUIRE(ds4_session_pos(session) == prompt.len);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    checkpoint = fopen(slot.skills->checkpoint_path, "r+b");
    REQUIRE(checkpoint != NULL);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    REQUIRE(fputc(last_byte, checkpoint) != EOF);
    REQUIRE(fclose(checkpoint) == 0);
    err[0] = '\0';
    REQUIRE(ds4_session_sync(session, &unrelated, err, sizeof(err)) == 0);
    REQUIRE(skill_frame_restore(&slot, slot.skills, NULL, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    fprintf(stderr, "PASS: system/skill restore, corruption rejection, and compact rollback are bit-exact\n");
    rc = 0;
done:
    skill_frames_clear_locked(&srv, &slot);
    if (skill_dir_created) rmdir(skill_dir);
    pthread_mutex_destroy(&srv.tool_mu);
    pthread_mutex_destroy(&srv.inference_mu);
    resident_kv_frontier_clear(&slot.system_frontier);
    ds4_tokens_free(&slot.active_system_prefix);
    free(expected);
    free(actual);
    free(rendered_prompt);
    resident_kv_frontier_clear(&local);
    resident_kv_frontier_clear(&frontier);
    ds4_tokens_free(&prefix);
    ds4_tokens_free(&tail);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&unrelated);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
