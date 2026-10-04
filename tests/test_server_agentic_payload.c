#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include "../ds4_gpu.h"


/* The server saves full portable payloads for skills and opaque agent contexts.
 * This gate deliberately does not call the qwen35-only compact rollback API. */
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #c, err); \
    goto done; } } while (0)

static bool same_frontier(ds4_session *session, const ds4_tokens *tokens,
                          const float *expected, float *actual, int vocab) {
    const ds4_tokens *live = ds4_session_tokens(session);
    return live && live->len == tokens->len &&
           !memcmp(live->v, tokens->v, (size_t)live->len * sizeof(int)) &&
           ds4_session_copy_logits(session, actual, vocab) == vocab &&
           !memcmp(expected, actual, (size_t)vocab * sizeof(float));
}

/* Compare the recent ranking, excluding physical cache addresses/occupancy.
 * A child may leave useful weight bytes cached, but must not leave its history
 * in the restored parent's policy. */
static uint64_t expert_history_hash(void) {
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    const uint64_t n = ds4_gpu_qwen4_expert_state_bytes();
    if (!n) return 0;
    unsigned char *bytes = malloc(n);
    if (!bytes || !ds4_gpu_qwen4_expert_state_export(bytes, n)) { free(bytes); return 0; }
    const uint32_t count = (uint32_t)bytes[16] | (uint32_t)bytes[17] << 8 | (uint32_t)bytes[18] << 16 | (uint32_t)bytes[19] << 24;
    uint64_t end = 32, hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < count; i++) {
        if (n - end < 32) { free(bytes); return 0; }
        uint32_t selected = 0, rows = 0;
        for (unsigned k = 0; k < 4; k++) {
            selected |= (uint32_t)bytes[end + 20 + k] << (8 * k);
            rows |= (uint32_t)bytes[end + 24 + k] << (8 * k);
        }
        end += 32 + 4ull * selected * rows;
        if (end > n) { free(bytes); return 0; }
    }
    for (uint64_t i = 32; i < end; i++) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    free(bytes); return hash;
#else
    return 0;
#endif
}

int main(int argc, char **argv) {

    bool mtp = false, embedded_mtp = false;
    const char *vision = NULL, *image = NULL;
    if (argc < 2) return 2;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp")) mtp = true;
        else if (!strcmp(argv[i], "--embedded-mtp")) mtp = embedded_mtp = true;
        else if (!strcmp(argv[i], "--vision") && i + 1 < argc) vision = argv[++i];
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) image = argv[++i];
        else { fprintf(stderr, "usage: %s model [--mtp|--embedded-mtp] [--vision FILE --image FILE]\n", argv[0]); return 2; }
    }
    if ((vision != NULL) != (image != NULL)) return 2;
    const char *ctx_env = getenv("DS4_TEST_CTX");
    int ctx = ctx_env ? parse_int_arg(ctx_env, "DS4_TEST_CTX") : 2048;
    const char *prefix_env = getenv("DS4_TEST_PREFIX_TOKENS");
    int warm_prefix = prefix_env ? parse_int_arg(prefix_env, "DS4_TEST_PREFIX_TOKENS") : 0;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_engine_options options = {
        .model_path = argv[1], .mtp_auto = mtp && !embedded_mtp, .glm_mtp = embedded_mtp, .vision_path = vision,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .prefill_chunk = 256,
    };
    server srv = {0}; server_slot slot = {0};
    pthread_mutex_init(&srv.tool_mu, NULL);
    pthread_mutex_init(&srv.inference_mu, NULL);
    srv.slots = &slot; srv.slot_count = 1; slot.id = 0;
    char dir[] = "/tmp/ds4-agentic-payload.XXXXXX";
    char err[256] = {0}; bool dir_created = false; int rc = 1;
    ds4_vision_span span = {0};
    ds4_vision_embedding embedding = {0};
    size_t image_count = image ? 1u : 0u;
    ds4_tokens prefix = {0}, tail = {0}, continuation = {0};
    ds4_tokens nested_tokens[3] = {{0}};
    float *expected = NULL, *actual = NULL, *nested_logits[3] = {0};
    skill_frame *frames[3] = {0};
    uint64_t free_before_save = 0, free_after_save = 0, free_after_release = 0;
    uint64_t parent_history = 0, nested_history[3] = {0};
    resident_kv_frontier request_frontier = {0};
    REQUIRE(ds4_engine_open(&engine, &options) == 0);
    srv.engine = engine;
    REQUIRE((ds4_engine_mtp_draft_tokens(engine) > 0) == mtp);
    REQUIRE(ds4_session_create(&session, engine, ctx) == 0);
    slot.session = session;
    if (ds4_engine_is_qwen4(engine)) REQUIRE(ds4_session_prefill_cap(session) == 256);
    REQUIRE(mkdtemp(dir) != NULL); dir_created = true; srv.skill_dir = dir;
    ds4_chat_append_message(engine, &prefix, "system", "Keep the parent memory: amber.");
    if (image) {
        if (warm_prefix) {
            ds4_tokens filler = {0};
            ds4_tokenize_text(engine, " Parent memory remains amber.\n", &filler);
            REQUIRE(warm_prefix < ctx - 1400 && filler.len > 0);
            while (prefix.len < warm_prefix)
                ds4_tokens_push(&prefix, filler.v[prefix.len % filler.len]);
            ds4_tokens_free(&filler);
            REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
            int vocabulary = ds4_engine_vocab_size(engine);
            float *before = xmalloc((size_t)vocabulary * sizeof(float));
            float *after = xmalloc((size_t)vocabulary * sizeof(float));
            REQUIRE(ds4_session_copy_logits(session, before, vocabulary) == vocabulary);
            REQUIRE(ds4_engine_vision_encode_file(engine, image, &embedding, err, sizeof(err)));
            REQUIRE(same_frontier(session, &prefix, before, after, vocabulary));
            free(before); free(after);
            fprintf(stderr, "PASS vision parking: %d populated text tokens, logits bit-exact\n", warm_prefix);
        } else
        REQUIRE(ds4_engine_vision_encode_file(engine, image, &embedding, err, sizeof(err)));
        const char *parts[] = {"Describe this image.", ""};
        REQUIRE(ds4_chat_append_multimodal_message(engine, &prefix, "user", parts,
                    &embedding, 1, &span, err, sizeof(err)));
    } else ds4_chat_append_message(engine, &prefix, "user", "Return the saved word.");
    ds4_chat_append_assistant_prefix(engine, &prefix, DS4_THINK_NONE);
    ds4_tokenize_text(engine, " child branch temporary data", &tail);
    REQUIRE(prefix.len > 0 && prefix.len < ctx - 96 && tail.len > 0);
    REQUIRE(ds4_session_sync_multimodal(session, &prefix, &span, image_count, err, sizeof(err)) == 0);
    if (mtp) {
        int accepted[3];
        REQUIRE(ds4_session_eval_speculative_argmax(session, ds4_session_argmax(session),
                    3, -1, accepted, 3, err, sizeof(err)) > 0);
        ds4_tokens_free(&prefix);
        ds4_tokens_copy(&prefix, ds4_session_tokens(session));
    }
    int vocab = ds4_engine_vocab_size(engine);
    expected = xmalloc((size_t)vocab * sizeof(float));
    actual = xmalloc((size_t)vocab * sizeof(float));
    REQUIRE(ds4_session_copy_logits(session, expected, vocab) == vocab);
    if (image) {
        /* Encoding another image must preserve the parent graph and identity.
         * Repeat pixels exercise cached decoder state, not the HTTP image cache. */
        REQUIRE(ds4_engine_vision_encode_file(engine, image, &embedding, err, sizeof(err)));
        ds4_vision_embedding_free(&embedding);
        REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
        REQUIRE(ds4_session_has_vision_state(session));
        REQUIRE(ds4_session_vision_state_matches(session, &span, 1u));
        ds4_vision_span wrong = span; wrong.embedding.fingerprint[0] ^= 1u;
        REQUIRE(!ds4_session_vision_prefix_matches(session, &wrong, 1u));
    }
    ds4_tokens_copy(&slot.active_system_prefix, &prefix);
    snprintf(slot.agentic_registry_sha, sizeof(slot.agentic_registry_sha), "%040d", 0);
    slot.responses_live.valid = true; slot.responses_live.live_tokens = prefix.len;
    id_list_push_unique(&slot.responses_live.call_ids, "call_parent");
    slot.responses_live.visible_text = xstrdup("parent [image] tool call");
    slot.responses_live.images.count = image_count;
    if (image_count) slot.responses_live.images.offsets[0] = 7u;
    slot.responses_live.visible_len = strlen(slot.responses_live.visible_text);
    parent_history = expert_history_hash();
    REQUIRE(resident_kv_frontier_capture(session, &request_frontier, true, err, sizeof(err)));
    if (ds4_engine_is_qwen4(engine) || image) REQUIRE(request_frontier.full_payload);
    ds4_tokens_copy(&continuation, &prefix);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&continuation, tail.v[i]);
    REQUIRE(ds4_session_sync_multimodal(session, &continuation, &span, image_count, err, sizeof(err)) == 0);
    REQUIRE(resident_kv_frontier_restore(session, &request_frontier, err, sizeof(err)));
    REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
    REQUIRE(expert_history_hash() == parent_history);
    if (ds4_engine_is_qwen4(engine)) {
        FILE *legacy = tmpfile(); REQUIRE(legacy != NULL);
        const uint64_t payload_bytes = ds4_session_payload_bytes(session);
        REQUIRE(ds4_session_save_payload(session, legacy, err, sizeof(err)) == 0);
        rewind(legacy);
        unsigned char header[32];
        REQUIRE(fread(header, 1, sizeof(header), legacy) == sizeof(header));
        uint32_t words[8] = {0};
        for (unsigned i = 0; i < 8; i++)
            for (unsigned b = 0; b < 4; b++) words[i] |= (uint32_t)header[4 * i + b] << (8 * b);
        /* VMCP wraps the family header with image identities/positions.
         * Its word 3 is the body length, not the historical prefill chunk. */
        const uint64_t body_offset = words[0] == UINT32_C(0x50434d56) ?
            32u + (uint64_t)words[4] * 40u + (uint64_t)words[6] * 12u : 0;
        REQUIRE(body_offset + 16u < payload_bytes);
        REQUIRE(fseeko(legacy, (off_t)(body_offset + 3u * 4u), SEEK_SET) == 0);
        const unsigned char old_cap[4] = {(unsigned char)ctx, (unsigned char)(ctx >> 8),
            (unsigned char)(ctx >> 16), (unsigned char)(ctx >> 24)};
        REQUIRE(fwrite(old_cap, 1, sizeof(old_cap), legacy) == sizeof(old_cap));
        rewind(legacy);
        REQUIRE(ds4_session_load_payload(session, legacy, payload_bytes, err, sizeof(err)) == 0);
        REQUIRE(fclose(legacy) == 0);
        REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
        REQUIRE(expert_history_hash() == parent_history);
        REQUIRE(ds4_session_prefill_cap(session) == 256);
        fprintf(stderr, "PASS legacy Flash prefill-cap header: logits and expert history bit-exact\n");
    }
    resident_kv_frontier_clear(&request_frontier);
    fprintf(stderr, "PASS resident pre-thinking rollback: logits and expert history bit-exact\n");
    free_before_save = ds4_gpu_tier_free_vram(0);
    tool_call parent_call = {.id = "skill_parent", .name = "mock-skill"};
    REQUIRE(skill_frame_save(&srv, &slot, &parent_call, NULL, err, sizeof(err)));
    skill_frame *parent = slot.skills;
    REQUIRE(agent_context_checkpoint_save(&srv, &slot, "parent", err, sizeof(err)));
    agent_context_checkpoint *cp = slot.agent_contexts;
    free_after_save = ds4_gpu_tier_free_vram(0);
    ds4_tokens_copy(&continuation, &prefix);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&continuation, tail.v[i]);
    REQUIRE(ds4_session_sync_multimodal(session, &continuation, &span, image_count, err, sizeof(err)) == 0);
    REQUIRE(skill_frame_restore(&slot, parent, NULL, err, sizeof(err)));
    REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
    REQUIRE(expert_history_hash() == parent_history);
    if (image) {
        int appended = 0;
        REQUIRE(append_rendered_suffix_to_live_session(&srv, &slot, "{\"color\":", &appended, err, sizeof(err)));
        REQUIRE(appended > 0 && ds4_session_has_vision_state(session));
        REQUIRE(ds4_session_vision_state_matches(session, &span, 1u));
        REQUIRE(skill_frame_restore(&slot, parent, NULL, err, sizeof(err)));
        REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
    }
    fprintf(stderr, "PASS skill restore: %d tokens, %d logits bit-exact\n", prefix.len, vocab);

    /* The authenticated server wrapper must reject damage before mutating KV. */
    FILE *fp = fopen(parent->checkpoint_path, "r+b"); REQUIRE(fp != NULL);
    REQUIRE(fseek(fp, -1, SEEK_END) == 0); int byte = fgetc(fp); REQUIRE(byte != EOF);
    REQUIRE(fseek(fp, -1, SEEK_END) == 0 && fputc(byte ^ 1, fp) != EOF);
    REQUIRE(fclose(fp) == 0);
    REQUIRE(!skill_frame_restore(&slot, parent, NULL, err, sizeof(err)));
    REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
    fp = fopen(parent->checkpoint_path, "r+b"); REQUIRE(fp != NULL);
    REQUIRE(fseek(fp, -1, SEEK_END) == 0 && fputc(byte, fp) != EOF);
    REQUIRE(fclose(fp) == 0); err[0] = '\0';

    /* A new session owns a fresh graph; weights remain in the same engine. */
    ds4_session_free(session); session = NULL; slot.session = NULL;
    REQUIRE(ds4_gpu_synchronize());
    free_after_release = ds4_gpu_tier_free_vram(0);
    REQUIRE(ds4_session_create(&session, engine, ctx) == 0); slot.session = session;
    live_tool_state_clear_locked(&slot.responses_live);
    REQUIRE(agent_context_checkpoint_restore(&srv, &slot, cp, NULL, err, sizeof(err)));
    REQUIRE(same_frontier(session, &prefix, expected, actual, vocab));
    REQUIRE(ds4_session_vision_state_matches(session, &span, image_count));
    REQUIRE(expert_history_hash() == parent_history);
    REQUIRE(slot.responses_live.images.count == image_count &&
            (!image_count || slot.responses_live.images.offsets[0] == 7u));
    REQUIRE(slot.responses_live.valid && slot.responses_live.call_ids.len == 1 &&
            !strcmp(slot.responses_live.call_ids.v[0], "call_parent"));
    REQUIRE(ds4_session_sync_multimodal(session, &continuation, &span, image_count, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(session, expected, vocab) == vocab);
    REQUIRE(agent_context_checkpoint_restore(&srv, &slot, cp, NULL, err, sizeof(err)));
    REQUIRE(ds4_session_sync_multimodal(session, &continuation, &span, image_count, err, sizeof(err)) == 0);
    REQUIRE(same_frontier(session, &continuation, expected, actual, vocab));
    fprintf(stderr, "PASS subagent/recreated-session restore and continuation bit-exact\n");

    /* Three nested skills must remain independent after child overwrite. */
    for (int level = 0; level < 3; level++) {
        char id[32]; snprintf(id, sizeof(id), "nested_%d", level);
        tool_call call = {.id = id, .name = "nested-skill"};
        ds4_tokens_copy(&nested_tokens[level], ds4_session_tokens(session));
        nested_history[level] = expert_history_hash();
        nested_logits[level] = xmalloc((size_t)vocab * sizeof(float));
        REQUIRE(ds4_session_copy_logits(session, nested_logits[level], vocab) == vocab);
        REQUIRE(skill_frame_save(&srv, &slot, &call, NULL, err, sizeof(err)));
        frames[level] = slot.skills;
        ds4_tokens_free(&continuation); ds4_tokens_copy(&continuation, &nested_tokens[level]);
        for (int i = 0; i < tail.len; i++) ds4_tokens_push(&continuation, tail.v[i]);
        REQUIRE(ds4_session_sync_multimodal(session, &continuation, &span, image_count, err, sizeof(err)) == 0);
    }
    for (int level = 2; level >= 0; level--) {
        REQUIRE(skill_frame_restore(&slot, frames[level], NULL, err, sizeof(err)));
        REQUIRE(same_frontier(session, &nested_tokens[level], nested_logits[level], actual, vocab));
        REQUIRE(expert_history_hash() == nested_history[level]);
    }
    printf("AGENTIC_PAYLOAD_REPORT {\"populated_text_prefix\":%d,\"images\":%zu,\"ctx\":%d,\"mtp\":%s,\"parent_tokens\":%d,\"vocab\":%d,"
           "\"skill_bytes\":%llu,\"subagent_bytes\":%llu,\"skill_restore_bit_exact\":true,"
           "\"corruption_rejected_without_mutation\":true,\"recreated_session_bit_exact\":true,"
           "\"continuation_bit_exact\":true,\"nested_three_levels_bit_exact\":true,\"expert_recent_history_restored\":true,"
           "\"free_vram_before_save\":%llu,\"free_vram_after_save\":%llu,\"free_vram_after_release\":%llu}\n",
           warm_prefix, image_count, ctx, mtp ? "true" : "false", prefix.len, vocab,
           (unsigned long long)parent->checkpoint_bytes, (unsigned long long)cp->checkpoint_bytes,
           (unsigned long long)free_before_save, (unsigned long long)free_after_save,
           (unsigned long long)free_after_release);
    rc = 0;
done:
    resident_kv_frontier_clear(&request_frontier);
    skill_frames_clear_locked(&srv, &slot);
    agent_context_checkpoints_clear(&slot, true);
    live_tool_state_free(&slot.responses_live);
    ds4_tokens_free(&slot.active_system_prefix);
    for (int i = 0; i < 3; i++) { ds4_tokens_free(&nested_tokens[i]); free(nested_logits[i]); }
    if (dir_created) rmdir(dir);
    ds4_vision_embedding_free(&embedding); ds4_vision_embedding_free(&span.embedding);
    ds4_tokens_free(&prefix); ds4_tokens_free(&tail); ds4_tokens_free(&continuation);
    free(expected); free(actual); ds4_session_free(session); ds4_engine_close(engine);
    pthread_mutex_destroy(&srv.tool_mu); pthread_mutex_destroy(&srv.inference_mu);
    return rc;
}