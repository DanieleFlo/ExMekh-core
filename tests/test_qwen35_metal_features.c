/* Real Metal execution on a 4-layer synthetic model. Tests API state semantics,
 * not pretrained Qwen27 quality, HTTP routing, or full-model qualification. */
#include "../ds4.c"
#include "qwen35_metal_fixture.h"

static char err[256];
static const int prompt_tokens[] = {1, 7, 3, 11, 2, 17, 5, 23, 9, 29, 4, 6, 8, 12, 13};

static void check_mtp_capabilities(void) {
    ds4_engine engine = {.backend = DS4_BACKEND_METAL,
                         .shape = DS4_SHAPE_QWEN4_EXP, .glm_mtp = true};
    CHECK(!ds4_runtime_create(&engine.runtime, NULL, NULL));
    CHECK(ds4_engine_has_mtp(&engine) && ds4_engine_mtp_draft_tokens(&engine) == 2);
    engine.glm_mtp = false;
    CHECK(!ds4_engine_has_mtp(&engine) && !ds4_engine_mtp_draft_tokens(&engine));
    engine.glm_mtp = true;
    engine.backend = DS4_BACKEND_CPU;
    CHECK(!ds4_engine_has_mtp(&engine) && !ds4_engine_mtp_draft_tokens(&engine));
    engine.backend = DS4_BACKEND_METAL;
    engine.shape = DS4_SHAPE_QWEN38_27B;
    engine.mtp_ready = true;
    engine.mtp_draft_tokens = 2;
    CHECK(ds4_engine_has_mtp(&engine) && ds4_engine_mtp_draft_tokens(&engine) == 2);
    CHECK(!ds4_engine_has_mtp(NULL) && !ds4_engine_mtp_draft_tokens(NULL));
    ds4_runtime_release(engine.runtime);
}

static ds4_tokens prompt(unsigned count) {
    return (ds4_tokens){.v = (int *)prompt_tokens, .len = (int)count, .cap = 15};
}
static void check_host_transfer_release(void) {
    /* C callers can keep a worker alive indefinitely without an outer
     * Objective-C pool. Host access must not retain a freed graph buffer. */
    enum { HOST_CONTENTS, HOST_WRITE, HOST_READ, HOST_FILL };
    const uint64_t bytes = 32u << 20;
    const uint64_t driver_overhead_allowance = 1u << 20;
    for (unsigned operation = HOST_CONTENTS; operation <= HOST_FILL; operation++) {
        ds4_gpu_metal_memory before, after;
        CHECK(ds4_gpu_metal_memory_query(&before));
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *source = NULL;
        CHECK(tensor);
        float value = 1.25f;
        if (operation == HOST_CONTENTS) {
            float *data = ds4_gpu_tensor_contents(tensor);
            CHECK(data);
            *data = value;
        } else if (operation == HOST_WRITE) {
            CHECK(ds4_gpu_tensor_write(tensor, 0, &value, sizeof(value)));
        } else if (operation == HOST_READ) {
            CHECK(ds4_gpu_tensor_write(tensor, 0, &value, sizeof(value)));
            CHECK(ds4_gpu_tensor_read(tensor, 0, &value, sizeof(value)));
            CHECK(value == 1.25f);
        } else {
            source = ds4_gpu_tensor_alloc(bytes);
            CHECK(source && ds4_gpu_tensor_fill_f32(source, 0, bytes / sizeof(float)));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_tensor_copy(tensor, 0, source, 0, bytes));
            CHECK(ds4_gpu_tensor_fill_f32(tensor, value, bytes / sizeof(float)));
            CHECK(ds4_gpu_commands_active() && ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(tensor, bytes - sizeof(value), &value, sizeof(value)));
            CHECK(value == 1.25f);
        }
        ds4_gpu_tensor_free(tensor);
        ds4_gpu_tensor_free(source);
        CHECK(ds4_gpu_metal_memory_query(&after));
        CHECK(after.allocated <= before.allocated + driver_overhead_allowance);
    }
    puts("PASS host access releases Metal buffers before the C worker returns");
}
static ds4_session_snapshot snapshot(ds4_session *s) {
    ds4_session_snapshot result = {0};
    CHECK(!ds4_session_save_snapshot(s, &result, err, sizeof(err)));
    return result;
}
static void same_snapshot(ds4_session *s, const ds4_session_snapshot *expected) {
    ds4_session_snapshot got = snapshot(s);
    CHECK(got.len == expected->len && !memcmp(got.ptr, expected->ptr, got.len));
    ds4_session_snapshot_free(&got);
}
static void same_tensor_prefix(ds4_gpu_tensor *a, ds4_gpu_tensor *b, uint64_t count) {
    float *av = malloc(count * sizeof(float)), *bv = malloc(count * sizeof(float));
    CHECK(av && bv && ds4_gpu_tensor_read(a, 0, av, count * sizeof(float)) &&
          ds4_gpu_tensor_read(b, 0, bv, count * sizeof(float)));
    for (uint64_t i = 0; i < count; i++)
        CHECK(isfinite(av[i]) && isfinite(bv[i]) &&
              fabs(av[i] - bv[i]) <= 2e-5 + 5e-5 * fabs(bv[i]));
    free(av);
    free(bv);
}
static void check_tensor_copy(void) {
    ds4_qwen4_vision_weights encoder = {
        .n_embd = 1152, .n_head = 16, .n_ff = 4304, .n_patch = 16, .n_merge = 2, .n_out = 5120};
    CHECK(ds4_gpu_qwen4_vision_scratch_bytes(&encoder, 256) > 0);
    CHECK(ds4_gpu_qwen4_vision_scratch_bytes(NULL, 256) == UINT64_MAX);
    CHECK(ds4_gpu_qwen4_vision_scratch_bytes(&encoder, 0) == UINT64_MAX);
    CHECK(ds4_gpu_qwen4_vision_scratch_bytes(&encoder, 255) == UINT64_MAX);
    encoder.n_head = 0;
    CHECK(ds4_gpu_qwen4_vision_scratch_bytes(&encoder, 256) == UINT64_MAX);

    float values[8] = {1, 2, 3, 4, 5, 6, 7, 8}, actual[8];
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(sizeof(values));
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(sizeof(values));
    CHECK(a && b && ds4_gpu_tensor_write(a, 0, values, sizeof(values)));
    CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, sizeof(values)));
    CHECK(ds4_gpu_tensor_read(b, 0, actual, sizeof(actual)) &&
          !memcmp(values, actual, sizeof(values)));
    CHECK(!ds4_gpu_tensor_copy(b, 1, a, 0, sizeof(values)));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_copy(a, 0, b, 0, sizeof(values)));
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(a, 0, actual, sizeof(actual)) &&
          !memcmp(values, actual, sizeof(values)));
    /* A fill after an encoded producer must win, just like a host write. */
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_copy(a, 0, b, 0, sizeof(values)));
    CHECK(!ds4_gpu_tensor_fill_f32(a, -7.0f, 9));
    CHECK(ds4_gpu_tensor_fill_f32(a, -7.0f, 0));
    CHECK(ds4_gpu_tensor_fill_f32(a, -7.0f, 8));
    CHECK(ds4_gpu_commands_active());
    CHECK(ds4_gpu_tensor_read(a, 0, actual, sizeof(actual)));
    for (unsigned i = 0; i < 8; i++) CHECK(actual[i] == -7.0f);
    CHECK(ds4_gpu_commands_active() && ds4_gpu_end_commands());
    /* A host override must follow an already encoded producer. Vision replaces
     * image placeholder embeddings this way before the first decoder layer. */
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_fill_f32(a, -3.0f, 8));
    CHECK(ds4_gpu_tensor_write(a, 0, values, sizeof(values)));
    CHECK(ds4_gpu_commands_active());
    CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, sizeof(values)));
    CHECK(ds4_gpu_tensor_read(b, 0, actual, sizeof(actual)) &&
          !memcmp(values, actual, sizeof(values)));
    CHECK(ds4_gpu_commands_active() && ds4_gpu_end_commands());
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(b);
    puts("PASS tensor copies inside/outside command batches and invalid bounds");
}
static void same_frontier(ds4_session *a, ds4_session *b) {
    CHECK(ds4_session_pos(a) == ds4_session_pos(b));
    for (unsigned i = 0; i < VOCAB; i++)
        CHECK(isfinite(a->logits[i]) && isfinite(b->logits[i]) &&
              fabs(a->logits[i] - b->logits[i]) <= 2e-5 + 5e-5 * fabs(b->logits[i]));
    CHECK(a->qwen_graph.mtp_cache_len == b->qwen_graph.mtp_cache_len);
    ds4_qwen_gpu_graph *ag = &a->qwen_graph, *bg = &b->qwen_graph;
    const uint64_t kv = (uint64_t)ds4_session_pos(a) * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    for (unsigned il = 0; il < LAYERS; il++) {
        if (qwen_layer_is_recurrent(il)) {
            same_tensor_prefix(ag->conv_state[il], bg->conv_state[il],
                               ds4_gpu_tensor_bytes(ag->conv_state[il]) / sizeof(float));
            same_tensor_prefix(ag->recurrent_state[il], bg->recurrent_state[il],
                               ds4_gpu_tensor_bytes(ag->recurrent_state[il]) / sizeof(float));
        } else {
            same_tensor_prefix(ag->key_cache[il], bg->key_cache[il], kv);
            same_tensor_prefix(ag->value_cache[il], bg->value_cache[il], kv);
        }
    }
    if (ag->mtp_ready && bg->mtp_ready) {
        same_tensor_prefix(ag->mtp_pending_h, bg->mtp_pending_h, WIDTH);
        same_tensor_prefix(ag->mtp_key_cache, bg->mtp_key_cache, kv);
        same_tensor_prefix(ag->mtp_value_cache, bg->mtp_value_cache, kv);
    }
    CHECK(ag->rope_len == bg->rope_len && ag->mrope_delta == bg->mrope_delta);
    if (ag->rope_len)
        CHECK(!memcmp(ag->host_pos3, bg->host_pos3, ag->rope_len * 12u));
}
static void check_mtp_allocation(ds4_session *s) {
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    ds4_session_snapshot expected = snapshot(s);
    ds4_resource_totals before, after;
    ds4_resource_budget cap, denied;
    ds4_resources *resources = ds4_gpu_resources();
    ds4_resources_totals(resources, 0, &before);
    CHECK(!ds4_resources_get_budget(resources, DS4_MEMORY_DEVICE, &cap));
    denied = cap;
    denied.limit = before.allocated[DS4_MEMORY_DEVICE] + 64 * 1024;
    CHECK(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &denied));
    CHECK(!qwen_graph_enable_mtp(g, 2));
    CHECK(ds4_gpu_synchronize());
    ds4_resources_totals(resources, 0, &after);
    CHECK(after.roots == before.roots && after.views == before.views &&
          after.allocated[DS4_MEMORY_DEVICE] == before.allocated[DS4_MEMORY_DEVICE]);
    same_snapshot(s, &expected);
    CHECK(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &cap));
    CHECK(qwen_graph_enable_mtp(g, 2));
    ds4_resources_totals(resources, 0, &after);
    ds4_context_memory mtp = qwen_graph_mtp_memory_estimate(CONTEXT, PREFILL, 2);
    CHECK(after.allocated[DS4_MEMORY_DEVICE] - before.allocated[DS4_MEMORY_DEVICE] ==
          mtp.total_bytes);
    CHECK(qwen_graph_enable_mtp(g, 2)); /* Repeated enable owns the same allocations. */
    CHECK(!qwen_graph_enable_mtp(g, 3));
    qwen_graph_free_mtp_workspace(g);
    same_snapshot(s, &expected); /* Successful enable also preserves target state. */
    ds4_session_snapshot_free(&expected);
    puts("PASS MTP allocation: rollback, retry, exact estimate and target "
         "preservation");
}
static void same_file(FILE *a, FILE *b) {
    rewind(a);
    rewind(b);
    uint8_t av[4096], bv[4096];
    for (;;) {
        size_t an = fread(av, 1, sizeof(av), a), bn = fread(bv, 1, sizeof(bv), b);
        CHECK(an == bn && !memcmp(av, bv, an));
        if (!an)
            break;
    }
    CHECK(!ferror(a) && !ferror(b));
}
static void check_inactive_mtp_exports(ds4_session *s, const ds4_tokens *parent) {
    CHECK(!s->mtp_draft_valid);
    ds4_session_snapshot expected = snapshot(s);
    FILE *before = tmpfile(), *after = tmpfile();
    uint64_t written = 0;
    CHECK(before && after &&
          !ds4_session_save_skill_state(s, before, &written, NULL, err, sizeof(err)));
    /* Deliberately poison unused scratch; public exports must neither expose
     * these bytes nor modify the live buffers to make comparisons pass. */
    for (unsigned i = 0; i < VOCAB; i++)
        s->mtp_logits[i] = NAN;
    s->mtp_draft_token = VOCAB + 1;
    same_snapshot(s, &expected);
    CHECK(isnan(s->mtp_logits[0]) && s->mtp_draft_token == VOCAB + 1);
    CHECK(!ds4_session_save_skill_state(s, after, &written, NULL, err, sizeof(err)));
    same_file(before, after);
    ds4_session_device_frontier *frontier = NULL;
    CHECK(!ds4_session_capture_device_frontier(s, &frontier, err, sizeof(err)));
    CHECK(!ds4_session_restore_device_frontier(s, frontier, parent, err, sizeof(err)));
    CHECK(!s->mtp_draft_valid && s->mtp_draft_token == 0);
    for (unsigned i = 0; i < VOCAB; i++)
        CHECK(s->mtp_logits[i] == 0);
    same_snapshot(s, &expected);
    ds4_session_device_frontier_free(frontier);
    ds4_session_snapshot_free(&expected);
    CHECK(!fclose(before) && !fclose(after));
    puts("PASS inactive MTP scratch excluded from full, skill and device exports");
}
static bool cancelled(void *ud) { return *(bool *)ud; }
static void check_skill_and_checkpoint(ds4_engine *engine) {
    ds4_session *s = NULL, *canonical = NULL;
    CHECK(!ds4_session_create(&s, engine, CONTEXT));
    CHECK(!ds4_session_create(&canonical, engine, CONTEXT));
    ds4_tokens parent = prompt(5);
    CHECK(!ds4_session_sync(s, &parent, err, sizeof(err)));
    CHECK(!ds4_session_sync(canonical, &parent, err, sizeof(err)));
    check_inactive_mtp_exports(s, &parent);
    ds4_session_snapshot saved = snapshot(s);
    FILE *skill = tmpfile();
    uint64_t bytes = 0;
    CHECK(skill && !ds4_session_save_skill_state(s, skill, &bytes, NULL, err, sizeof(err)));
    CHECK(bytes == ds4_session_skill_state_bytes(s));
    CHECK(!ds4_session_eval(s, 19, err, sizeof(err))); /* child-only token */
    rewind(skill);
    CHECK(!ds4_session_load_skill_state(s, skill, &parent, NULL, err, sizeof(err)));
    CHECK(!fclose(skill));
    same_snapshot(s, &saved);
    CHECK(!ds4_session_eval(s, 21, err, sizeof(err))); /* parent result */
    CHECK(!ds4_session_eval(canonical, 21, err, sizeof(err)));
    same_frontier(s, canonical);
    bool stop = true;
    ds4_session_set_cancel(s, cancelled, &stop);
    ds4_session_snapshot before_cancel = snapshot(s);
    ds4_tokens longer = prompt(10);
    CHECK(ds4_session_sync(s, &longer, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    ds4_session_set_cancel(s, NULL, NULL);
    CHECK(!ds4_session_load_snapshot(s, &before_cancel, err, sizeof(err)));
    same_snapshot(s, &before_cancel);
    ds4_session_checkpoint *checkpoint = NULL;
    uint64_t rng = 42, restored_rng = 0;
    size_t metadata_bytes = sizeof(restored_rng);
    CHECK(ds4_session_checkpoint_capture(s, "/tmp", &rng, sizeof(rng), &checkpoint) ==
          DS4_STATUS_OK);
    ds4_session_free(s);
    s = NULL; /* restore into a fresh session */
    CHECK(!ds4_session_create(&s, engine, CONTEXT));
    CHECK(ds4_session_checkpoint_restore(s, checkpoint, &restored_rng, &metadata_bytes) ==
          DS4_STATUS_OK);
    CHECK(restored_rng == rng && metadata_bytes == sizeof(rng));
    same_snapshot(s, &before_cancel);
    CHECK(!ds4_session_eval(s, 22, err, sizeof(err)));
    CHECK(!ds4_session_eval(canonical, 22, err, sizeof(err)));
    same_frontier(s, canonical);
    ds4_session_checkpoint_free(checkpoint);
    ds4_session_snapshot_free(&before_cancel);
    ds4_session_snapshot_free(&saved);
    ds4_session_free(canonical);
    ds4_session_free(s);
    puts("PASS parent -> skill -> return, cancellation, SSD/new-session restore, "
         "RNG metadata");
}

typedef struct {
    int draft[2], abort_row, calls;
    bool invalid;
} sample_state;
static int sample_row(void *ud, const float *logits, int vocab, const int *prefix, int len,
                      uint64_t *rng) {
    (void)logits;
    (void)prefix;
    sample_state *st = ud;
    st->calls++;
    if (st->abort_row == 0)
        return (st->draft[0] + 1) % vocab;
    *rng += 123;
    if (len == st->abort_row)
        return st->invalid ? vocab : -1;
    return st->draft[(len - 1) % 2];
}
static void check_speculation(ds4_engine *engine) {
    ds4_session *s = NULL;
    CHECK(!ds4_session_create(&s, engine, CONTEXT));
    ds4_tokens input = prompt(5);
    CHECK(!ds4_session_sync(s, &input, err, sizeof(err)));
    ds4_session_snapshot saved = snapshot(s), expected = {0};
    int first = ds4_session_argmax(s), drafts[2];
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    CHECK(qwen_graph_mtp_step(g, &engine->mtp_model, &engine->qwen_mtp_weights, first, input.len,
                              g->mtp_pending_h, g->mtp_draft_h[0], NULL, &drafts[0],
                              DS4_QWEN_STAGE_MTP_DRAFT));
    CHECK(qwen_graph_mtp_step(g, &engine->mtp_model, &engine->qwen_mtp_weights, drafts[0],
                              input.len + 1, g->mtp_draft_h[0], g->mtp_draft_h[1], NULL, &drafts[1],
                              DS4_QWEN_STAGE_MTP_DRAFT));
    for (int test = 0; test < 6; test++) {
        CHECK(!ds4_session_load_snapshot(s, &saved, err, sizeof(err)));
        sample_state st = {.draft = {drafts[0], drafts[1]},
                           .abort_row = test == 5 ? -1 : (test > 2 ? test - 2 : test),
                           .invalid = test > 2 && test < 5};
        uint64_t rng = 42;
        int accepted[3], next = -1;
        int n = ds4_session_eval_speculative_constrained_sample(s, first, 1, 0, 1, 0, &rng, 3, -1,
                                                                accepted, 3, &next, sample_row, &st,
                                                                err, sizeof(err));
        if (test < 5) {
            CHECK(n == 1 && accepted[0] == first && rng == 42);
            CHECK(st.calls == (st.abort_row == 2 ? 2 : 1));
            if (!test)
                expected = snapshot(s);
            else
                same_snapshot(s, &expected);
        } else {
            CHECK(n == 3 && accepted[0] == first && accepted[1] == drafts[0] &&
                  accepted[2] == drafts[1]);
            CHECK(rng > 42);
            ds4_session *canonical = NULL;
            CHECK(!ds4_session_create(&canonical, engine, CONTEXT));
            CHECK(!ds4_session_load_snapshot(canonical, &saved, err, sizeof(err)));
            for (int i = 0; i < n; i++)
                CHECK(!ds4_session_eval(canonical, accepted[i], err, sizeof(err)));
            same_frontier(s, canonical);
            ds4_session_free(canonical);
        }
        CHECK(g->mtp_cache_len == (unsigned)ds4_session_pos(s));
    }
    CHECK(!ds4_session_load_snapshot(s, &saved, err, sizeof(err)));
    input = prompt(CONTEXT - 2);
    CHECK(!ds4_session_sync(s, &input, err, sizeof(err)));
    int accepted[3];
    int n = ds4_session_eval_speculative_argmax(s, ds4_session_argmax(s), 3, -1, accepted, 3, err,
                                                sizeof(err));
    CHECK(n >= 1 && ds4_session_pos(s) < CONTEXT);
    ds4_session_snapshot_free(&expected);
    ds4_session_snapshot_free(&saved);
    ds4_session_free(s);
    puts("PASS MTP acceptance/rejection, declined/invalid verifier rows, RNG "
         "rollback and context boundary");
}
static void check_multimodal(ds4_engine *engine) {
    engine->vision_ready = true;
    engine->vision_kind = DS4_VISION_QWEN4;
    engine->vision_image_token = 0;
    int tokens[] = {1, 7, 0, 0, 0, 0, 5, 11, 13};
    ds4_tokens input = {.v = tokens, .len = 8, .cap = 9};
    float embedding[4 * WIDTH];
    for (unsigned i = 0; i < 4 * WIDTH; i++)
        embedding[i] = sinf(i * .01f);
    ds4_vision_span image = {.token_start = 2,
                             .embedding = {.data = embedding,
                                           .token_count = 4,
                                           .grid_width = 2,
                                           .grid_height = 2,
                                           .fingerprint = {1}}};
    ds4_session *s = NULL, *fresh = NULL;
    CHECK(!ds4_session_create(&s, engine, CONTEXT));
    CHECK(!ds4_session_create(&fresh, engine, CONTEXT));
    ds4_tokens text = prompt(2);
    CHECK(!ds4_session_sync(s, &text, err, sizeof(err)));
    CHECK(!ds4_session_sync_multimodal(s, &input, &image, 1, err, sizeof(err)));
    CHECK(ds4_session_has_vision_state(s) && s->qwen_graph.mrope_delta < 0);
    float zero_embedding[4 * WIDTH] = {0};
    ds4_vision_span zero_image = image;
    zero_image.embedding.data = zero_embedding;
    zero_image.embedding.fingerprint[0] = 99;
    CHECK(!ds4_session_sync_multimodal(fresh, &input, &zero_image, 1, err, sizeof(err)));
    float image_effect = 0;
    for (unsigned i = 0; i < VOCAB; i++)
        image_effect = fmaxf(image_effect, fabsf(s->logits[i] - fresh->logits[i]));
    CHECK(isfinite(image_effect) && image_effect > 1e-5f);
    ds4_session_snapshot saved = snapshot(s);
    engine->vision_ready = false;
    engine->vision_kind = DS4_VISION_NONE;
    CHECK(!ds4_session_load_snapshot(fresh, &saved, err, sizeof(err)));
    CHECK(ds4_session_vision_state_matches(fresh, &image, 1));
    ds4_session_snapshot legacy = snapshot(fresh);
    const uint8_t old_version[4] = {1, 0, 0, 0};
    memcpy((uint8_t *)legacy.ptr + 4, old_version, sizeof(old_version));
    CHECK(ds4_session_load_snapshot(fresh, &legacy, err, sizeof(err)) != 0);
    same_snapshot(fresh, &saved);
    ds4_session_snapshot_free(&legacy);
    CHECK(!ds4_session_eval(s, 13, err, sizeof(err)));
    CHECK(!ds4_session_eval(fresh, 13, err, sizeof(err)));
    same_frontier(s, fresh);
    ds4_session_snapshot continued = snapshot(s);
    same_snapshot(fresh, &continued);
    ds4_session_snapshot_free(&continued);
    engine->vision_ready = true;
    engine->vision_kind = DS4_VISION_QWEN4;
    image.embedding.fingerprint[0] = 2;
    embedding[0] += 1;
    CHECK(!ds4_session_vision_state_matches(fresh, &image, 1));
    input.len = 9;
    CHECK(!ds4_session_sync_multimodal(s, &input, &image, 1, err, sizeof(err)));
    ds4_session_invalidate(fresh);
    CHECK(!ds4_session_sync_multimodal(fresh, &input, &image, 1, err, sizeof(err)));
    same_frontier(s, fresh);
    CHECK(ds4_session_park(s, "/tmp") == DS4_STATUS_OK);
    CHECK(ds4_session_resume(s) == DS4_STATUS_OK);
    CHECK(ds4_session_vision_state_matches(s, &image, 1));
    same_frontier(s, fresh);
    /* Inject an encoder peak that cannot fit. Admission must fail before any
     * nonexistent synthetic vision weight is read, then restore both sessions. */
    ds4_session_snapshot before_encoder = snapshot(s), before_other = snapshot(fresh);
    engine->vision_map_ready = true;
    engine->qwen4_vision_weights = (ds4_qwen4_vision_weights){.n_embd = 1152,
                                                              .n_head = 16,
                                                              .n_ff = 1u << 24,
                                                              .n_patch = 16,
                                                              .n_merge = 2,
                                                              .n_out = WIDTH};
    ds4_vision_embedding denied = {0};
    CHECK(!ds4_engine_vision_encode_file(engine, "tests/vision-fixtures/qwen38/maple.png", &denied,
                                         err, sizeof(err)));
    CHECK(strstr(err, "peak exceeds") && !denied.data);
    CHECK(!s->lifecycle_parked && !fresh->lifecycle_parked);
    same_snapshot(s, &before_encoder);
    same_snapshot(fresh, &before_other);
    ds4_session_snapshot_free(&before_encoder);
    ds4_session_snapshot_free(&before_other);
    engine->vision_map_ready = false;
    ds4_session_snapshot_free(&saved);
    ds4_session_free(fresh);
    ds4_session_free(s);
    engine->vision_ready = false;
    puts("PASS text/image/text, MTP mRoPE catch-up, restore, changed pixels and "
         "park/resume");
}
int main(void) {
    check_mtp_capabilities();
    configure_fixture_shape();
    g_qwen_file_type = DS4_QWEN_FTYPE_Q4_K_S;
    CHECK(ds4_gpu_init());
    g_suite_gpu_ready = true;
    CHECK(ds4_gpu_resources_refresh(DS4_METAL_SAFETY_RESERVE_BYTES));
    ds4_resource_budget cap;
    CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    cap.limit = DEVICE_CAP_BYTES;
    cap.margin = 0;
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    check_host_transfer_release();
    check_tensor_copy();
    qwen35_fixture fixture;
    fixture_create(&fixture);
    ds4_engine engine = {.backend = DS4_BACKEND_METAL,
                         .shape = g_ds4_shape,
                         .metal_ready = true,
                         .prefill_chunk = PREFILL};
    fixture_mtp(&fixture, &engine.qwen_mtp_weights);
    engine.model = engine.mtp_model = fixture.model;
    engine.weights = fixture.weights;
    CHECK(ds4_gpu_set_model_map(fixture.model.map, fixture.model.size));
    ds4_session *plain = NULL;
    CHECK(!ds4_session_create(&plain, &engine, CONTEXT));
    ds4_tokens input = prompt(5);
    CHECK(!ds4_session_sync(plain, &input, err, sizeof(err)));
    check_mtp_allocation(plain);
    ds4_session_free(plain);
    engine.support_kind = DS4_SUPPORT_QWEN35_MTP;
    engine.mtp_ready = true;
    engine.mtp_draft_tokens = 2;
    check_skill_and_checkpoint(&engine);
    check_speculation(&engine);
    check_multimodal(&engine);
    CHECK(!engine.sessions);
    ds4_gpu_cleanup();
    fixture_destroy(&fixture);
    puts("PASS bounded Metal T04/T05 session features (synthetic weights)");
}
