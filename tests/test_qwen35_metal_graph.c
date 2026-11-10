/* Production scheduler on a reduced synthetic hybrid model, not a 27B
 * qualification. Keep real 27B attention/GDN head dimensions; reduce only
 * depth, embedding/FFN width and vocabulary to bound fixture storage. */
#include "../ds4.c"

#include "qwen35_metal_fixture.h"

/* Frozen before execution: FP32 kernel/reduction roundoff, with per-value
 * finite checks. Exact replay of the same path is tested separately. */
static double compare(const float *got, const float *ref, uint64_t count, bool exact) {
    if (exact)
        CHECK(!memcmp(got, ref, count * sizeof(float)));
    double worst = 0;
    for (uint64_t i = 0; i < count; i++) {
        double error = fabs((double)got[i] - ref[i]);
        CHECK(isfinite(got[i]) && isfinite(ref[i]));
        if (!exact)
            CHECK(error <= 2e-5 + 5e-5 * fabs(ref[i]));
        if (error > worst)
            worst = error;
    }
    return worst;
}
static double compare_tensor(ds4_gpu_tensor *a, ds4_gpu_tensor *b, bool exact) {
    CHECK(a && b && ds4_gpu_tensor_bytes(a) == ds4_gpu_tensor_bytes(b));
    uint64_t bytes = ds4_gpu_tensor_bytes(a);
    float *av = malloc(bytes), *bv = malloc(bytes);
    CHECK(av && bv);
    CHECK(ds4_gpu_tensor_read(a, 0, av, bytes));
    CHECK(ds4_gpu_tensor_read(b, 0, bv, bytes));
    double error = compare(av, bv, bytes / 4, exact);
    free(av);
    free(bv);
    return error;
}
static double compare_state(ds4_qwen_gpu_graph *a, ds4_qwen_gpu_graph *b, bool exact) {
    double worst = 0;
    for (unsigned il = 0; il < LAYERS; il++) {
        ds4_gpu_tensor *as[] = {a->conv_state[il], a->recurrent_state[il], a->key_cache[il],
                                a->value_cache[il]};
        ds4_gpu_tensor *bs[] = {b->conv_state[il], b->recurrent_state[il], b->key_cache[il],
                                b->value_cache[il]};
        for (unsigned j = 0; j < 4; j++)
            if (as[j]) {
                double error = compare_tensor(as[j], bs[j], exact);
                if (error > worst)
                    worst = error;
            }
    }
    return worst;
}
static void same_resources(const ds4_resource_totals *before) {
    ds4_resource_totals after;
    ds4_resources_totals(ds4_gpu_resources(), 0, &after);
    CHECK(after.roots == before->roots && after.views == before->views);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        CHECK(after.allocated[i] == before->allocated[i] &&
              after.reserved[i] == before->reserved[i] &&
              after.requested[i] == before->requested[i] && after.live[i] == before->live[i]);
}

static const int tokens[STEPS] = {1, 7, 3, 11, 2, 17, 5, 23, 9, 29, 4};

static void check_chunk_replay(ds4_qwen_gpu_graph *candidate, ds4_qwen_gpu_graph *reference,
                               const qwen35_fixture *fixture, const float logits[STEPS][VOCAB]) {
    float actual[VOCAB];
    /* Every chunk size leaves a different tail. Resume with single-token
     * decode, then compare the entire recurrent/KV state, including KV tails. */
    for (unsigned chunk = 1; chunk <= PREFILL; chunk++) {
        CHECK(qwen_graph_reset(candidate));
        double worst = 0;
        for (unsigned p = 0; p < STEPS;) {
            unsigned remaining = STEPS - 1 - p;
            unsigned n = chunk < remaining ? chunk : remaining;
            if (remaining) {
                CHECK(qwen_graph_forward_rows(candidate, &fixture->model, &fixture->weights,
                                              tokens + p, p, n, DS4_QWEN_STAGE_PREFILL, actual,
                                              NULL));
            } else {
                n = 1;
                CHECK(qwen_graph_forward_token(candidate, &fixture->model, &fixture->weights,
                                               tokens[p], p, actual));
            }
            double error = compare(actual, logits[p + n - 1], VOCAB, false);
            if (error > worst)
                worst = error;
            p += n;
        }
        double state_error = compare_state(candidate, reference, false);
        printf("PASS hybrid chunk=%u logits max_abs=%.9g state max_abs=%.9g\n", chunk, worst,
               state_error);
    }
}
static void check_exact_replay(ds4_qwen_gpu_graph *candidate, ds4_qwen_gpu_graph *reference,
                               const qwen35_fixture *fixture, const float logits[STEPS][VOCAB]) {
    float actual[VOCAB];
    CHECK(qwen_graph_reset(candidate));
    for (unsigned position = 0; position < STEPS; position++) {
        CHECK(qwen_graph_forward_token(candidate, &fixture->model, &fixture->weights,
                                       tokens[position], position, actual));
        compare(actual, logits[position], VOCAB, true);
    }
    compare_state(candidate, reference, true);
}
static void check_invalid_inputs(ds4_qwen_gpu_graph *candidate, ds4_qwen_gpu_graph *reference,
                                 const qwen35_fixture *fixture) {
    float actual[VOCAB];
    /* Invalid tokens, overflow and capacity failures cannot mutate state. */
    int bad = -1;
    CHECK(!qwen_graph_forward_rows(candidate, &fixture->model, &fixture->weights, &bad, 0, 1,
                                   DS4_QWEN_STAGE_PREFILL, actual, NULL));
    CHECK(!qwen_graph_forward_rows(candidate, &fixture->model, &fixture->weights, tokens,
                                   UINT32_MAX, 2, DS4_QWEN_STAGE_PREFILL, actual, NULL));
    CHECK(
        !qwen_graph_forward_token(candidate, &fixture->model, &fixture->weights, VOCAB, 0, actual));
    compare_state(candidate, reference, true);
}
static void check_partial_allocation_cleanup(const ds4_resource_totals *before) {
    ds4_resources *resources = ds4_gpu_resources();
    ds4_resource_budget saved, denied;
    CHECK(!ds4_resources_get_budget(resources, DS4_MEMORY_DEVICE, &saved));
    denied = saved;
    /* Projection and SwiGLU aliases exist before the first recurrent cache
     * allocation fails. Unwinding must retire both roots and views. */
    denied.limit = 512u * 1024u;
    CHECK(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &denied));
    ds4_qwen_gpu_graph failed = {0};
    CHECK(!qwen_graph_alloc(&failed, CONTEXT, PREFILL));
    CHECK(ds4_gpu_synchronize());
    same_resources(before);
    CHECK(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &saved));
}
static void configure_device_budget(void) {
    CHECK(ds4_gpu_resources_refresh(DS4_METAL_SAFETY_RESERVE_BYTES));
    ds4_gpu_metal_memory memory;
    CHECK(ds4_gpu_metal_memory_query(&memory));
    printf("Metal device: unified=%u host=%llu recommended=%llu allocated=%llu max_buffer=%llu "
           "pressure=%u\n",
           memory.unified, (unsigned long long)memory.host_available,
           (unsigned long long)memory.recommended, (unsigned long long)memory.allocated,
           (unsigned long long)memory.max_buffer, memory.pressure);
    CHECK(memory.max_buffer && !ds4_gpu_tensor_alloc(memory.max_buffer + 1));
    ds4_resource_budget cap;
    CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    cap.limit = DEVICE_CAP_BYTES;
    cap.margin = 0;
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
}
static void check_abandoned_layer_binding(const qwen35_fixture *fixture,
                                           ds4_gpu_tensor *input) {
    uint64_t owned, reads;
    ds4_gpu_tensor *short_output = ds4_gpu_tensor_alloc(sizeof(float));
    CHECK(short_output && ds4_gpu_tensor_fill_f32(input, 0, WIDTH));
    /* A deliberately undersized output abandons a UD binding before encoding
     * any kernel. Its source bytes are never interpreted as quantized data. */
    CHECK(!ds4_gpu_commands_active());
    CHECK(!ds4_gpu_matmul_quant_tensor(short_output, fixture->model.map, fixture->model.size,
        fixture->weights.layer[0].qwen_attn_qkv->abs_offset, 11, WIDTH, WIDTH, input, 1));
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(reads > 0);
    uint64_t abandoned_reads = reads;
    CHECK(ds4_gpu_qwen4_stream_layer(fixture->model.map, fixture->model.size));
    CHECK(!ds4_gpu_matmul_quant_tensor(short_output, fixture->model.map, fixture->model.size,
        fixture->weights.layer[2].qwen_attn_qkv->abs_offset, 11, WIDTH, WIDTH, input, 1));
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(reads > abandoned_reads);
    CHECK(ds4_gpu_qwen4_stream_layer(fixture->model.map, fixture->model.size));
    ds4_gpu_tensor_free(short_output);
}
static void check_layer_read_recovery(ds4_qwen_gpu_graph *candidate,
    ds4_qwen_gpu_graph *reference, const qwen35_fixture *fixture,
    const float logits[STEPS][VOCAB], uint64_t cap) {
    uint64_t owned, reads;
    /* Warm embedding/head remain usable, but the first layer must refill a
     * slot last consumed by layer 2. Repeated EOF cannot publish or allocate
     * a second copy; restoring the file must recover on the same owner. */
    uint8_t *saved = malloc(fixture->model.size);
    CHECK(saved);
    memcpy(saved, fixture->model.map, fixture->model.size);
    const ds4_tensor *partial = fixture->weights.layer[0].qwen_ssm_out;
    CHECK(!ftruncate(fixture->model.fd, (off_t)(partial->abs_offset + partial->bytes / 2)));
    float actual[VOCAB];
    uint64_t failed_owned = 0;
    for (unsigned i = 0; i < 2; i++) {
        CHECK(qwen_graph_reset(candidate));
        CHECK(!qwen_graph_forward_token(candidate, &fixture->model, &fixture->weights,
                                         tokens[0], 0, actual));
        ds4_gpu_qwen4_stream_memory(&owned, &reads);
        CHECK(owned <= cap);
        if (i) CHECK(owned == failed_owned);
        failed_owned = owned;
    }
    CHECK(!ftruncate(fixture->model.fd, FIXTURE_BYTES));
    CHECK(pwrite(fixture->model.fd, saved, fixture->model.size, 0) ==
          (ssize_t)fixture->model.size);
    free(saved);
    check_exact_replay(candidate, reference, fixture, logits);
}
static void check_layer_resize(ds4_qwen_gpu_graph *candidate, ds4_qwen_gpu_graph *reference,
    const qwen35_fixture *fixture, const float logits[STEPS][VOCAB], ds4_gpu_tensor *input,
    uint64_t cap) {
    uint64_t owned, reads;
    float actual[VOCAB];
    /* Growing transfers the warm head instead of re-reading it. Resize also
     * fences a live batch and resumes it before returning to the caller. */
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_fill_f32(input, 1, WIDTH));
    CHECK(ds4_gpu_qwen4_stream_resize(fixture->model.map, fixture->model.size, 28u << 20));
    CHECK(ds4_gpu_commands_active() && ds4_gpu_end_commands());
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    uint64_t head_reads = reads;
    CHECK(ds4_gpu_embed_token_quant_tensor(input, fixture->model.map, fixture->model.size,
        fixture->weights.output->abs_offset, fixture->weights.output->type, VOCAB, 0, WIDTH));
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(reads == head_reads);
    check_exact_replay(candidate, reference, fixture, logits);
    CHECK(ds4_gpu_qwen4_stream_resize(fixture->model.map, fixture->model.size, 4u << 20));
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(owned <= (4u << 20));
    check_exact_replay(candidate, reference, fixture, logits);
    CHECK(ds4_gpu_qwen4_stream_resize(fixture->model.map, fixture->model.size, 0));
    CHECK(ds4_gpu_qwen4_stream_active());
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(!owned);
    CHECK(qwen_graph_reset(candidate));
    CHECK(!qwen_graph_forward_token(candidate, &fixture->model, &fixture->weights,
                                     tokens[0], 0, actual));
    CHECK(ds4_gpu_qwen4_stream_resize(fixture->model.map, fixture->model.size, cap));
    check_exact_replay(candidate, reference, fixture, logits);
}
static void check_layer_stream(ds4_qwen_gpu_graph *candidate, ds4_qwen_gpu_graph *reference,
                                const qwen35_fixture *fixture,
                                const float logits[STEPS][VOCAB]) {
    const uint64_t cap = 16u << 20;
    CHECK(ds4_gpu_qwen4_stream_open(fixture->model.map, fixture->model.size,
                                     fixture->model.fd, cap));
    CHECK(!ds4_gpu_qwen35_stream_configure(fixture->model.map, fixture->model.size, NULL));
    ds4_gpu_weight_region invalid = {fixture->model.size, 1};
    ds4_gpu_qwen35_stream_plan bad = {
        .head = {fixture->weights.output->abs_offset, fixture->weights.output->bytes},
        .embedding = {fixture->weights.token_embd->abs_offset, fixture->weights.token_embd->bytes},
        .layers = &invalid, .layer_count = 1, .embedding_type = DS4_TENSOR_Q8_0,
        .vocab_size = VOCAB};
    CHECK(!ds4_gpu_qwen35_stream_configure(fixture->model.map, fixture->model.size, &bad));
    invalid = bad.head;
    CHECK(ds4_gpu_qwen35_stream_configure(fixture->model.map, fixture->model.size, &bad) == -1);
    CHECK(qwen35_metal_stream_configure_layers(&fixture->model, &fixture->weights) == 1);
    uint64_t owned, reads;
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(!owned && !reads);
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(WIDTH * sizeof(float));
    CHECK(input);
    check_abandoned_layer_binding(fixture, input);
    check_chunk_replay(candidate, reference, fixture, logits);
    check_exact_replay(candidate, reference, fixture, logits);
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(owned <= cap && reads > fixture->model.size);

    check_layer_read_recovery(candidate, reference, fixture, logits, cap);
    check_layer_resize(candidate, reference, fixture, logits, input, cap);
    ds4_gpu_tensor_free(input);
    CHECK(ds4_gpu_qwen4_stream_close());
}
int main(void) {
    configure_fixture_shape();
    CHECK(ds4_gpu_init());
    configure_device_budget();
    ds4_resource_totals before;
    ds4_resources_totals(ds4_gpu_resources(), 0, &before);
    check_partial_allocation_cleanup(&before);

    qwen35_fixture fixture;
    fixture_create(&fixture);
    CHECK(ds4_gpu_set_model_map(fixture.model.map, fixture.model.size));
    ds4_qwen_gpu_graph reference = {0}, candidate = {0};
    CHECK(qwen_graph_alloc(&reference, CONTEXT, PREFILL));
    CHECK(qwen_graph_alloc(&candidate, CONTEXT, PREFILL));
    float logits[STEPS][VOCAB];
    for (unsigned position = 0; position < STEPS; position++)
        CHECK(qwen_graph_forward_token(&reference, &fixture.model, &fixture.weights,
                                       tokens[position], position, logits[position]));
    check_chunk_replay(&candidate, &reference, &fixture, logits);
    check_exact_replay(&candidate, &reference, &fixture, logits);

    /* Verify the alias against a separate SwiGLU destination on Metal. */
    ds4_gpu_tensor_free(candidate.ffn_mid);
    candidate.ffn_mid = ds4_gpu_tensor_alloc(PREFILL * WIDTH * sizeof(float));
    CHECK(candidate.ffn_mid);
    check_exact_replay(&candidate, &reference, &fixture, logits);
    check_invalid_inputs(&candidate, &reference, &fixture);
    CHECK(ds4_gpu_qwen4_stream_open(fixture.model.map, fixture.model.size,
                                     fixture.model.fd, 4u << 20));
    CHECK(qwen35_metal_stream_configure_layers(&fixture.model, &fixture.weights) == 1);
    check_chunk_replay(&candidate, &reference, &fixture, logits);
    check_exact_replay(&candidate, &reference, &fixture, logits);
    uint64_t owned = 0, reads = 0;
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(owned <= (4u << 20) && reads > fixture.model.size);
    CHECK(ds4_gpu_qwen4_stream_open(fixture.model.map, fixture.model.size,
                                     fixture.model.fd, 12u << 20));
    uint64_t invalid_offset = fixture.model.size, invalid_bytes = 1;
    CHECK(!ds4_gpu_qwen4_stream_retain(fixture.model.map, fixture.model.size,
                                       &invalid_offset, &invalid_bytes, 1, 1));
    CHECK(qwen35_metal_stream_retain(&fixture.model, &fixture.weights));
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(!owned && !reads); /* Configuration is lazy, with no hidden pack. */
    check_exact_replay(&candidate, &reference, &fixture, logits);
    uint64_t first_reads = 0;
    ds4_gpu_qwen4_stream_memory(&owned, &first_reads);
    CHECK(owned > 0 && owned <= (12u << 20));
    check_exact_replay(&candidate, &reference, &fixture, logits);
    ds4_gpu_qwen4_stream_memory(&owned, &reads);
    CHECK(reads - first_reads < first_reads && owned <= (12u << 20));
    /* A fresh embedding row forces an actual read after retained weights
     * have warmed. Failed entries must not poison the next request. */
    uint8_t *retained_weights = malloc(fixture.model.size);
    CHECK(retained_weights);
    memcpy(retained_weights, fixture.model.map, fixture.model.size);
    CHECK(!ftruncate(fixture.model.fd, 0));
    CHECK(qwen_graph_reset(&candidate));
    float failed_logits[VOCAB];
    CHECK(!qwen_graph_forward_token(&candidate, &fixture.model, &fixture.weights,
                                    VOCAB - 2, 0, failed_logits));
    CHECK(!ftruncate(fixture.model.fd, FIXTURE_BYTES));
    CHECK(pwrite(fixture.model.fd, retained_weights, fixture.model.size, 0) ==
          (ssize_t)fixture.model.size);
    free(retained_weights);
    CHECK(qwen_graph_reset(&candidate));
    CHECK(qwen_graph_forward_token(&candidate, &fixture.model, &fixture.weights,
                                   VOCAB - 2, 0, failed_logits));
    check_exact_replay(&candidate, &reference, &fixture, logits);
    CHECK(ds4_gpu_qwen4_stream_open(fixture.model.map, fixture.model.size,
                                     fixture.model.fd, 32768));
    CHECK(qwen_graph_reset(&candidate));
    float rejected[VOCAB];
    CHECK(!qwen_graph_forward_token(&candidate, &fixture.model, &fixture.weights,
                                    tokens[0], 0, rejected));
    CHECK(ds4_gpu_qwen4_stream_open(fixture.model.map, fixture.model.size,
                                     fixture.model.fd, 4u << 20));
    check_exact_replay(&candidate, &reference, &fixture, logits);
    /* A short read must fail closed and the next rebuilt graph must recover. */
    uint8_t *saved_weights = malloc(fixture.model.size);
    CHECK(saved_weights);
    memcpy(saved_weights, fixture.model.map, fixture.model.size);
    CHECK(ds4_gpu_qwen4_stream_layer(fixture.model.map, fixture.model.size));
    CHECK(!ftruncate(fixture.model.fd, 0));
    CHECK(qwen_graph_reset(&candidate));
    CHECK(!qwen_graph_forward_token(&candidate, &fixture.model, &fixture.weights,
                                    tokens[0], 0, rejected));
    CHECK(!ftruncate(fixture.model.fd, FIXTURE_BYTES));
    CHECK(pwrite(fixture.model.fd, saved_weights, fixture.model.size, 0) == (ssize_t)fixture.model.size);
    free(saved_weights);
    CHECK(ds4_gpu_qwen4_stream_layer(fixture.model.map, fixture.model.size));
    check_exact_replay(&candidate, &reference, &fixture, logits);
    CHECK(ds4_gpu_qwen4_stream_close());
    check_layer_stream(&candidate, &reference, &fixture, logits);

    qwen_graph_free(&candidate);
    qwen_graph_free(&reference);
    CHECK(ds4_gpu_synchronize());
    same_resources(&before);
    ds4_gpu_cleanup();
    fixture_destroy(&fixture);
    puts("PASS: Qwen27 reduced hybrid replay, aliases, reset, rejected input and partial "
         "allocation cleanup");
    return 0;
}
