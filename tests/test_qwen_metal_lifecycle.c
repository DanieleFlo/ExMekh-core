/* Full Qwen27/Flash state on real Metal storage, without GGUF inference.
 * Synthetic descriptors satisfy allocation only; the checkpoint/park/restore
 * paths and their byte accounting are the production implementation. */
#include "../ds4.c"
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static void payload_hash(ds4_session *s, uint8_t hash[32]) {
    FILE *f = tmpfile();
    char error[256] = {0};
    CHECK(f);
    CHECK(!ds4_session_save_payload(s, f, error, sizeof(error)));
    rewind(f);
    ds4_sha256 sha;
    ds4_sha256_init(&sha);
    uint8_t chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)))
        ds4_sha256_update(&sha, chunk, n);
    CHECK(!ferror(f) && !fclose(f));
    ds4_sha256_final(&sha, hash);
}
static void fill_tensor(ds4_gpu_tensor *tensor, float value) {
    if (tensor)
        CHECK(ds4_gpu_tensor_fill_f32(tensor, value, ds4_gpu_tensor_bytes(tensor) / 4));
}
static void same_resources(const ds4_resource_totals *expected) {
    ds4_resource_totals actual;
    ds4_resources_totals(ds4_gpu_resources(), 0, &actual);
    CHECK(actual.roots == expected->roots && actual.views == expected->views);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++) {
        CHECK(actual.allocated[i] == expected->allocated[i]);
        CHECK(actual.reserved[i] == expected->reserved[i]);
    }
}
static void stale_capacity(ds4_resource_budget previous[2]) {
    const ds4_memory_location locations[] = {DS4_MEMORY_HOST, DS4_MEMORY_DEVICE};
    for (unsigned i = 0; i < 2; i++) {
        CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), locations[i], previous + i));
        CHECK(!ds4_resources_reconcile(ds4_gpu_resources(), locations[i], 0, true));
    }
}
static void fresh_capacity(const ds4_resource_budget previous[2]) {
    const ds4_memory_location locations[] = {DS4_MEMORY_HOST, DS4_MEMORY_DEVICE};
    for (unsigned i = 0; i < 2; i++) {
        ds4_resource_budget current;
        CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), locations[i], &current));
        CHECK(current.global_known && current.global_free > 0);
        CHECK(current.limit == previous[i].limit && current.margin == previous[i].margin);
    }
}
static void seed_state(ds4_session *s, bool flash) {
    for (unsigned i = 0; i < DS4_N_LAYER; i++) {
        /* Distinct, exactly representable values expose layer/field swaps. */
        float value = .125f + (float)i / 512;
        if (flash) {
            ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
            ds4_gpu_tensor *t[] = {g->layer_lin_state[i], g->layer_lin_hist[i],
                g->layer_k_cache[i], g->layer_v_cache[i], g->layer_ik_cache[i],
                g->layer_block_key[i]};
            for (unsigned j = 0; j < sizeof(t) / sizeof(t[0]); j++)
                fill_tensor(t[j], value + (float)j / 16);
        } else {
            ds4_qwen_gpu_graph *g = &s->qwen_graph;
            fill_tensor(g->key_cache[i], value);
            fill_tensor(g->value_cache[i], value + .25f);
            fill_tensor(g->conv_state[i], value + .5f);
            fill_tensor(g->recurrent_state[i], value + .75f);
        }
    }
    if (flash) {
        fill_tensor(s->qwen4_graph.ple_hist, .25f);
        s->qwen4_graph.pos = 3;
        s->qwen4_graph.mtp_pos = 2;
        s->qwen4_graph.mrope_delta = 7;
    } else {
        s->qwen_graph.mrope_delta = 7;
    }
    for (unsigned i = 0; i < DS4_N_VOCAB; i++)
        s->logits[i] = i * .00001f;
    for (unsigned i = 0; i < 3; i++)
        token_vec_push(&s->checkpoint, i + 1);
    s->checkpoint_valid = true;
}
int main(int argc, char **argv) {
    CHECK(argc == 2 && (!strcmp(argv[1], "qwen27") || !strcmp(argv[1], "flash") ||
                        !strcmp(argv[1], "qwen27-stream")));
    bool flash = !strcmp(argv[1], "flash");
    bool streamed = flash || !strcmp(argv[1], "qwen27-stream");
    g_ds4_shape = flash ? DS4_SHAPE_QWEN4_EXP : DS4_SHAPE_QWEN38_27B;
    g_qwen_file_type = DS4_QWEN_FTYPE_Q4_K_S;
    CHECK(ds4_gpu_init());
    g_suite_gpu_ready = true;
    CHECK(ds4_gpu_resources_refresh(2ull << 30));
    ds4_gpu_metal_memory device;
    CHECK(ds4_gpu_metal_memory_query(&device));
    printf("Metal memory: host_available=%llu allocated=%llu recommended=%llu max_buffer=%llu\n",
        (unsigned long long)device.host_available, (unsigned long long)device.allocated,
        (unsigned long long)device.recommended, (unsigned long long)device.max_buffer);
    /* Cap registered device storage, including live and candidate restore
     * graphs. The shared physical HOST budget retains the 2 GiB reserve. */
    ds4_resource_budget cap;
    CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    cap.limit = 1ull << 30;
    cap.margin = 0;
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &cap));
    ds4_resource_totals before;
    ds4_resources_totals(ds4_gpu_resources(), 0, &before);
    FILE *backing = tmpfile();
    CHECK(backing && !ftruncate(fileno(backing), 4096));
    void *map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
    CHECK(map != MAP_FAILED);
    ds4_engine e = {0};
    e.backend = DS4_BACKEND_METAL;
    e.shape = g_ds4_shape;
    e.metal_ready = true;
    e.ssd_streaming = streamed;
    e.ssd_streaming_cache_bytes = 4096; /* Synthetic model bypasses engine admission. */
    e.prefill_chunk = 2;
    e.glm_mtp = flash;
    e.share_session_prefill_workspace = true;
    e.model = (ds4_model){.map = map, .size = 4096, .fd = fileno(backing), .tensor_data_pos = 64};
    /* Separate auxiliary mapping: releasing target weights must also release,
     * then reload, the resident MTP view. This scalar is an ownership fixture. */
    void *auxiliary = NULL;
    if (!flash) {
        auxiliary = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
        CHECK(auxiliary != MAP_FAILED);
        *(float *)((char *)auxiliary + 64) = 3.25f;
        CHECK(!msync(auxiliary, 4096, MS_SYNC));
        e.mtp_model = (ds4_model){.map = auxiliary, .size = 4096, .fd = fileno(backing),
                                  .tensor_data_pos = 64, .max_tensor_bytes = 4};
        CHECK(ds4_gpu_set_model_map_range(auxiliary, 4096, 64, 4032, 4));
    }
    ds4_tensor dense = {.type = DS4_TENSOR_F32, .abs_offset = 64, .bytes = sizeof(float)},
               ngram = {.type = DS4_TENSOR_BF16};
    e.weights.token_embd = e.weights.output = e.weights.output_hc_down = e.weights.output_hc_up =
        &dense;
    e.weights.ple_embd = &ngram;
    for (unsigned i = 0; i < DS4_N_LAYER; i++) {
        ds4_layer_weights *l = &e.weights.layer[i];
        l->hc_attn_up = l->hc_attn_inject = l->hc_ffn_up = l->hc_ffn_inject = &dense;
        l->hc_attn_down = l->hc_ffn_down = l->lin_alpha = l->lin_beta = &dense;
        l->ffn_gate_exps = l->ffn_up_exps = l->ffn_down_exps = &dense;
        l->nextn_eh_proj = l->nextn_hc_head_down = l->nextn_hc_head_up = &dense;
    }
    if (streamed)
        CHECK(ds4_gpu_qwen4_stream_open(map, 4096, fileno(backing), e.ssd_streaming_cache_bytes));
    const uint64_t initial_cache = e.ssd_streaming_cache_bytes;
    e.metal_stream_fixed_bytes = streamed ? e.model.size : 0;
    e.metal_stream_cache_ceiling = 512;
    ds4_session *invalid = (ds4_session *)(uintptr_t)1;
    CHECK(ds4_session_create(NULL, &e, 16));
    CHECK(ds4_session_create(&invalid, &e, 0) && !invalid);
    invalid = (ds4_session *)(uintptr_t)1;
    CHECK(ds4_session_create(&invalid, NULL, 16) && !invalid);
    if (!flash) {
        invalid = (ds4_session *)(uintptr_t)1;
        CHECK(ds4_session_create(&invalid, &e, DS4_CONTEXT_LENGTH + 1) && !invalid);
    }
    CHECK(e.ssd_streaming_cache_bytes == initial_cache && !e.sessions);
    e.metal_stream_fixed_bytes = 0;
    e.metal_stream_cache_ceiling = 0;
    ds4_session *s = NULL;
    CHECK(!ds4_session_create(&s, &e, 16));
    if (flash)
        CHECK(s->qwen4_graph.owns_scratch && s->qwen4_slot == -1 && !e.qwen4_shared_workspace);
    else
        CHECK(s->qwen_graph_ready && !s->qwen_graph.mtp_ready);
    seed_state(s, flash);
    uint8_t original[32], restored[32];
    payload_hash(s, original);
    CHECK(ds4_engine_trim_cache(NULL) == DS4_STATUS_INVALID);
    uint64_t ceiling = e.metal_stream_cache_ceiling;
    CHECK(ds4_engine_trim_cache(&e) == (streamed ? DS4_STATUS_OK : DS4_STATUS_UNSUPPORTED));
    CHECK(!e.weights_evicted && e.metal_stream_cache_ceiling == ceiling);
    if (streamed) {
        uint64_t owned = UINT64_MAX;
        ds4_gpu_qwen4_stream_memory(&owned, NULL);
        CHECK(ds4_gpu_qwen4_stream_active() && !owned && !e.ssd_streaming_cache_bytes);
        CHECK(!ds4_engine_trim_cache(&e)); /* Idempotent, no session mutation. */
        CHECK(ds4_gpu_qwen4_stream_resize(map, 4096, initial_cache));
        e.ssd_streaming_cache_bytes = initial_cache;
    }
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    char directory[] = "/tmp/ds4-qwen-metal.XXXXXX";
    CHECK(mkdtemp(directory));
    ds4_session_checkpoint *checkpoint = NULL;
    uint64_t metadata[] = {20261029, 0x123456789abcdef0ull}, got[2] = {0};
    CHECK(!ds4_session_checkpoint_capture(s, directory, metadata, sizeof(metadata), &checkpoint));
    const char *checkpoint_path = ds4_session_checkpoint_path(checkpoint);
    CHECK(checkpoint_path);
    size_t bytes = sizeof(got);
    CHECK(!ds4_session_checkpoint_restore(s, checkpoint, got, &bytes));
    CHECK(bytes == sizeof(metadata) && !memcmp(got, metadata, bytes));
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    /* Recreate the session before restore: tokens alone cannot reconstruct
     * the recurrent state or KV. No model forward may hide a missing field. */
    ds4_session_free(s);
    CHECK(!ds4_session_create(&s, &e, 16));
    bytes = sizeof(got);
    CHECK(!ds4_session_checkpoint_restore(s, checkpoint, got, &bytes));
    CHECK(bytes == sizeof(metadata) && !memcmp(got, metadata, bytes));
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    ds4_resource_budget saved, denied;
    ds4_resource_totals live;
    CHECK(!ds4_resources_get_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &saved));
    ds4_resources_totals(ds4_gpu_resources(), 0, &live);
    denied = saved;
    /* Small graph allocations fit while larger ones fail. This also checks
     * cleanup of partially built graphs and their scratch aliases. */
    denied.limit = live.allocated[DS4_MEMORY_DEVICE] + 4096;
    denied.margin = 0;
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &denied));
    memset(got, 0xa5, sizeof(got));
    uint64_t untouched[2];
    memcpy(untouched, got, sizeof(got));
    bytes = sizeof(got);
    CHECK(ds4_session_checkpoint_restore(s, checkpoint, got, &bytes) == DS4_STATUS_NOMEM);
    CHECK(!memcmp(got, untouched, sizeof(got)));
    same_resources(&live);
    CHECK(!ds4_resources_set_budget(ds4_gpu_resources(), DS4_MEMORY_DEVICE, &saved));
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    FILE *corrupt = fopen(checkpoint_path, "r+b");
    CHECK(corrupt);
    int first = fgetc(corrupt);
    rewind(corrupt);
    CHECK(fputc(first ^ 1, corrupt) != EOF && !fclose(corrupt));
    CHECK(ds4_session_checkpoint_restore(s, checkpoint, got, &bytes) == DS4_STATUS_INVALID);
    CHECK(!memcmp(got, untouched, sizeof(got)));
    same_resources(&live);
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    CHECK(!truncate(checkpoint_path, DS4_CHECKPOINT_HEADER_BYTES - 1));
    CHECK(ds4_session_checkpoint_restore(s, checkpoint, got, &bytes) == DS4_STATUS_INVALID);
    CHECK(!memcmp(got, untouched, sizeof(got)));
    same_resources(&live);
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    CHECK(!unlink(checkpoint_path));
    CHECK(ds4_session_checkpoint_restore(s, checkpoint, got, &bytes) == DS4_STATUS_RESTORE);
    CHECK(!memcmp(got, untouched, sizeof(got)));
    same_resources(&live);
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    ds4_session_checkpoint_free(checkpoint);
    CHECK(!ds4_session_park(s, directory));
    ds4_session_memory memory;
    CHECK(!ds4_session_memory_query(s, &memory));
    CHECK(memory.parked && memory.capacity_known && !memory.device_capacity);
    ds4_resource_budget previous[2];
    stale_capacity(previous);
    CHECK(!ds4_engine_weights_release(&e));
    fresh_capacity(previous);
    CHECK(ds4_session_resume(s) == DS4_STATUS_NOT_RESIDENT);
    if (streamed) {
        /* A zero effective cap must still allow reopening the reader.
         * Resume then measures capacity, respecting the original ceiling. */
        e.ssd_streaming_cache_bytes = 0;
        e.metal_stream_fixed_bytes = e.model.size;
        e.metal_stream_cache_ceiling = flash ? e.model.size / 2 : 0;
    }
    CHECK(!ds4_engine_weights_load(&e));
    if (streamed) CHECK(ds4_gpu_qwen4_stream_active() && !e.ssd_streaming_cache_bytes);
    CHECK(!ds4_session_resume(s));
    if (streamed) {
        CHECK(e.ssd_streaming_cache_bytes == (flash ? e.model.size / 2 : e.model.size));
        e.metal_stream_fixed_bytes = 0;
    }
    if (auxiliary) {
        ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(4), *output = ds4_gpu_tensor_alloc(4);
        float value = 2, actual = 0;
        CHECK(input && output && ds4_gpu_tensor_write(input, 0, &value, 4));
        CHECK(ds4_gpu_matmul_f32_tensor(output, auxiliary, 4096, 64, 1, 1, input, 1));
        CHECK(ds4_gpu_tensor_read(output, 0, &actual, 4) && actual == 6.5f);
        ds4_gpu_tensor_free(output);
        ds4_gpu_tensor_free(input);
    }
    payload_hash(s, restored);
    CHECK(!memcmp(original, restored, 32));
    ds4_session_free(s);
    CHECK(!e.sessions);
    CHECK(!ds4_session_create(&s, &e, 16));
    stale_capacity(previous);
    CHECK(!ds4_session_park(s, directory));
    fresh_capacity(previous);
    CHECK(!ds4_session_resume(s));
    CHECK(!s->checkpoint_valid && (flash ? s->qwen4_graph_ready : s->qwen_graph_ready));
    ds4_session_free(s);
    CHECK(!ds4_engine_weights_release(&e));
    same_resources(&before);
    CHECK(!rmdir(directory));
    if (auxiliary) CHECK(!munmap(auxiliary, 4096));
    munmap(map, 4096);
    fclose(backing);
    ds4_gpu_cleanup();
    g_suite_gpu_ready = false;
    printf("%s Metal lifecycle: payload bit-exact in new session; OOM/corrupt/truncated/missing "
           "restore atomic; park/release/resume and empty session PASS (1 GiB device cap)\n",
           flash ? "Flash" : "Qwen27");
    return 0;
}
