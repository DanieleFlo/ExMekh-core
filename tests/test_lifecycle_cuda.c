#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void progress(void *opaque, const char *event, int current, int total) {
    (void)event; (void)current; (void)total; (*(int *)opaque)++;
}
static void restore(ds4_session *s, ds4_session_checkpoint *p, uint64_t expected,
                    ds4_resources *registry) {
    ds4_resource_totals total_before, total_after;
    ds4_session_memory owned_before, owned_after;
    ds4_resources_totals(registry, 0, &total_before);
    assert(!ds4_session_memory_query(s, &owned_before));
    uint64_t rng = 0; size_t size = sizeof(rng);
    assert(!ds4_session_checkpoint_restore(s, p, &rng, &size));
    assert(size == sizeof(rng) && rng == expected);
    ds4_resources_totals(registry, 0, &total_after);
    assert(!ds4_session_memory_query(s, &owned_after));
    /* A restored graph's lazy vision buffers belong to that session too.
     * Otherwise the next executor lease undercounts its restore capacity. */
    assert(total_before.allocated[DS4_MEMORY_DEVICE] - owned_before.device_capacity ==
           total_after.allocated[DS4_MEMORY_DEVICE] - owned_after.device_capacity);
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) return 2;
    char error[256] = {0};
    ds4_engine_options options = {.model_path=argv[1], .backend=DS4_BACKEND_CUDA,
        .context_size=2048, .prefill_chunk=128,
        .mtp_path=argc > 2 && strcmp(argv[2], "-") ? argv[2] : NULL,
        .vision_path=argc > 3 ? argv[3] : NULL};
    ds4_engine *e = NULL; ds4_session *s = NULL, *sibling = NULL, *empty = NULL;
    assert(!ds4_engine_open(&e, &options));
    ds4_resources *registry = ds4_engine_resources(e); assert(registry);
    ds4_resource_totals initial, final;
    ds4_resources_totals(registry, 0, &initial);
    ds4_gpu_tensor *base = ds4_gpu_tensor_alloc(4096), *copy = ds4_gpu_tensor_alloc(4096);
    assert(base && copy);
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(base, 0, 4096); assert(view);
    float value = 17.25f, copied = 0;
    assert(ds4_gpu_tensor_write(base, 0, &value, sizeof(value)));
    ds4_gpu_tensor_free(base); /* Alias retains the actual CUDA allocation. */
    assert(ds4_gpu_tensor_copy_async(copy, view, 4096));
    ds4_gpu_tensor_free(view); /* Drains the marked event before returning budget. */
    assert(ds4_gpu_tensor_read(copy, 0, &copied, sizeof(copied)) && value == copied);
    ds4_gpu_tensor_free(copy);
    ds4_gpu_tensor in_place = {0};
    assert(!ds4_gpu_tensor_alloc_on(&in_place, 0, 4096));
    assert(ds4_gpu_tensor_write(&in_place, 0, &value, sizeof(value)));
    view = ds4_gpu_tensor_view(&in_place, 0, 4096); assert(view);
    ds4_gpu_tensor_free_in_place(&in_place);
    ds4_gpu_tensor_free_in_place(&in_place);
    assert(!in_place.ptr && !in_place.account && !in_place.bytes && !in_place.owner);
    assert(ds4_gpu_tensor_read(view, 0, &copied, sizeof(copied)) && copied == value);
    ds4_gpu_tensor_free(view);
    ds4_resources_totals(registry, 0, &final);
    assert(initial.allocated[0] == final.allocated[0] && initial.views == final.views);
    /* WDDM can accept virtual allocations beyond physical VRAM. A dimension
     * beyond its address space exercises native failure without filling it. */
    ds4_resource_budget saved_budget, unlimited = {.limit=UINT64_MAX};
    assert(!ds4_resources_get_budget(registry, DS4_MEMORY_DEVICE, &saved_budget));
    assert(!ds4_resources_set_budget(registry, DS4_MEMORY_DEVICE, &unlimited));
    assert(!ds4_gpu_tensor_alloc(UINT64_MAX / 2u));
    ds4_resources_totals(registry, 0, &final);
    assert(initial.allocated[0] == final.allocated[0]);
    assert(!ds4_resources_get_budget(registry, DS4_MEMORY_DEVICE, &unlimited));
    assert(unlimited.global_known && unlimited.global_free > 0);
    assert(!ds4_resources_set_budget(registry, DS4_MEMORY_DEVICE, &saved_budget));
    assert(!ds4_session_create(&s, e, 2048));
    assert(!ds4_session_create(&sibling, e, 2048));
    assert(!ds4_session_create(&empty, e, 2048));
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, "A red square beside a blue circle. Describe the colors.\n", &prompt);
    ds4_vision_embedding embedding = {0}; ds4_vision_span span = {0};
    if (options.vision_path) {
        assert(ds4_engine_vision_encode_file(e, "tests/vision-fixtures/glm53/spatial.png", &embedding, error, sizeof(error)));
        assert(ds4_prompt_append_vision(e, &prompt, &span, &embedding, error, sizeof(error)));
        assert(!ds4_session_sync_multimodal(s, &prompt, &span, 1, error, sizeof(error)));
    } else assert(!ds4_session_sync(s, &prompt, error, sizeof(error)));
    assert(!ds4_session_sync(sibling, &prompt, error, sizeof(error)));
    int callback_count = 0;
    ds4_session_set_progress(s, progress, &callback_count);
    const int vocab = ds4_engine_vocab_size(e);
    float *reference = malloc((size_t)vocab * sizeof(float));
    float *actual = malloc((size_t)vocab * sizeof(float)); assert(reference && actual);
    const uint64_t rng = UINT64_C(0x123456789abcdef);
    ds4_session_checkpoint *p = NULL, *second = NULL;
    assert(!ds4_session_checkpoint_capture(s, "/tmp", &rng, sizeof(rng), &p));
    assert(!ds4_session_checkpoint_capture(s, "/tmp", &rng, sizeof(rng), &second));
    const int token = ds4_session_argmax(s);
    int expected_tokens[8], actual_tokens[8];
    const int count = ds4_session_eval_speculative_argmax(s, token, 4, -1, expected_tokens, 8, error, sizeof(error));
    assert(count > 0 && ds4_session_copy_logits(s, reference, vocab) == vocab);
    for (int mode = 0; mode < 3; mode++) {
        restore(s, p, rng, registry);
        ds4_session_memory before, during, after;
        assert(!ds4_session_memory_query(s, &before) && before.capacity_known && before.device_capacity);
        ds4_resource_totals owned;
        ds4_resources_totals(registry, (uint64_t)(uintptr_t)s, &owned);
        assert(before.device_capacity == owned.allocated[DS4_MEMORY_DEVICE]);
        if (mode != 0) {
            assert(!ds4_session_park(s, "/tmp"));
            assert(ds4_session_pos(s) == prompt.len && ds4_session_checkpoint_valid(s));
            assert(!ds4_session_memory_query(s, &during) && !during.device_capacity);
        }
        if (mode != 1) {
            assert(!ds4_engine_weights_release(e));
            assert(!ds4_engine_weights_release(e));
            ds4_resource_totals weights;
            ds4_resources_totals(registry, DS4_RESOURCE_OWNER_LEGACY_WEIGHTS, &weights);
            assert(!weights.allocated[DS4_MEMORY_DEVICE] && !weights.allocated[DS4_MEMORY_PINNED]);
            assert(weights.allocated[DS4_MEMORY_MMAP]);
            assert(ds4_session_eval(s, token, error, sizeof(error)) != 0);
            assert(ds4_session_eval_speculative_argmax(s, token, 4, -1, actual_tokens, 8, error, sizeof(error)) < 0);
            assert(ds4_session_pos(s) == prompt.len);
            assert(!ds4_session_memory_query(s, &during));
            if (mode == 0) assert(during.device_capacity == before.device_capacity && !during.weights_resident);
            assert(!ds4_engine_weights_load(e));
        }
        if (mode != 0) for (unsigned repeat = 0; repeat < 3; repeat++) {
            uint64_t demand[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE]=before.device_capacity};
            ds4_reservation *lease = NULL;
            assert(!ds4_resources_reserve(registry, 900, demand, &lease));
            ds4_reservation_scope working = ds4_reservation_enter(lease);
            assert(!ds4_session_resume(s));
            ds4_reservation_leave(&working);
            assert(!ds4_reservation_cancel(lease));
            assert(!ds4_session_memory_query(s, &after));
            assert(after.device_capacity == before.device_capacity);
            if (repeat != 2) assert(!ds4_session_park(s, "/tmp"));
        }
        assert(!ds4_session_memory_query(s, &after) && !after.parked);
        assert(ds4_session_eval_speculative_argmax(s, token, 4, -1, actual_tokens, 8, error, sizeof(error)) == count);
        assert(!memcmp(expected_tokens, actual_tokens, (size_t)count * sizeof(int)));
        assert(ds4_session_copy_logits(s, actual, vocab) == vocab);
        assert(!memcmp(reference, actual, (size_t)vocab * sizeof(float)));
        assert(!ds4_session_trim(s));
        printf("{\"mode\":%d,\"capacity\":%" PRIu64 ",\"payload\":%" PRIu64
            ",\"scratch\":%" PRIu64 ",\"recurrent\":%" PRIu64 ",\"mtp\":%" PRIu64 ",\"bit_exact\":true}\n",
            mode, before.device_capacity, before.payload, before.scratch, before.recurrent, before.mtp);
        fflush(stdout);
    }
    ds4_session_report_progress(s, "test", 1, 1); assert(callback_count == 1);
    restore(s, p, rng, registry);
    /* SSD admission failure keeps authoritative graph and logits. */
    ds4_resource_budget budget = {.limit=0};
    assert(!ds4_resources_set_budget(registry, DS4_MEMORY_CHECKPOINT, &budget));
    assert(ds4_session_park(s, "/tmp") == DS4_STATUS_NOMEM);
    assert(ds4_session_pos(s) == prompt.len);
    budget.limit = UINT64_MAX; assert(!ds4_resources_set_budget(registry, DS4_MEMORY_CHECKPOINT, &budget));
    assert(!ds4_session_park(s, "/tmp"));
    assert(!ds4_session_park(sibling, "/tmp"));
    assert(!ds4_session_park(empty, "/tmp"));
    budget.limit = 0; assert(!ds4_resources_set_budget(registry, DS4_MEMORY_DEVICE, &budget));
    assert(ds4_session_resume(s) != DS4_STATUS_OK);
    assert(ds4_session_checkpoint_valid(s) && ds4_session_pos(s) == prompt.len);
    budget.limit = UINT64_MAX; assert(!ds4_resources_set_budget(registry, DS4_MEMORY_DEVICE, &budget));
    assert(!ds4_session_resume(s)); assert(!ds4_session_resume(sibling)); assert(!ds4_session_resume(empty));
    /* Corrupt a separate checkpoint; neither live state nor caller metadata changes. */
    FILE *fp = fopen(ds4_session_checkpoint_path(second), "r+b"); assert(fp);
    assert(!fseek(fp, 140, SEEK_SET)); int byte = fgetc(fp); assert(byte != EOF);
    assert(!fseek(fp, 140, SEEK_SET)); assert(fputc(byte ^ 1, fp) != EOF); assert(!fclose(fp));
    assert(ds4_session_copy_logits(s, reference, vocab) == vocab);
    uint64_t metadata = 999; size_t bytes = sizeof(metadata);
    assert(ds4_session_checkpoint_restore(s, second, &metadata, &bytes) == DS4_STATUS_INVALID);
    assert(metadata == 999 && !access(ds4_session_checkpoint_path(second), F_OK));
    assert(ds4_session_copy_logits(s, actual, vocab) == vocab && !memcmp(reference, actual, (size_t)vocab * sizeof(float)));
    assert(!ds4_session_park(s, "/tmp"));
    ds4_session_invalidate(s);
    assert(!ds4_session_pos(s) && !ds4_session_checkpoint_valid(s));
    assert(!ds4_session_resume(s));
    assert(!ds4_session_pos(s) && !ds4_session_checkpoint_valid(s));
    ds4_session_checkpoint_free(second); ds4_session_checkpoint_free(p);
    ds4_session_free(s); ds4_session_free(sibling); ds4_session_free(empty);
    ds4_vision_embedding_free(&span.embedding); ds4_vision_embedding_free(&embedding); ds4_tokens_free(&prompt);
    ds4_engine_close(e); free(reference); free(actual);
    ds4_resource_totals totals; ds4_resources_totals(registry, 0, &totals);
    assert(!totals.roots && !totals.views);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        assert(!totals.allocated[i] && !totals.reserved[i]);
    puts("PASS: lifecycle bit-exact, multiple checkpoints/sessions, corruption, failed restore and SSD admission");
    return 0;
}
