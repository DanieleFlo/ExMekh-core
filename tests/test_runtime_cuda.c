/* A real legacy LLM plus independently owned component tensors. This is a
 * lifecycle test, not a claim that the future image/embedding kernels exist. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <math.h>

typedef struct { ds4_gpu_tensor *tensor; float value; bool quality; int released; } component;
static ds4_status activate(void *opaque) {
    component *c = opaque;
    ds4_gpu_set_quality(c->quality);
    ds4_gpu_set_glm_model(false);
    ds4_gpu_set_qwen_model(false);
    return DS4_STATUS_OK;
}
static void release_component(void *opaque) {
    component *c = opaque;
    ds4_gpu_tensor_free(c->tensor); c->tensor = NULL; c->released++;
}
static ds4_runtime_model *make_component(ds4_runtime *runtime, component *c, const char *id) {
    ds4_runtime_model *handle = NULL;
    const ds4_runtime_descriptor descriptor = {DS4_SUITE_ABI, id, "fixture-v1", "f32", "encoder",
        DS4_OPERATION_EMBED, DS4_MODALITY_ALL, 8192, 768};
    const ds4_runtime_model_ops ops = {activate, release_component};
    assert(ds4_runtime_model_create(runtime, &descriptor, &ops, c, &handle) == DS4_STATUS_OK);
    ds4_runtime_scope scope = {0};
    assert(ds4_runtime_model_enter(handle, &scope) == DS4_STATUS_OK);
    c->tensor = ds4_gpu_tensor_alloc(4096);
    assert(c->tensor && ds4_gpu_tensor_write(c->tensor, 0, &c->value, sizeof(float)));
    ds4_runtime_scope_leave(&scope);
    return handle;
}
static void check_component(ds4_runtime_model *handle, component *c) {
    ds4_runtime_scope scope = {0}; float value = 0;
    assert(ds4_runtime_model_enter(handle, &scope) == DS4_STATUS_OK);
    assert(ds4_gpu_tensor_read(c->tensor, 0, &value, sizeof(value)) && value == c->value);
    ds4_runtime_scope_leave(&scope);
}

static void check_standalone_vision(const char *model) {
    char path[] = "/tmp/ds4-suite-vision-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    float *reference = NULL;
    uint32_t dimensions[4] = {0};
    for (int repeat = 0; repeat < 2; repeat++) {
        assert(ds4_qwen4_vision_dump(model, "tests/vision-fixtures/glm53/spatial.png", path, 64, 128));
        FILE *stream = fopen(path, "rb");
        uint32_t header[4];
        assert(stream && fread(header, sizeof(header), 1, stream) == 1);
        assert(header[0] >= 64 && header[0] <= 128 && header[1] > 0);
        const size_t count = (size_t)header[0] * header[1];
        float *values = malloc(count * sizeof(float));
        assert(values && fread(values, sizeof(float), count, stream) == count);
        assert(fgetc(stream) == EOF);
        fclose(stream);
        for (size_t i = 0; i < count; i++) assert(isfinite(values[i]));
        if (!repeat) {
            memcpy(dimensions, header, sizeof(header));
            reference = values;
        } else {
            assert(!memcmp(dimensions, header, sizeof(header)));
            assert(!memcmp(reference, values, count * sizeof(float)));
            free(values);
        }
    }
    free(reference);
    unlink(path);
    /* An output-open failure must release the private runtime too. */
    assert(!ds4_qwen4_vision_dump(model, "tests/vision-fixtures/glm53/spatial.png", "/", 64, 128));
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) return 2;
    if (argc == 4) check_standalone_vision(argv[3]);
    ds4_engine_options options = {.model_path=argv[1], .backend=DS4_BACKEND_CUDA,
        .context_size=256, .prefill_chunk=32};
    ds4_engine *e = NULL, *refused = NULL;
    ds4_session *s = NULL;
    ds4_tokens prompt = {0}; char err[256] = {0};
    assert(!ds4_qwen4_vision_dump("missing-vision.gguf", "missing-image.png", "unused-dump.bin", 0, 0));
    assert(!ds4_engine_open(&e, &options));
    ds4_runtime *runtime = ds4_engine_runtime(e);
    assert(!ds4_qwen4_vision_dump(NULL, NULL, NULL, 0, 0));
    assert(ds4_engine_open(&refused, &options) != 0 && !refused);
    assert(ds4_engine_open_with_runtime(&refused, runtime, &options) != 0 && !refused);
    assert(!ds4_session_create(&s, e, 256));
    assert(ds4_engine_close_checked(e) == DS4_STATUS_BUSY);
    ds4_tokenize_text(e, "A red square beside a blue circle.", &prompt);
    assert(!ds4_session_sync(s, &prompt, err, sizeof(err)));
    ds4_session_snapshot snapshot = {0};
    assert(!ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)));
    const int token = ds4_session_argmax(s);
    assert(!ds4_session_eval(s, token, err, sizeof(err)));
    const int n = ds4_engine_vocab_size(e), dim = ds4_engine_embd_dim(e);
    float *before = malloc((size_t)n * sizeof(float)), *after = malloc((size_t)n * sizeof(float));
    assert(before && after && ds4_session_copy_logits(s, before, n) == n);
    component a = {NULL, 17.25f, true, 0}, b = {NULL, -42.5f, false, 0};
    assert(runtime);
    ds4_runtime_model *ha = make_component(runtime, &a, "fixture-image");
    ds4_runtime_model *hb = make_component(runtime, &b, "fixture-audio");
    ds4_runtime_scope blocked = {0};
    assert(ds4_runtime_model_enter(hb, &blocked) == DS4_STATUS_OK);
    assert(!ds4_engine_tp_vocab_split(e));
    assert(!ds4_engine_tp_bind(e, NULL, err, sizeof(err)));
    ds4_runtime_scope_leave(&blocked);
    void *compute_stream = g_gpu[0].stream, *blas = g_gpu[0].cublas;
    for (int i = 0; i < 4; i++) {
        check_component(ha, &a); check_component(hb, &b);
        assert(!ds4_session_load_snapshot(s, &snapshot, err, sizeof(err)));
        assert(!ds4_session_eval(s, token, err, sizeof(err)));
        assert(ds4_engine_embd_dim(e) == dim && ds4_engine_vocab_size(e) == n);
        assert(ds4_session_copy_logits(s, after, n) == n);
        assert(!memcmp(before, after, (size_t)n * sizeof(float)));
    }
    assert(ds4_runtime_model_close(ha) == DS4_STATUS_OK && a.released == 1);
    check_component(hb, &b);
    ds4_session_free(s); ds4_engine_close(e);
    /* Closing the LLM releases its ranges but must preserve runtime-owned
     * streams and another component's persistent allocations. */
    check_component(hb, &b);
    int lock = open("/tmp/ds4.lock", O_RDWR);
    assert(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) != 0);
    close(lock);
    assert(ds4_engine_open(&refused, &options) != 0 && !refused);
    ds4_runtime_scope nested = {0};
    assert(ds4_runtime_model_enter(hb, &nested) == DS4_STATUS_OK);
    assert(ds4_engine_open_with_runtime(&refused, runtime, &options) != 0 && !refused);
    ds4_runtime_scope_leave(&nested);
    if (argc >= 3) {
        /* Optional real shape/variant switch, using the existing Flash tiered
         * launcher environment. The original Qwen snapshot is restored next. */
        ds4_engine_options flash = options; flash.model_path = argv[2];
        assert(!ds4_engine_open_with_runtime(&e, runtime, &flash));
        assert(ds4_engine_is_qwen4(e));
        assert(g_gpu[0].stream == compute_stream && g_gpu[0].cublas == blas);
        assert(!ds4_session_create(&s, e, 256));
        ds4_tokens text = {0}; ds4_tokenize_text(e, "Describe a red square.", &text);
        assert(!ds4_session_sync(s, &text, err, sizeof(err)));
        int vocab = ds4_engine_vocab_size(e); float *row = malloc((size_t)vocab*sizeof(float));
        assert(row && ds4_session_copy_logits(s, row, vocab) == vocab);
        for (int i = 0; i < vocab; i++) assert(isfinite(row[i]));
        free(row); ds4_tokens_free(&text); ds4_session_free(s);
        assert(ds4_engine_close_checked(e) == DS4_STATUS_OK);
        check_component(hb, &b);
    }
    assert(!ds4_engine_open_with_runtime(&e, runtime, &options));
    assert(g_gpu[0].stream == compute_stream && g_gpu[0].cublas == blas);
    check_component(hb, &b);
    assert(!ds4_session_create(&s, e, 256));
    assert(!ds4_session_load_snapshot(s, &snapshot, err, sizeof(err)));
    assert(ds4_session_tokens(s)->len == prompt.len);
    assert(!ds4_session_eval(s, token, err, sizeof(err)));
    assert(ds4_session_copy_logits(s, after, n) == n);
    assert(!memcmp(before, after, (size_t)n * sizeof(float)));
    ds4_session_free(s); assert(ds4_engine_close_checked(e) == DS4_STATUS_OK);
    check_component(hb, &b);
    assert(ds4_runtime_model_close(hb) == DS4_STATUS_OK && b.released == 1);
    lock = open("/tmp/ds4.lock", O_RDWR);
    assert(lock >= 0 && !flock(lock, LOCK_EX | LOCK_NB));
    assert(!flock(lock, LOCK_UN)); close(lock);
    /* A component can initialize the service before any LLM is loaded. */
    assert(ds4_compute_runtime_open(&runtime) == DS4_STATUS_OK);
    compute_stream = g_gpu[0].stream; blas = g_gpu[0].cublas;
    b.released = 0; b.quality = true;
    hb = make_component(runtime, &b, "fixture-audio-first");
    check_component(hb, &b);
    assert(!ds4_engine_open_with_runtime(&e, runtime, &options));
    assert(g_gpu[0].stream == compute_stream && g_gpu[0].cublas == blas);
    assert(!ds4_session_create(&s, e, 256));
    assert(!ds4_session_sync(s, &prompt, err, sizeof(err)));
    assert(!ds4_session_eval(s, token, err, sizeof(err)));
    assert(ds4_session_copy_logits(s, after, n) == n);
    assert(!memcmp(before, after, (size_t)n * sizeof(float)));
    ds4_session_free(s); ds4_engine_close(e);
    check_component(hb, &b);
    assert(ds4_runtime_model_close(hb) == DS4_STATUS_OK && b.released == 1);
    lock = open("/tmp/ds4.lock", O_RDWR);
    assert(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) != 0); close(lock);
    ds4_runtime_release(runtime);
    lock = open("/tmp/ds4.lock", O_RDWR);
    assert(lock >= 0 && !flock(lock, LOCK_EX | LOCK_NB));
    assert(!flock(lock, LOCK_UN)); close(lock);
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens_free(&prompt); free(before); free(after);
    ds4_resource_totals totals;
    ds4_resources_totals(ds4_gpu_resources(), 0, &totals);
    assert(!totals.roots && !totals.views);
    for (unsigned i = 0; i < DS4_MEMORY_COUNT; i++)
        assert(!totals.allocated[i] && !totals.reserved[i]);
    puts("runtime CUDA ownership/activation/legacy reopen: PASS");
    return 0;
}
