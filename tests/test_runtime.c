#include "ds4_runtime.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int fences, shutdowns, active, shape, quality, value[2], releases[2];
    bool fence_failure, activation_failure;
    ds4_runtime *runtime;
} backend;
typedef struct { backend *backend; int id, shape, quality; ds4_runtime_model *handle; } owner;

static ds4_status fence(void *opaque) {
    backend *b = opaque;
    b->fences++;
    ds4_runtime_scope nested = {0};
    assert(ds4_runtime_begin_update(b->runtime, &nested) == DS4_STATUS_BUSY);
    return b->fence_failure ? DS4_STATUS_INFERENCE : DS4_STATUS_OK;
}
static void shutdown_backend(void *opaque) { ((backend *)opaque)->shutdowns++; }
static ds4_status activate(void *opaque) {
    owner *o = opaque;
    o->backend->active = o->id;
    o->backend->shape = o->shape;
    o->backend->quality = o->quality;
    ds4_runtime_scope nested = {0};
    assert(ds4_runtime_model_enter(o->handle, &nested) == DS4_STATUS_BUSY);
    assert(ds4_runtime_model_close(o->handle) == DS4_STATUS_BUSY);
    return o->backend->activation_failure ? DS4_STATUS_NOT_READY : DS4_STATUS_OK;
}
static void release_owner(void *opaque) {
    owner *o = opaque;
    assert(o->backend->active == o->id);
    assert(ds4_runtime_model_close(o->handle) == DS4_STATUS_BUSY);
    ds4_runtime_scope nested = {0};
    assert(ds4_runtime_model_enter(o->handle, &nested) == DS4_STATUS_OK);
    ds4_runtime_scope_leave(&nested);
    o->backend->value[o->id] = 0;
    o->backend->releases[o->id]++;
}
static void *worker(void *opaque) {
    owner *o = opaque;
    for (int i = 0; i < 10000; i++) {
        ds4_runtime_scope scope = {0};
        assert(ds4_runtime_model_enter(o->handle, &scope) == DS4_STATUS_OK);
        assert(o->backend->active == o->id && o->backend->shape == o->shape);
        assert(o->backend->quality == o->quality);
        o->backend->value[o->id]++;
        ds4_runtime_scope_leave(&scope);
    }
    return NULL;
}

int main(void) {
    backend b = {0};
    ds4_runtime *runtime = NULL;
    const ds4_runtime_ops ops = {fence, shutdown_backend};
    const ds4_runtime_model_ops model_ops = {activate, release_owner};
    assert(ds4_runtime_create(NULL, &ops, &b) == DS4_STATUS_INVALID);
    assert(ds4_runtime_create(&runtime, &ops, &b) == DS4_STATUS_OK);
    b.runtime = runtime;
    char name[] = "text-model";
    ds4_runtime_descriptor descriptor = {DS4_SUITE_ABI, name, "sha-a", "f32", "decoder",
        DS4_OPERATION_CHAT, DS4_MODALITY_TEXT, 2048, 0};
    owner a = {&b, 0, 128, 0, NULL}, c = {&b, 1, 768, 1, NULL};
    assert(ds4_runtime_model_create(runtime, &descriptor, &model_ops, &a, &a.handle) == DS4_STATUS_OK);
    name[0] = 'X';
    assert(!strcmp(ds4_runtime_model_descriptor(a.handle)->model_id, "text-model"));
    descriptor.model_id = "embedding-model"; descriptor.component = "encoder";
    descriptor.operations = DS4_OPERATION_EMBED; descriptor.input_modalities = DS4_MODALITY_ALL;
    descriptor.embedding_dimensions = 768;
    assert(ds4_runtime_model_create(runtime, &descriptor, &model_ops, &c, &c.handle) == DS4_STATUS_OK);
    descriptor.abi++;
    ds4_runtime_model *bad = (void *)1;
    assert(ds4_runtime_model_create(runtime, &descriptor, NULL, NULL, &bad) == DS4_STATUS_INVALID && !bad);
    ds4_runtime_release(runtime); /* Models retain shared backend ownership. */
    assert(!b.shutdowns);
    ds4_runtime_scope outer = {0}, inner = {0}, other = {0};
    assert(ds4_runtime_model_enter(a.handle, &outer) == DS4_STATUS_OK);
    assert(ds4_runtime_model_enter(a.handle, &inner) == DS4_STATUS_OK);
    assert(ds4_runtime_model_enter(c.handle, &other) == DS4_STATUS_BUSY);
    assert(ds4_runtime_begin_update(runtime, &other) == DS4_STATUS_BUSY);
    assert(ds4_runtime_model_close(a.handle) == DS4_STATUS_BUSY);
    ds4_runtime_scope_leave(&inner); ds4_runtime_scope_leave(&outer);
    b.activation_failure = true;
    assert(ds4_runtime_model_enter(c.handle, &other) == DS4_STATUS_NOT_READY && !other.model);
    b.activation_failure = false;
    assert(ds4_runtime_model_enter(a.handle, &outer) == DS4_STATUS_OK && b.shape == 128);
    ds4_runtime_scope_leave(&outer);
    b.fence_failure = true;
    assert(ds4_runtime_begin_update(runtime, &other) == DS4_STATUS_INFERENCE && !other.runtime);
    assert(ds4_runtime_model_enter(c.handle, &other) == DS4_STATUS_INFERENCE);
    assert(ds4_runtime_model_close(c.handle) == DS4_STATUS_INFERENCE && !b.releases[1]);
    b.fence_failure = false;
    b.activation_failure = true;
    assert(ds4_runtime_model_close(c.handle) == DS4_STATUS_NOT_READY && !b.releases[1]);
    b.activation_failure = false;
    assert(ds4_runtime_begin_update(runtime, &other) == DS4_STATUS_OK);
    assert(ds4_runtime_model_enter(a.handle, &outer) == DS4_STATUS_BUSY);
    assert(ds4_runtime_model_close(a.handle) == DS4_STATUS_BUSY);
    ds4_runtime_scope_leave(&other);
    pthread_t t[2];
    assert(!pthread_create(&t[0], NULL, worker, &a));
    assert(!pthread_create(&t[1], NULL, worker, &c));
    assert(!pthread_join(t[0], NULL) && !pthread_join(t[1], NULL));
    assert(b.value[0] == 10000 && b.value[1] == 10000);
    assert(ds4_runtime_model_close(a.handle) == DS4_STATUS_OK);
    assert(b.value[1] == 10000 && !b.shutdowns);
    worker(&c);
    assert(b.value[1] == 20000);
    assert(ds4_runtime_model_close(c.handle) == DS4_STATUS_OK);
    assert(b.shutdowns == 1 && b.releases[0] == 1 && b.releases[1] == 1);
    assert(ds4_runtime_model_close(NULL) == DS4_STATUS_OK);
    backend update_backend = {0};
    assert(ds4_runtime_create(&runtime, &ops, &update_backend) == DS4_STATUS_OK);
    update_backend.runtime = runtime;
    ds4_runtime_scope update = {0};
    assert(ds4_runtime_begin_update(runtime, &update) == DS4_STATUS_OK);
    ds4_runtime_release(runtime);
    assert(!update_backend.shutdowns);
    ds4_runtime_scope_leave(&update);
    assert(update_backend.shutdowns == 1);
    puts("runtime: PASS");
    return 0;
}
