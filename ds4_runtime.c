#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "ds4_runtime.h"
#include "ds4_resources.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct ds4_runtime {
    pthread_mutex_t mutex;
    size_t references;
    ds4_runtime_ops ops;
    void *opaque;
    ds4_runtime_model *active;
    ds4_runtime_model *closing;
    bool transitioning;
    size_t depth;
};

struct ds4_runtime_model {
    ds4_runtime *runtime;
    ds4_runtime_descriptor descriptor;
    ds4_runtime_model_ops ops;
    void *opaque;
    char *identity;
};

ds4_status ds4_runtime_create(ds4_runtime **out, const ds4_runtime_ops *ops, void *opaque) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    ds4_runtime *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return DS4_STATUS_NOMEM;
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr)) { free(runtime); return DS4_STATUS_NOT_READY; }
    int rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (!rc) rc = pthread_mutex_init(&runtime->mutex, &attr);
    pthread_mutexattr_destroy(&attr);
    if (rc) { free(runtime); return DS4_STATUS_NOT_READY; }
    runtime->references = 1;
    runtime->opaque = opaque;
    if (ops) runtime->ops = *ops;
    *out = runtime;
    return DS4_STATUS_OK;
}

void ds4_runtime_retain(ds4_runtime *runtime) {
    if (!runtime) return;
    pthread_mutex_lock(&runtime->mutex);
    runtime->references++;
    pthread_mutex_unlock(&runtime->mutex);
}

void ds4_runtime_release(ds4_runtime *runtime) {
    if (!runtime) return;
    pthread_mutex_lock(&runtime->mutex);
    if (--runtime->references) { pthread_mutex_unlock(&runtime->mutex); return; }
    if (runtime->ops.shutdown) runtime->ops.shutdown(runtime->opaque);
    pthread_mutex_unlock(&runtime->mutex);
    pthread_mutex_destroy(&runtime->mutex);
    free(runtime);
}

ds4_status ds4_runtime_model_create(ds4_runtime *runtime,
        const ds4_runtime_descriptor *descriptor, const ds4_runtime_model_ops *ops,
        void *opaque, ds4_runtime_model **out) {
    if (!out) return DS4_STATUS_INVALID;
    *out = NULL;
    if (!runtime || !descriptor || descriptor->abi != DS4_SUITE_ABI ||
        !descriptor->model_id || !descriptor->model_id[0] ||
        !descriptor->revision || !descriptor->revision[0] ||
        !descriptor->profile || !descriptor->profile[0] ||
        !descriptor->component || !descriptor->component[0] ||
        !descriptor->operations || (descriptor->operations & ~DS4_OPERATION_ALL) ||
        !descriptor->input_modalities || (descriptor->input_modalities & ~DS4_MODALITY_ALL))
        return DS4_STATUS_INVALID;
    ds4_runtime_model *model = calloc(1, sizeof(*model));
    if (!model) return DS4_STATUS_NOMEM;
    const char *strings[] = {descriptor->model_id, descriptor->revision,
                            descriptor->profile, descriptor->component};
    size_t bytes = 0;
    for (size_t i = 0; i < 4; i++) {
        size_t length = strlen(strings[i]);
        if (length == SIZE_MAX || bytes > SIZE_MAX - length - 1) {
            free(model); return DS4_STATUS_INVALID;
        }
        bytes += length + 1;
    }
    model->identity = malloc(bytes);
    if (!model->identity) { free(model); return DS4_STATUS_NOMEM; }
    model->descriptor = *descriptor;
    const char **destinations[] = {&model->descriptor.model_id, &model->descriptor.revision,
                                  &model->descriptor.profile, &model->descriptor.component};
    char *cursor = model->identity;
    for (size_t i = 0; i < 4; i++) {
        size_t length = strlen(strings[i]) + 1;
        memcpy(cursor, strings[i], length);
        *destinations[i] = cursor;
        cursor += length;
    }
    model->runtime = runtime;
    model->opaque = opaque;
    if (ops) model->ops = *ops;
    ds4_runtime_retain(runtime);
    *out = model;
    return DS4_STATUS_OK;
}

const ds4_runtime_descriptor *ds4_runtime_model_descriptor(const ds4_runtime_model *model) {
    return model ? &model->descriptor : NULL;
}

/* Used only while holding the runtime mutex and outside an execution scope.
 * Both switches and destruction install the owner's context after a fence. */
static ds4_status runtime_activate_model(ds4_runtime *runtime, ds4_runtime_model *model) {
    ds4_resource_owner_scope owner __attribute__((cleanup(ds4_resource_owner_leave))) =
        ds4_resource_owner_enter((uint64_t)(uintptr_t)model);
    runtime->transitioning = true;
    ds4_status status = runtime->ops.fence ? runtime->ops.fence(runtime->opaque) : DS4_STATUS_OK;
    if (status == DS4_STATUS_OK && model->ops.activate)
        status = model->ops.activate(model->opaque);
    runtime->transitioning = false;
    /* A failed callback may already have changed backend state. */
    runtime->active = status == DS4_STATUS_OK ? model : NULL;
    return status;
}

static ds4_status runtime_model_enter(ds4_runtime_model *model, ds4_runtime_scope *scope, bool exclusive) {
    if (!scope || scope->model || scope->runtime || !model) return DS4_STATUS_INVALID;
    ds4_runtime *runtime = model->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->transitioning || (exclusive && runtime->depth) ||
        (runtime->depth && runtime->active != model) ||
        (runtime->closing && runtime->closing != model)) {
        pthread_mutex_unlock(&runtime->mutex);
        return DS4_STATUS_BUSY;
    }
    if (!runtime->depth && runtime->active != model) {
        ds4_status status = runtime_activate_model(runtime, model);
        if (status != DS4_STATUS_OK) {
            pthread_mutex_unlock(&runtime->mutex);
            return status;
        }
    }
    runtime->depth++;
    scope->model = model;
    scope->runtime = runtime;
    scope->previous_resource_owner = ds4_resource_current_owner();
    (void)ds4_resource_owner_enter((uint64_t)(uintptr_t)model);
    return DS4_STATUS_OK;
}

ds4_status ds4_runtime_model_enter(ds4_runtime_model *model, ds4_runtime_scope *scope) {
    return runtime_model_enter(model, scope, false);
}
ds4_status ds4_runtime_model_enter_exclusive(ds4_runtime_model *model, ds4_runtime_scope *scope) {
    return runtime_model_enter(model, scope, true);
}

ds4_status ds4_runtime_begin_update(ds4_runtime *runtime, ds4_runtime_scope *scope) {
    if (!runtime || !scope || scope->model || scope->runtime) return DS4_STATUS_INVALID;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->depth || runtime->closing || runtime->transitioning) {
        pthread_mutex_unlock(&runtime->mutex);
        return DS4_STATUS_BUSY;
    }
    /* The update lease itself keeps the runtime alive even when its caller
     * drops the service reference before leaving the scope. */
    runtime->references++;
    runtime->transitioning = true;
    ds4_status status = runtime->ops.fence ? runtime->ops.fence(runtime->opaque) : DS4_STATUS_OK;
    runtime->transitioning = false;
    runtime->active = NULL;
    if (status != DS4_STATUS_OK) {
        pthread_mutex_unlock(&runtime->mutex);
        ds4_runtime_release(runtime);
        return status;
    }
    runtime->depth = 1;
    scope->runtime = runtime;
    scope->previous_resource_owner = ds4_resource_current_owner();
    (void)ds4_resource_owner_enter((uint64_t)(uintptr_t)runtime);
    return DS4_STATUS_OK;
}

void ds4_runtime_scope_leave(ds4_runtime_scope *scope) {
    if (!scope || !scope->runtime) return;
    ds4_runtime *runtime = scope->runtime;
    const bool update = !scope->model;
    ds4_resource_owner_scope owner = {scope->previous_resource_owner};
    ds4_resource_owner_leave(&owner);
    runtime->depth--;
    if (!scope->model) runtime->active = NULL;
    scope->model = NULL;
    scope->runtime = NULL;
    pthread_mutex_unlock(&runtime->mutex);
    if (update) ds4_runtime_release(runtime);
}

ds4_status ds4_runtime_model_close(ds4_runtime_model *model) {
    if (!model) return DS4_STATUS_OK;
    ds4_runtime *runtime = model->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->depth || runtime->closing || runtime->transitioning) {
        pthread_mutex_unlock(&runtime->mutex); return DS4_STATUS_BUSY;
    }
    runtime->closing = model;
    /* Destructors may use the owner's backend context too. Do not release
     * another model while a previous owner's shape/quality is installed. */
    ds4_status status = runtime_activate_model(runtime, model);
    if (status != DS4_STATUS_OK) {
        runtime->closing = NULL;
        pthread_mutex_unlock(&runtime->mutex);
        return status;
    }
    ds4_resource_owner_scope owner __attribute__((cleanup(ds4_resource_owner_leave))) =
        ds4_resource_owner_enter((uint64_t)(uintptr_t)model);
    if (model->ops.release) model->ops.release(model->opaque);
    /* A legacy destructor can call other public functions on its own engine. */
    if (runtime->active == model) runtime->active = NULL;
    runtime->closing = NULL;
    pthread_mutex_unlock(&runtime->mutex);
    free(model->identity);
    free(model);
    ds4_runtime_release(runtime);
    return DS4_STATUS_OK;
}
