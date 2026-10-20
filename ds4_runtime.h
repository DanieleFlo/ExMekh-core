#ifndef DS4_RUNTIME_H
#define DS4_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_SUITE_ABI 1u

typedef enum {
    DS4_STATUS_OK = 0,
    DS4_STATUS_INVALID,
    DS4_STATUS_UNSUPPORTED,
    DS4_STATUS_NOT_RESIDENT,
    DS4_STATUS_NOT_READY,
    DS4_STATUS_BUSY,
    DS4_STATUS_NOMEM,
    DS4_STATUS_CANCELLED,
    DS4_STATUS_INFERENCE,
    DS4_STATUS_RESTORE
} ds4_status;

enum {
    DS4_MODALITY_TEXT = 1u,
    DS4_MODALITY_IMAGE = 2u,
    DS4_MODALITY_VIDEO = 4u,
    DS4_MODALITY_AUDIO = 8u,
    DS4_MODALITY_ALL = DS4_MODALITY_TEXT | DS4_MODALITY_IMAGE |
                       DS4_MODALITY_VIDEO | DS4_MODALITY_AUDIO
};
enum {
    DS4_OPERATION_CHAT = 1u,
    DS4_OPERATION_IMAGE_GENERATE = 2u,
    DS4_OPERATION_IMAGE_EDIT = 4u,
    DS4_OPERATION_EMBED = 8u,
    DS4_OPERATION_ALL = DS4_OPERATION_CHAT | DS4_OPERATION_IMAGE_GENERATE |
                        DS4_OPERATION_IMAGE_EDIT | DS4_OPERATION_EMBED
};

/* Immutable identity, separate from both device residency and session state.
 * Strings are copied on registration; descriptor lifetime is the model's. */
typedef struct {
    uint32_t abi;
    const char *model_id;
    const char *revision;
    const char *profile;
    const char *component;
    uint32_t operations;
    uint32_t input_modalities;
    uint32_t context_tokens;
    uint32_t embedding_dimensions;
} ds4_runtime_descriptor;

typedef struct ds4_runtime ds4_runtime;
typedef struct ds4_runtime_model ds4_runtime_model;

/* Backend owns initialization separately. fence orders cross-model switches;
 * shutdown runs once, after the last runtime/model reference, never at eviction. */
typedef struct {
    ds4_status (*fence)(void *opaque);
    void (*shutdown)(void *opaque);
} ds4_runtime_ops;

typedef struct {
    ds4_status (*activate)(void *opaque);
    void (*release)(void *opaque);
} ds4_runtime_model_ops;

ds4_status ds4_runtime_create(ds4_runtime **out, const ds4_runtime_ops *ops, void *opaque);
void ds4_runtime_retain(ds4_runtime *runtime);
void ds4_runtime_release(ds4_runtime *runtime);
ds4_status ds4_runtime_model_create(ds4_runtime *runtime,
        const ds4_runtime_descriptor *descriptor, const ds4_runtime_model_ops *ops,
        void *opaque, ds4_runtime_model **out);
const ds4_runtime_descriptor *ds4_runtime_model_descriptor(const ds4_runtime_model *model);
ds4_status ds4_runtime_model_close(ds4_runtime_model *model);

/* Initialize scopes with {0}. They serialize execution, including recursive calls on the same model.
 * Cross-model recursion is rejected: a callback cannot change the active
 * model underneath its caller. Switching between completed scopes fences.
 * The caller must join users of a handle before closing it. */
typedef struct {
    ds4_runtime_model *model;
    ds4_runtime *runtime;
} ds4_runtime_scope;
ds4_status ds4_runtime_model_enter(ds4_runtime_model *model, ds4_runtime_scope *scope);
/* Exclusive construction/update, before a model handle exists. Rejects nested
 * execution, fences the previous owner, and invalidates the active context. */
ds4_status ds4_runtime_begin_update(ds4_runtime *runtime, ds4_runtime_scope *scope);
/* Leave either kind of scope on its entering thread; safe for an empty scope. */
void ds4_runtime_scope_leave(ds4_runtime_scope *scope);

/* Adapter contract for T03–T05. No HTTP or tensor layout appears here.
 * An estimate is unknown unless known=true; zero is a valid known estimate.
 * Plans own the referenced arrays for the duration of an operation. */
typedef struct {
    uint64_t device_bytes, host_bytes, pinned_bytes, checkpoint_bytes;
    bool known;
} ds4_runtime_estimate;
typedef struct {
    uint32_t id;
    const uint32_t *dependencies;
    size_t dependency_count;
    const uint32_t *weight_groups;
    size_t weight_group_count;
    ds4_runtime_estimate peak;
} ds4_runtime_phase;
typedef struct {
    ds4_status (*prepare)(void *, const void *, ds4_runtime_phase **, size_t *);
    ds4_status (*execute)(void *, uint32_t, void *);
    void (*release_plan)(void *, ds4_runtime_phase *, size_t);
    ds4_status (*estimate)(void *, const void *, ds4_runtime_estimate *);
    ds4_status (*load)(void *, uint32_t);
    ds4_status (*release)(void *, uint32_t);
    ds4_status (*quiesce)(void *);
    ds4_status (*checkpoint)(void *, void **);
    ds4_status (*restore)(void *, const void *);
    ds4_status (*validate)(void *);
} ds4_runtime_adapter;

#ifdef __cplusplus
}
#endif
#endif
