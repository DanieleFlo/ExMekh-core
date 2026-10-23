#ifndef DS4_BUNDLE_H
#define DS4_BUNDLE_H
#include "ds4_resources.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Offline-verified directory over original GGUF/safetensors bytes. Dimensions
 * and byte strides use GGUF order (fastest dimension first). Quantized stride
 * zero is the block byte size, with block_elements values per block. */
#define DS4_WEIGHT_MAX_DIMS 8u
typedef struct ds4_bundle ds4_bundle;
typedef struct ds4_weight_group ds4_weight_group;
typedef struct ds4_bundle_hint ds4_bundle_hint;
typedef struct {
    const char *name;
    uint32_t file, component, group, type, rank, block_elements, block_bytes, alignment;
    uint64_t offset, bytes, shape[DS4_WEIGHT_MAX_DIMS], stride[DS4_WEIGHT_MAX_DIMS];
} ds4_weight_tensor;
typedef struct { const char *id, *revision, *compute_dtype; } ds4_bundle_component;
typedef struct { const char *path, *revision, *format; uint64_t bytes; uint8_t sha256[32]; } ds4_bundle_file;

ds4_status ds4_bundle_open(ds4_bundle **, const char *index_path, const char *artifact_root,
                          ds4_resources *, uint64_t owner);
ds4_status ds4_bundle_close(ds4_bundle *);
const char *ds4_bundle_model(const ds4_bundle *);
const char *ds4_bundle_revision(const ds4_bundle *);
const char *ds4_bundle_profile(const ds4_bundle *);
size_t ds4_bundle_tensor_count(const ds4_bundle *);
size_t ds4_bundle_component_count(const ds4_bundle *);
size_t ds4_bundle_file_count(const ds4_bundle *);
const ds4_weight_tensor *ds4_bundle_tensor(const ds4_bundle *, size_t);
const ds4_bundle_component *ds4_bundle_component_at(const ds4_bundle *, size_t);
const ds4_bundle_file *ds4_bundle_file_at(const ds4_bundle *, size_t);
const void *ds4_bundle_tensor_data(const ds4_bundle *, size_t);
const void *ds4_bundle_file_data(const ds4_bundle *, size_t);
/* An architecture adapter must bind an exhaustive expected inventory before
 * inference. No unknown tensor, type or shape is silently skipped. */
ds4_status ds4_bundle_validate_component(const ds4_bundle *, size_t component,
        const ds4_weight_tensor *expected, size_t count);
int ds4_bundle_dump(const ds4_bundle *, FILE *);
/* Explicit, cancellable CPU/page-cache hint on an already verified bundle.
 * No paths or partial generated tool text are accepted. The byte bound is
 * advisory I/O, not a promise of resident RAM. close drains before unmapping. */
ds4_status ds4_bundle_hint_start(ds4_bundle_hint **, ds4_bundle *, uint32_t component,
                                 uint64_t max_bytes);
ds4_status ds4_bundle_hint_close(ds4_bundle_hint *);

/* Backend allocation is responsible for resource accounting. Original mapped
 * bytes are authoritative and never dequantized by the loader. */
typedef struct {
    void *context;
    ds4_status (*allocate)(void *, uint64_t, void **);
    ds4_status (*write)(void *, void *, uint64_t, const void *, uint64_t);
    ds4_status (*quiesce)(void *);
    ds4_status (*release)(void *, void *);
} ds4_weight_backend;
/* Optional availability barrier for newly created, unloaded groups. Keeps the
 * v1 backend layout intact. quiesce still drains last use/failed uploads. */
ds4_status ds4_weight_group_availability(ds4_weight_group *, ds4_status (*ready)(void *));
ds4_status ds4_weight_group_create(ds4_weight_group **, ds4_bundle *, uint32_t component,
                                  uint32_t group, const ds4_weight_backend *);
/* Architecture-defined execution groups may split an offline storage group.
 * Indices must be unique and belong to component; the same upload/lifetime
 * machinery applies. This avoids loading an unused autoregressive head. */
ds4_status ds4_weight_group_create_selected(ds4_weight_group **, ds4_bundle *, uint32_t component,
        const size_t *tensor_indices, size_t count, const ds4_weight_backend *);
uint64_t ds4_weight_group_bytes(const ds4_weight_group *);
uint64_t ds4_weight_group_payload(const ds4_weight_group *);
bool ds4_weight_group_resident(const ds4_weight_group *);
ds4_status ds4_weight_group_load(ds4_weight_group *);
ds4_status ds4_weight_group_release(ds4_weight_group *);
ds4_status ds4_weight_group_close(ds4_weight_group *);
ds4_status ds4_weight_group_tensor(const ds4_weight_group *, const char *name,
                                  const void **storage, uint64_t *offset);
#ifdef __cplusplus
}
#endif
#endif
