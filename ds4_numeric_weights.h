#ifndef DS4_NUMERIC_WEIGHTS_H
#define DS4_NUMERIC_WEIGHTS_H
#include "ds4_bundle.h"
#include "ds4_numeric.h"
#include "ds4_weight_cache.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Bridge between generic numeric storage and the bundle's group protocol.
 * The numeric core does not depend on model formats or weight inventories. */
ds4_weight_backend ds4_numeric_weight_backend(ds4_numeric *);
/* Async availability is opt-in for new groups; the v1 backend struct and
 * existing synchronous constructors keep their original ABI. */
ds4_status ds4_numeric_weight_group_selected(ds4_weight_group **, ds4_bundle *,
                                             uint32_t component, const size_t *,
                                             size_t count, ds4_numeric *);
ds4_num_view ds4_numeric_weight_view(const void *storage, uint64_t byte_offset,
                                     uint64_t count, uint32_t type);
/* limit=0 queries only; UINT64_MAX restores unbounded residency. Mutations
 * require the adapter's exclusive runtime scope and never touch other owners.
 */
ds4_status ds4_weight_cache_configure(ds4_weight_cache *, ds4_runtime_model *,
                                      uint64_t limit, ds4_weight_budget *);
#ifdef __cplusplus
}
#endif
#endif
