#ifndef DS4_NUMERIC_WEIGHTS_H
#define DS4_NUMERIC_WEIGHTS_H
#include "ds4_bundle.h"
#include "ds4_numeric.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Bridge between generic numeric storage and the bundle's group protocol.
 * The numeric core does not depend on model formats or weight inventories. */
ds4_weight_backend ds4_numeric_weight_backend(ds4_numeric *);
ds4_num_view ds4_numeric_weight_view(const void *storage, uint64_t byte_offset,
                                     uint64_t count, uint32_t type);
#ifdef __cplusplus
}
#endif
#endif
