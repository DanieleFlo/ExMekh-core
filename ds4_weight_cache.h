#ifndef DS4_WEIGHT_CACHE_H
#define DS4_WEIGHT_CACHE_H
#include "ds4_bundle.h"

/* Adapter-owned, serial cache. A visit ends at the next acquire: borrowed
 * weight views must not cross that boundary. Release fences the last GPU use.
 * Byte limits include alignment in the packed groups, never tensor estimates.
 * No policy here depends on a model, layer number or accelerator capacity. */
enum { DS4_WEIGHT_CACHE_GROUPS = 80 };
/* Embedded storage; callers use the cache operations below, not these fields.
 */
typedef struct {
  ds4_weight_group **groups;
  size_t count;
  uint64_t limit, clock, stamp[DS4_WEIGHT_CACHE_GROUPS],
      visits[DS4_WEIGHT_CACHE_GROUPS];
  bool keep[DS4_WEIGHT_CACHE_GROUPS];
  size_t last;
  uint64_t loads, hits, evictions, transferred;
} ds4_weight_cache;

typedef struct {
  uint64_t total, minimum, resident, limit;
  uint64_t loads, hits, evictions, transferred;
} ds4_weight_budget;

#ifdef __cplusplus
extern "C" {
#endif
ds4_status ds4_weight_cache_init(ds4_weight_cache *, ds4_weight_group **,
                                 size_t);
uint64_t ds4_weight_cache_minimum(const ds4_weight_cache *);
uint64_t ds4_weight_cache_resident(const ds4_weight_cache *);
uint64_t ds4_weight_cache_total(const ds4_weight_cache *);
uint64_t ds4_weight_cache_missing(const ds4_weight_cache *);
ds4_status ds4_weight_cache_budget(const ds4_weight_cache *,
                                   ds4_weight_budget *);
/* Warm a prefix only when the whole cache fits. A bounded cache acquires
 * groups on demand so preloading cannot evict its next working group. */
ds4_status ds4_weight_cache_preload(ds4_weight_cache *, size_t count);
/* Read-only admission first. Failed trimming preserves all unreleased groups.
 */
ds4_status ds4_weight_cache_limit(ds4_weight_cache *, uint64_t);
ds4_status ds4_weight_cache_acquire(ds4_weight_cache *, size_t);
ds4_status ds4_weight_cache_release(ds4_weight_cache *);
#ifdef __cplusplus
}
#endif
#endif
