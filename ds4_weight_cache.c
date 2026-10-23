#include "ds4_weight_cache.h"
#include <string.h>

ds4_status ds4_weight_cache_init(ds4_weight_cache *c, ds4_weight_group **groups,
                                 size_t count) {
  if (!c || !groups || !count || count > DS4_WEIGHT_CACHE_GROUPS)
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < count; i++)
    if (!groups[i])
      return DS4_STATUS_INVALID;
  *c = (ds4_weight_cache){
      .groups = groups, .count = count, .limit = UINT64_MAX, .last = count};
  return DS4_STATUS_OK;
}
uint64_t ds4_weight_cache_minimum(const ds4_weight_cache *c) {
  uint64_t n = 0;
  if (c)
    for (size_t i = 0; i < c->count; i++) {
      uint64_t bytes = ds4_weight_group_bytes(c->groups[i]);
      if (bytes > n)
        n = bytes;
    }
  return n;
}
uint64_t ds4_weight_cache_resident(const ds4_weight_cache *c) {
  uint64_t n = 0;
  if (c)
    for (size_t i = 0; i < c->count; i++)
      if (ds4_weight_group_resident(c->groups[i]))
        n += ds4_weight_group_bytes(c->groups[i]);
  return n;
}
uint64_t ds4_weight_cache_total(const ds4_weight_cache *c) {
  uint64_t n = 0;
  if (c)
    for (size_t i = 0; i < c->count; i++)
      n += ds4_weight_group_bytes(c->groups[i]);
  return n;
}
uint64_t ds4_weight_cache_missing(const ds4_weight_cache *c) {
  if (!c)
    return 0;
  uint64_t n = ds4_weight_cache_total(c), live = ds4_weight_cache_resident(c);
  if (n > c->limit)
    n = c->limit;
  return n > live ? n - live : 0;
}
ds4_status ds4_weight_cache_budget(const ds4_weight_cache *c,
                                   ds4_weight_budget *out) {
  if (!c || !c->groups || !out)
    return DS4_STATUS_INVALID;
  *out = (ds4_weight_budget){ds4_weight_cache_total(c),
                             ds4_weight_cache_minimum(c),
                             ds4_weight_cache_resident(c),
                             c->limit,
                             c->loads,
                             c->hits,
                             c->evictions,
                             c->transferred};
  return DS4_STATUS_OK;
}
ds4_status ds4_weight_cache_preload(ds4_weight_cache *c, size_t count) {
  if (!c || !c->groups || count > c->count)
    return DS4_STATUS_INVALID;
  if (c->limit < ds4_weight_cache_total(c))
    return DS4_STATUS_OK;
  for (size_t i = 0; i < count; i++) {
    ds4_status s = ds4_weight_cache_acquire(c, i);
    if (s)
      return s;
  }
  return DS4_STATUS_OK;
}
static ds4_status trim(ds4_weight_cache *c, uint64_t needed, size_t protected) {
  uint64_t live = ds4_weight_cache_resident(c);
  while (live > c->limit - needed) {
    size_t victim = c->count;
    for (size_t i = 0; i < c->count; i++)
      if (i != protected && !c->keep[i] &&
          ds4_weight_group_resident(c->groups[i]) &&
          (victim == c->count || c->stamp[i] < c->stamp[victim]))
        victim = i;
    if (victim == c->count)
      return DS4_STATUS_NOMEM;
    ds4_status s = ds4_weight_group_release(c->groups[victim]);
    if (s)
      return s;
    live -= ds4_weight_group_bytes(c->groups[victim]);
    c->evictions++;
  }
  return DS4_STATUS_OK;
}
ds4_status ds4_weight_cache_limit(ds4_weight_cache *c, uint64_t limit) {
  if (!c || !c->groups || limit < ds4_weight_cache_minimum(c))
    return DS4_STATUS_INVALID;
  c->limit = limit;
  memset(c->keep, 0, sizeof(c->keep));
  /* Keep the most revisited groups, leaving one largest-group slot for
   * progress. Cold ties use adapter order. This is a byte policy, not a
   * fixed number of resident blocks. Reconsider only at request boundaries. */
  uint64_t room = limit - ds4_weight_cache_minimum(c);
  for (size_t k = 0; k < c->count; k++) {
    size_t best = c->count;
    for (size_t i = 0; i < c->count; i++)
      if (!c->keep[i] && ds4_weight_group_bytes(c->groups[i]) <= room &&
          (best == c->count || c->visits[i] > c->visits[best]))
        best = i;
    if (best == c->count)
      break;
    c->keep[best] = true;
    room -= ds4_weight_group_bytes(c->groups[best]);
  }
  return trim(c, 0, c->count);
}
ds4_status ds4_weight_cache_acquire(ds4_weight_cache *c, size_t group) {
  if (!c || !c->groups || group >= c->count)
    return DS4_STATUS_INVALID;
  if (c->last != group) {
    if (c->visits[group] != UINT64_MAX)
      c->visits[group]++;
    c->last = group;
  }
  if (c->clock == UINT64_MAX) {
    /* Saturation preserves order; it must not wrap into a fresh-old stamp. */
    for (size_t i = 0; i < c->count; i++)
      c->stamp[i] >>= 1;
    c->clock >>= 1;
  }
  c->stamp[group] = ++c->clock;
  if (ds4_weight_group_resident(c->groups[group])) {
    c->hits++;
    return DS4_STATUS_OK;
  }
  uint64_t bytes = ds4_weight_group_bytes(c->groups[group]);
  if (bytes > c->limit)
    return DS4_STATUS_NOMEM;
  ds4_status s = trim(c, bytes, group);
  if (!s)
    s = ds4_weight_group_load(c->groups[group]);
  if (!s) {
    c->loads++;
    c->transferred += ds4_weight_group_payload(c->groups[group]);
  }
  return s;
}
ds4_status ds4_weight_cache_release(ds4_weight_cache *c) {
  if (!c || !c->groups)
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < c->count; i++) {
    ds4_status s = ds4_weight_group_release(c->groups[i]);
    if (s)
      return s;
  }
  return DS4_STATUS_OK;
}
