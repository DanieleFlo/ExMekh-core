#include "ds4_numeric_weights.h"
ds4_status ds4_weight_cache_configure(ds4_weight_cache *c, ds4_runtime_model *m,
                                      uint64_t limit, ds4_weight_budget *out) {
  if (!c || (!limit && !out))
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = DS4_STATUS_OK;
  if (limit) {
    s = ds4_runtime_model_enter_exclusive(m, &scope);
    if (!s)
      s = ds4_weight_cache_limit(c, limit);
  }
  if (out) {
    ds4_status query = ds4_weight_cache_budget(c, out);
    if (!s)
      s = query;
  }
  ds4_runtime_scope_leave(&scope);
  return s;
}
static ds4_status weight_alloc(void *p, uint64_t n, void **out) {
  if (n == UINT64_MAX)
    return DS4_STATUS_INVALID;
  ds4_num_buffer *b = NULL;
  ds4_status s = ds4_num_alloc(p, (n + 1) / 2, DS4_NUM_BF16, &b);
  /* Round storage to BF16 words; byte writes and borrowed tensor views go
   * through the numeric owner's API, without interpreting its representation.
   */
  if (!s)
    *out = b;
  return s;
}
static ds4_status weight_write(void *p, void *storage, uint64_t off,
                               const void *src, uint64_t n) {
  (void)p;
  return ds4_num_buffer_stage(storage, off, src, n);
}
static ds4_status weight_free(void *p, void *b) {
  (void)p;
  return ds4_num_free(b);
}
static ds4_status weight_fence(void *p) { return ds4_numeric_fence(p); }
static ds4_status weight_ready(void *p) { return ds4_numeric_weights_ready(p); }
ds4_weight_backend ds4_numeric_weight_backend(ds4_numeric *c) {
  return (ds4_weight_backend){c, weight_alloc, weight_write, weight_fence,
                              weight_free};
}
ds4_status ds4_numeric_weight_group_selected(ds4_weight_group **out,
                                             ds4_bundle *b, uint32_t component,
                                             const size_t *ids, size_t count,
                                             ds4_numeric *n) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!n)
    return DS4_STATUS_INVALID;
  ds4_weight_backend backend = ds4_numeric_weight_backend(n);
  ds4_status s =
      ds4_weight_group_create_selected(out, b, component, ids, count, &backend);
  if (!s)
    s = ds4_weight_group_availability(*out, weight_ready);
  return s;
}
ds4_num_view ds4_numeric_weight_view(const void *storage, uint64_t off,
                                     uint64_t n, uint32_t type) {
  return ds4_num_reinterpret(ds4_num_data(storage), off, n, type);
}
