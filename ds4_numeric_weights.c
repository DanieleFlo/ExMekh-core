#include "ds4_numeric_weights.h"
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
  return ds4_num_buffer_write(storage, off, src, n);
}
static ds4_status weight_free(void *p, void *b) {
  (void)p;
  return ds4_num_free(b);
}
static ds4_status weight_fence(void *p) { return ds4_numeric_fence(p); }
ds4_weight_backend ds4_numeric_weight_backend(ds4_numeric *c) {
  return (ds4_weight_backend){c, weight_alloc, weight_write, weight_fence,
                              weight_free};
}
ds4_num_view ds4_numeric_weight_view(const void *storage, uint64_t off,
                                     uint64_t n, uint32_t type) {
  return ds4_num_reinterpret(ds4_num_data(storage), off, n, type);
}
