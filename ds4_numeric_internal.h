#ifndef DS4_NUMERIC_INTERNAL_H
#define DS4_NUMERIC_INTERNAL_H
#include "ds4_numeric.h"
typedef struct {
  ds4_status (*allocate)(void *, uint64_t, void **);
  ds4_status (*release)(void *, void *);
  ds4_status (*copy)(void *, void *, const void *, uint64_t, int);
  ds4_status (*fence)(void *);
  void (*close)(void *);
  ds4_status (*linear)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                       ds4_num_view, uint32_t, uint32_t, uint32_t);
  ds4_status (*norm)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                     ds4_num_view, uint32_t, float, uint32_t, uint32_t, bool);
  ds4_status (*point)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                      uint32_t, float);
  ds4_status (*rope)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                     ds4_num_view, uint32_t, uint32_t, bool, uint32_t);
  ds4_status (*attend)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                       ds4_num_view, const uint32_t *,
                       const ds4_num_attention *);
  ds4_status (*conv)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                     ds4_num_view, const ds4_num_convolution *);
  ds4_status (*reorder)(void *, ds4_num_view, ds4_num_view,
                        const ds4_num_reindex *);
  ds4_status (*pool)(void *, ds4_num_view, ds4_num_view, const uint32_t *,
                     uint32_t, uint32_t, uint32_t, float);
  ds4_status (*nearest)(void *, ds4_num_view, ds4_num_view,
                        const ds4_num_resample *);
  ds4_status (*softmax)(void *, ds4_num_view, ds4_num_view, const uint32_t *,
                        uint32_t, uint32_t, float);
  ds4_status (*matmul_batch)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                             uint32_t, uint32_t, uint32_t, uint32_t, bool,
                             bool);
  ds4_status (*channel_l2)(void *, ds4_num_view, ds4_num_view, uint32_t,
                           uint32_t, float);
  ds4_status (*channel_rms)(void *, ds4_num_view, ds4_num_view, ds4_num_view,
                            uint32_t, uint32_t, float);
  ds4_status (*weight_copy)(void *, void *, const void *, uint64_t);
  ds4_status (*weight_ready)(void *);
  /* Fallible shutdown retains a usable backend on failure. close() runs only
   * after this succeeds and the context metadata event has completed. */
  ds4_status (*prepare_close)(void *);
} ds4_numeric_ops;
struct ds4_numeric {
  const ds4_numeric_ops *ops;
  void *state;
  ds4_resources *resources;
  uint64_t owner;
  size_t buffers;
  bool device;
  uint64_t live_bytes, peak_bytes;
  uint64_t h2d_bytes, d2h_bytes;
  uint64_t backend_bytes;
  ds4_resource *metadata;
  ds4_num_buffer *deferred;
};
#ifdef __cplusplus
extern "C" {
#endif
ds4_status ds4_numeric_create(ds4_numeric **, const ds4_numeric_ops *, void *,
                              ds4_resources *, uint64_t, bool);
uint32_t ds4_num_type_bytes(uint32_t);
ds4_status ds4_numeric_cudnn_create(void **, void *stream, ds4_resources *,
                                    uint64_t owner);
void ds4_numeric_cudnn_close(void *);
ds4_status ds4_numeric_cudnn_conv(void *, ds4_numeric *, ds4_num_view,
                                  ds4_num_view, ds4_num_view, ds4_num_view,
                                  const ds4_num_convolution *);
#ifdef __cplusplus
}
#endif
#endif
