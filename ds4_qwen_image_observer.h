#ifndef DS4_QWEN_IMAGE_OBSERVER_H
#define DS4_QWEN_IMAGE_OBSERVER_H
#include "ds4_numeric.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  void *opaque;
  bool (*cancelled)(void *);
  ds4_status (*trace)(void *, const char *stage, uint32_t layer, ds4_num_view);
} ds4_qwen_image_observer;
/* Trace receives a fenced borrowed view. A cancellation requested by trace
 * is observed before the adapter can publish its result. */
ds4_status ds4_qwen_image_observe(ds4_numeric *,
                                  const ds4_qwen_image_observer *,
                                  const char *stage, uint32_t layer,
                                  ds4_num_view);
#ifdef __cplusplus
}
#endif
#endif
