#ifndef DS4_QWEN_IMAGE_RGBA_H
#define DS4_QWEN_IMAGE_RGBA_H
#include "ds4_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_rgba ds4_qwen_image_rgba;
typedef struct {
  uint32_t width, height;
  const float *pixels; /* borrowed top-to-bottom straight RGBA, [0,1] */
} ds4_qwen_image_rgba_view;
/* Host results outlive adapters/numeric contexts; their resource registry
 * must remain alive until free. PNG never retains or accesses a GPU model. */
ds4_qwen_image_rgba_view ds4_qwen_image_rgba_data(const ds4_qwen_image_rgba *);
void ds4_qwen_image_rgba_free(ds4_qwen_image_rgba *);
ds4_status ds4_qwen_image_rgba_save_png(const ds4_qwen_image_rgba *,
                                        const char *);
#ifdef __cplusplus
}
#endif
#endif
