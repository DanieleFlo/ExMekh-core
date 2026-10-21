#ifndef DS4_QWEN_IMAGE_INPUT_H
#define DS4_QWEN_IMAGE_INPUT_H
#include "ds4_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_input ds4_qwen_image_input;
typedef struct {
  const uint8_t *pixels;
  uint32_t width, height, channels; /* immutable RGB or straight RGBA */
} ds4_qwen_image_reference;
/* Borrowed read-only data, valid until input_free. Ownership stays in input. */
typedef struct {
  const uint32_t *tokens, *positions; /* positions: [count,3] T,H,W */
  const uint8_t *image_mask;
  uint32_t count, drop, images, patches, left_padding;
  const uint32_t *grids; /* [images,2] H,W, before spatial merge */
  const float *pixels;   /* [patches,3,2,16,16] */
  const char *prompt;
} ds4_qwen_image_input_view;
ds4_qwen_image_input_view
ds4_qwen_image_input_data(const ds4_qwen_image_input *);
void ds4_qwen_image_input_free(ds4_qwen_image_input *);
#ifdef __cplusplus
}
#endif
#endif
