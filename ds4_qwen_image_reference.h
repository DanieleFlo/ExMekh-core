#ifndef DS4_QWEN_IMAGE_REFERENCE_H
#define DS4_QWEN_IMAGE_REFERENCE_H
#include "ds4_qwen_image_input.h"
/* Private deterministic reference preprocessing. A zero resolution requires
 * already aligned explicit pixels. Positive resolution follows the pinned
 * Diffusers area/aspect recipe and Pillow Lanczos RGBA arithmetic. */
ds4_status qi_reference_plan(const ds4_qwen_image_reference *,
                             uint32_t resolution, uint32_t *width,
                             uint32_t *height);
ds4_status qi_reference_resize(const ds4_qwen_image_reference *, uint8_t *rgba,
                               uint32_t width, uint32_t height);
/* Temporary host bytes, excluding the caller-owned output. Same-size copies
 * need no workspace. Geometry must have passed qi_reference_plan first. */
uint64_t qi_reference_workspace(const ds4_qwen_image_reference *,
                                uint32_t width, uint32_t height);
#endif
