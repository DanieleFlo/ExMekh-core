#ifndef DS4_QWEN_IMAGE_RGBA_INTERNAL_H
#define DS4_QWEN_IMAGE_RGBA_INTERNAL_H
#include "ds4_qwen_image_rgba.h"
#include "ds4_resources.h"
/* Construction boundary: only the decoder receives writable pixel storage.
 * The result owns it immediately; publish only after a successful decode. */
uint64_t ds4_qwen_image_rgba_bytes(uint32_t width, uint32_t height);
ds4_status ds4_qwen_image_rgba_create(ds4_resources *, uint64_t owner,
                                      uint32_t width, uint32_t height,
                                      ds4_qwen_image_rgba **, float **pixels);
#endif
