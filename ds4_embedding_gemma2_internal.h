#ifndef DS4_EMBEDDING_GEMMA2_INTERNAL_H
#define DS4_EMBEDDING_GEMMA2_INTERNAL_H
/* Private bridge: copy projected visual tokens into a prepared backbone input.
 * The text adapter owns token/embedding layout and validates every placeholder.
 * Callers supply ordered features, never access the input representation. */
#include "ds4_embedding_gemma2.h"
typedef struct {
  const uint16_t *images, *videos;
  uint32_t image_tokens, video_tokens;
} gemma_visual_tokens;
ds4_status gemma_input_visual(ds4_gemma_input *, size_t item,
                              gemma_visual_tokens);
ds4_status gemma_input_visual_validate(const ds4_gemma_input *, size_t item,
                                       uint32_t image_tokens,
                                       uint32_t video_tokens);
ds4_status gemma_input_visual_clear(ds4_gemma_input *);
#endif
