#ifndef DS4_GEMMA_MULTIMODAL_H
#define DS4_GEMMA_MULTIMODAL_H
#include "ds4_gemma_visual.h"
#ifdef __cplusplus
extern "C" {
#endif
/* The former visual adapter now owns all three branches. These aliases keep
 * one implementation and preserve existing source/binary callers. */
typedef ds4_gemma_visual ds4_gemma_multimodal;
typedef ds4_gemma_visual_input ds4_gemma_multimodal_input;
#define ds4_gemma_multimodal_open ds4_gemma_visual_open
#define ds4_gemma_multimodal_close ds4_gemma_visual_close
#define ds4_gemma_multimodal_input_free ds4_gemma_visual_input_free
#define ds4_gemma_multimodal_input_tokens ds4_gemma_visual_input_tokens
#define ds4_gemma_multimodal_estimate ds4_gemma_visual_estimate
#define ds4_gemma_multimodal_plan ds4_gemma_visual_plan
#define ds4_gemma_multimodal_embed ds4_gemma_visual_embed
#define ds4_gemma_multimodal_release_vision ds4_gemma_visual_release_vision
#define ds4_gemma_multimodal_release_audio ds4_gemma_visual_release_audio
#define ds4_gemma_multimodal_release_text ds4_gemma_visual_release_text
#define ds4_gemma_multimodal_resident_weights ds4_gemma_visual_resident_weights
typedef struct {
  /* Exactly one field is non-NULL. Text is literal content; reserved model
   * markers are rejected. The adapter inserts all media separators itself. */
  const char *text;
  ds4_gemma_media *media;
} ds4_gemma_content;
typedef struct {
  /* Optional explicit source segment. Each request still yields one vector;
   * callers decode each segment with ds4_media_decode before preparation. */
  uint64_t id;
  double start_seconds, end_seconds;
} ds4_gemma_segment;
typedef struct {
  const ds4_gemma_content *content;
  size_t count;
  uint32_t task;
  bool add_timestamps;
  ds4_gemma_segment segment;
} ds4_gemma_multimodal_request;
/* C composition contract v2: ordered typed parts, heterogeneous batches and
 * explicit segments. 8192 operational tokens per output, overflow is an error.
 * Segments never average together. HTTP v1 remains overflow=error (T17). */
ds4_status ds4_gemma_multimodal_prepare(ds4_gemma_multimodal *,
                                        const ds4_gemma_multimodal_request *,
                                        size_t, ds4_gemma_multimodal_input **);
ds4_status ds4_gemma_multimodal_prepare_tokens(ds4_gemma_multimodal *,
    const uint32_t *const *, const uint32_t *, size_t,
    ds4_gemma_multimodal_input **);
ds4_status ds4_gemma_multimodal_segment(const ds4_gemma_multimodal_input *,
                                        size_t, ds4_gemma_segment *);
/* Stable SHA-256 identity includes revision/profile/composition version,
 * ordered tokens, media preprocessing keys, task, dimension, L2 pooling recipe
 * and explicit segment. Same key denotes the same embedding space and input. */
ds4_status ds4_gemma_multimodal_key(const ds4_gemma_multimodal_input *, size_t,
                                    uint32_t dimensions, uint8_t key[32]);
#ifdef __cplusplus
}
#endif
#endif
