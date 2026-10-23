#ifndef DS4_GEMMA_VISION_H
#define DS4_GEMMA_VISION_H
#include "ds4_embedding_gemma2.h"
#include "ds4_gemma_media.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_gemma_vision ds4_gemma_vision;
ds4_status ds4_gemma_vision_weights(ds4_gemma_vision *, uint64_t limit,
                                    ds4_weight_budget *);
typedef struct ds4_gemma_visual_features ds4_gemma_visual_features;
typedef struct {
  const uint16_t *tokens; /* Detached BF16, [soft_tokens,512]. */
  const uint32_t *frame_tokens;
  uint32_t soft_tokens, frames;
} ds4_gemma_visual_features_view;
/* Private adapter boundary. No audio/text weights are bound or loaded here.
 * Numeric/bundle/runtime outlive the encoder; registry outlives host features.
 */
ds4_status ds4_gemma_vision_open(ds4_gemma_vision **, ds4_bundle *,
                                 ds4_numeric *, ds4_runtime *,
                                 const ds4_gemma_processor *);
ds4_status ds4_gemma_vision_close(ds4_gemma_vision *);
ds4_status ds4_gemma_vision_release_weights(ds4_gemma_vision *);
uint64_t ds4_gemma_vision_resident_weights(const ds4_gemma_vision *);
ds4_status ds4_gemma_vision_estimate(ds4_gemma_vision *,
                                     const ds4_gemma_media *,
                                     ds4_runtime_estimate *);
ds4_status ds4_gemma_vision_encode(ds4_gemma_vision *, const ds4_gemma_media *,
                                   const ds4_gemma_observer *,
                                   ds4_gemma_visual_features **);
ds4_gemma_visual_features_view
ds4_gemma_visual_features_data(const ds4_gemma_visual_features *);
void ds4_gemma_visual_features_free(ds4_gemma_visual_features *);
#ifdef __cplusplus
}
#endif
#endif
