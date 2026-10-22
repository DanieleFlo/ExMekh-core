#ifndef DS4_GEMMA_AUDIO_H
#define DS4_GEMMA_AUDIO_H
#include "ds4_embedding_gemma2.h"
#include "ds4_gemma_media.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_gemma_audio ds4_gemma_audio;
typedef struct ds4_gemma_audio_features ds4_gemma_audio_features;
typedef struct {
  const uint16_t *tokens; /* Detached valid BF16 rows, width 512. */
  uint32_t soft_tokens;
} ds4_gemma_audio_features_view;
/* Borrows bundle/numeric/runtime. No weights load until encode. Calls and
 * reference mutations are serialized by the caller, as for the vision tower. */
ds4_status ds4_gemma_audio_open(ds4_gemma_audio **, ds4_bundle *, ds4_numeric *,
                                ds4_runtime *);
ds4_status ds4_gemma_audio_close(ds4_gemma_audio *);
ds4_status ds4_gemma_audio_release_weights(ds4_gemma_audio *);
uint64_t ds4_gemma_audio_resident_weights(const ds4_gemma_audio *);
ds4_status ds4_gemma_audio_estimate(ds4_gemma_audio *, const ds4_gemma_media *,
                                    ds4_runtime_estimate *);
ds4_status ds4_gemma_audio_encode(ds4_gemma_audio *, const ds4_gemma_media *,
                                  const ds4_gemma_observer *,
                                  ds4_gemma_audio_features **);
ds4_gemma_audio_features_view
ds4_gemma_audio_features_data(const ds4_gemma_audio_features *);
void ds4_gemma_audio_features_free(ds4_gemma_audio_features *);
#ifdef __cplusplus
}
#endif
#endif
