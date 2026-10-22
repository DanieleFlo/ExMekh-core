#ifndef DS4_GEMMA_MEDIA_H
#define DS4_GEMMA_MEDIA_H
#include "ds4_bundle.h"
#include "ds4_media.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_gemma_processor ds4_gemma_processor;
typedef struct ds4_gemma_media ds4_gemma_media;
typedef struct {
  /* Zero chooses the packaged image/video budget. Supported explicit budgets
   * are 70/140/280/560/1120, exactly as in the frozen processor. */
  uint32_t soft_tokens;
} ds4_gemma_media_recipe;
typedef struct {
  ds4_media_type type;
  uint32_t frames, max_patches, soft_token_budget, soft_tokens, audio_frames,
      audio_soft_tokens;
  const float *patches;     /* [frames,max_patches,16*16*3], FP32 [0,1]. */
  const int32_t *positions; /* [frames,max_patches,2], x,y; padding=-1. */
  const uint32_t *frame_soft_tokens;
  const float *audio_features; /* [audio_frames,128], masked padding=0. */
  const uint8_t *audio_mask;
  const double *timestamps, *processor_timestamps;
  uint64_t host_bytes;
  uint8_t key[32];
} ds4_gemma_media_view;
typedef struct {
  uint64_t hits, misses, bytes, limit;
  uint32_t entries;
} ds4_gemma_cache_stats;
/* Host-only deterministic processing. Bundle/registry outlive the processor;
 * prepared media may outlive its processor but still borrows the registry.
 * Processor/cache calls and reference-count mutations are externally serialized.
 * The optional cache retains immutable handles, bounded by bytes and 64
 * entries. No encoder weights or GPU scopes are touched. */
ds4_status ds4_gemma_processor_open(ds4_gemma_processor **, ds4_bundle *,
                                    ds4_resources *, uint64_t owner,
                                    const char *compiled_asset,
                                    uint64_t cache_bytes);
void ds4_gemma_processor_close(ds4_gemma_processor *);
void ds4_gemma_processor_clear(ds4_gemma_processor *);
ds4_gemma_cache_stats ds4_gemma_processor_cache(const ds4_gemma_processor *);
ds4_media_video_recipe
ds4_gemma_processor_video_recipe(const ds4_gemma_processor *);
ds4_status ds4_gemma_media_plan(ds4_gemma_processor *, const ds4_media *,
                                const ds4_gemma_media_recipe *,
                                ds4_runtime_estimate *, uint32_t *soft_tokens);
ds4_status ds4_gemma_media_prepare(ds4_gemma_processor *, const ds4_media *,
                                   const ds4_gemma_media_recipe *,
                                   ds4_gemma_media **);
ds4_status ds4_gemma_media_retain(ds4_gemma_media *);
void ds4_gemma_media_free(ds4_gemma_media *);
ds4_gemma_media_view ds4_gemma_media_data(const ds4_gemma_media *);
/* Frozen vision RoPE frequencies, copied without exposing processor storage. */
ds4_status ds4_gemma_processor_vision_rope(const ds4_gemma_processor *,
                                           float frequencies[16]);
#ifdef __cplusplus
}
#endif
#endif
