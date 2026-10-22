#ifndef DS4_GEMMA_VISUAL_H
#define DS4_GEMMA_VISUAL_H
#include "ds4_embedding_gemma2.h"
#include "ds4_gemma_media.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_gemma_visual ds4_gemma_visual;
typedef struct ds4_gemma_visual_input ds4_gemma_visual_input;
typedef struct {
  /* NULL: one image/video marker per supplied media item. */
  const char *prompt;
  uint32_t task;
  /* Immutable handles, original per-item order. */
  ds4_gemma_media *const *media;
  size_t media_count;
  bool add_timestamps; /* Explicit mm:ss prefixes from processor timestamps. */
} ds4_gemma_visual_request;
/* Bundle, numeric context and runtime outlive this adapter. The processor is
 * borrowed only during open. Calls on a handle are externally serialized. Only
 * text and vision branches are bound. No audio tower or Python serving.
 * Prepared requests retain processor outputs; close is BUSY while they live. */
ds4_status ds4_gemma_visual_open(ds4_gemma_visual **, ds4_bundle *,
                                 ds4_numeric *, ds4_runtime *,
                                 const char *text_asset,
                                 const ds4_gemma_processor *);
ds4_status ds4_gemma_visual_close(ds4_gemma_visual *);
ds4_status ds4_gemma_visual_release_vision(ds4_gemma_visual *);
ds4_status ds4_gemma_visual_release_text(ds4_gemma_visual *);
/* Host-only preparation validates media, placeholders and the 8192-token
 * operational context without loading any model weights. */
ds4_status ds4_gemma_visual_prepare(ds4_gemma_visual *,
                                    const ds4_gemma_visual_request *,
                                    size_t count, ds4_gemma_visual_input **);
ds4_status ds4_gemma_visual_input_free(ds4_gemma_visual_input *);
ds4_status ds4_gemma_visual_input_tokens(const ds4_gemma_visual_input *, size_t,
                                         const uint32_t **, uint32_t *);
ds4_status ds4_gemma_visual_estimate(ds4_gemma_visual *,
                                     const ds4_gemma_visual_input *,
                                     ds4_runtime_estimate *);
/* Serial media groups with eager frame batching, then one isolated backbone per
 * ragged item. Detached features are request temporaries; no KV survives the
 * operation. */
ds4_status ds4_gemma_visual_embed(ds4_gemma_visual *, ds4_gemma_visual_input *,
                                  uint32_t dimensions,
                                  const ds4_gemma_observer *,
                                  ds4_gemma_result **);
#ifdef __cplusplus
}
#endif
#endif
