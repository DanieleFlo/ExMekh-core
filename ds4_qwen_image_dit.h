#ifndef DS4_QWEN_IMAGE_DIT_H
#define DS4_QWEN_IMAGE_DIT_H
#include "ds4_qwen_image_encoder.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_dit ds4_qwen_image_dit;
typedef struct ds4_qwen_image_dit_input ds4_qwen_image_dit_input;
typedef struct {
  uint32_t height, width;
} ds4_qwen_image_latent_shape;
typedef struct {
  uint64_t weights, prefix_cache, first_step, cached_step, host;
  uint32_t groups[33];       /* common projections, then block 0..31 */
  uint32_t dependencies[33]; /* UINT32_MAX for common, then previous block */
  uint64_t group_bytes[33];
} ds4_qwen_image_dit_memory;
/* Bundle/context/runtime outlive the adapter; prepared input owns copies of
 * conditioning and reference latents. Shapes are in unpatched latent pixels,
 * reference images first, target last; both axes must be even. The adapter
 * outlives its prepared inputs; close returns BUSY while one is alive. */
ds4_status ds4_qwen_image_dit_open(ds4_qwen_image_dit **, ds4_bundle *,
                                   ds4_numeric *, ds4_runtime *);
ds4_status ds4_qwen_image_dit_close(ds4_qwen_image_dit *);
ds4_status ds4_qwen_image_dit_release_weights(ds4_qwen_image_dit *);
ds4_status ds4_qwen_image_dit_prepare(ds4_qwen_image_dit *,
                                      ds4_qwen_image_conditioning_view,
                                      ds4_num_view references,
                                      const ds4_qwen_image_latent_shape *,
                                      size_t shapes,
                                      ds4_qwen_image_dit_input **);
ds4_status ds4_qwen_image_dit_input_free(ds4_qwen_image_dit_input *);
ds4_status ds4_qwen_image_dit_estimate(ds4_qwen_image_dit *,
                                       const ds4_qwen_image_dit_input *,
                                       ds4_qwen_image_dit_memory *);
/* Clear owns only detached prefix K/V. The next forward recomputes the full
 * prefix. Useful for an exact uncached reference as well as request rollback.
 */
ds4_status ds4_qwen_image_dit_cache_clear(ds4_qwen_image_dit_input *);
/* Residency only, selected before execution. HOST keeps exact detached BF16
 * K/V in bounded registered RAM and copies one layer synchronously on reuse.
 * No recomputation, precision change, asynchronous prefetch or weight offload.
 */
ds4_status ds4_qwen_image_dit_cache_place(ds4_qwen_image_dit_input *,
                                          ds4_memory_location);
ds4_status ds4_qwen_image_dit_forward(ds4_qwen_image_dit *,
                                      ds4_qwen_image_dit_input *,
                                      ds4_num_view target, float sigma,
                                      const ds4_qwen_image_observer *,
                                      ds4_num_buffer **velocity);
/* Eight official Turbo Euler updates, CFG=1, terminal sigma=0. Noise and
 * returned packed target are FP32 [height*width,64]; initial noise and each
 * update round to BF16 following the official scheduler. NULL noise uses the
 * version-1 SplitMix64/Box-Muller RNG; no LLM/global RNG is touched. */
ds4_status ds4_qwen_image_dit_generate(ds4_qwen_image_dit *,
                                       ds4_qwen_image_dit_input *,
                                       const float *noise, uint64_t seed,
                                       const ds4_qwen_image_observer *,
                                       ds4_num_buffer **);
void ds4_qwen_image_turbo_sigmas(float sigmas[9]);
ds4_status ds4_qwen_image_noise(float *, size_t count, uint64_t seed);
#ifdef __cplusplus
}
#endif
#endif
