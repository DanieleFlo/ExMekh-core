#ifndef DS4_QWEN_IMAGE_VAE_H
#define DS4_QWEN_IMAGE_VAE_H
#include "ds4_bundle.h"
#include "ds4_qwen_image_input.h"
#include "ds4_qwen_image_observer.h"
#include "ds4_qwen_image_rgba.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_vae ds4_qwen_image_vae;
/* Read-only plan snapshot; use the constructors below to choose a tile.
 * Decode rejects edited/stale byte estimates or decoder geometry. */
typedef struct {
  uint32_t latent_height, latent_width, core; /* core=0: full decode */
  uint32_t halo, overlap; /* latent pixels; exact local decoder support */
  uint64_t weights, device, host;
} ds4_qwen_image_vae_plan;
ds4_status ds4_qwen_image_vae_open(ds4_qwen_image_vae **, ds4_bundle *,
                                   ds4_numeric *, ds4_runtime *);
ds4_status ds4_qwen_image_vae_release_weights(ds4_qwen_image_vae *);
ds4_status ds4_qwen_image_vae_close(ds4_qwen_image_vae *);
/* Aligned RGB/RGBA uint8 reference; RGB supplies opaque alpha. No compositing,
 * resize, EXIF transform or posterior sampling is implicit. Output is packed
 * [H/16*W/16,64] normalized BF16, independent of the adapter's weights. */
ds4_status ds4_qwen_image_vae_encode(ds4_qwen_image_vae *,
                                     const ds4_qwen_image_reference *,
                                     const ds4_qwen_image_observer *,
                                     ds4_num_buffer **);
/* Budget includes resident VAE weights. The full low-resolution attention is
 * retained; high-resolution decoder tiles have a 14-latent halo and one-latent
 * linear overlap. No global-attention approximation is made by tiling. */
ds4_status ds4_qwen_image_vae_plan_decode(ds4_qwen_image_vae *,
                                          uint32_t latent_h, uint32_t latent_w,
                                          uint64_t device_budget,
                                          ds4_qwen_image_vae_plan *);
ds4_status ds4_qwen_image_vae_plan_tiled(ds4_qwen_image_vae *,
                                         uint32_t latent_h, uint32_t latent_w,
                                         uint32_t core,
                                         ds4_qwen_image_vae_plan *);
ds4_status ds4_qwen_image_vae_decode(ds4_qwen_image_vae *, ds4_num_view,
                                     const ds4_qwen_image_vae_plan *,
                                     const ds4_qwen_image_observer *,
                                     ds4_qwen_image_rgba **);
#ifdef __cplusplus
}
#endif
#endif
