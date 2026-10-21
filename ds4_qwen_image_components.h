#ifndef DS4_QWEN_IMAGE_COMPONENTS_H
#define DS4_QWEN_IMAGE_COMPONENTS_H
/* Private component API. Memory queries never allocate or enter model scopes. */
#include "ds4_qwen_image_dit.h"
#include "ds4_qwen_image_vae.h"
void qi_encoder_memory(ds4_qwen_image_encoder *, bool vision, uint64_t *weights,
                       uint64_t *resident);
void qi_dit_memory(ds4_qwen_image_dit *, uint32_t total, uint32_t target,
                   uint32_t prefix, uint32_t query_span,
                   ds4_qwen_image_dit_memory *, uint64_t *resident);
void qi_vae_memory(ds4_qwen_image_vae *, uint64_t *weights, uint64_t *resident);
/* Validated reference geometry; transient memory excludes resident weights. */
void qi_vae_encode_memory(const ds4_qwen_image_reference *,
                          ds4_runtime_estimate *);
ds4_status qi_encoder_load(ds4_qwen_image_encoder *, bool vision);
ds4_status qi_dit_load(ds4_qwen_image_dit *);
ds4_status qi_vae_load(ds4_qwen_image_vae *);
#endif
