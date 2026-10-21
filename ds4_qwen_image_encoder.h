#ifndef DS4_QWEN_IMAGE_ENCODER_H
#define DS4_QWEN_IMAGE_ENCODER_H
#include "ds4_bundle.h"
#include "ds4_numeric.h"
#include "ds4_qwen_image_input.h"
#include "ds4_qwen_image_observer.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_encoder ds4_qwen_image_encoder;
typedef struct ds4_qwen_image_conditioning ds4_qwen_image_conditioning;
typedef struct {
  ds4_num_view hidden; /* borrowed [count,4096] BF16, pre-final RMSNorm */
  const uint8_t *image_mask;
  uint32_t count;
} ds4_qwen_image_conditioning_view;
ds4_qwen_image_conditioning_view
ds4_qwen_image_conditioning_data(const ds4_qwen_image_conditioning *);

/* Bundle, numeric context and runtime outlive the adapter. No GGUF LLM
 * engine is required. Open checks the complete 750-tensor architecture. */
ds4_status ds4_qwen_image_encoder_open(ds4_qwen_image_encoder **, ds4_bundle *,
                                       ds4_numeric *, ds4_runtime *,
                                       const char *compiled_tokenizer);
ds4_status ds4_qwen_image_encoder_close(ds4_qwen_image_encoder *);
ds4_status ds4_qwen_image_encoder_release_weights(ds4_qwen_image_encoder *);
ds4_status ds4_qwen_image_prepare(ds4_qwen_image_encoder *, const char *prompt,
                                  const ds4_qwen_image_reference *,
                                  size_t references, ds4_qwen_image_input **);
/* Explicit left padding for a single forward. Zero means the natural length;
 * shorter targets never truncate. Outputs contain only valid conditioning. */
ds4_status ds4_qwen_image_prepare_padded(ds4_qwen_image_encoder *, const char *,
                                         const ds4_qwen_image_reference *,
                                         size_t references, uint32_t pad_to,
                                         ds4_qwen_image_input **);
ds4_status ds4_qwen_image_encode(ds4_qwen_image_encoder *,
                                 const ds4_qwen_image_input *,
                                 const ds4_qwen_image_observer *,
                                 ds4_qwen_image_conditioning **);
ds4_status ds4_qwen_image_conditioning_free(ds4_qwen_image_conditioning *);
/* Conservative resident-weight/device activation estimate, without eviction.
 * The caller may reserve it through the common executor before execution. */
ds4_status ds4_qwen_image_encoder_estimate(ds4_qwen_image_encoder *,
                                           const ds4_qwen_image_input *,
                                           ds4_runtime_estimate *);
#ifdef __cplusplus
}
#endif
#endif
