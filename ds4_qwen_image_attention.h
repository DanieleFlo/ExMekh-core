#ifndef DS4_QWEN_IMAGE_ATTENTION_H
#define DS4_QWEN_IMAGE_ATTENTION_H
#include "ds4_numeric.h"
/* FP32 SDPA intermediates, no BF16 QK/probability rounding. Fixed query
 * tiles; supports the VAE's 1152-channel head as well as the DiT's 128. */
ds4_status ds4_qwen_image_attention(ds4_numeric *, ds4_num_view out,
                                    ds4_num_view q, ds4_num_view k,
                                    ds4_num_view v, uint32_t queries,
                                    uint32_t keys, uint32_t heads, uint32_t dim,
                                    const uint32_t *bounds);
#endif
