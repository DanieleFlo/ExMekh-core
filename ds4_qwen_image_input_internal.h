#ifndef DS4_QWEN_IMAGE_INPUT_INTERNAL_H
#define DS4_QWEN_IMAGE_INPUT_INTERNAL_H
#include "ds4_numeric.h"
#include "ds4_qwen_image_input.h"
#include "ds4_qwen_image_tokenizer.h"
/* Only the adapter's construction boundary needs the tokenizer. */
ds4_status ds4_qwen_image_input_create(ds4_numeric *,
                                       const ds4_qwen_image_tokenizer *,
                                       const char *prompt,
                                       const ds4_qwen_image_reference *,
                                       size_t references, uint32_t pad_to,
                                       ds4_qwen_image_input **);
#endif
