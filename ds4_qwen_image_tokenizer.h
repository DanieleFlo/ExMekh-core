#ifndef DS4_QWEN_IMAGE_TOKENIZER_H
#define DS4_QWEN_IMAGE_TOKENIZER_H
#include "ds4_runtime.h"
typedef struct ds4_qwen_image_tokenizer ds4_qwen_image_tokenizer;
ds4_status ds4_qwen_image_tokenizer_open(ds4_qwen_image_tokenizer **,
                                         const char *,
                                         const uint8_t source_sha[32]);
void ds4_qwen_image_tokenizer_close(ds4_qwen_image_tokenizer *);
uint64_t ds4_qwen_image_tokenizer_bytes(const ds4_qwen_image_tokenizer *);
ds4_status ds4_qwen_image_tokenize(const ds4_qwen_image_tokenizer *,
                                   const char *, uint32_t **, size_t *);
ds4_status ds4_qwen_image_prompt(const ds4_qwen_image_tokenizer *, const char *,
                                 const uint32_t *image_tokens, size_t images,
                                 uint32_t **tokens, size_t *count,
                                 size_t *drop);
#endif
