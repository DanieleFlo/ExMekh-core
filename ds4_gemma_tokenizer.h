#ifndef DS4_GEMMA_TOKENIZER_H
#define DS4_GEMMA_TOKENIZER_H
#include "ds4_embedding_gemma2.h"
/* Model-private compiled configuration. Each layer reads its own head shape,
 * locality and frozen FP32 RoPE frequencies. */
typedef struct {
  uint32_t dim, kv, local, heads;
  float frequency[256];
} gemma_layer;
typedef struct gemma_tokenizer gemma_tokenizer;
typedef struct {
  gemma_layer layers[24];
  uint32_t window;
  float eps;
} gemma_config;
ds4_status gemma_tokenizer_open(gemma_tokenizer **, const char *,
                                const uint8_t hashes[96]);
void gemma_tokenizer_close(gemma_tokenizer *);
uint64_t gemma_tokenizer_bytes(const gemma_tokenizer *);
const gemma_config *gemma_tokenizer_config(const gemma_tokenizer *);
ds4_status gemma_tokenize(const gemma_tokenizer *, const char *, uint32_t task,
                          uint32_t **tokens, uint32_t *count);
#endif
