#ifndef DS4_EMBEDDING_GEMMA2_H
#define DS4_EMBEDDING_GEMMA2_H
#include "ds4_bundle.h"
#include "ds4_numeric.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_embedding_gemma2 ds4_embedding_gemma2;
typedef struct ds4_gemma_input ds4_gemma_input;
typedef struct ds4_gemma_result ds4_gemma_result;
enum {
  DS4_GEMMA_RAW,
  DS4_GEMMA_QUERY,
  DS4_GEMMA_DOCUMENT,
  DS4_GEMMA_CODE,
  DS4_GEMMA_SIMILARITY
};
typedef struct {
  const char *text;
  uint32_t task;
} ds4_gemma_text;
typedef struct {
  void *opaque;
  bool (*cancelled)(void *);
  ds4_status (*trace)(void *, const char *, uint32_t item, uint32_t layer,
                      ds4_num_view);
} ds4_gemma_observer;
/* Immutable bundle, numeric context and runtime outlive this text adapter.
 * compiled_asset comes from compile_embedding_gemma2.py. BF16 operations with
 * FP32 accumulation; no FP16 fallback and no persistent KV between requests.
 * The resource registry outlives prepared inputs and host results, which can
 * be freed after adapter close. */
ds4_status ds4_embedding_gemma2_open(ds4_embedding_gemma2 **, ds4_bundle *,
                                     ds4_numeric *, ds4_runtime *,
                                     const char *compiled_asset);
ds4_status ds4_embedding_gemma2_close(ds4_embedding_gemma2 *);
ds4_status ds4_embedding_gemma2_release_weights(ds4_embedding_gemma2 *);
/* Read-only owned weight storage, without exposing groups or tensor layout. */
uint64_t ds4_embedding_gemma2_resident_weights(const ds4_embedding_gemma2 *);
ds4_status ds4_gemma_prepare(ds4_embedding_gemma2 *, const ds4_gemma_text *,
                             size_t count, ds4_gemma_input **);
/* Complete tokenizer sequences, without implicit BOS/EOS or task prefixes.
 * PAD is excluded; media control tokens require the typed media interface. */
ds4_status ds4_gemma_prepare_tokens(ds4_embedding_gemma2 *,
                                    const uint32_t *const *, const uint32_t *,
                                    size_t count, ds4_gemma_input **);
void ds4_gemma_input_free(ds4_gemma_input *);
/* Borrowed BOS/content/EOS ids, including task prefix. No padding is stored;
 * execution isolates ragged items, equivalent to right-padded attention. */
ds4_status ds4_gemma_input_tokens(const ds4_gemma_input *, size_t item,
                                  const uint32_t **, uint32_t *count);
ds4_status ds4_gemma_estimate(ds4_embedding_gemma2 *, const ds4_gemma_input *,
                              ds4_runtime_estimate *);
ds4_status ds4_gemma_embed(ds4_embedding_gemma2 *, const ds4_gemma_input *,
                           uint32_t dimensions, const ds4_gemma_observer *,
                           ds4_gemma_result **);
typedef struct {
  const float *vectors;
  uint32_t count, dimensions;
} ds4_gemma_result_view;
ds4_gemma_result_view ds4_gemma_result_data(const ds4_gemma_result *);
void ds4_gemma_result_free(ds4_gemma_result *);
#ifdef __cplusplus
}
#endif
#endif
