#ifndef DS4_NUMERIC_H
#define DS4_NUMERIC_H
#include "ds4_resources.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Suite kernels use contiguous row-major views, never the legacy GGUF offset
 * cache. A view borrows storage while its owner lives. Keep the owner until
 * queued work completes; a fence orders work without invalidating the view.
 * BF16 is storage AND operation output precision, with FP32 accumulation. */
enum { DS4_NUM_F32 = 0, DS4_NUM_BF16 = 30 };
typedef struct {
  void *data;
  uint64_t count;
  uint32_t type;
} ds4_num_view;
typedef struct ds4_numeric ds4_numeric;
typedef struct ds4_num_buffer ds4_num_buffer;
ds4_status ds4_numeric_cpu(ds4_numeric **, ds4_resources *, uint64_t owner);
/* CUDA context owns a stream and a cuBLAS handle, never the device/runtime.
 * Use under the caller's serial runtime lease; destruction never resets CUDA.
 */
ds4_status ds4_numeric_cuda(ds4_numeric **, ds4_resources *, uint64_t owner);
ds4_status ds4_numeric_close(ds4_numeric *);
ds4_status ds4_numeric_fence(ds4_numeric *);
/* Owned tensor storage only; driver/cuBLAS allocations remain in the device
 * sample. Peak includes weights and transient kernel workspace in this context.
 */
void ds4_numeric_memory(const ds4_numeric *, uint64_t *live, uint64_t *peak);
ds4_resources *ds4_numeric_resources(ds4_numeric *, uint64_t *owner);
ds4_status ds4_num_alloc(ds4_numeric *, uint64_t count, uint32_t type,
                         ds4_num_buffer **);
ds4_status ds4_num_free(ds4_num_buffer *);
/* Transfer ownership even when an event, fence or backend free blocks release.
 * The context retains the
 * allocation until recover/close succeeds; callers must discard the handle. */
ds4_status ds4_num_dispose(ds4_num_buffer *);
ds4_status ds4_numeric_recover(ds4_numeric *);
ds4_num_view ds4_num_data(const ds4_num_buffer *);
ds4_num_view ds4_num_slice(ds4_num_view, uint64_t offset, uint64_t count);
/* Reinterpret storage bytes without converting values. Offset and address
 * must satisfy the target type's alignment; the returned view is borrowed. */
ds4_num_view ds4_num_reinterpret(ds4_num_view, uint64_t byte_offset,
                                 uint64_t count, uint32_t type);
/* Storage owns its context; writers need not know the buffer representation. */
ds4_status ds4_num_buffer_write(ds4_num_buffer *, uint64_t byte_offset,
                                const void *, uint64_t bytes);
ds4_status ds4_num_write(ds4_numeric *, ds4_num_view, const void *,
                         uint64_t bytes);
ds4_status ds4_num_read(ds4_numeric *, ds4_num_view, void *, uint64_t bytes);
ds4_status ds4_num_copy(ds4_numeric *, ds4_num_view, ds4_num_view);
/* Operations write only the declared result prefix of y and preserve unused
 * capacity. Partial overlap is rejected; exact in-place is supported only by
 * norm, pointwise operations and softmax. CAST is the explicit dtype change.
 * y[rows,out] = x[rows,in] @ w[out,in]^T + bias[out]. No TF32,
 * FP16, reduced-precision reduction or fast-math dispatch is implicit. */
ds4_status ds4_num_linear(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                          ds4_num_view w, ds4_num_view bias, uint32_t rows,
                          uint32_t in, uint32_t out);
enum { DS4_NUM_RMS, DS4_NUM_LAYER, DS4_NUM_L2 };
enum { DS4_NUM_SCALE_NONE, DS4_NUM_SCALE_DIRECT, DS4_NUM_SCALE_ZERO_CENTERED };
ds4_status ds4_num_norm(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                        ds4_num_view scale, ds4_num_view bias, uint32_t width,
                        float eps, uint32_t kind, uint32_t scale_mode,
                        bool round_before_scale);
enum {
  DS4_NUM_ADD,
  DS4_NUM_MUL,
  DS4_NUM_SILU,
  DS4_NUM_GELU,
  DS4_NUM_GELU_TANH,
  DS4_NUM_SIGMOID,
  DS4_NUM_SCALE,
  DS4_NUM_CAST,
  DS4_NUM_RELU,
  DS4_NUM_SOFTPLUS,
  DS4_NUM_TANH,
  DS4_NUM_MIN,
  DS4_NUM_MAX,
  DS4_NUM_SIN,
  DS4_NUM_COS
};
ds4_status ds4_num_point(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                         ds4_num_view b, uint32_t op, float scalar);
/* cos/sin contain [tokens,dim/2]. split-half (Qwen/Gemma) and
 * adjacent-pair (DiT) rotations are distinct. */
ds4_status ds4_num_rope(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                        ds4_num_view cos, ds4_num_view sin, uint32_t heads,
                        uint32_t dim, bool adjacent, bool round_products);
/* Bounds are inclusive/exclusive key ranges per batch/query, so padding,
 * causal, block-causal, local/full and segmented vision need no score mask.
 * q/k/v layout: [batch,tokens,heads,dim]; GQA maps contiguous query groups.
 * Empty rows produce zero. FP32 uses online softmax with O(dim) per query;
 * BF16 preserves rounded scores/probabilities in 256-query tiles and packed
 * K/V. Scratch grows linearly with key length, never quadratically in N. */
typedef struct {
  uint32_t batch, queries, keys, q_heads, kv_heads, dim;
  float scale, softcap;
  bool round_scores, round_probabilities;
  /* Audio Transformer-XL relative term: q dot relative_keys[k-q+offset].
   * Out-of-table indices contribute zero, matching the blocked relative shift.
   * Relative keys are [relative_rows,q_heads,dim], not a dense score bias. */
  ds4_num_view relative_keys;
  uint32_t relative_rows;
  int32_t relative_offset;
} ds4_num_attention;
ds4_status ds4_num_attend(ds4_numeric *, ds4_num_view y, ds4_num_view q,
                          ds4_num_view k, ds4_num_view v,
                          const uint32_t *host_bounds,
                          const ds4_num_attention *);
/* Row-wise softmax over already projected QK scores, with explicit scale and
 * optional bounds. Used independently by numerical boundary diagnostics. */
ds4_status ds4_num_softmax(ds4_numeric *, ds4_num_view y, ds4_num_view scores,
                           const uint32_t *host_bounds, uint32_t rows,
                           uint32_t width, float scale);
/* N,C,D,H,W convolution, grouped/depthwise, explicit zero padding and
 * dilation. Singleton axes cover audio/2D; temporal left padding covers VAE.
 * No im2col of the entire input is allocated. */
typedef struct {
  uint32_t batch, in_channels, out_channels, groups;
  uint32_t input[3], kernel[3], stride[3], dilation[3], pad_before[3],
      pad_after[3];
} ds4_num_convolution;
ds4_status ds4_num_conv(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                        ds4_num_view w, ds4_num_view bias,
                        const ds4_num_convolution *);
/* Bounded storage conversion: unsupported GGML types are rejected. The
 * caller provides only the requested whole-block slice, never a full model. */
ds4_status ds4_num_dequantize(float *out, uint64_t count, const void *storage,
                              uint64_t bytes, uint32_t ggml_type);
/* Positive-stride reindex covers transpose, depth/space rearrangements and
 * broadcast/repeat without arithmetic. shape/strides describe output axes in
 * source elements. Zero strides explicitly repeat an axis; no implicit cast. */
typedef struct {
  uint32_t rank;
  uint64_t shape[8], stride[8];
} ds4_num_reindex;
ds4_status ds4_num_reorder(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                           const ds4_num_reindex *);
/* Pool [rows,group] selected source rows, UINT32_MAX denotes zero padding.
 * Explicit divisor preserves Gemma's k*k convention even at padded cells. */
ds4_status ds4_num_pool(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                        const uint32_t *host_indices, uint32_t rows,
                        uint32_t group, uint32_t width, float divisor);
typedef struct {
  uint32_t batch, channels, input[3], output[3];
} ds4_num_resample;
ds4_status ds4_num_nearest(ds4_numeric *, ds4_num_view y, ds4_num_view x,
                           const ds4_num_resample *);
#ifdef __cplusplus
}
#endif
#endif
