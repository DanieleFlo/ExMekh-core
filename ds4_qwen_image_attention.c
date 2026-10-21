#include "ds4_qwen_image_attention.h"
#include "ds4_qwen_image_graph.h"
ds4_status ds4_qwen_image_attention(ds4_numeric *num, ds4_num_view out,
                                    ds4_num_view q, ds4_num_view k,
                                    ds4_num_view v, uint32_t nq, uint32_t nk,
                                    uint32_t heads, uint32_t dim,
                                    const uint32_t *bounds) {
  if (!num || !nq || nq > 262144 || !nk || nk > 262144 || !heads ||
      heads > 32 || !dim || dim > 1152 || !bounds || q.type != DS4_NUM_BF16 ||
      k.type != DS4_NUM_BF16 || v.type != DS4_NUM_BF16 ||
      out.type != DS4_NUM_BF16 || q.count != (uint64_t)nq * heads * dim ||
      out.count != q.count || k.count != (uint64_t)nk * heads * dim ||
      v.count != k.count)
    return DS4_STATUS_INVALID;
  for (uint32_t i = 0; i < nq; i++)
    if (bounds[i * 2] > bounds[i * 2 + 1] || bounds[i * 2 + 1] > nk)
      return DS4_STATUS_INVALID;
  qi_work work = {.num = num};
  ds4_status status;
  uint32_t *ranges = NULL;
  ds4_resource *range_account = NULL;
  /* A wide single head (VAE) fits the eager full GEMM below this ceiling.
   * Query tiling changed BF16 ties despite exact full-key softmax. Multi-head
   * DiT retains its bounded 256-row workspace and request-sized memory plan. */
  uint64_t full_score = (uint64_t)heads * nq * nk * sizeof(float);
  uint32_t tile = heads == 1 && full_score <= 512ull * 1024 * 1024 ? nq : 256;
  ds4_num_view qt, kt, vt, qp, kp, vp, scores, values, output, bits, query;
  uint64_t qs = (uint64_t)nq * heads * dim, ks = (uint64_t)nk * heads * dim;
  QI_TRY(qi_alloc(&work, qs, DS4_NUM_F32, &qt));
  QI_TRY(qi_alloc(&work, ks, DS4_NUM_F32, &kt));
  QI_TRY(qi_alloc(&work, ks, DS4_NUM_F32, &vt));
  QI_TRY(ds4_num_point(num, qt, q, QI_NONE, DS4_NUM_CAST, 0));
  QI_TRY(ds4_num_point(num, kt, k, QI_NONE, DS4_NUM_CAST, 0));
  QI_TRY(ds4_num_point(num, vt, v, QI_NONE, DS4_NUM_CAST, 0));
  /* Math SDPA distributes sqrt(scale) over Q and K before the GEMM. */
  float scale = (float)sqrt(1.0 / sqrt((double)dim));
  QI_TRY(ds4_num_point(num, qt, qt, QI_NONE, DS4_NUM_SCALE, scale));
  QI_TRY(ds4_num_point(num, kt, kt, QI_NONE, DS4_NUM_SCALE, scale));
  QI_TRY(qi_alloc(&work, qs, DS4_NUM_F32, &qp));
  QI_TRY(qi_alloc(&work, ks, DS4_NUM_F32, &kp));
  QI_TRY(qi_alloc(&work, ks, DS4_NUM_F32, &vp));
  ds4_num_reindex ix = {.rank = 3,
                        .shape = {heads, nq, dim},
                        .stride = {dim, (uint64_t)heads * dim, 1}};
  QI_TRY(ds4_num_reorder(num, qp, qt, &ix));
  ix.shape[1] = nk;
  QI_TRY(ds4_num_reorder(num, kp, kt, &ix));
  QI_TRY(ds4_num_reorder(num, vp, vt, &ix));
  QI_TRY(qi_alloc(&work, (uint64_t)heads * tile * nk, DS4_NUM_F32, &scores));
  QI_TRY(qi_alloc(&work, (uint64_t)heads * tile * dim, DS4_NUM_F32, &values));
  QI_TRY(qi_alloc(&work, (uint64_t)heads * tile * dim, DS4_NUM_F32, &query));
  QI_TRY(qi_host(num, (uint64_t)heads * tile * 8, (void **)&ranges,
                 &range_account));
  QI_TRY(qi_alloc(&work, qs, DS4_NUM_F32, &output));
  QI_TRY(qi_alloc(&work, qs, DS4_NUM_BF16, &bits));
  for (uint32_t first = 0; first < nq; first += tile) {
    uint32_t rows = nq - first < tile ? nq - first : tile;
    ds4_num_view score = ds4_num_slice(scores, 0, (uint64_t)heads * rows * nk),
                 value = ds4_num_slice(values, 0, (uint64_t)heads * rows * dim),
                 qrows = ds4_num_slice(query, 0, (uint64_t)heads * rows * dim);
    ix = (ds4_num_reindex){.rank = 3,
                           .shape = {heads, rows, dim},
                           .stride = {(uint64_t)nq * dim, dim, 1}};
    QI_TRY(ds4_num_reorder(num, qrows,
                           ds4_num_slice(qp, (uint64_t)first * dim,
                                         qp.count - (uint64_t)first * dim),
                           &ix));
    QI_TRY(ds4_num_matmul_batch(num, score, qrows, kp, heads, rows, dim, nk,
                                true));
    for (uint32_t h = 0; h < heads; h++)
      memcpy(ranges + h * rows * 2, bounds + 2 * first,
             rows * 2 * sizeof(uint32_t));
    QI_TRY(ds4_num_softmax(num, score, score, ranges, heads * rows, nk, 1));
    QI_TRY(ds4_num_matmul_batch(num, value, score, vp, heads, rows, nk, dim,
                                false));
    for (uint32_t h = 0; h < heads; h++)
      QI_TRY(
          ds4_num_copy(num,
                       ds4_num_slice(output, ((uint64_t)h * nq + first) * dim,
                                     (uint64_t)rows * dim),
                       ds4_num_slice(value, (uint64_t)h * rows * dim,
                                     (uint64_t)rows * dim)));
  }
  QI_TRY(ds4_num_point(num, bits, output, QI_NONE, DS4_NUM_CAST, 0));
  ix = (ds4_num_reindex){.rank = 3,
                         .shape = {nq, heads, dim},
                         .stride = {dim, (uint64_t)nq * dim, 1}};
  QI_TRY(ds4_num_reorder(num, out, bits, &ix));
cleanup: {
  ds4_status s = qi_clear(&work);
  qi_host_free(ranges, range_account);
  return status ? status : s;
}
}
