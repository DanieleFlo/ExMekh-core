/* Gemma4 vision equations from the pinned Transformers source (Apache-2.0).
 * Uses the suite numeric API; model mechanics and weight ownership stay here.
 */
#include "ds4_gemma_vision.h"
#include "ds4_host_memory.h"
#include "ds4_numeric_weights.h"
#include <math.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
#define NONE ((ds4_num_view){0})
enum { LAYERS = 16, WIDTH = 768, GROUPS = 18 };
typedef struct {
  size_t index;
  uint32_t group;
} binding;
struct ds4_gemma_vision {
  ds4_bundle *bundle;
  ds4_numeric *num;
  ds4_runtime_model *model;
  ds4_resources *resources;
  uint64_t owner;
  uint32_t vision, projection, count;
  size_t positions;
  binding bindings[256];
  ds4_weight_group *groups[GROUPS];
  float rope[16];
  ds4_host_buffer account;
};
struct ds4_gemma_visual_features {
  ds4_gemma_visual_features_view view;
  ds4_host_buffer account, tokens, counts;
};
uint64_t ds4_gemma_vision_resident_weights(const ds4_gemma_vision *e) {
  uint64_t bytes = 0;
  if (!e)
    return 0;
  for (unsigned i = 0; i < GROUPS; i++) {
    uint64_t n = ds4_weight_group_bytes(e->groups[i]);
    if (ds4_weight_group_resident(e->groups[i]))
      bytes += n;
  }
  return bytes;
}
typedef struct {
  ds4_numeric *num;
  ds4_num_buffer *buffers[28];
  uint32_t count;
} work;
static uint16_t bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
static float from_bf16(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
static ds4_status allocate(work *w, uint64_t count, uint32_t type,
                           ds4_num_view *v) {
  if (w->count == 28)
    return DS4_STATUS_INVALID;
  ds4_num_buffer *b = NULL;
  ds4_status s = ds4_num_alloc(w->num, count, type, &b);
  if (!s) {
    w->buffers[w->count++] = b;
    *v = ds4_num_data(b);
  }
  return s;
}
static ds4_status clear(work *w) {
  ds4_status s = DS4_STATUS_OK;
  while (w->count) {
    ds4_status t = ds4_num_dispose(w->buffers[--w->count]);
    if (!s)
      s = t;
  }
  return s;
}
static ds4_status bind(ds4_gemma_vision *e, const char *name, uint32_t group,
                       uint32_t rank, uint64_t a, uint64_t b, uint64_t c) {
  for (size_t i = 0; i < ds4_bundle_tensor_count(e->bundle); i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, i);
    uint32_t component = group == 17 ? e->projection : e->vision;
    if (t->component != component || strcmp(t->name, name))
      continue;
    if (e->count == 256 || t->rank != rank || t->type != DS4_NUM_BF16 ||
        t->block_elements != 1 || t->block_bytes != 2 || t->shape[0] != a ||
        t->stride[0] != 2 ||
        (rank > 1 && (t->shape[1] != b || t->stride[1] != 2 * a)) ||
        (rank > 2 && (t->shape[2] != c || t->stride[2] != 2 * a * b)))
      return DS4_STATUS_UNSUPPORTED;
    e->bindings[e->count++] = (binding){i, group};
    if (rank == 3)
      e->positions = i;
    return DS4_STATUS_OK;
  }
  return DS4_STATUS_INVALID;
}
static ds4_status weight(ds4_gemma_vision *e, const char *name,
                         ds4_num_view *out) {
  for (uint32_t i = 0; i < e->count; i++) {
    const ds4_weight_tensor *t =
        ds4_bundle_tensor(e->bundle, e->bindings[i].index);
    if (strcmp(t->name, name))
      continue;
    ds4_weight_group *g = e->groups[e->bindings[i].group];
    ds4_status s = ds4_weight_group_load(g);
    const void *storage;
    uint64_t off;
    if (!s)
      s = ds4_weight_group_tensor(g, name, &storage, &off);
    if (!s)
      *out = ds4_numeric_weight_view(storage, off, t->bytes / 2, DS4_NUM_BF16);
    return s;
  }
  return DS4_STATUS_INVALID;
}
static ds4_status linear(ds4_gemma_vision *e, const char *name, ds4_num_view y,
                         ds4_num_view x, uint32_t rows, uint32_t in,
                         uint32_t out) {
  ds4_num_view w;
  ds4_status s = weight(e, name, &w);
  return s ? s : ds4_num_linear(e->num, y, x, w, NONE, rows, in, out);
}
static ds4_status norm(ds4_gemma_vision *e, const char *name, ds4_num_view y,
                       ds4_num_view x, uint32_t width) {
  ds4_num_view w = NONE;
  ds4_status s = name ? weight(e, name, &w) : DS4_STATUS_OK;
  return s ? s
           : ds4_num_norm(e->num, y, x, w, NONE, width, 1e-6f, DS4_NUM_RMS,
                          name ? DS4_NUM_SCALE_DIRECT : DS4_NUM_SCALE_NONE,
                          false);
}
static ds4_status observe(ds4_gemma_vision *e, const ds4_gemma_observer *o,
                          const char *stage, uint32_t frame, uint32_t layer,
                          ds4_num_view v) {
  if (o && o->cancelled && o->cancelled(o->opaque))
    return DS4_STATUS_CANCELLED;
  ds4_status s = ds4_numeric_fence(e->num);
  if (!s && o && o->trace)
    s = o->trace(o->opaque, stage, frame, layer, v);
  if (!s && o && o->cancelled && o->cancelled(o->opaque))
    s = DS4_STATUS_CANCELLED;
  return s;
}
static ds4_status observe_batch(ds4_gemma_vision *e,
                                const ds4_gemma_observer *o, const char *stage,
                                uint32_t layer, ds4_num_view v,
                                ds4_gemma_media_view media, bool soft) {
  uint64_t offset = 0, width = soft ? v.count / media.soft_tokens : 0;
  for (uint32_t i = 0; i < media.frames; i++) {
    uint64_t count = soft ? (uint64_t)media.frame_soft_tokens[i] * width
                          : v.count / media.frames;
    ds4_status s =
        observe(e, o, stage, i, layer, ds4_num_slice(v, offset, count));
    if (s)
      return s;
    offset += count;
  }
  return DS4_STATUS_OK;
}
static ds4_status activate(void *opaque) {
  return ds4_numeric_fence(((ds4_gemma_vision *)opaque)->num);
}
ds4_status ds4_gemma_vision_release_weights(ds4_gemma_vision *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(e->model, &scope);
  if (s)
    return s;
  for (uint32_t i = 0; i < GROUPS && !s; i++)
    s = ds4_weight_group_release(e->groups[i]);
  ds4_runtime_scope_leave(&scope);
  return s;
}
ds4_status ds4_gemma_vision_close(ds4_gemma_vision *e) {
  if (!e)
    return DS4_STATUS_OK;
  ds4_status s = e->model ? ds4_gemma_vision_release_weights(e) : DS4_STATUS_OK;
  if (s)
    return s;
  if (e->model && (s = ds4_runtime_model_close(e->model)))
    return s;
  e->model = NULL;
  for (uint32_t i = 0; i < GROUPS; i++) {
    if ((s = ds4_weight_group_close(e->groups[i])))
      return s;
    e->groups[i] = NULL;
  }
  ds4_host_buffer account = e->account;
  ds4_host_free(&account);
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_vision_open(ds4_gemma_vision **out, ds4_bundle *b,
                                 ds4_numeric *n, ds4_runtime *runtime,
                                 const ds4_gemma_processor *processor) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!b || !n || !runtime || !processor)
    return DS4_STATUS_INVALID;
  if (strcmp(ds4_bundle_model(b), "google/embeddinggemma-2") ||
      strcmp(ds4_bundle_revision(b),
             "914f7f89142e33e77833254d9c9b90c3cef7303b") ||
      strcmp(ds4_bundle_profile(b), "bf16-native-v1"))
    return DS4_STATUS_UNSUPPORTED;
  ds4_gemma_vision *e = NULL;
  ds4_host_buffer account = {0};
  ds4_runtime_scope scope = {0};
  ds4_status status;
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(n, &owner);
  TRY(ds4_host_alloc(rs, owner, sizeof(*e), &account));
  e = account.data;
  e->account = account;
  e->bundle = b;
  e->num = n;
  e->resources = rs;
  e->owner = owner;
  e->vision = e->projection = UINT32_MAX;
  TRY(ds4_runtime_begin_update(runtime, &scope));
  TRY(ds4_gemma_processor_vision_rope(processor, e->rope));
  for (size_t i = 0; i < ds4_bundle_component_count(b); i++) {
    const ds4_bundle_component *c = ds4_bundle_component_at(b, i);
    if (strcmp(c->revision, ds4_bundle_revision(b)) ||
        strcmp(c->compute_dtype, "bfloat16"))
      continue;
    if (!strcmp(c->id, "vision"))
      e->vision = i;
    if (!strcmp(c->id, "vision_projection"))
      e->projection = i;
  }
  if (e->vision == UINT32_MAX || e->projection == UINT32_MAX) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  TRY(bind(e, "vision_tower.patch_embedder.input_proj.weight", 0, 2, 768, 768,
           0));
  TRY(bind(e, "vision_tower.patch_embedder.position_embedding_table", 0, 3, 768,
           10240, 2));
  TRY(bind(e, "embed_vision.embedding_projection.weight", 17, 2, 768, 512, 0));
  const char *norms[] = {
      "input_layernorm.weight",           "post_attention_layernorm.weight",
      "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
      "self_attn.q_norm.weight",          "self_attn.k_norm.weight"};
  const char *linears[] = {
      "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight",
      "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight",
      "mlp.gate_proj.linear.weight",    "mlp.up_proj.linear.weight",
      "mlp.down_proj.linear.weight"};
  uint32_t ins[] = {768, 768, 768, 768, 768, 768, 3072},
           outs[] = {768, 768, 768, 768, 3072, 3072, 768};
  for (uint32_t i = 0; i < LAYERS; i++) {
    char name[256];
    for (unsigned j = 0; j < 6; j++) {
      snprintf(name, sizeof(name), "vision_tower.encoder.layers.%u.%s", i,
               norms[j]);
      TRY(bind(e, name, i + 1, 1, j < 4 ? 768 : 64, 0, 0));
    }
    for (unsigned j = 0; j < 7; j++) {
      snprintf(name, sizeof(name), "vision_tower.encoder.layers.%u.%s", i,
               linears[j]);
      TRY(bind(e, name, i + 1, 2, ins[j], outs[j], 0));
    }
  }
  size_t count = 0;
  for (size_t i = 0; i < ds4_bundle_tensor_count(b); i++) {
    uint32_t c = ds4_bundle_tensor(b, i)->component;
    if (c == e->vision || c == e->projection)
      count++;
  }
  if (count != e->count) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  ds4_weight_backend backend = ds4_numeric_weight_backend(n);
  for (uint32_t g = 0; g < GROUPS; g++) {
    size_t indices[256], used = 0;
    for (uint32_t j = 0; j < e->count; j++)
      if (e->bindings[j].group == g && e->bindings[j].index != e->positions)
        indices[used++] = e->bindings[j].index;
    TRY(ds4_weight_group_create_selected(e->groups + g, b,
                                         g == 17 ? e->projection : e->vision,
                                         indices, used, &backend));
  }
  ds4_runtime_descriptor descriptor = {DS4_SUITE_ABI,
                                       ds4_bundle_model(b),
                                       ds4_bundle_revision(b),
                                       ds4_bundle_profile(b),
                                       "vision",
                                       DS4_OPERATION_EMBED,
                                       DS4_MODALITY_IMAGE | DS4_MODALITY_VIDEO,
                                       8192,
                                       512};
  ds4_runtime_model_ops ops = {activate, NULL};
  TRY(ds4_runtime_model_create(runtime, &descriptor, &ops, e, &e->model));
  *out = e;
  e = NULL;
cleanup:
  ds4_runtime_scope_leave(&scope);
  if (e)
    ds4_gemma_vision_close(e);
  return status;
}
ds4_status ds4_gemma_vision_estimate(ds4_gemma_vision *e,
                                     const ds4_gemma_media *media,
                                     ds4_runtime_estimate *out) {
  if (!e || !media || !out)
    return DS4_STATUS_INVALID;
  ds4_gemma_media_view v = ds4_gemma_media_data(media);
  if (!v.frames || !v.max_patches || v.soft_tokens > 8192)
    return DS4_STATUS_INVALID;
  uint64_t rows = (uint64_t)v.max_patches * v.frames;
  /* One media group executes at a time. Preserve eager frame batching, since
   * changing batched attention GEMM geometry changes BF16 rounding ties. */
  *out = (ds4_runtime_estimate){
      .known = true,
      .device_bytes = rows * 180000 + 64ull * 1024 * 1024,
      .host_bytes = rows * 16384 + (uint64_t)v.soft_tokens * 1024 + 65536};
  if (v.max_patches > 256 &&
      12ull * v.max_patches * v.max_patches * 2 <= 128ull * 1024 * 1024)
    out->device_bytes += (uint64_t)v.frames * (v.max_patches - 256) * 12 *
                         (v.max_patches + 64) * 2;
  for (uint32_t i = 0; i < GROUPS; i++)
    if (!ds4_weight_group_resident(e->groups[i]))
      out->device_bytes += ds4_weight_group_bytes(e->groups[i]);
  return DS4_STATUS_OK;
}
static ds4_status rotary(ds4_gemma_vision *e, ds4_num_view out,
                         ds4_num_view input, ds4_num_view packed,
                         ds4_num_view rotated, ds4_num_view co, ds4_num_view si,
                         uint32_t n) {
  ds4_num_reindex ix = {
      .rank = 4, .shape = {n, 2, 12, 32}, .stride = {768, 32, 64, 1}};
  ds4_status s = ds4_num_reorder(e->num, packed, input, &ix);
  if (!s)
    s = ds4_num_rope(e->num, rotated, packed, co, si, 12, 32, false, true);
  ix = (ds4_num_reindex){
      .rank = 4, .shape = {n, 12, 2, 32}, .stride = {768, 32, 384, 1}};
  return s ? s : ds4_num_reorder(e->num, out, rotated, &ix);
}
static ds4_status media_forward(ds4_gemma_vision *e, ds4_gemma_media_view media,
                                const ds4_gemma_observer *o, uint16_t *output) {
  ds4_status status = DS4_STATUS_OK;
  work w = {.num = e->num};
  ds4_host_buffer host = {0}, bounds = {0}, pool_weights = {0};
  uint32_t n = media.max_patches, soft = media.soft_tokens,
           rows = n * media.frames;
  const int32_t *positions = media.positions;
  const float *pixels = media.patches;
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)rows * 768 * 2, &host));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)rows * 8, &bounds));
  uint16_t *bits = host.data;
  for (uint64_t i = 0; i < (uint64_t)rows * 768; i++)
    bits[i] = bf16(pixels[i]);
  ds4_num_view x, y, z, q, k, v, qr, kr, packed, rotated, attn, gate, up, co,
      si, half;
  ds4_num_view f32, weights, pooled, pooled_bits, scaled, projected;
#define ALLOC(name, count, type) TRY(allocate(&w, count, type, &name))
  ALLOC(x, (uint64_t)rows * 768, DS4_NUM_BF16);
  ALLOC(y, x.count, DS4_NUM_BF16);
  ALLOC(z, x.count, DS4_NUM_BF16);
  ALLOC(q, x.count, DS4_NUM_BF16);
  ALLOC(k, x.count, DS4_NUM_BF16);
  ALLOC(v, x.count, DS4_NUM_BF16);
  ALLOC(qr, x.count, DS4_NUM_BF16);
  ALLOC(kr, x.count, DS4_NUM_BF16);
  ALLOC(packed, x.count, DS4_NUM_BF16);
  ALLOC(rotated, x.count, DS4_NUM_BF16);
  ALLOC(attn, x.count, DS4_NUM_BF16);
  ALLOC(gate, (uint64_t)rows * 3072, DS4_NUM_BF16);
  ALLOC(up, gate.count, DS4_NUM_BF16);
  ALLOC(co, (uint64_t)rows * 32, DS4_NUM_BF16);
  ALLOC(si, co.count, DS4_NUM_BF16);
  ALLOC(half, 1, DS4_NUM_BF16);
  TRY(ds4_num_write(e->num, y, bits, y.count * 2));
  uint16_t minus_half = bf16(-0.5f);
  TRY(ds4_num_write(e->num, half, &minus_half, 2));
  TRY(ds4_num_point(e->num, y, y, half, DS4_NUM_ADD, 0));
  TRY(ds4_num_point(e->num, y, y, NONE, DS4_NUM_SCALE, 2));
  TRY(linear(e, "vision_tower.patch_embedder.input_proj.weight", x, y, rows,
             768, 768));
  const uint16_t *table = ds4_bundle_tensor_data(e->bundle, e->positions);
  for (uint32_t i = 0; i < rows; i++)
    for (uint32_t d = 0; d < 768; d++)
      bits[(uint64_t)i * 768 + d] =
          positions[2 * i] < 0
              ? 0
              : bf16(from_bf16(table[(uint64_t)positions[2 * i] * 768 + d]) +
                     from_bf16(
                         table[((uint64_t)10240 + positions[2 * i + 1]) * 768 +
                               d]));
  TRY(ds4_num_write(e->num, y, bits, y.count * 2));
  TRY(ds4_num_point(e->num, x, x, y, DS4_NUM_ADD, 0));
  TRY(observe_batch(e, o, "vision-patches", 0, x, media, false));
  for (uint32_t i = 0; i < rows; i++) {
    ((uint32_t *)bounds.data)[2 * i] = 0;
    ((uint32_t *)bounds.data)[2 * i + 1] = media.frame_soft_tokens[i / n] * 9;
    for (uint32_t axis = 0; axis < 2; axis++)
      for (uint32_t d = 0; d < 16; d++)
        bits[((uint64_t)i * 2 + axis) * 16 + d] =
            bf16(cosf(positions[2 * i + axis] * e->rope[d]));
  }
  TRY(ds4_num_write(e->num, co, bits, co.count * 2));
  for (uint32_t i = 0; i < rows; i++)
    for (uint32_t axis = 0; axis < 2; axis++)
      for (uint32_t d = 0; d < 16; d++)
        bits[((uint64_t)i * 2 + axis) * 16 + d] =
            bf16(sinf(positions[2 * i + axis] * e->rope[d]));
  TRY(ds4_num_write(e->num, si, bits, si.count * 2));
  for (uint32_t layer = 0; layer < 16; layer++) {
    char key[256];
#define NAME(s)                                                                \
  (snprintf(key, sizeof(key), "vision_tower.encoder.layers.%u.%s", layer, s),  \
   key)
    TRY(norm(e, NAME("input_layernorm.weight"), y, x, 768));
    if (!layer)
      TRY(observe_batch(e, o, "vision-norm", layer, y, media, false));
    TRY(linear(e, NAME("self_attn.q_proj.linear.weight"), q, y, rows, 768,
               768));
    TRY(linear(e, NAME("self_attn.k_proj.linear.weight"), k, y, rows, 768,
               768));
    TRY(linear(e, NAME("self_attn.v_proj.linear.weight"), v, y, rows, 768,
               768));
    TRY(norm(e, NAME("self_attn.q_norm.weight"), q, q, 64));
    TRY(norm(e, NAME("self_attn.k_norm.weight"), k, k, 64));
    TRY(norm(e, NULL, v, v, 64));
    if (!layer) {
      TRY(observe_batch(e, o, "vision-q", layer, q, media, false));
      TRY(observe_batch(e, o, "vision-k", layer, k, media, false));
      TRY(observe_batch(e, o, "vision-v", layer, v, media, false));
    }
    TRY(rotary(e, qr, q, packed, rotated, co, si, rows));
    TRY(rotary(e, kr, k, packed, rotated, co, si, rows));
    if (!layer) {
      TRY(observe_batch(e, o, "vision-rope-q", layer, qr, media, false));
      TRY(observe_batch(e, o, "vision-rope-k", layer, kr, media, false));
    }
    ds4_num_attention a = {.batch = media.frames,
                           .queries = n,
                           .keys = n,
                           .q_heads = 12,
                           .kv_heads = 12,
                           .dim = 64,
                           .scale = 1,
                           .round_scores = true,
                           .round_probabilities = true};
    TRY(ds4_num_attend(e->num, attn, qr, kr, v, bounds.data, &a));
    if (!layer)
      TRY(observe_batch(e, o, "vision-attention", layer, attn, media, false));
    TRY(linear(e, NAME("self_attn.o_proj.linear.weight"), z, attn, rows, 768,
               768));
    TRY(norm(e, NAME("post_attention_layernorm.weight"), z, z, 768));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    if (!layer)
      TRY(observe_batch(e, o, "vision-attn-out", layer, x, media, false));
    TRY(norm(e, NAME("pre_feedforward_layernorm.weight"), y, x, 768));
    TRY(linear(e, NAME("mlp.gate_proj.linear.weight"), gate, y, rows, 768,
               3072));
    TRY(linear(e, NAME("mlp.up_proj.linear.weight"), up, y, rows, 768, 3072));
    TRY(ds4_num_point(e->num, gate, gate, NONE, DS4_NUM_GELU_TANH, 0));
    TRY(ds4_num_point(e->num, gate, gate, up, DS4_NUM_MUL, 0));
    if (!layer)
      TRY(observe_batch(e, o, "vision-gate", layer, gate, media, false));
    TRY(linear(e, NAME("mlp.down_proj.linear.weight"), z, gate, rows, 3072,
               768));
    TRY(norm(e, NAME("post_feedforward_layernorm.weight"), z, z, 768));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    TRY(observe_batch(e, o, "vision-layer", layer, x, media, false));
#undef NAME
  }
  /* Eager pooling retains the full budget and a transposed one-hot layout.
   * Repacking it changes cuBLAS accumulation and BF16 ties. Round the FP32
   * product, then scale sqrt(768) in FP32 and round again before projection. */
  uint32_t pool_length = media.soft_token_budget;
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)pool_length * rows * 4,
                     &pool_weights));
  for (uint32_t frame = 0; frame < media.frames; frame++) {
    uint32_t valid = media.frame_soft_tokens[frame] * 9;
    uint64_t base = (uint64_t)frame * n;
    uint32_t grid_w = positions[(base + valid - 1) * 2] + 1;
    for (uint32_t i = 0; i < valid; i++) {
      uint32_t index = positions[(base + i) * 2] / 3 +
                       (grid_w / 3) * (positions[(base + i) * 2 + 1] / 3);
      ((float *)pool_weights.data)[(base + i) * pool_length + index] = 1.f / 9;
    }
  }
  ALLOC(f32, x.count, DS4_NUM_F32);
  ALLOC(weights, (uint64_t)pool_length * rows, DS4_NUM_F32);
  ALLOC(pooled, (uint64_t)media.frames * pool_length * 768, DS4_NUM_F32);
  ALLOC(pooled_bits, pooled.count, DS4_NUM_BF16);
  ALLOC(scaled, pooled.count, DS4_NUM_F32);
  ALLOC(projected, (uint64_t)soft * 512, DS4_NUM_BF16);
  TRY(ds4_num_point(e->num, f32, x, NONE, DS4_NUM_CAST, 0));
  TRY(ds4_num_write(e->num, weights, pool_weights.data, weights.count * 4));
  TRY(ds4_num_matmul_batch_layout(e->num, pooled, weights, f32, media.frames,
                                  pool_length, n, 768, true, false));
  TRY(ds4_num_point(e->num, pooled_bits, pooled, NONE, DS4_NUM_CAST, 0));
  TRY(ds4_num_point(e->num, scaled, pooled_bits, NONE, DS4_NUM_CAST, 0));
  TRY(ds4_num_point(e->num, scaled, scaled, NONE, DS4_NUM_SCALE,
                    (float)sqrt(768.0)));
  TRY(ds4_num_point(e->num, pooled_bits, scaled, NONE, DS4_NUM_CAST, 0));
  TRY(ds4_num_read(e->num, pooled_bits, bits, pooled_bits.count * 2));
  uint64_t offset = 0;
  for (uint32_t frame = 0; frame < media.frames; frame++) {
    uint64_t count = (uint64_t)media.frame_soft_tokens[frame] * 768;
    memmove(bits + offset, bits + (uint64_t)frame * pool_length * 768,
            count * 2);
    offset += count;
  }
  pooled_bits.count = (uint64_t)soft * 768;
  TRY(ds4_num_write(e->num, pooled_bits, bits, pooled_bits.count * 2));
  TRY(observe_batch(e, o, "vision-pooled", 16, pooled_bits, media, true));
  TRY(norm(e, NULL, pooled_bits, pooled_bits, 768));
  TRY(linear(e, "embed_vision.embedding_projection.weight", projected,
             pooled_bits, soft, 768, 512));
  TRY(observe_batch(e, o, "vision-projected", 16, projected, media, true));
  TRY(ds4_num_read(e->num, projected, output, projected.count * 2));
cleanup: {
  ds4_status s = clear(&w);
  if (!status)
    status = s;
  ds4_host_free(&host);
  ds4_host_free(&bounds);
  ds4_host_free(&pool_weights);
  return status;
}
#undef ALLOC
}
ds4_gemma_visual_features_view
ds4_gemma_visual_features_data(const ds4_gemma_visual_features *f) {
  return f ? f->view : (ds4_gemma_visual_features_view){0};
}
void ds4_gemma_visual_features_free(ds4_gemma_visual_features *f) {
  if (!f)
    return;
  ds4_host_free(&f->tokens);
  ds4_host_free(&f->counts);
  ds4_host_buffer account = f->account;
  ds4_host_free(&account);
}
ds4_status ds4_gemma_vision_encode(ds4_gemma_vision *e,
                                   const ds4_gemma_media *media,
                                   const ds4_gemma_observer *o,
                                   ds4_gemma_visual_features **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !media)
    return DS4_STATUS_INVALID;
  ds4_status status;
  ds4_runtime_scope scope = {0};
  ds4_reservation *lease = NULL;
  ds4_reservation_scope working = {0};
  ds4_gemma_visual_features *f = NULL;
  ds4_host_buffer account = {0};
  TRY(ds4_runtime_model_enter(e->model, &scope));
  ds4_runtime_estimate estimate;
  TRY(ds4_gemma_vision_estimate(e, media, &estimate));
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_HOST] = estimate.host_bytes,
                                      [DS4_MEMORY_DEVICE] =
                                          estimate.device_bytes};
  TRY(ds4_resources_reserve(e->resources, e->owner, bytes, &lease));
  working = ds4_reservation_enter(lease);
  ds4_gemma_media_view v = ds4_gemma_media_data(media);
  v.soft_tokens -= v.audio_soft_tokens;
  TRY(ds4_host_alloc(e->resources, e->owner, sizeof(*f), &account));
  f = account.data;
  f->account = account;
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)v.soft_tokens * 512 * 2,
                     &f->tokens));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)v.frames * 4,
                     &f->counts));
  memcpy(f->counts.data, v.frame_soft_tokens, v.frames * 4);
  f->view = (ds4_gemma_visual_features_view){f->tokens.data, f->counts.data,
                                             v.soft_tokens, v.frames};
  TRY(media_forward(e, v, o, f->tokens.data));
cleanup:
  ds4_reservation_leave(&working);
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status)
    ds4_gemma_visual_features_free(f);
  else
    *out = f;
  ds4_runtime_scope_leave(&scope);
  return status;
}
