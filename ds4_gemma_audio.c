/* Gemma4 audio equations from the pinned Transformers source (Apache-2.0).
 * Uses the suite numeric API; model mechanics and weight ownership stay here.
 */
#include "ds4_gemma_audio.h"
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
enum { LAYERS = 12, WIDTH = 1024, GROUPS = 14 };
typedef struct {
  size_t index;
  uint32_t group;
} binding;
struct ds4_gemma_audio {
  ds4_bundle *bundle;
  ds4_numeric *num;
  ds4_runtime_model *model;
  ds4_resources *resources;
  uint64_t owner;
  uint32_t audio, projection, count;
  binding bindings[1024];
  ds4_weight_group *groups[GROUPS];
  ds4_weight_cache cache;
  ds4_host_buffer account;
};
struct ds4_gemma_audio_features {
  ds4_gemma_audio_features_view view;
  ds4_host_buffer account, tokens;
};
uint64_t ds4_gemma_audio_resident_weights(const ds4_gemma_audio *e) {
  return e ? ds4_weight_cache_resident(&e->cache) : 0;
}
typedef struct {
  ds4_numeric *num;
  ds4_num_buffer *buffers[32];
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
  if (w->count == sizeof(w->buffers) / sizeof(w->buffers[0]))
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
static ds4_status bind(ds4_gemma_audio *e, const char *name, uint32_t group,
                       uint32_t rank, uint64_t a, uint64_t b, uint64_t c) {
  for (size_t i = 0; i < ds4_bundle_tensor_count(e->bundle); i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, i);
    uint32_t component = group == 13 ? e->projection : e->audio;
    if (t->component != component || strcmp(t->name, name))
      continue;
    if (e->count == 1024 || t->rank != rank || t->type != DS4_NUM_BF16 ||
        t->block_elements != 1 || t->block_bytes != 2 ||
        (rank && (t->shape[0] != a || t->stride[0] != 2)) ||
        (rank > 1 && (t->shape[1] != b || t->stride[1] != 2 * a)) ||
        (rank > 2 && (t->shape[2] != c || t->stride[2] != 2 * a * b)))
      return DS4_STATUS_UNSUPPORTED;
    e->bindings[e->count++] = (binding){i, group};
    return DS4_STATUS_OK;
  }
  return DS4_STATUS_INVALID;
}
static const binding *find_weight(const ds4_gemma_audio *e, const char *name) {
  for (uint32_t i = 0; i < e->count; i++)
    if (!strcmp(ds4_bundle_tensor(e->bundle, e->bindings[i].index)->name, name))
      return e->bindings + i;
  return NULL;
}
static ds4_status scalar_weight(const ds4_gemma_audio *e, const char *name,
                                float *out) {
  const binding *b = find_weight(e, name);
  if (!b)
    return DS4_STATUS_INVALID;
  uint16_t bits;
  memcpy(&bits, ds4_bundle_tensor_data(e->bundle, b->index), sizeof(bits));
  *out = from_bf16(bits);
  return isnan(*out) ? DS4_STATUS_INVALID : DS4_STATUS_OK;
}
static ds4_status weight(ds4_gemma_audio *e, const char *name,
                         ds4_num_view *out) {
  const binding *b = find_weight(e, name);
  if (!b)
    return DS4_STATUS_INVALID;
  const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, b->index);
  ds4_weight_group *g = e->groups[b->group];
  ds4_status s = ds4_weight_cache_acquire(&e->cache, b->group);
  const void *storage;
  uint64_t off;
  if (!s)
    s = ds4_weight_group_tensor(g, name, &storage, &off);
  if (!s)
    *out = ds4_numeric_weight_view(storage, off, t->bytes / 2, DS4_NUM_BF16);
  return s;
}
static ds4_status linear(ds4_gemma_audio *e, const char *name, ds4_num_view y,
                         ds4_num_view x, uint32_t rows, uint32_t in,
                         uint32_t out) {
  ds4_num_view w;
  ds4_status s = weight(e, name, &w);
  return s ? s : ds4_num_linear(e->num, y, x, w, NONE, rows, in, out);
}
/* Clipping limits are immutable scalars in the mmap bundle. Apply input and
 * output limits at the same BF16 boundaries as Gemma4ClippableLinear. */
typedef enum { AUDIO_ROWS, AUDIO_TRANSPOSED_ROWS } audio_linear_layout;
static ds4_status clipped(ds4_gemma_audio *e, uint32_t layer, const char *part,
                          ds4_num_view y, ds4_num_view x, ds4_num_view scratch,
                          uint32_t rows, uint32_t in, uint32_t out,
                          audio_linear_layout layout) {
  char name[256];
  const char *limits[] = {"input_min", "input_max", "output_min", "output_max"};
  float bound[4];
  ds4_status status;
  for (unsigned j = 0; j < 4; j++) {
    snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.%s", layer, part,
             limits[j]);
    TRY(scalar_weight(e, name, bound + j));
  }
  ds4_num_view workspace = scratch;
  scratch = ds4_num_slice(scratch, 0, (uint64_t)rows * in);
  TRY(ds4_num_copy(e->num, scratch, x));
  for (unsigned j = 0; j < 2; j++)
    if (isfinite(bound[j])) {
      ds4_num_view limit;
      snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.%s", layer, part,
               limits[j]);
      TRY(weight(e, name, &limit));
      TRY(ds4_num_point(e->num, scratch, scratch, limit,
                        j ? DS4_NUM_MIN : DS4_NUM_MAX, 0));
    }
  snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.linear.weight", layer,
           part);
  if (layout == AUDIO_TRANSPOSED_ROWS) {
    /* The channel-major norm/SiLU retain a transposed view in eager linear.
     * Preserve its GEMM strides: packing rows first changes rare BF16 ties. */
    ds4_num_view packed =
        ds4_num_slice(workspace, (uint64_t)rows * in, (uint64_t)rows * in);
    ds4_num_view weights;
    TRY(weight(e, name, &weights));
    ds4_num_reindex ix = {.rank = 2, .shape = {in, rows}, .stride = {1, in}};
    TRY(ds4_num_reorder(e->num, packed, scratch, &ix));
    TRY(ds4_num_matmul_batch_layout(e->num, y, packed, weights, 1, rows, in,
                                    out, true, true));
  } else {
    TRY(linear(e, name, y, scratch, rows, in, out));
  }
  for (unsigned j = 2; j < 4; j++)
    if (isfinite(bound[j])) {
      ds4_num_view limit;
      snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.%s", layer, part,
               limits[j]);
      TRY(weight(e, name, &limit));
      TRY(ds4_num_point(e->num, y, y, limit, j == 3 ? DS4_NUM_MIN : DS4_NUM_MAX,
                        0));
    }
cleanup:
  return status;
}
static ds4_status norm(ds4_gemma_audio *e, const char *name, ds4_num_view y,
                       ds4_num_view x, uint32_t width) {
  ds4_num_view w = NONE;
  ds4_status s = name ? weight(e, name, &w) : DS4_STATUS_OK;
  return s ? s
           : ds4_num_norm(e->num, y, x, w, NONE, width, 1e-6f, DS4_NUM_RMS,
                          name ? DS4_NUM_SCALE_DIRECT : DS4_NUM_SCALE_NONE,
                          false);
}
static ds4_status observe(ds4_gemma_audio *e, const ds4_gemma_observer *o,
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
static ds4_status activate(void *opaque) {
  return ds4_numeric_fence(((ds4_gemma_audio *)opaque)->num);
}
ds4_status ds4_gemma_audio_weights(ds4_gemma_audio *e, uint64_t limit,
                                   ds4_weight_budget *out) {
  return e ? ds4_weight_cache_configure(&e->cache, e->model, limit, out)
           : DS4_STATUS_INVALID;
}
ds4_status ds4_gemma_audio_release_weights(ds4_gemma_audio *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(e->model, &scope);
  if (s)
    return s;
  s = ds4_weight_cache_release(&e->cache);
  ds4_runtime_scope_leave(&scope);
  return s;
}
ds4_status ds4_gemma_audio_close(ds4_gemma_audio *e) {
  if (!e)
    return DS4_STATUS_OK;
  ds4_status s = e->model ? ds4_gemma_audio_release_weights(e) : DS4_STATUS_OK;
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
ds4_status ds4_gemma_audio_open(ds4_gemma_audio **out, ds4_bundle *b,
                                ds4_numeric *n, ds4_runtime *runtime) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!b || !n || !runtime)
    return DS4_STATUS_INVALID;
  if (strcmp(ds4_bundle_model(b), "google/embeddinggemma-2") ||
      strcmp(ds4_bundle_revision(b),
             "914f7f89142e33e77833254d9c9b90c3cef7303b") ||
      strcmp(ds4_bundle_profile(b), "bf16-native-v1"))
    return DS4_STATUS_UNSUPPORTED;
  ds4_gemma_audio *e = NULL;
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
  e->audio = e->projection = UINT32_MAX;
  TRY(ds4_runtime_begin_update(runtime, &scope));
  for (size_t i = 0; i < ds4_bundle_component_count(b); i++) {
    const ds4_bundle_component *c = ds4_bundle_component_at(b, i);
    if (strcmp(c->revision, ds4_bundle_revision(b)) ||
        strcmp(c->compute_dtype, "bfloat16"))
      continue;
    if (!strcmp(c->id, "audio"))
      e->audio = i;
    if (!strcmp(c->id, "audio_projection"))
      e->projection = i;
  }
  if (e->audio == UINT32_MAX || e->projection == UINT32_MAX) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  /* Every tensor of the pinned audio branch is bound, including clipping
   * scalars. Shapes are checked below before any device allocation. */
  const char *ss = "audio_tower.subsample_conv_projection";
  char name[256];
  for (unsigned i = 0; i < 2; i++) {
    snprintf(name, sizeof(name), "%s.layer%u.conv.weight", ss, i);
    /* safetensors [out,in,3,3] has reversed bundle dimensions. */
    for (size_t j = 0; j < ds4_bundle_tensor_count(b); j++) {
      const ds4_weight_tensor *t = ds4_bundle_tensor(b, j);
      if (strcmp(t->name, name) || t->component != e->audio)
        continue;
      if (t->rank != 4 || t->type != DS4_NUM_BF16 || t->shape[0] != 3 ||
          t->shape[1] != 3 || t->shape[2] != (i ? 128u : 1u) ||
          t->shape[3] != (i ? 32u : 128u) || t->bytes != (i ? 73728u : 2304u)) {
        status = DS4_STATUS_UNSUPPORTED;
        goto cleanup;
      }
      e->bindings[e->count++] = (binding){j, 0};
    }
    snprintf(name, sizeof(name), "%s.layer%u.norm.weight", ss, i);
    TRY(bind(e, name, 0, 1, i ? 32 : 128, 0, 0));
  }
  TRY(bind(e, "audio_tower.subsample_conv_projection.input_proj_linear.weight",
           0, 2, 1024, 1024, 0));
  TRY(bind(e, "audio_tower.output_proj.weight", 0, 2, 1024, 1536, 0));
  TRY(bind(e, "audio_tower.output_proj.bias", 0, 1, 1536, 0, 0));
  TRY(bind(e, "embed_audio.embedding_projection.weight", 13, 2, 1536, 512, 0));
  const char *norms[] = {"feed_forward1.pre_layer_norm.weight",
                         "feed_forward1.post_layer_norm.weight",
                         "feed_forward2.pre_layer_norm.weight",
                         "feed_forward2.post_layer_norm.weight",
                         "lconv1d.pre_layer_norm.weight",
                         "lconv1d.conv_norm.weight",
                         "norm_pre_attn.weight",
                         "norm_post_attn.weight",
                         "norm_out.weight"};
  const char *linears[] = {
      "feed_forward1.ffw_layer_1", "feed_forward1.ffw_layer_2",
      "feed_forward2.ffw_layer_1", "feed_forward2.ffw_layer_2",
      "lconv1d.linear_start",      "lconv1d.linear_end",
      "self_attn.q_proj",          "self_attn.k_proj",
      "self_attn.v_proj",          "self_attn.post"};
  uint32_t ins[] = {1024, 4096, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024};
  uint32_t outs[] = {4096, 1024, 4096, 1024, 2048,
                     1024, 1024, 1024, 1024, 1024};
  const char *limits[] = {"input_min", "input_max", "output_min", "output_max"};
  for (uint32_t i = 0; i < LAYERS; i++) {
    for (unsigned j = 0; j < 9; j++) {
      snprintf(name, sizeof(name), "audio_tower.layers.%u.%s", i, norms[j]);
      TRY(bind(e, name, i + 1, 1, 1024, 0, 0));
    }
    for (unsigned j = 0; j < 10; j++) {
      snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.linear.weight", i,
               linears[j]);
      TRY(bind(e, name, i + 1, 2, ins[j], outs[j], 0));
      for (unsigned k = 0; k < 4; k++) {
        snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.%s", i,
                 linears[j], limits[k]);
        TRY(bind(e, name, i + 1, 0, 1, 0, 0));
      }
    }
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.self_attn.relative_k_proj.weight", i);
    TRY(bind(e, name, i + 1, 2, 1024, 1024, 0));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.self_attn.per_dim_scale", i);
    TRY(bind(e, name, i + 1, 1, 128, 0, 0));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.lconv1d.depthwise_conv1d.weight", i);
    TRY(bind(e, name, i + 1, 3, 5, 1, 1024));
  }
  size_t count = 0;
  for (size_t i = 0; i < ds4_bundle_tensor_count(b); i++) {
    uint32_t c = ds4_bundle_tensor(b, i)->component;
    if (c == e->audio || c == e->projection)
      count++;
  }
  if (count != e->count) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  for (uint32_t g = 0; g < GROUPS; g++) {
    size_t indices[1024], used = 0;
    for (uint32_t j = 0; j < e->count; j++)
      if (e->bindings[j].group == g)
        indices[used++] = e->bindings[j].index;
    TRY(ds4_numeric_weight_group_selected(e->groups + g, b,
                                         g == 13 ? e->projection : e->audio,
                                         indices, used, n));
  }
  TRY(ds4_weight_cache_init(&e->cache, e->groups, GROUPS));
  ds4_runtime_descriptor descriptor = {DS4_SUITE_ABI,
                                       ds4_bundle_model(b),
                                       ds4_bundle_revision(b),
                                       ds4_bundle_profile(b),
                                       "audio",
                                       DS4_OPERATION_EMBED,
                                       DS4_MODALITY_AUDIO,
                                       8192,
                                       512};
  ds4_runtime_model_ops ops = {activate, NULL};
  TRY(ds4_runtime_model_create(runtime, &descriptor, &ops, e, &e->model));
  *out = e;
  e = NULL;
cleanup:
  ds4_runtime_scope_leave(&scope);
  if (e)
    ds4_gemma_audio_close(e);
  return status;
}
ds4_status ds4_gemma_audio_estimate(ds4_gemma_audio *e,
                                    const ds4_gemma_media *media,
                                    ds4_runtime_estimate *out) {
  if (!e || !media || !out)
    return DS4_STATUS_INVALID;
  ds4_gemma_media_view v = ds4_gemma_media_data(media);
  if (!v.audio_frames || v.audio_frames > 3000 || !v.audio_features ||
      !v.audio_mask || !v.audio_soft_tokens || v.audio_soft_tokens > 750)
    return DS4_STATUS_INVALID;
  /* Two SSCP convolutions, 12 conformer blocks, small causal attention and
   * detached projected tokens. cuDNN workspace has an explicit margin. */
  *out = (ds4_runtime_estimate){
      .known = true,
      .device_bytes = (uint64_t)v.audio_frames * 300000 + (64ull << 20),
      .host_bytes = (uint64_t)v.audio_frames * 32768 + (1ull << 20)};
  out->device_bytes += ds4_weight_cache_missing(&e->cache);
  return DS4_STATUS_OK;
}

static ds4_status layer_norm(ds4_gemma_audio *e, uint32_t layer,
                             const char *part, ds4_num_view y, ds4_num_view x) {
  char name[256];
  snprintf(name, sizeof(name), "audio_tower.layers.%u.%s.weight", layer, part);
  return norm(e, name, y, x, 1024);
}
static ds4_status feed_forward(ds4_gemma_audio *e, uint32_t layer,
                               unsigned which, uint32_t n, ds4_num_view x,
                               ds4_num_view y, ds4_num_view wide,
                               ds4_num_view scratch) {
  ds4_status status;
  char part[96];
  snprintf(part, sizeof(part), "feed_forward%u.pre_layer_norm", which);
  TRY(layer_norm(e, layer, part, y, x));
  snprintf(part, sizeof(part), "feed_forward%u.ffw_layer_1", which);
  TRY(clipped(e, layer, part, wide, y, scratch, n, 1024, 4096, AUDIO_ROWS));
  TRY(ds4_num_point(e->num, wide, wide, NONE, DS4_NUM_SILU, 0));
  snprintf(part, sizeof(part), "feed_forward%u.ffw_layer_2", which);
  TRY(clipped(e, layer, part, y, wide, scratch, n, 4096, 1024, AUDIO_ROWS));
  snprintf(part, sizeof(part), "feed_forward%u.post_layer_norm", which);
  TRY(layer_norm(e, layer, part, y, y));
  TRY(ds4_num_point(e->num, y, y, NONE, DS4_NUM_SCALE, 0.5f));
  TRY(ds4_num_point(e->num, x, x, y, DS4_NUM_ADD, 0));
cleanup:
  return status;
}

/* Preserve the official blocked GEMM geometry, including the relative shift.
 * FP32 scores are small (8 * ceil(T/12) * 12 * 24), so a bounded host staging
 * buffer can express the shift without introducing a model-specific GPU API. */
static ds4_status attention(ds4_gemma_audio *e, ds4_num_view out,
                            ds4_num_view q, ds4_num_view k, ds4_num_view v,
                            ds4_num_view relative, uint32_t n, uint32_t valid,
                            uint32_t layer, const ds4_gemma_observer *o) {
  ds4_status status = DS4_STATUS_OK;
  work w = {.num = e->num};
  ds4_host_buffer host = {0}, indices = {0}, bounds = {0};
  uint32_t blocks = (n + 11) / 12, p = blocks * 12, batch = 8 * blocks;
  uint64_t scores = (uint64_t)batch * 12 * 24, bdcount = (uint64_t)8 * p * 13;
  TRY(ds4_host_alloc(e->resources, e->owner, (scores + bdcount) * 4, &host));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)batch * 24 * 4,
                     &indices));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)batch * 12 * 8,
                     &bounds));
  ds4_num_view qp, kp, vp, rp, ac, bd, prob, result, ordered, kt;
#define A(v, c) TRY(allocate(&w, (c), DS4_NUM_F32, &(v)))
  A(qp, (uint64_t)8 * p * 128);
  A(kp, (uint64_t)batch * 24 * 128);
  A(vp, (uint64_t)batch * 24 * 128);
  A(rp, 8 * 13 * 128);
  A(ac, scores);
  A(bd, bdcount);
  A(prob, scores);
  A(result, (uint64_t)8 * p * 128);
  A(ordered, (uint64_t)p * 1024);
  A(kt, (uint64_t)batch * 24 * 128);
  uint32_t *idx = indices.data, *range = bounds.data;
  for (uint32_t h = 0; h < 8; h++)
    for (uint32_t b = 0; b < blocks; b++)
      for (uint32_t r = 0; r < 12; r++) {
        uint32_t t = b * 12 + r;
        idx[(h * blocks + b) * 12 + r] = t < n ? t * 8 + h : UINT32_MAX;
      }
  TRY(ds4_num_pool(e->num, qp, q, idx, 8 * p, 1, 128, 1));
  for (uint32_t h = 0; h < 8; h++)
    for (uint32_t b = 0; b < blocks; b++)
      for (uint32_t c = 0; c < 24; c++) {
        int t = (int)b * 12 - 12 + (int)c;
        idx[(h * blocks + b) * 24 + c] =
            t >= 0 && t < (int)n ? t * 8 + h : UINT32_MAX;
      }
  TRY(ds4_num_pool(e->num, kp, k, idx, batch * 24, 1, 128, 1));
  TRY(ds4_num_pool(e->num, vp, v, idx, batch * 24, 1, 128, 1));
  ds4_num_reindex ix = {
      .rank = 3, .shape = {8, 13, 128}, .stride = {128, 1024, 1}};
  TRY(ds4_num_reorder(e->num, rp, relative, &ix));
  ix = (ds4_num_reindex){
      .rank = 3, .shape = {batch, 128, 24}, .stride = {24 * 128, 1, 128}};
  TRY(ds4_num_reorder(e->num, kt, kp, &ix));
  TRY(ds4_num_matmul_batch(e->num, ac, qp, kt, batch, 12, 128, 24, false));
  TRY(ds4_num_matmul_batch(e->num, bd, qp, rp, 8, p, 128, 13, true));
  float *score = host.data, *relscore = score + scores;
  TRY(ds4_num_read(e->num, ac, score, scores * 4));
  TRY(ds4_num_read(e->num, bd, relscore, bdcount * 4));
  for (uint32_t h = 0; h < 8; h++)
    for (uint32_t b = 0; b < blocks; b++)
      for (uint32_t r = 0; r < 12; r++) {
        uint64_t row = (h * blocks + b) * 12 + r;
        int qtime = b * 12 + r;
        int lo = qtime - 11, hi = qtime + 1;
        int begin = (int)b * 12 - 12;
        lo = lo < 0 ? 0 : lo;
        hi = hi > (int)valid ? (int)valid : hi;
        if (lo > hi || qtime >= (int)n)
          lo = hi = 0;
        range[2 * row] = lo < hi ? lo - begin : 0;
        range[2 * row + 1] = lo < hi ? hi - begin : 24;
        for (uint32_t c = 0; c < 24; c++) {
          int d = (int)c - (int)r;
          if (d >= 0 && d < 13)
            score[row * 24 + c] +=
                relscore[((uint64_t)h * p + b * 12 + r) * 13 + d];
          if (lo >= hi)
            score[row * 24 + c] = 0;
          score[row * 24 + c] /= 50.f;
        }
      }
  TRY(ds4_num_write(e->num, ac, score, scores * 4));
  TRY(ds4_num_point(e->num, ac, ac, NONE, DS4_NUM_TANH, 0));
  TRY(ds4_num_point(e->num, ac, ac, NONE, DS4_NUM_SCALE, 50));
  TRY(ds4_num_softmax(e->num, prob, ac, range, batch * 12, 24, 1));
  TRY(observe(e, o, "audio-probabilities", 0, layer, prob));
  TRY(ds4_num_matmul_batch(e->num, result, prob, vp, batch, 12, 24, 128,
                           false));
  ix = (ds4_num_reindex){
      .rank = 3, .shape = {p, 8, 128}, .stride = {128, (uint64_t)p * 128, 1}};
  TRY(ds4_num_reorder(e->num, ordered, result, &ix));
  TRY(ds4_num_copy(e->num, out, ds4_num_slice(ordered, 0, (uint64_t)n * 1024)));
cleanup: {
  ds4_status s = clear(&w);
  if (!status)
    status = s;
}
  ds4_host_free(&host);
  ds4_host_free(&indices);
  ds4_host_free(&bounds);
  return status;
#undef A
}

/* Construct the checkpoint's BF16 relative position table. Allocation and
 * cleanup use the same numeric workspace as the audio forward. */
static ds4_status relative_positions(ds4_gemma_audio *e, work *w,
                                     uint16_t *bits, ds4_num_view position) {
  ds4_status status;
  ds4_num_view phase, co, si;
  TRY(allocate(w, 13 * 512, DS4_NUM_BF16, &phase));
  TRY(allocate(w, 13 * 512, DS4_NUM_BF16, &co));
  TRY(allocate(w, 13 * 512, DS4_NUM_BF16, &si));
  /* inv_timescales is cast to BF16 when the pinned model dtype is selected. */
  for (uint32_t r = 0; r < 13; r++)
    for (uint32_t d = 0; d < 512; d++)
      bits[r * 512 + d] =
          bf16((12 - r) * from_bf16(bf16(expf(d * (-logf(10000.f) / 511.f)))));
  TRY(ds4_num_write(e->num, phase, bits, 13 * 512 * 2));
  TRY(ds4_num_point(e->num, co, phase, NONE, DS4_NUM_COS, 0));
  TRY(ds4_num_point(e->num, si, phase, NONE, DS4_NUM_SIN, 0));
  TRY(ds4_num_read(e->num, si, bits, 13 * 512 * 2));
  for (int r = 12; r >= 0; r--)
    memmove(bits + r * 1024, bits + r * 512, 512 * 2);
  for (uint32_t r = 0; r < 13; r++)
    TRY(ds4_num_read(e->num, ds4_num_slice(co, r * 512, 512),
                     bits + r * 1024 + 512, 1024));
  TRY(ds4_num_write(e->num, position, bits, 13 * 1024 * 2));
cleanup:
  return status;
}

static ds4_status media_forward(ds4_gemma_audio *e, ds4_gemma_media_view m,
                                const ds4_gemma_observer *o, uint16_t *output) {
  ds4_status status = DS4_STATUS_OK;
  work w = {.num = e->num};
  ds4_host_buffer host = {0};
  uint32_t t0 = m.audio_frames, t1 = (t0 + 1) / 2, n = (t1 + 1) / 2;
  uint64_t max = (uint64_t)t1 * 64 * 128;
  if (max < 13 * 1024)
    max = 13 * 1024;
  TRY(ds4_host_alloc(e->resources, e->owner, max * 2, &host));
  uint16_t *bits = host.data;
  uint32_t valid = 0;
  for (uint32_t i = 0; i < n; i++) {
    /* Processor masks are prefix-valid. Keep actual subsampled validity,
     * never infer soft token count from the padded tensor length. */
    bool live = m.audio_mask[i * 4] != 0;
    if (live && valid != i) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    valid += live;
  }
  if (valid != m.audio_soft_tokens) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  ds4_num_view input, conv, nhwc, flat, x, y, z, wide, scratch, gate, sig,
      channels, conved, q, k, v, attn, q32, k32, v32, a32, scale, scale32,
      expanded, rel, rel32, position, projected;
#define ALLOC(v, c, t) TRY(allocate(&w, (c), (t), &(v)))
  ALLOC(input, (uint64_t)t0 * 128, DS4_NUM_BF16);
  ALLOC(conv, max, DS4_NUM_BF16);
  ALLOC(nhwc, max, DS4_NUM_BF16);
  ALLOC(flat, (uint64_t)n * 1024, DS4_NUM_BF16);
  for (uint64_t i = 0; i < (uint64_t)t0 * 128; i++)
    bits[i] = bf16(m.audio_mask[i / 128] ? m.audio_features[i] : 0);
  TRY(ds4_num_write(e->num, input, bits, (uint64_t)t0 * 128 * 2));
  for (uint32_t l = 0; l < 2; l++) {
    uint32_t ti = l ? t1 : t0, to = l ? n : t1;
    uint32_t fi = l ? 64 : 128, fo = fi / 2, ci = l ? 128 : 1,
             cout = l ? 32 : 128;
    char name[256];
    ds4_num_view weight_view;
    snprintf(name, sizeof(name),
             "audio_tower.subsample_conv_projection.layer%u.conv.weight", l);
    TRY(weight(e, name, &weight_view));
    ds4_num_convolution c = {.batch = 1,
                             .in_channels = ci,
                             .out_channels = cout,
                             .groups = 1,
                             .input = {1, ti, fi},
                             .kernel = {1, 3, 3},
                             .stride = {1, 2, 2},
                             .dilation = {1, 1, 1},
                             .pad_before = {0, 1, 1},
                             .pad_after = {0, 1, 1}};
    ds4_num_view src =
        l ? ds4_num_slice(nhwc, 0, (uint64_t)ci * ti * fi) : input;
    TRY(ds4_num_conv(e->num, conv, src, weight_view, NONE, &c));
    ds4_num_reindex ix = {.rank = 3,
                          .shape = {to, fo, cout},
                          .stride = {fo, 1, (uint64_t)to * fo}};
    uint64_t count = (uint64_t)to * fo * cout;
    TRY(ds4_num_reorder(e->num, nhwc, ds4_num_slice(conv, 0, count), &ix));
    TRY(observe(e, o, "audio-sscp-conv", 0, l, ds4_num_slice(nhwc, 0, count)));
    snprintf(name, sizeof(name),
             "audio_tower.subsample_conv_projection.layer%u.norm.weight", l);
    TRY(weight(e, name, &weight_view));
    TRY(ds4_num_norm(e->num, ds4_num_slice(nhwc, 0, count),
                     ds4_num_slice(nhwc, 0, count), weight_view, NONE, cout,
                     1e-6f, DS4_NUM_LAYER, DS4_NUM_SCALE_DIRECT, false));
    TRY(ds4_num_point(e->num, ds4_num_slice(nhwc, 0, count),
                      ds4_num_slice(nhwc, 0, count), NONE, DS4_NUM_RELU, 0));
    TRY(observe(e, o, "audio-subsample", 0, l, ds4_num_slice(nhwc, 0, count)));
    if (!l) {
      /* Mask again before the second convolution, in [time,freq,channel]. */
      TRY(ds4_num_read(e->num, ds4_num_slice(nhwc, 0, count), bits, count * 2));
      for (uint32_t t = 0; t < to; t++)
        if (!m.audio_mask[t * 2])
          memset(bits + (uint64_t)t * fo * cout, 0, (uint64_t)fo * cout * 2);
      TRY(ds4_num_write(e->num, ds4_num_slice(conv, 0, count), bits,
                        count * 2));
      ix = (ds4_num_reindex){.rank = 3,
                             .shape = {cout, to, fo},
                             .stride = {1, (uint64_t)fo * cout, cout}};
      TRY(ds4_num_reorder(e->num, nhwc, ds4_num_slice(conv, 0, count), &ix));
    } else {
      TRY(ds4_num_copy(e->num, flat, ds4_num_slice(nhwc, 0, count)));
    }
  }
  /* Keep flat rows separate from the SSCP workspace. */
  ALLOC(x, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(y, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(z, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(wide, (uint64_t)n * 4096, DS4_NUM_BF16);
  ALLOC(scratch, (uint64_t)n * 4096, DS4_NUM_BF16);
  ALLOC(gate, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(sig, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(channels, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(conved, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(q, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(k, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(v, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(attn, (uint64_t)n * 1024, DS4_NUM_BF16);
  ALLOC(q32, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(k32, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(v32, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(a32, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(scale, 128, DS4_NUM_BF16);
  ALLOC(scale32, 128, DS4_NUM_F32);
  ALLOC(expanded, (uint64_t)n * 1024, DS4_NUM_F32);
  ALLOC(rel, 13 * 1024, DS4_NUM_BF16);
  ALLOC(rel32, 13 * 1024, DS4_NUM_F32);
  ALLOC(position, 13 * 1024, DS4_NUM_BF16);
  TRY(relative_positions(e, &w, bits, position));
  TRY(linear(e,
             "audio_tower.subsample_conv_projection.input_proj_linear.weight",
             x, flat, n, 1024, 1024));
  TRY(observe(e, o, "audio-input", 0, 0, x));
  for (uint32_t l = 0; l < LAYERS; l++) {
    char name[256];
    TRY(feed_forward(e, l, 1, n, x, y, wide, scratch));
    TRY(observe(e, o, "audio-ff1", 0, l, x));
    TRY(layer_norm(e, l, "norm_pre_attn", y, x));
    TRY(clipped(e, l, "self_attn.q_proj", q, y, scratch, n, 1024, 1024,
                AUDIO_ROWS));
    TRY(clipped(e, l, "self_attn.k_proj", k, y, scratch, n, 1024, 1024,
                AUDIO_ROWS));
    TRY(clipped(e, l, "self_attn.v_proj", v, y, scratch, n, 1024, 1024,
                AUDIO_ROWS));
    TRY(ds4_num_point(e->num, q32, q, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_point(e->num, k32, k, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_point(e->num, v32, v, NONE, DS4_NUM_CAST, 0));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.self_attn.per_dim_scale", l);
    ds4_num_view per;
    TRY(weight(e, name, &per));
    TRY(ds4_num_point(e->num, scale, per, NONE, DS4_NUM_SOFTPLUS, 0));
    TRY(ds4_num_point(e->num, scale32, scale, NONE, DS4_NUM_CAST, 0));
    ds4_num_reindex ix = {
        .rank = 2, .shape = {(uint64_t)n * 8, 128}, .stride = {0, 1}};
    TRY(ds4_num_reorder(e->num, expanded, scale32, &ix));
    TRY(ds4_num_point(e->num, q32, q32, NONE, DS4_NUM_SCALE,
                      (float)(1 / sqrt(128.) / log(2.))));
    TRY(ds4_num_point(e->num, q32, q32, expanded, DS4_NUM_MUL, 0));
    TRY(ds4_num_point(e->num, k32, k32, NONE, DS4_NUM_SCALE,
                      (float)(log(1 + exp(1.)) / log(2.))));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.self_attn.relative_k_proj.weight", l);
    TRY(linear(e, name, rel, position, 13, 1024, 1024));
    TRY(ds4_num_point(e->num, rel32, rel, NONE, DS4_NUM_CAST, 0));
    if (!l) {
      TRY(observe(e, o, "audio-position", 0, 0, position));
      TRY(observe(e, o, "audio-relative", 0, 0, rel));
      TRY(observe(e, o, "audio-query", 0, 0, q32));
      TRY(observe(e, o, "audio-key", 0, 0, k32));
    }
    TRY(attention(e, a32, q32, k32, v32, rel32, n, valid, l, o));
    TRY(ds4_num_point(e->num, attn, a32, NONE, DS4_NUM_CAST, 0));
    if (!l)
      TRY(observe(e, o, "audio-context", 0, 0, attn));
    TRY(clipped(e, l, "self_attn.post", y, attn, scratch, n, 1024, 1024,
                AUDIO_ROWS));
    TRY(layer_norm(e, l, "norm_post_attn", y, y));
    TRY(ds4_num_point(e->num, x, x, y, DS4_NUM_ADD, 0));
    TRY(observe(e, o, "audio-attention", 0, l, x));
    TRY(layer_norm(e, l, "lconv1d.pre_layer_norm", y, x));
    ds4_num_view doubled = ds4_num_slice(wide, 0, (uint64_t)n * 2048);
    TRY(clipped(e, l, "lconv1d.linear_start", doubled, y, scratch, n, 1024,
                2048, AUDIO_ROWS));
    ix = (ds4_num_reindex){.rank = 2, .shape = {n, 1024}, .stride = {2048, 1}};
    TRY(ds4_num_reorder(e->num, y, doubled, &ix));
    TRY(ds4_num_reorder(
        e->num, z, ds4_num_slice(doubled, 1024, doubled.count - 1024), &ix));
    /* GLU is one BF16 operation: avoid rounding sigmoid before the multiply. */
    TRY(ds4_num_point(e->num, gate, y, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_point(e->num, sig, z, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_point(e->num, sig, sig, NONE, DS4_NUM_SIGMOID, 0));
    TRY(ds4_num_point(e->num, gate, gate, sig, DS4_NUM_MUL, 0));
    TRY(ds4_num_point(e->num, y, gate, NONE, DS4_NUM_CAST, 0));
    if (!l)
      TRY(observe(e, o, "audio-glu", 0, 0, y));
    ix = (ds4_num_reindex){.rank = 2, .shape = {1024, n}, .stride = {1, 1024}};
    TRY(ds4_num_reorder(e->num, channels, y, &ix));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.lconv1d.depthwise_conv1d.weight", l);
    ds4_num_view cw;
    TRY(weight(e, name, &cw));
    ds4_num_convolution c = {.batch = 1,
                             .in_channels = 1024,
                             .out_channels = 1024,
                             .groups = 1024,
                             .input = {1, 1, n},
                             .kernel = {1, 1, 5},
                             .stride = {1, 1, 1},
                             .dilation = {1, 1, 1},
                             .pad_before = {0, 0, 4}};
    TRY(ds4_num_conv(e->num, conved, channels, cw, NONE, &c));
    ix = (ds4_num_reindex){.rank = 2, .shape = {n, 1024}, .stride = {1, n}};
    TRY(ds4_num_reorder(e->num, y, conved, &ix));
    if (!l)
      TRY(observe(e, o, "audio-depthwise", 0, 0, y));
    snprintf(name, sizeof(name),
             "audio_tower.layers.%u.lconv1d.conv_norm.weight", l);
    TRY(weight(e, name, &cw));
    TRY(ds4_num_channel_rms(e->num, y, conved, cw, 1024, n, 1e-6f));
    if (!l)
      TRY(observe(e, o, "audio-convnorm", 0, 0, y));
    TRY(ds4_num_point(e->num, y, y, NONE, DS4_NUM_SILU, 0));
    if (!l)
      TRY(observe(e, o, "audio-silu", 0, 0, y));
    TRY(clipped(e, l, "lconv1d.linear_end", z, y, scratch, n, 1024, 1024,
                AUDIO_TRANSPOSED_ROWS));
    TRY(observe(e, o, "audio-conv-end", 0, l, z));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    TRY(observe(e, o, "audio-convolution", 0, l, x));
    TRY(feed_forward(e, l, 2, n, x, y, wide, scratch));
    TRY(layer_norm(e, l, "norm_out", x, x));
    TRY(observe(e, o, "audio-layer", 0, l, x));
  }
  /* Reuse the FFN buffer for the 1536-wide output projection. */
  projected = ds4_num_slice(wide, 0, (uint64_t)n * 1536);
  ds4_num_view ow, ob;
  TRY(weight(e, "audio_tower.output_proj.weight", &ow));
  TRY(weight(e, "audio_tower.output_proj.bias", &ob));
  TRY(ds4_num_linear(e->num, projected, x, ow, ob, n, 1024, 1536));
  TRY(observe(e, o, "audio-output", 0, 12, projected));
  TRY(ds4_num_norm(e->num, projected, projected, NONE, NONE, 1536, 1e-6f,
                   DS4_NUM_RMS, DS4_NUM_SCALE_NONE, false));
  ds4_num_view features = ds4_num_slice(y, 0, (uint64_t)n * 512);
  TRY(linear(e, "embed_audio.embedding_projection.weight", features, projected,
             n, 1536, 512));
  TRY(observe(e, o, "audio-projected", 0, 12, features));
  TRY(ds4_num_read(e->num, ds4_num_slice(features, 0, (uint64_t)valid * 512),
                   output, (uint64_t)valid * 1024));
cleanup: {
  ds4_status s = clear(&w);
  if (!status)
    status = s;
}
  ds4_host_free(&host);
  return status;
#undef ALLOC
}

ds4_gemma_audio_features_view
ds4_gemma_audio_features_data(const ds4_gemma_audio_features *f) {
  return f ? f->view : (ds4_gemma_audio_features_view){0};
}
void ds4_gemma_audio_features_free(ds4_gemma_audio_features *f) {
  if (!f)
    return;
  ds4_host_free(&f->tokens);
  ds4_host_buffer account = f->account;
  ds4_host_free(&account);
}
ds4_status ds4_gemma_audio_encode(ds4_gemma_audio *e,
                                  const ds4_gemma_media *media,
                                  const ds4_gemma_observer *o,
                                  ds4_gemma_audio_features **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !media)
    return DS4_STATUS_INVALID;
  ds4_status status;
  ds4_runtime_scope scope = {0};
  ds4_reservation *lease = NULL;
  ds4_reservation_scope working = {0};
  ds4_gemma_audio_features *f = NULL;
  ds4_host_buffer account = {0};
  TRY(ds4_runtime_model_enter(e->model, &scope));
  ds4_runtime_estimate estimate;
  TRY(ds4_gemma_audio_estimate(e, media, &estimate));
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE] =
                                          estimate.device_bytes,
                                      [DS4_MEMORY_HOST] = estimate.host_bytes};
  TRY(ds4_resources_reserve(e->resources, e->owner, bytes, &lease));
  working = ds4_reservation_enter(lease);
  ds4_gemma_media_view m = ds4_gemma_media_data(media);
  TRY(ds4_host_alloc(e->resources, e->owner, sizeof(*f), &account));
  f = account.data;
  f->account = account;
  TRY(ds4_host_alloc(e->resources, e->owner,
                     (uint64_t)m.audio_soft_tokens * 1024, &f->tokens));
  f->view =
      (ds4_gemma_audio_features_view){f->tokens.data, m.audio_soft_tokens};
  TRY(media_forward(e, m, o, f->tokens.data));
cleanup:
  ds4_reservation_leave(&working);
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status)
    ds4_gemma_audio_features_free(f);
  else
    *out = f;
  ds4_runtime_scope_leave(&scope);
  return status;
}
