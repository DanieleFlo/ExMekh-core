/* Projection-only PLE and bidirectional encoder semantics follow the frozen
 * Transformers d4fcbccd reference (Apache-2.0); see the T11/T12 provenance.
 * This graph uses the suite's qualified C numeric API, never causal LLM KV. */
#include "ds4_embedding_gemma2_internal.h"
#include "ds4_gemma_tokenizer.h"
#include "ds4_host_memory.h"
#include "ds4_numeric_weights.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
#define NONE ((ds4_num_view){0})
struct ds4_embedding_gemma2 {
  ds4_bundle *bundle;
  ds4_numeric *num;
  ds4_runtime_model *model;
  gemma_tokenizer *tokenizer;
  const gemma_config *config;
  ds4_weight_group *groups[25];
  size_t indices[512];
  uint32_t group[512], count, component;
  size_t embedding;
  ds4_resource *account;
};
struct ds4_gemma_input {
  uint32_t **tokens, *lengths, count;
  ds4_resource *account;
  ds4_resources *resources;
  uint64_t owner;
  ds4_host_buffer *visual;
};
struct ds4_gemma_result {
  float *vectors;
  uint32_t count, dimensions;
  ds4_resource *account;
};
uint64_t ds4_embedding_gemma2_resident_weights(const ds4_embedding_gemma2 *e) {
  uint64_t bytes = 0;
  if (!e)
    return 0;
  for (unsigned i = 0; i < 25; i++) {
    uint64_t n = ds4_weight_group_bytes(e->groups[i]);
    if (ds4_weight_group_resident(e->groups[i]))
      bytes += n;
  }
  return bytes;
}
static ds4_status bind(ds4_embedding_gemma2 *e, const char *name,
                       uint32_t group, uint32_t rank, uint64_t a, uint64_t b) {
  for (size_t i = 0; i < ds4_bundle_tensor_count(e->bundle); i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, i);
    if (t->component != e->component || strcmp(t->name, name))
      continue;
    if (e->count == 512 || t->type != DS4_NUM_BF16 || t->block_elements != 1 ||
        t->block_bytes != 2 || t->rank != rank || t->shape[0] != a ||
        t->stride[0] != 2 ||
        (rank == 2 && (t->shape[1] != b || t->stride[1] != 2 * a)))
      return DS4_STATUS_UNSUPPORTED;
    for (uint32_t j = 0; j < e->count; j++)
      if (e->indices[j] == i)
        return DS4_STATUS_INVALID;
    e->indices[e->count] = i;
    e->group[e->count++] = group;
    if (!strcmp(name, "language_model.embed_tokens.weight"))
      e->embedding = i;
    return DS4_STATUS_OK;
  }
  return DS4_STATUS_INVALID;
}
static ds4_status weight(ds4_embedding_gemma2 *e, const char *name,
                         ds4_num_view *v) {
  for (uint32_t i = 0; i < e->count; i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, e->indices[i]);
    if (strcmp(t->name, name))
      continue;
    ds4_weight_group *g = e->groups[e->group[i]];
    ds4_status s = ds4_weight_group_load(g);
    const void *storage;
    uint64_t off;
    if (!s)
      s = ds4_weight_group_tensor(g, name, &storage, &off);
    if (!s)
      *v = ds4_numeric_weight_view(storage, off, t->bytes / 2, DS4_NUM_BF16);
    return s;
  }
  return DS4_STATUS_INVALID;
}
static ds4_status linear(ds4_embedding_gemma2 *e, const char *name,
                         ds4_num_view y, ds4_num_view x, uint32_t rows,
                         uint32_t in, uint32_t out) {
  ds4_num_view w;
  ds4_status s = weight(e, name, &w);
  return s ? s : ds4_num_linear(e->num, y, x, w, NONE, rows, in, out);
}
static ds4_status norm(ds4_embedding_gemma2 *e, const char *name,
                       ds4_num_view y, ds4_num_view x, uint32_t width) {
  ds4_num_view scale = NONE;
  ds4_status s = name ? weight(e, name, &scale) : DS4_STATUS_OK;
  return s ? s
           : ds4_num_norm(
                 e->num, y, x, scale, NONE, width, e->config->eps, DS4_NUM_RMS,
                 name ? DS4_NUM_SCALE_DIRECT : DS4_NUM_SCALE_NONE, false);
}
static ds4_status activate(void *p) {
  return ds4_numeric_fence(((ds4_embedding_gemma2 *)p)->num);
}
ds4_status ds4_embedding_gemma2_release_weights(ds4_embedding_gemma2 *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(e->model, &scope);
  if (s)
    return s;
  for (uint32_t i = 0; i < 25 && !s; i++)
    s = ds4_weight_group_release(e->groups[i]);
  ds4_runtime_scope_leave(&scope);
  return s;
}
ds4_status ds4_embedding_gemma2_close(ds4_embedding_gemma2 *e) {
  if (!e)
    return DS4_STATUS_OK;
  ds4_status s =
      e->model ? ds4_embedding_gemma2_release_weights(e) : DS4_STATUS_OK;
  if (s)
    return s;
  if (e->model && (s = ds4_runtime_model_close(e->model)))
    return s;
  e->model = NULL;
  for (uint32_t i = 0; i < 25; i++) {
    if ((s = ds4_weight_group_close(e->groups[i])))
      return s;
    e->groups[i] = NULL;
  }
  gemma_tokenizer_close(e->tokenizer);
  ds4_resource *r = e->account;
  free(e);
  if (r)
    ds4_resource_release(r);
  return DS4_STATUS_OK;
}
ds4_status ds4_embedding_gemma2_open(ds4_embedding_gemma2 **out, ds4_bundle *b,
                                     ds4_numeric *n, ds4_runtime *rt,
                                     const char *asset) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!b || !n || !rt || !asset)
    return DS4_STATUS_INVALID;
  if (strcmp(ds4_bundle_model(b), "google/embeddinggemma-2") ||
      strcmp(ds4_bundle_revision(b),
             "914f7f89142e33e77833254d9c9b90c3cef7303b") ||
      strcmp(ds4_bundle_profile(b), "bf16-native-v1"))
    return DS4_STATUS_UNSUPPORTED;
  ds4_embedding_gemma2 *e = calloc(1, sizeof(*e));
  if (!e)
    return DS4_STATUS_NOMEM;
  e->bundle = b;
  e->num = n;
  e->component = UINT32_MAX;
  ds4_status status = DS4_STATUS_OK;
  ds4_runtime_scope scope = {0};
  TRY(ds4_runtime_begin_update(rt, &scope));
  for (size_t i = 0; i < ds4_bundle_component_count(b); i++) {
    const ds4_bundle_component *c = ds4_bundle_component_at(b, i);
    if (!strcmp(c->id, "text") &&
        !strcmp(c->revision, ds4_bundle_revision(b)) &&
        !strcmp(c->compute_dtype, "bfloat16"))
      e->component = i;
  }
  if (e->component == UINT32_MAX) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  uint8_t hashes[96];
  uint32_t found = 0;
  const char *names[] = {
      "google/embeddinggemma-2/tokenizer.json",
      "google/embeddinggemma-2/config.json",
      "google/embeddinggemma-2/config_sentence_transformers.json"};
  for (size_t i = 0; i < ds4_bundle_file_count(b); i++)
    for (uint32_t j = 0; j < 3; j++) {
      const ds4_bundle_file *f = ds4_bundle_file_at(b, i);
      if (!strcmp(f->path, names[j])) {
        memcpy(hashes + j * 32, f->sha256, 32);
        found |= 1u << j;
      }
    }
  if (found != 7) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  TRY(gemma_tokenizer_open(&e->tokenizer, asset, hashes));
  e->config = gemma_tokenizer_config(e->tokenizer);
  TRY(bind(e, "language_model.embed_tokens.weight", 0, 2, 512, 262144));
  TRY(bind(e, "language_model.embedding_projection.weight", 0, 2, 512, 768));
  TRY(bind(e, "language_model.norm.weight", 0, 1, 512, 0));
  TRY(bind(e, "language_model.ple.per_layer_model_projection.weight", 0, 2, 512,
           12288));
  TRY(bind(e, "language_model.ple.per_layer_projection_norm.weight", 0, 1, 512,
           0));
  for (uint32_t i = 0; i < 24; i++) {
    const gemma_layer *l = e->config->layers + i;
    char key[256];
    const char *norms[] = {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "ple_block.post_per_layer_input_norm.weight"};
    for (uint32_t j = 0; j < 5; j++) {
      snprintf(key, sizeof(key), "language_model.layers.%u.%s", i, norms[j]);
      TRY(bind(e, key, i + 1, 1, 512, 0));
    }
    const char *linears[] = {"self_attn.q_proj.weight",
                             "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight",
                             "self_attn.o_proj.weight",
                             "mlp.gate_proj.weight",
                             "mlp.up_proj.weight",
                             "mlp.down_proj.weight",
                             "ple_block.per_layer_input_gate.weight",
                             "ple_block.per_layer_projection.weight"};
    uint32_t ins[] = {512, 512, 512, 4 * l->dim, 512, 512, 2048, 512, 512};
    uint32_t outs[] = {
        4 * l->dim, l->kv * l->dim, l->kv * l->dim, 512, 2048, 2048, 512, 512,
        512};
    for (uint32_t j = 0; j < 9; j++) {
      snprintf(key, sizeof(key), "language_model.layers.%u.%s", i, linears[j]);
      TRY(bind(e, key, i + 1, 2, ins[j], outs[j]));
    }
    for (uint32_t j = 0; j < 2; j++) {
      snprintf(key, sizeof(key),
               "language_model.layers.%u.self_attn.%c_norm.weight", i,
               j ? 'k' : 'q');
      TRY(bind(e, key, i + 1, 1, l->dim, 0));
    }
    snprintf(key, sizeof(key), "language_model.layers.%u.layer_scalar", i);
    TRY(bind(e, key, i + 1, 1, 1, 0));
  }
  size_t total = 0;
  for (size_t i = 0; i < ds4_bundle_tensor_count(b); i++)
    if (ds4_bundle_tensor(b, i)->component == e->component)
      total++;
  if (total != e->count) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  ds4_weight_backend backend = ds4_numeric_weight_backend(n);
  for (uint32_t g = 0; g < 25; g++) {
    size_t ids[512], count = 0;
    for (uint32_t j = 0; j < e->count; j++)
      if (e->group[j] == g && e->indices[j] != e->embedding)
        ids[count++] = e->indices[j];
    TRY(ds4_weight_group_create_selected(&e->groups[g], b, e->component, ids,
                                         count, &backend));
  }
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(n, &owner);
  TRY(ds4_resource_register_host(
      rs, owner, sizeof(*e) + gemma_tokenizer_bytes(e->tokenizer),
      &e->account));
  ds4_runtime_descriptor d = {DS4_SUITE_ABI,
                              ds4_bundle_model(b),
                              ds4_bundle_revision(b),
                              ds4_bundle_profile(b),
                              "text",
                              DS4_OPERATION_EMBED,
                              DS4_MODALITY_TEXT,
                              8192,
                              768};
  ds4_runtime_model_ops ops = {activate, NULL};
  TRY(ds4_runtime_model_create(rt, &d, &ops, e, &e->model));
  *out = e;
  e = NULL;
cleanup:
  ds4_runtime_scope_leave(&scope);
  if (e)
    ds4_embedding_gemma2_close(e);
  return status;
}
void ds4_gemma_input_free(ds4_gemma_input *in) {
  if (in) {
    if (in->tokens)
      for (uint32_t i = 0; i < in->count; i++)
        free(in->tokens[i]);
    free(in->tokens);
    free(in->lengths);
    if (in->visual)
      for (uint32_t i = 0; i < in->count; i++)
        ds4_host_free(in->visual + i);
    free(in->visual);
    ds4_resource *r = in->account;
    free(in);
    if (r)
      ds4_resource_release(r);
  }
}
ds4_status ds4_gemma_prepare(ds4_embedding_gemma2 *e,
                             const ds4_gemma_text *items, size_t count,
                             ds4_gemma_input **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !items || !count || count > 1024)
    return DS4_STATUS_INVALID;
  ds4_gemma_input *in = calloc(1, sizeof(*in));
  if (!in)
    return DS4_STATUS_NOMEM;
  in->count = count;
  in->tokens = calloc(count, sizeof(uint32_t *));
  in->lengths = calloc(count, 4);
  in->visual = calloc(count, sizeof(*in->visual));
  ds4_status status = DS4_STATUS_NOMEM;
  ds4_reservation *temporary = NULL;
  ds4_reservation_scope staging = {0};
  if (!in->tokens || !in->lengths || !in->visual)
    goto cleanup;
  uint64_t bytes =
      sizeof(*in) +
      count * (sizeof(uint32_t *) + sizeof(ds4_host_buffer) + 4 + 8192 * 4);
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(e->num, &owner);
  in->resources = rs;
  in->owner = owner;
  TRY(ds4_resource_register_host(rs, owner, bytes, &in->account));
  uint64_t max_text = 0;
  for (size_t i = 0; i < count; i++) {
    if (!items[i].text || items[i].task > DS4_GEMMA_SIMILARITY) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    size_t length = strnlen(items[i].text, 1024 * 1024 + 1);
    if (length > 1024 * 1024) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    if (length > max_text)
      max_text = length;
  }
  /* Character nodes, merge heap and prefix storage are bounded host staging,
   * separate from the persistent BOS/content/EOS arrays charged above. */
  uint64_t peak[DS4_MEMORY_COUNT] = {0};
  peak[DS4_MEMORY_HOST] = (max_text + 128) * 96 + 65536;
  TRY(ds4_resources_reserve(rs, owner, peak, &temporary));
  staging = ds4_reservation_enter(temporary);
  for (size_t i = 0; i < count; i++)
    TRY(gemma_tokenize(e->tokenizer, items[i].text, items[i].task,
                       &in->tokens[i], &in->lengths[i]));
cleanup:
  ds4_reservation_leave(&staging);
  if (temporary) {
    ds4_status s = ds4_reservation_cancel(temporary);
    if (!status)
      status = s;
  }
  if (status)
    ds4_gemma_input_free(in);
  else
    *out = in;
  return status;
}
ds4_status gemma_input_visual_clear(ds4_gemma_input *in) {
  if (!in)
    return DS4_STATUS_INVALID;
  for (uint32_t i = 0; i < in->count; i++)
    ds4_host_free(in->visual + i);
  return DS4_STATUS_OK;
}
ds4_status gemma_input_visual_validate(const ds4_gemma_input *in, size_t item,
                                       uint32_t image_tokens,
                                       uint32_t video_tokens,
                                       uint32_t audio_tokens) {
  if (!in || item >= in->count)
    return DS4_STATUS_INVALID;
  uint32_t images = 0, videos = 0, audio = 0;
  for (uint32_t i = 0; i < in->lengths[item]; i++) {
    images += in->tokens[item][i] == 258880;
    videos += in->tokens[item][i] == 258884;
    audio += in->tokens[item][i] == 258881;
  }
  return images == image_tokens && videos == video_tokens &&
                 audio == audio_tokens
             ? DS4_STATUS_OK
             : DS4_STATUS_INVALID;
}
ds4_status gemma_input_visual(ds4_gemma_input *in, size_t item,
                              gemma_visual_tokens features) {
  if ((features.image_tokens && !features.images) ||
      (features.video_tokens && !features.videos) ||
      (features.audio_tokens && !features.audio))
    return DS4_STATUS_INVALID;
  ds4_status s =
      gemma_input_visual_validate(in, item, features.image_tokens,
                                  features.video_tokens, features.audio_tokens);
  if (s)
    return s;
  ds4_host_buffer replacement = {0};
  s = ds4_host_alloc(in->resources, in->owner,
                     ((uint64_t)features.image_tokens + features.video_tokens +
                      features.audio_tokens) *
                         512 * 2,
                     &replacement);
  if (s)
    return s;
  uint32_t a = 0, b = 0, c = 0, row = 0;
  for (uint32_t i = 0; i < in->lengths[item]; i++) {
    const uint16_t *source =
        in->tokens[item][i] == 258880   ? features.images + (uint64_t)a++ * 512
        : in->tokens[item][i] == 258884 ? features.videos + (uint64_t)b++ * 512
        : in->tokens[item][i] == 258881 ? features.audio + (uint64_t)c++ * 512
                                        : NULL;
    if (source)
      memcpy((uint16_t *)replacement.data + (uint64_t)row++ * 512, source,
             1024);
  }
  ds4_host_free(in->visual + item);
  in->visual[item] = replacement;
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_input_tokens(const ds4_gemma_input *in, size_t i,
                                  const uint32_t **ids, uint32_t *count) {
  if (ids)
    *ids = NULL;
  if (count)
    *count = 0;
  if (!in || i >= in->count || !ids || !count)
    return DS4_STATUS_INVALID;
  *ids = in->tokens[i];
  *count = in->lengths[i];
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_estimate(ds4_embedding_gemma2 *e,
                              const ds4_gemma_input *in,
                              ds4_runtime_estimate *out) {
  if (!e || !in || !out)
    return DS4_STATUS_INVALID;
  uint32_t max = 0;
  for (uint32_t i = 0; i < in->count; i++)
    if (in->lengths[i] > max)
      max = in->lengths[i];
  *out = (ds4_runtime_estimate){
      .device_bytes = (uint64_t)max * 200000 + 64 * 1024 * 1024,
      .host_bytes =
          (uint64_t)max * 32768 + (uint64_t)in->count * 768 * 4 + 65536,
      .known = true};
  /* Full-query BF16 score/result rows replace the former 256-row tiles only
   * under the numeric backend's bounded score cap. Full layers use dim 512. */
  if (max > 256 && 4ull * max * max * 2 <= 128ull * 1024 * 1024)
    out->device_bytes += (max - 256) * 4ull * (max + 512) * 2;
  for (uint32_t i = 0; i < 25; i++)
    if (!ds4_weight_group_resident(e->groups[i]))
      out->device_bytes += ds4_weight_group_bytes(e->groups[i]);
  return DS4_STATUS_OK;
}
typedef struct {
  ds4_numeric *num;
  ds4_num_buffer *buffers[24];
  uint32_t count;
} work;
static ds4_status alloc_type(work *w, uint64_t count, uint32_t type,
                             ds4_num_view *v) {
  if (w->count == 24)
    return DS4_STATUS_INVALID;
  ds4_num_buffer *b = NULL;
  ds4_status s = ds4_num_alloc(w->num, count, type, &b);
  if (!s) {
    w->buffers[w->count++] = b;
    *v = ds4_num_data(b);
  }
  return s;
}
static ds4_status alloc(work *w, uint64_t count, ds4_num_view *v) {
  return alloc_type(w, count, DS4_NUM_BF16, v);
}
static ds4_status clear(work *w) {
  ds4_status s = 0;
  while (w->count) {
    ds4_status t = ds4_num_dispose(w->buffers[--w->count]);
    if (!s)
      s = t;
  }
  return s;
}
static ds4_status observe(ds4_embedding_gemma2 *e, const ds4_gemma_observer *o,
                          const char *stage, uint32_t item, uint32_t layer,
                          ds4_num_view v) {
  if (o && o->cancelled && o->cancelled(o->opaque))
    return DS4_STATUS_CANCELLED;
  ds4_status s = ds4_numeric_fence(e->num);
  if (!s && o && o->trace)
    s = o->trace(o->opaque, stage, item, layer, v);
  if (!s && o && o->cancelled && o->cancelled(o->opaque))
    s = DS4_STATUS_CANCELLED;
  return s;
}
static ds4_status forward(ds4_embedding_gemma2 *e, const uint32_t *tokens,
                          uint32_t n, uint32_t item, uint32_t dimensions,
                          const uint16_t *visual, const ds4_gemma_observer *o,
                          float *result) {
  ds4_status status = 0;
  work w = {.num = e->num};
  uint16_t *h = NULL;
  uint32_t *bounds = NULL;
  ds4_host_buffer host = {0}, mask = {0};
  uint64_t owner;
  ds4_resources *resources = ds4_numeric_resources(e->num, &owner);
  ds4_num_view x, y, z, ple, pi, q, k, v, qr, kr, attn, gate, up, co, si,
      projected, phase, trig;
  TRY(ds4_host_alloc(resources, owner, (uint64_t)n * 768 * 2, &host));
  TRY(ds4_host_alloc(resources, owner, (uint64_t)n * 8, &mask));
  h = host.data;
  bounds = mask.data;
  const uint16_t *table = ds4_bundle_tensor_data(e->bundle, e->embedding);
  for (uint32_t i = 0; i < n; i++) {
    uint32_t token = tokens[i];
    if (token == 258880 || token == 258884 || token == 258881) {
      if (!visual) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      token = 0; /* Match the oracle's placeholder PAD before scaled lookup. */
    }
    memcpy(h + (uint64_t)i * 512, table + (uint64_t)token * 512, 1024);
  }
  TRY(alloc(&w, (uint64_t)n * 512, &x));
  TRY(alloc(&w, (uint64_t)n * 512, &y));
  TRY(alloc(&w, (uint64_t)n * 512, &z));
  TRY(ds4_num_write(e->num, x, h, (uint64_t)n * 1024));
  TRY(ds4_num_point(e->num, x, x, NONE, DS4_NUM_SCALE, 22.625f));
  if (visual) {
    /* Vision projections already inhabit the scaled backbone embedding space.
     * Scatter after text scaling, before PLE; never scale soft tokens again. */
    TRY(ds4_num_read(e->num, x, h, (uint64_t)n * 1024));
    uint32_t row = 0;
    for (uint32_t i = 0; i < n; i++)
      if (tokens[i] == 258880 || tokens[i] == 258884 || tokens[i] == 258881)
        memcpy(h + (uint64_t)i * 512, visual + (uint64_t)row++ * 512, 1024);
    TRY(ds4_num_write(e->num, x, h, (uint64_t)n * 1024));
  }
  TRY(observe(e, o, "embeddings", item, 0, x));
  TRY(alloc(&w, (uint64_t)n * 12288, &ple));
  TRY(alloc(&w, (uint64_t)n * 512, &pi));
  TRY(linear(e, "language_model.ple.per_layer_model_projection.weight", ple, x,
             n, 512, 12288));
  TRY(ds4_num_point(e->num, ple, ple, NONE, DS4_NUM_SCALE,
                    0.04419417382415922f));
  TRY(norm(e, "language_model.ple.per_layer_projection_norm.weight", ple, ple,
           512));
  TRY(observe(e, o, "ple", item, 0, ple));
  TRY(alloc(&w, (uint64_t)n * 2048, &q));
  TRY(alloc(&w, (uint64_t)n * 512, &k));
  TRY(alloc(&w, (uint64_t)n * 512, &v));
  TRY(alloc(&w, (uint64_t)n * 2048, &qr));
  TRY(alloc(&w, (uint64_t)n * 512, &kr));
  TRY(alloc(&w, (uint64_t)n * 2048, &attn));
  TRY(alloc(&w, (uint64_t)n * 2048, &gate));
  TRY(alloc(&w, (uint64_t)n * 2048, &up));
  TRY(alloc(&w, (uint64_t)n * 256, &co));
  TRY(alloc(&w, (uint64_t)n * 256, &si));
  TRY(alloc_type(&w, (uint64_t)n * 256, DS4_NUM_F32, &phase));
  TRY(alloc_type(&w, (uint64_t)n * 256, DS4_NUM_F32, &trig));
  for (uint32_t i = 0; i < 24; i++) {
    const gemma_layer *l = e->config->layers + i;
    uint32_t dim = l->dim;
    char key[256];
    ds4_num_view qv = ds4_num_slice(q, 0, (uint64_t)n * 4 * dim),
                 kv = ds4_num_slice(k, 0, (uint64_t)n * l->kv * dim),
                 vv = ds4_num_slice(v, 0, (uint64_t)n * l->kv * dim),
                 qrv = ds4_num_slice(qr, 0, qv.count),
                 krv = ds4_num_slice(kr, 0, kv.count),
                 av = ds4_num_slice(attn, 0, qv.count),
                 cv = ds4_num_slice(co, 0, (uint64_t)n * dim / 2),
                 sv = ds4_num_slice(si, 0, cv.count),
                 pv = ds4_num_slice(phase, 0, cv.count),
                 tv = ds4_num_slice(trig, 0, cv.count);
#define NAME(s)                                                                \
  (snprintf(key, sizeof(key), "language_model.layers.%u.%s", i, s), key)
    TRY(norm(e, NAME("input_layernorm.weight"), y, x, 512));
    if (!i)
      TRY(observe(e, o, "backbone-norm", item, i, y));
    TRY(linear(e, NAME("self_attn.q_proj.weight"), qv, y, n, 512, 4 * dim));
    TRY(linear(e, NAME("self_attn.k_proj.weight"), kv, y, n, 512, l->kv * dim));
    TRY(linear(e, NAME("self_attn.v_proj.weight"), vv, y, n, 512, l->kv * dim));
    TRY(norm(e, NAME("self_attn.q_norm.weight"), qv, qv, dim));
    TRY(norm(e, NAME("self_attn.k_norm.weight"), kv, kv, dim));
    TRY(norm(e, NULL, vv, vv, dim));
    if (!i) {
      TRY(observe(e, o, "backbone-q", item, i, qv));
      TRY(observe(e, o, "backbone-k", item, i, kv));
      TRY(observe(e, o, "backbone-v", item, i, vv));
    }
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t d = 0; d < dim / 2; d++)
        ((float *)h)[(uint64_t)t * dim / 2 + d] = (float)t * l->frequency[d];
    /* Evaluate FP32 trigonometry on the compute backend. Host libm and CUDA
     * differ at rare BF16 ties on long positions; those ties amplify in PLE. */
    TRY(ds4_num_write(e->num, pv, h, pv.count * 4));
    TRY(ds4_num_point(e->num, tv, pv, NONE, DS4_NUM_COS, 0));
    TRY(ds4_num_point(e->num, cv, tv, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_point(e->num, tv, pv, NONE, DS4_NUM_SIN, 0));
    TRY(ds4_num_point(e->num, sv, tv, NONE, DS4_NUM_CAST, 0));
    TRY(ds4_num_rope(e->num, qrv, qv, cv, sv, 4, dim, false, true));
    TRY(ds4_num_rope(e->num, krv, kv, cv, sv, l->kv, dim, false, true));
    if (!i) {
      TRY(observe(e, o, "backbone-rope-q", item, i, qrv));
      TRY(observe(e, o, "backbone-rope-k", item, i, krv));
    }
    /* Symmetric local mask includes keys exactly +/- window. Full layers
     * have no causal mask. Ragged items never share an attention invocation. */
    for (uint32_t t = 0; t < n; t++) {
      bounds[2 * t] =
          l->local && t > e->config->window ? t - e->config->window : 0;
      uint32_t end = l->local ? t + e->config->window + 1 : n;
      bounds[2 * t + 1] = end < n ? end : n;
    }
    ds4_num_attention a = {.batch = 1,
                           .queries = n,
                           .keys = n,
                           .q_heads = 4,
                           .kv_heads = l->kv,
                           .dim = dim,
                           .scale = 1.f,
                           .round_scores = true,
                           .round_probabilities = true};
    TRY(ds4_num_attend(e->num, av, qrv, krv, vv, bounds, &a));
    if (!i)
      TRY(observe(e, o, "backbone-attention", item, i, av));
    TRY(linear(e, NAME("self_attn.o_proj.weight"), z, av, n, 4 * dim, 512));
    TRY(norm(e, NAME("post_attention_layernorm.weight"), z, z, 512));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    if (!i)
      TRY(observe(e, o, "backbone-attn-out", item, i, x));
    TRY(norm(e, NAME("pre_feedforward_layernorm.weight"), y, x, 512));
    TRY(linear(e, NAME("mlp.gate_proj.weight"), gate, y, n, 512, 2048));
    TRY(linear(e, NAME("mlp.up_proj.weight"), up, y, n, 512, 2048));
    TRY(ds4_num_point(e->num, gate, gate, NONE, DS4_NUM_GELU_TANH, 0));
    TRY(ds4_num_point(e->num, gate, gate, up, DS4_NUM_MUL, 0));
    TRY(linear(e, NAME("mlp.down_proj.weight"), z, gate, n, 2048, 512));
    TRY(norm(e, NAME("post_feedforward_layernorm.weight"), z, z, 512));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    TRY(linear(e, NAME("ple_block.per_layer_input_gate.weight"), y, x, n, 512,
               512));
    TRY(ds4_num_point(e->num, y, y, NONE, DS4_NUM_GELU_TANH, 0));
    ds4_num_reindex ix = {.rank = 2, .shape = {n, 512}, .stride = {12288, 1}};
    TRY(ds4_num_reorder(
        e->num, pi,
        ds4_num_slice(ple, (uint64_t)i * 512, ple.count - (uint64_t)i * 512),
        &ix));
    TRY(ds4_num_point(e->num, y, y, pi, DS4_NUM_MUL, 0));
    TRY(linear(e, NAME("ple_block.per_layer_projection.weight"), z, y, n, 512,
               512));
    TRY(norm(e, NAME("ple_block.post_per_layer_input_norm.weight"), z, z, 512));
    TRY(ds4_num_point(e->num, x, x, z, DS4_NUM_ADD, 0));
    ds4_num_view scalar;
    TRY(weight(e, NAME("layer_scalar"), &scalar));
    TRY(ds4_num_point(e->num, x, x, scalar, DS4_NUM_MUL, 0));
    TRY(observe(e, o, "layer", item, i, x));
#undef NAME
  }
  TRY(norm(e, "language_model.norm.weight", y, x, 512));
  TRY(alloc(&w, (uint64_t)n * 768, &projected));
  TRY(linear(e, "language_model.embedding_projection.weight", projected, y, n,
             512, 768));
  TRY(observe(e, o, "projected", item, 24, projected));
  TRY(ds4_num_read(e->num, projected, h, projected.count * 2));
  /* Pool every non-padding position, including BOS/EOS/task tokens, in FP32.
   * Truncate BEFORE L2 normalization for each Matryoshka dimension. */
  for (uint32_t d = 0; d < dimensions; d++) {
    float sum = 0;
    for (uint32_t t = 0; t < n; t++) {
      uint32_t bits = (uint32_t)h[(uint64_t)t * 768 + d] << 16;
      float f;
      memcpy(&f, &bits, 4);
      sum += f;
    }
    result[d] = sum / n;
  }
  double squared = 0;
  for (uint32_t d = 0; d < dimensions; d++) {
    if (!isfinite(result[d])) {
      status = DS4_STATUS_INFERENCE;
      goto cleanup;
    }
    squared += (double)result[d] * result[d];
  }
  if (!(squared > 0)) {
    status = DS4_STATUS_INFERENCE;
    goto cleanup;
  }
  float inv = 1.f / (float)sqrt(squared);
  for (uint32_t d = 0; d < dimensions; d++)
    result[d] *= inv;
cleanup: {
  ds4_status s = clear(&w);
  if (!status)
    status = s;
  ds4_host_free(&host);
  ds4_host_free(&mask);
  return status;
}
}
void ds4_gemma_result_free(ds4_gemma_result *r) {
  if (r) {
    free(r->vectors);
    ds4_resource *a = r->account;
    free(r);
    if (a)
      ds4_resource_release(a);
  }
}
ds4_gemma_result_view ds4_gemma_result_data(const ds4_gemma_result *r) {
  return r ? (ds4_gemma_result_view){r->vectors, r->count, r->dimensions}
           : (ds4_gemma_result_view){0};
}
ds4_status ds4_gemma_embed(ds4_embedding_gemma2 *e, const ds4_gemma_input *in,
                           uint32_t dimensions, const ds4_gemma_observer *o,
                           ds4_gemma_result **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !in ||
      (dimensions != 128 && dimensions != 256 && dimensions != 512 &&
       dimensions != 768))
    return DS4_STATUS_INVALID;
  for (uint32_t i = 0; i < in->count; i++)
    for (uint32_t j = 0; j < in->lengths[i]; j++) {
      uint32_t token = in->tokens[i][j];
      if ((token == 258880 || token == 258884 || token == 258881) &&
          !in->visual[i].data)
        return DS4_STATUS_INVALID;
    }
  ds4_status status;
  ds4_runtime_scope scope = {0};
  ds4_reservation *lease = NULL;
  ds4_gemma_result *r = NULL;
  TRY(ds4_runtime_model_enter(e->model, &scope));
  ds4_runtime_estimate estimate;
  TRY(ds4_gemma_estimate(e, in, &estimate));
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(e->num, &owner);
  uint64_t bytes[DS4_MEMORY_COUNT] = {0};
  bytes[DS4_MEMORY_DEVICE] = estimate.device_bytes;
  bytes[DS4_MEMORY_HOST] = estimate.host_bytes;
  TRY(ds4_resources_reserve(rs, owner, bytes, &lease));
  ds4_reservation_scope working = ds4_reservation_enter(lease);
  r = calloc(1, sizeof(*r));
  if (!r) {
    status = DS4_STATUS_NOMEM;
    goto finish;
  }
  r->count = in->count;
  r->dimensions = dimensions;
  uint64_t vector_bytes = (uint64_t)in->count * dimensions * 4;
  status = ds4_resource_register_host(rs, owner, sizeof(*r) + vector_bytes,
                                      &r->account);
  if (!status && !(r->vectors = calloc(1, vector_bytes)))
    status = DS4_STATUS_NOMEM;
  for (uint32_t i = 0; i < in->count && !status; i++)
    status =
        forward(e, in->tokens[i], in->lengths[i], i, dimensions,
                in->visual[i].data, o, r->vectors + (uint64_t)i * dimensions);
  if (!status)
    status = observe(e, o, "complete", in->count, 0, NONE);
finish:
  ds4_reservation_leave(&working);
cleanup:
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status)
    ds4_gemma_result_free(r);
  else
    *out = r;
  ds4_runtime_scope_leave(&scope);
  return status;
}
