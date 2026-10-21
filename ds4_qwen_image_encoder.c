/* Qwen3-VL conditioning for Qwen Image 2.1, not the conversational Qwen27.
 * Equations/layout follow Transformers d4fcbccd and Diffusers aea7b70b;
 * see the T07/T08 provenance record. All orchestration remains C. */
#include "ds4_qwen_image_encoder.h"
#include "ds4_numeric_weights.h"
#include "ds4_qwen_image_input_internal.h"
#include "ds4_qwen_image_tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_qwen_image_conditioning {
  ds4_num_buffer *hidden; /* [count,4096] BF16, pre-final RMSNorm */
  uint8_t *image_mask;
  uint32_t count;
  ds4_resource *account; /* private ownership record; do not modify */
};

#define TEXT_LAYERS 36
#define VISION_LAYERS 27
#define GROUPS 68
#define EMBED_GROUP UINT32_MAX
typedef struct {
  const ds4_weight_tensor *tensor;
  size_t index;
  uint32_t group;
} binding;
struct ds4_qwen_image_encoder {
  ds4_bundle *bundle;
  ds4_numeric *num;
  ds4_runtime_model *model;
  ds4_qwen_image_tokenizer *tokenizer;
  binding weights[750];
  size_t bound;
  ds4_weight_group *groups[GROUPS];
  uint32_t component;
  ds4_resources *resources;
  uint64_t owner;
  ds4_resource *metadata;
};
static ds4_status host_account(ds4_qwen_image_encoder *e, uint64_t bytes,
                               ds4_resource **out) {
  return ds4_resource_register_host(e->resources, e->owner, bytes, out);
}
static ds4_status host_allocate(ds4_qwen_image_encoder *e, uint64_t bytes,
                                void **out, ds4_resource **account) {
  *out = NULL;
  *account = NULL;
  ds4_status status = host_account(e, bytes, account);
  if (status)
    return status;
  *out = malloc(bytes);
  if (!*out) {
    ds4_resource_release(*account);
    *account = NULL;
    return DS4_STATUS_NOMEM;
  }
  return DS4_STATUS_OK;
}
static void host_free(void *p, ds4_resource *account) {
  free(p);
  if (account)
    ds4_resource_release(account);
}
static ds4_num_view none(void) { return (ds4_num_view){0}; }
static uint16_t bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
static float unbf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static const binding *lookup(const ds4_qwen_image_encoder *e,
                             const char *name) {
  for (size_t i = 0; i < e->bound; i++)
    if (!strcmp(e->weights[i].tensor->name, name))
      return &e->weights[i];
  return NULL;
}
static ds4_status bind_weight(ds4_qwen_image_encoder *e, const char *name,
                              uint32_t group, uint32_t rank,
                              const uint64_t *shape) {
  if (e->bound >= 750 || lookup(e, name))
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < ds4_bundle_tensor_count(e->bundle); i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(e->bundle, i);
    if (t->component != e->component || strcmp(t->name, name))
      continue;
    if (t->type != DS4_NUM_BF16 || t->rank != rank || t->block_elements != 1 ||
        t->block_bytes != 2)
      return DS4_STATUS_UNSUPPORTED;
    uint64_t stride = 2;
    for (uint32_t j = 0; j < rank; j++) {
      if (t->shape[j] != shape[j] || t->stride[j] != stride)
        return DS4_STATUS_INVALID;
      stride *= shape[j];
    }
    e->weights[e->bound++] = (binding){t, i, group};
    return DS4_STATUS_OK;
  }
  return DS4_STATUS_INVALID;
}
static ds4_status bind_vector(ds4_qwen_image_encoder *e, const char *prefix,
                              const char *name, uint32_t group,
                              uint32_t width) {
  char full[256];
  snprintf(full, sizeof(full), "%s.%s", prefix, name);
  uint64_t shape[] = {width};
  return bind_weight(e, full, group, 1, shape);
}
static ds4_status bind_linear(ds4_qwen_image_encoder *e, const char *prefix,
                              const char *name, uint32_t group, uint32_t in,
                              uint32_t out, bool bias) {
  char full[256];
  snprintf(full, sizeof(full), "%s.%s.weight", prefix, name);
  uint64_t shape[] = {in, out};
  ds4_status s = bind_weight(e, full, group, 2, shape);
  if (!s && bias) {
    snprintf(full, sizeof(full), "%s.%s.bias", prefix, name);
    shape[0] = out;
    s = bind_weight(e, full, group, 1, shape);
  }
  return s;
}
#define TRY(expr)                                                              \
  do {                                                                         \
    status = (expr);                                                           \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
static ds4_status bind_architecture(ds4_qwen_image_encoder *e) {
  ds4_status status;
  char prefix[128];
  uint64_t shape[] = {4096, 151936};
  TRY(bind_weight(e, "lm_head.weight", EMBED_GROUP, 2, shape));
  TRY(bind_weight(e, "model.language_model.embed_tokens.weight", EMBED_GROUP, 2,
                  shape));
  shape[0] = 4096;
  TRY(bind_weight(e, "model.language_model.norm.weight", EMBED_GROUP, 1,
                  shape));
  for (uint32_t i = 0; i < TEXT_LAYERS; i++) {
    snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u", i);
    TRY(bind_linear(e, prefix, "self_attn.q_proj", i, 4096, 4096, false));
    TRY(bind_linear(e, prefix, "self_attn.k_proj", i, 4096, 1024, false));
    TRY(bind_linear(e, prefix, "self_attn.v_proj", i, 4096, 1024, false));
    TRY(bind_linear(e, prefix, "self_attn.o_proj", i, 4096, 4096, false));
    TRY(bind_vector(e, prefix, "self_attn.q_norm.weight", i, 128));
    TRY(bind_vector(e, prefix, "self_attn.k_norm.weight", i, 128));
    TRY(bind_vector(e, prefix, "input_layernorm.weight", i, 4096));
    TRY(bind_vector(e, prefix, "post_attention_layernorm.weight", i, 4096));
    TRY(bind_linear(e, prefix, "mlp.gate_proj", i, 4096, 12288, false));
    TRY(bind_linear(e, prefix, "mlp.up_proj", i, 4096, 12288, false));
    TRY(bind_linear(e, prefix, "mlp.down_proj", i, 12288, 4096, false));
  }
  for (uint32_t i = 0; i < VISION_LAYERS; i++) {
    uint32_t g = TEXT_LAYERS + i;
    snprintf(prefix, sizeof(prefix), "model.visual.blocks.%u", i);
    TRY(bind_vector(e, prefix, "norm1.weight", g, 1152));
    TRY(bind_vector(e, prefix, "norm1.bias", g, 1152));
    TRY(bind_vector(e, prefix, "norm2.weight", g, 1152));
    TRY(bind_vector(e, prefix, "norm2.bias", g, 1152));
    TRY(bind_linear(e, prefix, "attn.qkv", g, 1152, 3456, true));
    TRY(bind_linear(e, prefix, "attn.proj", g, 1152, 1152, true));
    TRY(bind_linear(e, prefix, "mlp.linear_fc1", g, 1152, 4304, true));
    TRY(bind_linear(e, prefix, "mlp.linear_fc2", g, 4304, 1152, true));
  }
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t g = 63 + i;
    if (i == 3)
      snprintf(prefix, sizeof(prefix), "model.visual.merger");
    else
      snprintf(prefix, sizeof(prefix), "model.visual.deepstack_merger_list.%u",
               i);
    TRY(bind_vector(e, prefix, "norm.weight", g, i == 3 ? 1152 : 4608));
    TRY(bind_vector(e, prefix, "norm.bias", g, i == 3 ? 1152 : 4608));
    TRY(bind_linear(e, prefix, "linear_fc1", g, 4608, 4608, true));
    TRY(bind_linear(e, prefix, "linear_fc2", g, 4608, 4096, true));
  }
  {
    uint64_t pshape[] = {16, 16, 2, 3, 1152};
    TRY(bind_weight(e, "model.visual.patch_embed.proj.weight", 67, 5, pshape));
  }
  TRY(bind_vector(e, "model.visual.patch_embed.proj", "bias", 67, 1152));
  TRY(bind_linear(e, "model.visual", "pos_embed", 67, 1152, 2304, false));
  if (e->bound != 750) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  /* Exhaustive inventory check is independent of the construction above. */
  {
    size_t count = 0;
    for (size_t i = 0; i < ds4_bundle_tensor_count(e->bundle); i++)
      if (ds4_bundle_tensor(e->bundle, i)->component == e->component)
        count++;
    if (count != e->bound) {
      status = DS4_STATUS_UNSUPPORTED;
      goto cleanup;
    }
  }
  return DS4_STATUS_OK;
cleanup:
  return status;
}
static ds4_status fence(void *p) {
  return ds4_numeric_fence(((ds4_qwen_image_encoder *)p)->num);
}
ds4_status ds4_qwen_image_encoder_open(ds4_qwen_image_encoder **out,
                                       ds4_bundle *b, ds4_numeric *num,
                                       ds4_runtime *runtime,
                                       const char *tokenizer) {
  if (!out || !b || !num || !runtime || !tokenizer)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (strcmp(ds4_bundle_model(b), "Qwen/Qwen-Image-2.1-Turbo") ||
      strcmp(ds4_bundle_revision(b),
             "fc13ba1986fcb905854dc8de2f52eee395f49b5b"))
    return DS4_STATUS_UNSUPPORTED;
  ds4_qwen_image_encoder *e = calloc(1, sizeof(*e));
  if (!e)
    return DS4_STATUS_NOMEM;
  e->bundle = b;
  e->num = num;
  e->resources = ds4_numeric_resources(num, &e->owner);
  e->component = UINT32_MAX;
  for (size_t i = 0; i < ds4_bundle_component_count(b); i++)
    if (!strcmp(ds4_bundle_component_at(b, i)->id, "text_encoder"))
      e->component = i;
  ds4_status status = bind_architecture(e);
  if (status)
    goto cleanup;
  const uint8_t *sha = NULL;
  for (size_t i = 0; i < ds4_bundle_file_count(b); i++) {
    const ds4_bundle_file *f = ds4_bundle_file_at(b, i);
    if (strstr(f->path, "/processor/tokenizer.json"))
      sha = f->sha256;
  }
  if (!sha) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  TRY(ds4_qwen_image_tokenizer_open(&e->tokenizer, tokenizer, sha));
  TRY(host_account(e, sizeof(*e) + ds4_qwen_image_tokenizer_bytes(e->tokenizer),
                   &e->metadata));
  ds4_weight_backend backend = ds4_numeric_weight_backend(num);
  for (uint32_t g = 0; g < GROUPS; g++) {
    size_t ids[16], count = 0;
    for (size_t i = 0; i < e->bound; i++)
      if (e->weights[i].group == g)
        ids[count++] = e->weights[i].index;
    TRY(ds4_weight_group_create_selected(&e->groups[g], b, e->component, ids,
                                         count, &backend));
  }
  ds4_runtime_descriptor d = {DS4_SUITE_ABI,
                              ds4_bundle_model(b),
                              ds4_bundle_revision(b),
                              ds4_bundle_profile(b),
                              "text_encoder",
                              DS4_OPERATION_IMAGE_GENERATE |
                                  DS4_OPERATION_IMAGE_EDIT,
                              DS4_MODALITY_TEXT | DS4_MODALITY_IMAGE,
                              262144,
                              0};
  ds4_runtime_model_ops ops = {fence, NULL};
  TRY(ds4_runtime_model_create(runtime, &d, &ops, e, &e->model));
  *out = e;
  return DS4_STATUS_OK;
cleanup:
  for (uint32_t i = 0; i < GROUPS; i++)
    ds4_weight_group_close(e->groups[i]);
  ds4_qwen_image_tokenizer_close(e->tokenizer);
  if (e->metadata)
    ds4_resource_release(e->metadata);
  free(e);
  return status;
}
ds4_status ds4_qwen_image_encoder_release_weights(ds4_qwen_image_encoder *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status status = ds4_runtime_model_enter_exclusive(e->model, &scope);
  if (status)
    return status;
  for (uint32_t i = 0; i < GROUPS; i++) {
    status = ds4_weight_group_release(e->groups[i]);
    if (status)
      break;
  }
  ds4_runtime_scope_leave(&scope);
  return status;
}
ds4_status ds4_qwen_image_encoder_close(ds4_qwen_image_encoder *e) {
  if (!e)
    return DS4_STATUS_OK;
  ds4_status status =
      e->model ? ds4_qwen_image_encoder_release_weights(e) : DS4_STATUS_OK;
  if (status)
    return status;
  status = ds4_runtime_model_close(e->model);
  if (status)
    return status;
  e->model = NULL;
  for (uint32_t i = 0; i < GROUPS; i++) {
    status = ds4_weight_group_close(e->groups[i]);
    if (status)
      return status;
    e->groups[i] = NULL;
  }
  status = ds4_resource_release(e->metadata);
  if (status)
    return status;
  ds4_qwen_image_tokenizer_close(e->tokenizer);
  free(e);
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_prepare(ds4_qwen_image_encoder *e, const char *prompt,
                                  const ds4_qwen_image_reference *refs,
                                  size_t count, ds4_qwen_image_input **out) {
  return ds4_qwen_image_prepare_padded(e, prompt, refs, count, 0, out);
}
ds4_status ds4_qwen_image_prepare_padded(ds4_qwen_image_encoder *e,
                                         const char *prompt,
                                         const ds4_qwen_image_reference *refs,
                                         size_t count, uint32_t pad_to,
                                         ds4_qwen_image_input **out) {
  if (!e)
    return DS4_STATUS_INVALID;
  return ds4_qwen_image_input_create(e->num, e->tokenizer, prompt, refs, count,
                                     pad_to, out);
}

/* Small request workspace owns every activation. Weights live on the adapter
 * until explicitly released; deepstack outputs live through language layer 2.
 */
typedef struct {
  ds4_numeric *num;
  ds4_num_buffer *buffers[32];
  size_t count;
} workspace;
static ds4_status alloc_view(workspace *w, uint64_t count, uint32_t type,
                             ds4_num_view *out) {
  if (w->count >= 32)
    return DS4_STATUS_INVALID;
  ds4_num_buffer *b = NULL;
  ds4_status s = ds4_num_alloc(w->num, count, type, &b);
  if (!s) {
    w->buffers[w->count++] = b;
    *out = ds4_num_data(b);
  }
  return s;
}
static ds4_status work_free(workspace *w) {
  ds4_status status = DS4_STATUS_OK;
  while (w->count) {
    ds4_status next = ds4_num_dispose(w->buffers[--w->count]);
    if (!status)
      status = next;
  }
  return status;
}
static ds4_status weight(ds4_qwen_image_encoder *e, const char *prefix,
                         const char *suffix, ds4_num_view *v) {
  char name[256];
  snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
  const binding *b = lookup(e, name);
  if (!b || b->group == EMBED_GROUP)
    return DS4_STATUS_INVALID;
  ds4_status s = ds4_weight_group_load(e->groups[b->group]);
  if (s)
    return s;
  const void *storage;
  uint64_t off;
  s = ds4_weight_group_tensor(e->groups[b->group], name, &storage, &off);
  if (!s)
    *v = ds4_numeric_weight_view(storage, off, b->tensor->bytes / 2,
                                 DS4_NUM_BF16);
  return s;
}
static ds4_status linear(ds4_qwen_image_encoder *e, const char *prefix,
                         const char *name, ds4_num_view y, ds4_num_view x,
                         uint32_t rows, uint32_t in, uint32_t out, bool bias) {
  ds4_status status;
  ds4_num_view w, b = none();
  char suffix[128];
  snprintf(suffix, sizeof(suffix), "%s.weight", name);
  TRY(weight(e, prefix, suffix, &w));
  if (bias) {
    snprintf(suffix, sizeof(suffix), "%s.bias", name);
    TRY(weight(e, prefix, suffix, &b));
  }
  return ds4_num_linear(e->num, y, x, w, b, rows, in, out);
cleanup:
  return status;
}
static ds4_status norm(ds4_qwen_image_encoder *e, const char *prefix,
                       const char *name, ds4_num_view y, ds4_num_view x,
                       uint32_t width, bool layer) {
  ds4_status status;
  ds4_num_view w, b = none();
  char suffix[128];
  snprintf(suffix, sizeof(suffix), "%s.weight", name);
  TRY(weight(e, prefix, suffix, &w));
  if (layer) {
    snprintf(suffix, sizeof(suffix), "%s.bias", name);
    TRY(weight(e, prefix, suffix, &b));
  }
  return ds4_num_norm(e->num, y, x, w, b, width, 1e-6f,
                      layer ? DS4_NUM_LAYER : DS4_NUM_RMS, DS4_NUM_SCALE_DIRECT,
                      !layer);
cleanup:
  return status;
}
static ds4_status vision_merger(ds4_qwen_image_encoder *e, const char *prefix,
                                bool post, ds4_num_view x, ds4_num_view out,
                                uint32_t patches) {
  workspace w = {.num = e->num};
  ds4_status status;
  ds4_num_view n, h;
  TRY(alloc_view(&w, (uint64_t)patches * 1152, DS4_NUM_BF16, &n));
  TRY(alloc_view(&w, (uint64_t)patches * 1152, DS4_NUM_BF16, &h));
  TRY(norm(e, prefix, "norm", n, x, post ? 4608 : 1152, true));
  TRY(linear(e, prefix, "linear_fc1", h, n, patches / 4, 4608, 4608, true));
  TRY(ds4_num_point(e->num, h, h, none(), DS4_NUM_GELU, 0));
  TRY(linear(e, prefix, "linear_fc2", out, h, patches / 4, 4608, 4096, true));
cleanup: {
  ds4_status cleanup_status = work_free(&w);
  if (!status)
    status = cleanup_status;
}
  return status;
}
static ds4_status vision(ds4_qwen_image_encoder *e,
                         const ds4_qwen_image_input_view *in,
                         const ds4_qwen_image_observer *observer,
                         ds4_num_view features[4]) {
  uint32_t n = in->patches;
  workspace w = {.num = e->num};
  ds4_status status;
  ds4_num_view x, normed, qkv, q, k, v, qr, kr, attn, proj, mlp, co, si, patch;
  uint32_t *bounds = NULL;
  float *cosines = NULL, *sines = NULL, *position = NULL;
  uint16_t *packed = NULL;
  ds4_resource *host_accounts[5] = {0};
  TRY(host_allocate(e, (uint64_t)n * 8, (void **)&bounds, &host_accounts[0]));
  TRY(host_allocate(e, (uint64_t)n * 36 * 4, (void **)&cosines,
                    &host_accounts[1]));
  TRY(host_allocate(e, (uint64_t)n * 36 * 4, (void **)&sines,
                    &host_accounts[2]));
  TRY(host_allocate(e, (uint64_t)n * 1152 * 4, (void **)&position,
                    &host_accounts[3]));
  TRY(host_allocate(e, (uint64_t)n * 1536 * 2, (void **)&packed,
                    &host_accounts[4]));
  for (uint64_t i = 0; i < (uint64_t)n * 1536; i++)
    packed[i] = bf16(in->pixels[i]);
  TRY(alloc_view(&w, (uint64_t)n * 1536, DS4_NUM_BF16, &patch));
  TRY(ds4_num_write(e->num, patch, packed, (uint64_t)n * 1536 * 2));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &x));
  /* Conv3d's output is cast BEFORE bias. Use the native convolution backend
   * so its reduction and bias boundaries match the frozen encoder. */
  {
    ds4_num_view bias, filter;
    TRY(weight(e, "model.visual.patch_embed.proj", "weight", &filter));
    TRY(weight(e, "model.visual.patch_embed.proj", "bias", &bias));
    ds4_num_convolution conv = {.batch = n,
                                .in_channels = 3,
                                .out_channels = 1152,
                                .groups = 1,
                                .input = {2, 16, 16},
                                .kernel = {2, 16, 16},
                                .stride = {2, 16, 16},
                                .dilation = {1, 1, 1}};
    TRY(ds4_num_conv(e->num, x, patch, filter, bias, &conv));
  }
  {
    const binding *pos = lookup(e, "model.visual.pos_embed.weight");
    const uint16_t *table = ds4_bundle_tensor_data(e->bundle, pos->index);
    uint32_t offset = 0;
    for (uint32_t image = 0; image < in->images; image++) {
      uint32_t gh = in->grids[2 * image], gw = in->grids[2 * image + 1],
               end = offset + gh * gw;
      for (uint32_t by = 0; by < gh / 2; by++)
        for (uint32_t bx = 0; bx < gw / 2; bx++)
          for (uint32_t dy = 0; dy < 2; dy++)
            for (uint32_t dx = 0; dx < 2; dx++) {
              uint32_t py = 2 * by + dy, px = 2 * bx + dx,
                       p = offset + (by * (gw / 2) + bx) * 4 + dy * 2 + dx;
              bounds[2 * p] = offset;
              bounds[2 * p + 1] = end;
              for (uint32_t d = 0; d < 36; d++) {
                uint32_t coord = d < 18 ? py : px;
                float angle =
                    coord * (1.0f / powf(10000.0f, (float)(2 * (d % 18)) / 36));
                cosines[p * 36 + d] = angle;
                sines[p * 36 + d] = angle;
              }
              float fy = (float)py * 47 / (gh - 1),
                    fx = (float)px * 47 / (gw - 1);
              uint32_t y0 = (uint32_t)fy, x0 = (uint32_t)fx,
                       y1 = y0 < 47 ? y0 + 1 : 47, x1 = x0 < 47 ? x0 + 1 : 47;
              float wy = fy - y0, wx = fx - x0;
              float hy0 = 1 - fabsf(wy), hy1 = 1 - fabsf(wy - 1),
                    hx0 = 1 - fabsf(wx), hx1 = 1 - fabsf(wx - 1);
              float w00 = hy0 * hx0, w01 = hy0 * hx1, w10 = hy1 * hx0,
                    w11 = hy1 * hx1;
              for (uint32_t d = 0; d < 1152; d++)
                position[(uint64_t)p * 1152 + d] =
                    unbf16(table[((uint64_t)y0 * 48 + x0) * 1152 + d]) * w00 +
                    unbf16(table[((uint64_t)y0 * 48 + x1) * 1152 + d]) * w01 +
                    unbf16(table[((uint64_t)y1 * 48 + x0) * 1152 + d]) * w10 +
                    unbf16(table[((uint64_t)y1 * 48 + x1) * 1152 + d]) * w11;
            }
      offset = end;
    }
  }
  {
    ds4_num_view pos;
    TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &pos));
    for (uint64_t i = 0; i < (uint64_t)n * 1152; i++)
      packed[i] = bf16(position[i]);
    TRY(ds4_num_write(e->num, pos, packed, (uint64_t)n * 1152 * 2));
    TRY(ds4_num_point(e->num, x, x, pos, DS4_NUM_ADD, 0));
  }
  TRY(alloc_view(&w, (uint64_t)n * 36, DS4_NUM_F32, &co));
  TRY(alloc_view(&w, (uint64_t)n * 36, DS4_NUM_F32, &si));
  TRY(ds4_num_write(e->num, co, cosines, (uint64_t)n * 36 * 4));
  TRY(ds4_num_write(e->num, si, sines, (uint64_t)n * 36 * 4));
  TRY(ds4_num_point(e->num, co, co, none(), DS4_NUM_COS, 0));
  TRY(ds4_num_point(e->num, si, si, none(), DS4_NUM_SIN, 0));
  TRY(ds4_qwen_image_observe(e->num, observer, "vision_input", 0, x));
  TRY(ds4_qwen_image_observe(e->num, observer, "vision_cos", 0, co));
  TRY(ds4_qwen_image_observe(e->num, observer, "vision_sin", 0, si));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &normed));
  TRY(alloc_view(&w, (uint64_t)n * 3456, DS4_NUM_BF16, &qkv));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &q));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &k));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &v));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &qr));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &kr));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &attn));
  TRY(alloc_view(&w, (uint64_t)n * 1152, DS4_NUM_BF16, &proj));
  TRY(alloc_view(&w, (uint64_t)n * 4304, DS4_NUM_BF16, &mlp));
  for (uint32_t layer = 0; layer < VISION_LAYERS; layer++) {
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "model.visual.blocks.%u", layer);
    TRY(norm(e, prefix, "norm1", normed, x, 1152, true));
    TRY(linear(e, prefix, "attn.qkv", qkv, normed, n, 1152, 3456, true));
    for (uint32_t p = 0; p < n; p++) {
      TRY(ds4_num_copy(e->num, ds4_num_slice(q, (uint64_t)p * 1152, 1152),
                       ds4_num_slice(qkv, (uint64_t)p * 3456, 1152)));
      TRY(ds4_num_copy(e->num, ds4_num_slice(k, (uint64_t)p * 1152, 1152),
                       ds4_num_slice(qkv, (uint64_t)p * 3456 + 1152, 1152)));
      TRY(ds4_num_copy(e->num, ds4_num_slice(v, (uint64_t)p * 1152, 1152),
                       ds4_num_slice(qkv, (uint64_t)p * 3456 + 2304, 1152)));
    }
    TRY(ds4_num_rope(e->num, qr, q, co, si, 16, 72, false, false));
    TRY(ds4_num_rope(e->num, kr, k, co, si, 16, 72, false, false));
    ds4_num_attention a = {.batch = 1,
                           .queries = n,
                           .keys = n,
                           .q_heads = 16,
                           .kv_heads = 16,
                           .dim = 72,
                           .scale = 1 / sqrtf(72),
                           .round_scores = true,
                           .round_probabilities = true};
    TRY(ds4_num_attend(e->num, attn, qr, kr, v, bounds, &a));
    TRY(linear(e, prefix, "attn.proj", proj, attn, n, 1152, 1152, true));
    TRY(ds4_num_point(e->num, x, x, proj, DS4_NUM_ADD, 0));
    TRY(norm(e, prefix, "norm2", normed, x, 1152, true));
    TRY(linear(e, prefix, "mlp.linear_fc1", mlp, normed, n, 1152, 4304, true));
    TRY(ds4_num_point(e->num, mlp, mlp, none(), DS4_NUM_GELU_TANH, 0));
    TRY(linear(e, prefix, "mlp.linear_fc2", proj, mlp, n, 4304, 1152, true));
    TRY(ds4_num_point(e->num, x, x, proj, DS4_NUM_ADD, 0));
    TRY(ds4_qwen_image_observe(e->num, observer, "vision", layer, x));
    if (layer == 8 || layer == 16 || layer == 24) {
      uint32_t index = (layer - 8) / 8;
      snprintf(prefix, sizeof(prefix), "model.visual.deepstack_merger_list.%u",
               index);
      TRY(vision_merger(e, prefix, true, x, features[index], n));
      TRY(ds4_qwen_image_observe(e->num, observer, "deepstack", index,
                                 features[index]));
    }
  }
  TRY(vision_merger(e, "model.visual.merger", false, x, features[3], n));
  TRY(ds4_qwen_image_observe(e->num, observer, "vision_merged", 0,
                             features[3]));
cleanup:
  host_free(bounds, host_accounts[0]);
  host_free(cosines, host_accounts[1]);
  host_free(sines, host_accounts[2]);
  host_free(position, host_accounts[3]);
  host_free(packed, host_accounts[4]);
  {
    ds4_status cleanup_status = work_free(&w);
    if (!status)
      status = cleanup_status;
  }
  if (!status && observer && observer->cancelled &&
      observer->cancelled(observer->opaque))
    status = DS4_STATUS_CANCELLED;
  return status;
}

ds4_status ds4_qwen_image_encoder_estimate(ds4_qwen_image_encoder *e,
                                           const ds4_qwen_image_input *source,
                                           ds4_runtime_estimate *out) {
  if (!e || !out)
    return DS4_STATUS_INVALID;
  ds4_qwen_image_input_view input = ds4_qwen_image_input_data(source);
  const ds4_qwen_image_input_view *in = &input;
  if (!in->count)
    return DS4_STATUS_INVALID;
  memset(out, 0, sizeof(*out));
  for (uint32_t i = 0; i < GROUPS; i++)
    if ((in->images || i < TEXT_LAYERS) &&
        !ds4_weight_group_resident(e->groups[i]))
      out->device_bytes += ds4_weight_group_bytes(e->groups[i]);
  /* Includes simultaneous vision/text workspace, GEMM bias scratch, masks,
   * deepstack, host preprocessing and output. Conservative, not measured. */
  out->device_bytes +=
      (uint64_t)in->count * 100000 + (uint64_t)in->patches * 100000;
  out->host_bytes = (uint64_t)in->count * 32768 + (uint64_t)in->patches * 40000;
  out->known = true;
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_encode(ds4_qwen_image_encoder *e,
                                 const ds4_qwen_image_input *source,
                                 const ds4_qwen_image_observer *observer,
                                 ds4_qwen_image_conditioning **out) {
  if (!e || !out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_qwen_image_input_view input = ds4_qwen_image_input_data(source);
  const ds4_qwen_image_input_view *in = &input;
  if (!in->count)
    return DS4_STATUS_INVALID;
  ds4_status status = DS4_STATUS_OK;
  ds4_runtime_scope scope = {0};
  workspace w = {.num = e->num};
  ds4_qwen_image_conditioning *result = NULL;
  uint16_t *host = NULL, *trig = NULL;
  uint32_t *bounds = NULL;
  ds4_resource *host_accounts[3] = {0};
  ds4_num_view x, normed, q, k, v, qr, kr, attn, proj, gate, up, co, si,
      features[4] = {{0}};
  uint32_t n = in->count;
  TRY(ds4_runtime_model_enter(e->model, &scope));
  TRY(ds4_qwen_image_observe(e->num, observer, "begin", 0, none()));
  if (in->patches) {
    for (int i = 0; i < 4; i++)
      TRY(alloc_view(&w, (uint64_t)in->patches / 4 * 4096, DS4_NUM_BF16,
                     &features[i]));
    TRY(vision(e, in, observer, features));
  }
  TRY(host_allocate(e, (uint64_t)n * 4096 * 2, (void **)&host,
                    &host_accounts[0]));
  TRY(host_allocate(e, (uint64_t)n * 64 * 4, (void **)&trig,
                    &host_accounts[1]));
  TRY(host_allocate(e, (uint64_t)n * 8, (void **)&bounds, &host_accounts[2]));
  {
    const binding *emb = lookup(e, "model.language_model.embed_tokens.weight");
    const uint16_t *table = ds4_bundle_tensor_data(e->bundle, emb->index);
    for (uint32_t i = 0; i < n; i++)
      memcpy(host + (uint64_t)i * 4096, table + (uint64_t)in->tokens[i] * 4096,
             8192);
  }
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &x));
  TRY(ds4_num_write(e->num, x, host, (uint64_t)n * 8192));
  if (in->patches) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < n; i++)
      if (in->image_mask[i]) {
        TRY(ds4_num_copy(
            e->num, ds4_num_slice(x, (uint64_t)i * 4096, 4096),
            ds4_num_slice(features[3], (uint64_t)offset++ * 4096, 4096)));
      }
  }
  TRY(ds4_qwen_image_observe(e->num, observer, "embeddings", 0, x));
  for (uint32_t t = 0; t < n; t++) {
    bounds[2 * t] = t < in->left_padding ? 0 : in->left_padding;
    bounds[2 * t + 1] = t < in->left_padding ? 0 : t + 1;
    for (uint32_t d = 0; d < 64; d++) {
      uint32_t axis = d % 3 == 1 && d < 60 ? 1 : d % 3 == 2 && d < 60 ? 2 : 0;
      float angle = in->positions[3 * t + axis] *
                    (1.0f / powf(5000000.0f, (float)(2 * d) / 128));
      trig[t * 64 + d] = bf16(cosf(angle));
      trig[(uint64_t)n * 64 + t * 64 + d] = bf16(sinf(angle));
    }
  }
  TRY(alloc_view(&w, (uint64_t)n * 64, DS4_NUM_BF16, &co));
  TRY(alloc_view(&w, (uint64_t)n * 64, DS4_NUM_BF16, &si));
  TRY(ds4_num_write(e->num, co, trig, (uint64_t)n * 128));
  TRY(ds4_num_write(e->num, si, trig + (uint64_t)n * 64, (uint64_t)n * 128));
  TRY(ds4_qwen_image_observe(e->num, observer, "rope_cos", 0, co));
  TRY(ds4_qwen_image_observe(e->num, observer, "rope_sin", 0, si));
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &normed));
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &q));
  TRY(alloc_view(&w, (uint64_t)n * 1024, DS4_NUM_BF16, &k));
  TRY(alloc_view(&w, (uint64_t)n * 1024, DS4_NUM_BF16, &v));
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &qr));
  TRY(alloc_view(&w, (uint64_t)n * 1024, DS4_NUM_BF16, &kr));
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &attn));
  TRY(alloc_view(&w, (uint64_t)n * 4096, DS4_NUM_BF16, &proj));
  TRY(alloc_view(&w, (uint64_t)n * 12288, DS4_NUM_BF16, &gate));
  TRY(alloc_view(&w, (uint64_t)n * 12288, DS4_NUM_BF16, &up));
  for (uint32_t layer = 0; layer < TEXT_LAYERS; layer++) {
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u", layer);
    TRY(norm(e, prefix, "input_layernorm", normed, x, 4096, false));
    if (!layer)
      TRY(ds4_qwen_image_observe(e->num, observer, "input_norm", layer,
                                 normed));
    TRY(linear(e, prefix, "self_attn.q_proj", q, normed, n, 4096, 4096, false));
    TRY(linear(e, prefix, "self_attn.k_proj", k, normed, n, 4096, 1024, false));
    TRY(linear(e, prefix, "self_attn.v_proj", v, normed, n, 4096, 1024, false));
    TRY(norm(e, prefix, "self_attn.q_norm", q, q, 128, false));
    TRY(norm(e, prefix, "self_attn.k_norm", k, k, 128, false));
    if (!layer) {
      TRY(ds4_qwen_image_observe(e->num, observer, "q_norm", layer, q));
      TRY(ds4_qwen_image_observe(e->num, observer, "k_norm", layer, k));
      TRY(ds4_qwen_image_observe(e->num, observer, "v_proj", layer, v));
    }
    TRY(ds4_num_rope(e->num, qr, q, co, si, 32, 128, false, true));
    TRY(ds4_num_rope(e->num, kr, k, co, si, 8, 128, false, true));
    if (!layer) {
      TRY(ds4_qwen_image_observe(e->num, observer, "q_rope", layer, qr));
      TRY(ds4_qwen_image_observe(e->num, observer, "k_rope", layer, kr));
    }
    ds4_num_attention a = {.batch = 1,
                           .queries = n,
                           .keys = n,
                           .q_heads = 32,
                           .kv_heads = 8,
                           .dim = 128,
                           .scale = 1 / sqrtf(128),
                           .round_scores = true,
                           .round_probabilities = true};
    TRY(ds4_num_attend(e->num, attn, qr, kr, v, bounds, &a));
    if (!layer)
      TRY(ds4_qwen_image_observe(e->num, observer, "attention", layer, attn));
    TRY(linear(e, prefix, "self_attn.o_proj", proj, attn, n, 4096, 4096,
               false));
    TRY(ds4_num_point(e->num, x, x, proj, DS4_NUM_ADD, 0));
    if (!layer)
      TRY(ds4_qwen_image_observe(e->num, observer, "attention_residual", layer,
                                 x));
    TRY(norm(e, prefix, "post_attention_layernorm", normed, x, 4096, false));
    TRY(linear(e, prefix, "mlp.gate_proj", gate, normed, n, 4096, 12288,
               false));
    TRY(linear(e, prefix, "mlp.up_proj", up, normed, n, 4096, 12288, false));
    TRY(ds4_num_point(e->num, gate, gate, none(), DS4_NUM_SILU, 0));
    TRY(ds4_num_point(e->num, gate, gate, up, DS4_NUM_MUL, 0));
    if (!layer)
      TRY(ds4_qwen_image_observe(e->num, observer, "mlp_mid", layer, gate));
    TRY(linear(e, prefix, "mlp.down_proj", proj, gate, n, 12288, 4096, false));
    TRY(ds4_num_point(e->num, x, x, proj, DS4_NUM_ADD, 0));
    if (layer < 3 && in->patches) {
      uint32_t offset = 0;
      for (uint32_t i = 0; i < n; i++)
        if (in->image_mask[i]) {
          ds4_num_view row = ds4_num_slice(x, (uint64_t)i * 4096, 4096);
          TRY(ds4_num_point(
              e->num, row, row,
              ds4_num_slice(features[layer], (uint64_t)offset++ * 4096, 4096),
              DS4_NUM_ADD, 0));
        }
    }
    TRY(ds4_qwen_image_observe(e->num, observer, "text", layer, x));
  }
  /* Intentionally skip language_model.norm and lm_head. The DiT was trained
   * on the last decoder block BEFORE final RMSNorm; no AR generation occurs. */
  result = calloc(1, sizeof(*result));
  if (!result) {
    status = DS4_STATUS_NOMEM;
    goto cleanup;
  }
  result->count = n - in->drop;
  TRY(host_account(e, sizeof(*result) + result->count, &result->account));
  result->image_mask = malloc(result->count);
  if (!result->image_mask) {
    status = DS4_STATUS_NOMEM;
    goto cleanup;
  }
  memcpy(result->image_mask, in->image_mask + in->drop, result->count);
  TRY(ds4_num_alloc(e->num, (uint64_t)result->count * 4096, DS4_NUM_BF16,
                    &result->hidden));
  TRY(ds4_num_copy(e->num, ds4_num_data(result->hidden),
                   ds4_num_slice(x, (uint64_t)in->drop * 4096,
                                 (uint64_t)result->count * 4096)));
  TRY(ds4_numeric_fence(e->num));
  TRY(ds4_qwen_image_observe(e->num, observer, "conditioning", 0,
                             ds4_num_data(result->hidden)));
cleanup:
  host_free(host, host_accounts[0]);
  host_free(trig, host_accounts[1]);
  host_free(bounds, host_accounts[2]);
  {
    ds4_status cleanup_status = work_free(&w);
    if (!status)
      status = cleanup_status;
  }
  if (!status && observer && observer->cancelled &&
      observer->cancelled(observer->opaque))
    status = DS4_STATUS_CANCELLED;
  if (!status) {
    *out = result;
  } else if (result) {
    ds4_num_dispose(result->hidden);
    host_free(result->image_mask, result->account);
    free(result);
  }
  ds4_runtime_scope_leave(&scope);
  return status;
}
ds4_status ds4_qwen_image_conditioning_free(ds4_qwen_image_conditioning *c) {
  if (!c)
    return DS4_STATUS_OK;
  ds4_status s = ds4_num_free(c->hidden);
  if (s)
    return s;
  free(c->image_mask);
  if (c->account)
    ds4_resource_release(c->account);
  free(c);
  return DS4_STATUS_OK;
}

ds4_qwen_image_conditioning_view
ds4_qwen_image_conditioning_data(const ds4_qwen_image_conditioning *result) {
  return result
             ? (ds4_qwen_image_conditioning_view){ds4_num_data(result->hidden),
                                                  result->image_mask,
                                                  result->count}
             : (ds4_qwen_image_conditioning_view){0};
}
