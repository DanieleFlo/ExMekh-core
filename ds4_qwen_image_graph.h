#ifndef DS4_QWEN_IMAGE_GRAPH_H
#define DS4_QWEN_IMAGE_GRAPH_H
/* Private, model-local graph helpers. The numeric core and bundle loader keep
 * their own ownership; adapters describe an exhaustive architecture here. */
#include "ds4_numeric_weights.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define QI_TRY(x)                                                              \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
#define QI_NONE ((ds4_num_view){0})
typedef struct {
  ds4_bundle *bundle;
  ds4_numeric *num;
  ds4_runtime_model *model;
  uint32_t component, group_count;
  ds4_weight_group *groups[34];
  ds4_weight_cache cache;
  struct {
    size_t index;
    uint32_t group;
  } weights[1024];
  size_t count;
  ds4_resource *account;
} qi_graph;
typedef struct {
  ds4_numeric *num;
  ds4_num_buffer *buffers[40];
  size_t count;
} qi_work;
static inline uint16_t qi_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
static inline float qi_float(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
/* QwenImage21Pipeline casts scheduler time to BF16 before dividing by 1000.
 * Rounding sigma directly changes three Turbo steps and their time embeddings. */
static inline float qi_image_timestep(float sigma) {
  float time = qi_float(qi_bf16(sigma * 1000.f));
  return qi_float(qi_bf16(time * (1.f / 1000.f)));
}
static inline ds4_status qi_alloc(qi_work *w, uint64_t n, uint32_t type,
                                  ds4_num_view *v) {
  if (w->count == 40)
    return DS4_STATUS_INVALID;
  ds4_num_buffer *b = NULL;
  ds4_status s = ds4_num_alloc(w->num, n, type, &b);
  if (!s) {
    w->buffers[w->count++] = b;
    *v = ds4_num_data(b);
  }
  return s;
}
static inline ds4_status qi_clear(qi_work *w) {
  ds4_status s = 0;
  while (w->count) {
    ds4_status t = ds4_num_dispose(w->buffers[--w->count]);
    if (!s)
      s = t;
  }
  return s;
}
static inline ds4_status qi_host(ds4_numeric *n, uint64_t bytes, void **p,
                                 ds4_resource **r) {
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(n, &owner);
  *p = NULL;
  *r = NULL;
  ds4_status s = ds4_resource_register_host(rs, owner, bytes, r);
  if (!s && !(*p = calloc(1, bytes))) {
    ds4_resource_release(*r);
    *r = NULL;
    s = DS4_STATUS_NOMEM;
  }
  return s;
}
static inline void qi_host_free(void *p, ds4_resource *r) {
  free(p);
  if (r)
    ds4_resource_release(r);
}
static inline ds4_status qi_bind(qi_graph *g, const char *name, uint32_t group,
                                 uint32_t rank, const uint64_t *shape) {
  if (g->count == 1024 || group >= g->group_count)
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < ds4_bundle_tensor_count(g->bundle); i++) {
    const ds4_weight_tensor *t = ds4_bundle_tensor(g->bundle, i);
    if (t->component != g->component || strcmp(t->name, name))
      continue;
    if (t->type != DS4_NUM_BF16 || t->rank != rank || t->block_elements != 1 ||
        t->block_bytes != 2)
      return DS4_STATUS_UNSUPPORTED;
    uint64_t stride = 2;
    for (uint32_t d = 0; d < rank; d++) {
      if (t->shape[d] != shape[d] || t->stride[d] != stride)
        return DS4_STATUS_INVALID;
      stride *= shape[d];
    }
    for (size_t j = 0; j < g->count; j++)
      if (g->weights[j].index == i)
        return DS4_STATUS_INVALID;
    g->weights[g->count].index = i;
    g->weights[g->count++].group = group;
    return DS4_STATUS_OK;
  }
  return DS4_STATUS_INVALID;
}
static inline ds4_status qi_bind_linear(qi_graph *g, const char *name,
                                        uint32_t group, uint32_t in,
                                        uint32_t out, bool bias) {
  char key[256];
  uint64_t shape[] = {in, out};
  snprintf(key, sizeof(key), "%s.weight", name);
  ds4_status s = qi_bind(g, key, group, 2, shape);
  if (!s && bias) {
    snprintf(key, sizeof(key), "%s.bias", name);
    shape[0] = out;
    s = qi_bind(g, key, group, 1, shape);
  }
  return s;
}
static inline ds4_status qi_weight(qi_graph *g, const char *name,
                                   ds4_num_view *v) {
  for (size_t j = 0; j < g->count; j++) {
    const ds4_weight_tensor *t =
        ds4_bundle_tensor(g->bundle, g->weights[j].index);
    if (strcmp(t->name, name))
      continue;
    ds4_weight_group *group = g->groups[g->weights[j].group];
    ds4_status s = ds4_weight_cache_acquire(&g->cache, g->weights[j].group);
    if (s)
      return s;
    const void *p;
    uint64_t off;
    s = ds4_weight_group_tensor(group, name, &p, &off);
    if (!s)
      *v = ds4_numeric_weight_view(p, off, t->bytes / 2, DS4_NUM_BF16);
    return s;
  }
  return DS4_STATUS_INVALID;
}
static inline ds4_status qi_linear(qi_graph *g, const char *name,
                                   ds4_num_view y, ds4_num_view x,
                                   uint32_t rows, uint32_t in, uint32_t out,
                                   bool bias) {
  char key[256];
  ds4_num_view w, b = QI_NONE;
  snprintf(key, sizeof(key), "%s.weight", name);
  ds4_status s = qi_weight(g, key, &w);
  if (s)
    return s;
  if (bias) {
    snprintf(key, sizeof(key), "%s.bias", name);
    s = qi_weight(g, key, &b);
    if (s)
      return s;
  }
  return ds4_num_linear(g->num, y, x, w, b, rows, in, out);
}
static inline ds4_status qi_activate(void *p) {
  return ds4_numeric_fence(((qi_graph *)p)->num);
}
static inline ds4_status qi_init(qi_graph *g, ds4_bundle *b, ds4_numeric *n,
                                 const char *id, uint32_t groups) {
  g->bundle = b;
  g->num = n;
  g->component = UINT32_MAX;
  g->group_count = groups;
  if (strcmp(ds4_bundle_model(b), "Qwen/Qwen-Image-2.1-Turbo") ||
      strcmp(ds4_bundle_revision(b),
             "fc13ba1986fcb905854dc8de2f52eee395f49b5b") ||
      strcmp(ds4_bundle_profile(b), "bf16-native-v1"))
    return DS4_STATUS_UNSUPPORTED;
  for (size_t i = 0; i < ds4_bundle_component_count(b); i++) {
    const ds4_bundle_component *c = ds4_bundle_component_at(b, i);
    if (!strcmp(c->id, id) && !strcmp(c->compute_dtype, "bfloat16") &&
        !strcmp(c->revision, ds4_bundle_revision(b)))
      g->component = i;
  }
  return g->component == UINT32_MAX ? DS4_STATUS_UNSUPPORTED : DS4_STATUS_OK;
}
static inline ds4_status qi_finish(qi_graph *g, ds4_runtime *runtime,
                                   const char *id) {
  size_t count = 0;
  for (size_t i = 0; i < ds4_bundle_tensor_count(g->bundle); i++)
    if (ds4_bundle_tensor(g->bundle, i)->component == g->component)
      count++;
  if (count != g->count)
    return DS4_STATUS_UNSUPPORTED;
  for (uint32_t k = 0; k < g->group_count; k++) {
    size_t ids[1024], n = 0;
    for (size_t j = 0; j < g->count; j++)
      if (g->weights[j].group == k)
        ids[n++] = g->weights[j].index;
    ds4_status s = ds4_numeric_weight_group_selected(
        &g->groups[k], g->bundle, g->component, ids, n, g->num);
    if (s)
      return s;
  }
  ds4_status cache_status = ds4_weight_cache_init(&g->cache, g->groups, g->group_count);
  if (cache_status) return cache_status;
  ds4_runtime_descriptor d = {DS4_SUITE_ABI,
                              ds4_bundle_model(g->bundle),
                              ds4_bundle_revision(g->bundle),
                              ds4_bundle_profile(g->bundle),
                              id,
                              DS4_OPERATION_IMAGE_GENERATE |
                                  DS4_OPERATION_IMAGE_EDIT,
                              DS4_MODALITY_TEXT | DS4_MODALITY_IMAGE,
                              0,
                              0};
  ds4_runtime_model_ops ops = {qi_activate, NULL};
  return ds4_runtime_model_create(runtime, &d, &ops, g, &g->model);
}
static inline ds4_status qi_release(qi_graph *g) {
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(g->model, &scope);
  if (s)
    return s;
  s = ds4_weight_cache_release(&g->cache);
  ds4_runtime_scope_leave(&scope);
  return s;
}
static inline ds4_status qi_load_all(qi_graph *g) {
  return ds4_weight_cache_preload(&g->cache, g->group_count);
}
static inline uint64_t qi_group_bytes(qi_graph *g, uint32_t group) {
  return group < g->group_count ? ds4_weight_group_bytes(g->groups[group]) : 0;
}
static inline ds4_status qi_close(qi_graph *g) {
  ds4_status s = g->model ? qi_release(g) : DS4_STATUS_OK;
  if (s)
    return s;
  if (g->model && (s = ds4_runtime_model_close(g->model)))
    return s;
  g->model = NULL;
  for (uint32_t i = 0; i < g->group_count; i++) {
    if ((s = ds4_weight_group_close(g->groups[i])))
      return s;
    g->groups[i] = NULL;
  }
  return DS4_STATUS_OK;
}
static inline uint64_t qi_weight_bytes(qi_graph *g) {
  uint64_t bytes = 0;
  for (uint32_t i = 0; i < g->group_count; i++)
    bytes += qi_group_bytes(g, i);
  return bytes;
}
#endif
