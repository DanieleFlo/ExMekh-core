/* Single-stream Qwen Image 2.1. Equations and token order: pinned Diffusers
 * aea7b70b (Apache-2.0). No dual-stream or approximate block skipping. */
#include "ds4_qwen_image_dit.h"
#include "ds4_qwen_image_attention.h"
#include "ds4_qwen_image_components.h"
#include "ds4_qwen_image_dit_constants.h"
#include "ds4_qwen_image_graph.h"
#define D 4096u
#define FF 12288u
struct ds4_qwen_image_dit {
  qi_graph g;
  size_t inputs;
};
struct ds4_qwen_image_dit_input {
  ds4_qwen_image_dit *dit;
  uint32_t count, prefix, target, text, reference;
  int32_t *source;
  uint32_t *bounds;
  ds4_num_buffer *conditioning, *references, *cos, *sin;
  ds4_num_buffer *k[32], *v[32];
  bool cached;
  bool host_cache;
  uint16_t *host_kv;
  ds4_resource *host_account;
  ds4_resource *account, *metadata;
};
static ds4_status dit_bind(qi_graph *g) {
  ds4_status status;
  char name[256];
  uint64_t shape[] = {D};
  QI_TRY(qi_bind_linear(g, "img_in", 0, 64, D, false));
  QI_TRY(qi_bind(g, "txt_in.text_norm.weight", 0, 1, shape));
  QI_TRY(qi_bind_linear(g, "txt_in.in_layer", 0, D, D, false));
  QI_TRY(qi_bind_linear(g, "txt_in.out_layer", 0, D, D, false));
  QI_TRY(qi_bind_linear(g, "time_text_embed.timestep_embedder.linear_1", 0, 256,
                        D, false));
  QI_TRY(qi_bind_linear(g, "time_text_embed.timestep_embedder.linear_2", 0, D,
                        D, false));
  QI_TRY(qi_bind_linear(g, "modulation.1", 0, D, D * 4, false));
  QI_TRY(qi_bind_linear(g, "norm_out.linear", 0, D, D, false));
  QI_TRY(qi_bind_linear(g, "proj_out", 0, D, 64, false));
  for (uint32_t i = 0; i < 32; i++) {
    const char *linear[] = {
        "attn.to_q",    "attn.to_k",          "attn.to_v",  "attn.to_out.0",
        "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"};
    for (uint32_t j = 0; j < 7; j++) {
      snprintf(name, sizeof(name), "transformer_blocks.%u.%s", i, linear[j]);
      QI_TRY(qi_bind_linear(g, name, i + 1, j == 6 ? FF : D,
                            (j == 4 || j == 5) ? FF : D, false));
    }
    shape[0] = 128;
    snprintf(name, sizeof(name), "transformer_blocks.%u.attn.norm_q.weight", i);
    QI_TRY(qi_bind(g, name, i + 1, 1, shape));
    snprintf(name, sizeof(name), "transformer_blocks.%u.attn.norm_k.weight", i);
    QI_TRY(qi_bind(g, name, i + 1, 1, shape));
  }
cleanup:
  return status;
}
ds4_status ds4_qwen_image_dit_open(ds4_qwen_image_dit **out, ds4_bundle *b,
                                   ds4_numeric *num, ds4_runtime *runtime) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!b || !num || !runtime)
    return DS4_STATUS_INVALID;
  ds4_resource *account = NULL;
  ds4_qwen_image_dit *e = NULL;
  ds4_status status = qi_host(num, sizeof(*e), (void **)&e, &account);
  if (status)
    return status;
  e->g.account = account;
  QI_TRY(qi_init(&e->g, b, num, "dit", 33));
  QI_TRY(dit_bind(&e->g));
  QI_TRY(qi_finish(&e->g, runtime, "dit"));
  *out = e;
  return DS4_STATUS_OK;
cleanup:
  qi_close(&e->g);
  qi_host_free(e, account);
  return status;
}
ds4_status ds4_qwen_image_dit_release_weights(ds4_qwen_image_dit *e) {
  return e ? qi_release(&e->g) : DS4_STATUS_INVALID;
}
ds4_status ds4_qwen_image_dit_close(ds4_qwen_image_dit *e) {
  if (!e)
    return DS4_STATUS_OK;
  if (e->inputs)
    return DS4_STATUS_BUSY;
  ds4_status s = qi_close(&e->g);
  if (!s)
    qi_host_free(e, e->g.account);
  return s;
}
static ds4_status clear_cache(ds4_qwen_image_dit_input *p) {
  p->cached = false;
  ds4_status status = 0;
  for (uint32_t i = 0; i < 32; i++) {
    ds4_status s = ds4_num_dispose(p->k[i]);
    p->k[i] = NULL;
    if (!status)
      status = s;
    s = ds4_num_dispose(p->v[i]);
    p->v[i] = NULL;
    if (!status)
      status = s;
  }
  qi_host_free(p->host_kv, p->host_account);
  p->host_kv = NULL;
  p->host_account = NULL;
  return status;
}
ds4_status ds4_qwen_image_dit_cache_place(ds4_qwen_image_dit_input *p,
                                          ds4_memory_location location) {
  if (!p || (location != DS4_MEMORY_HOST && location != DS4_MEMORY_DEVICE))
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(p->dit->g.model, &scope);
  if (!s && p->cached)
    s = DS4_STATUS_BUSY;
  if (!s)
    p->host_cache = location == DS4_MEMORY_HOST;
  ds4_runtime_scope_leave(&scope);
  return s;
}
ds4_status ds4_qwen_image_dit_cache_clear(ds4_qwen_image_dit_input *p) {
  if (!p)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter_exclusive(p->dit->g.model, &scope);
  if (!s)
    s = clear_cache(p);
  ds4_runtime_scope_leave(&scope);
  return s;
}
static ds4_status input_dispose(ds4_qwen_image_dit_input *p) {
  ds4_status s = clear_cache(p), t;
  ds4_num_buffer *b[] = {p->conditioning, p->references, p->cos, p->sin};
  for (size_t i = 0; i < 4; i++) {
    t = ds4_num_dispose(b[i]);
    if (!s)
      s = t;
  }
  qi_host_free(p->source, p->metadata);
  qi_host_free(p, p->account);
  return s;
}
ds4_status ds4_qwen_image_dit_input_free(ds4_qwen_image_dit_input *p) {
  if (!p)
    return DS4_STATUS_OK;
  ds4_runtime_scope scope = {0};
  ds4_qwen_image_dit *e = p->dit;
  ds4_status s = ds4_runtime_model_enter_exclusive(e->g.model, &scope);
  if (s)
    return s;
  s = input_dispose(p);
  e->inputs--;
  ds4_runtime_scope_leave(&scope);
  return s;
}
ds4_status ds4_qwen_image_dit_prepare(ds4_qwen_image_dit *e,
                                      ds4_qwen_image_conditioning_view c,
                                      ds4_num_view refs,
                                      const ds4_qwen_image_latent_shape *shapes,
                                      size_t ns,
                                      ds4_qwen_image_dit_input **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !c.count || c.count > 262144 || !c.image_mask ||
      c.hidden.type != DS4_NUM_BF16 || !c.hidden.data ||
      c.hidden.count != (uint64_t)c.count * D || !shapes || !ns || ns > 65)
    return DS4_STATUS_INVALID;
  uint64_t image = 0;
  uint32_t lengths[65];
  for (size_t i = 0; i < ns; i++) {
    uint64_t n = (uint64_t)shapes[i].height * shapes[i].width;
    if (!n || n > 262144 || shapes[i].height % 2 || shapes[i].width % 2 ||
        shapes[i].height > 2048 || shapes[i].width > 2048)
      return DS4_STATUS_INVALID;
    lengths[i] = n;
    image += n;
  }
  uint32_t slots = 0;
  for (uint32_t i = 0; i < c.count; i++) {
    if (c.image_mask[i] > 1)
      return DS4_STATUS_INVALID;
    slots += c.image_mask[i];
  }
  uint32_t target = lengths[ns - 1];
  if (image - target != (uint64_t)slots * 4 ||
      (image > target && (!refs.data || refs.type != DS4_NUM_BF16 ||
                          refs.count != (image - target) * 64)))
    return DS4_STATUS_INVALID;
  uint64_t total = (uint64_t)c.count + 3 * slots + target;
  if (total > 262144)
    return DS4_STATUS_INVALID;
  ds4_runtime_scope scope = {0};
  ds4_status status = ds4_runtime_model_enter(e->g.model, &scope);
  if (status)
    return status;
  ds4_qwen_image_dit_input *p = NULL;
  ds4_resource *account = NULL;
  float *angles = NULL;
  ds4_resource *ha = NULL;
  QI_TRY(qi_host(e->g.num, sizeof(*p), (void **)&p, &account));
  p->account = account;
  p->dit = e;
  p->count = total;
  p->target = target;
  p->prefix = total - target;
  p->text = c.count;
  p->reference = image - target;
  QI_TRY(qi_host(e->g.num, total * (sizeof(int32_t) + 2 * sizeof(uint32_t)),
                 (void **)&p->source, &p->metadata));
  p->bounds = (uint32_t *)(p->source + total);
  uint32_t cursor = 0, im = 0;
  for (uint32_t i = 0; i < c.count; i++) {
    if (!c.image_mask[i])
      p->source[cursor++] = i;
    else
      for (uint32_t j = 0; j < 4; j++)
        p->source[cursor++] = -(int32_t)(++im);
  }
  for (uint32_t i = 0; i < target; i++)
    p->source[cursor++] = -(int32_t)(++im);
  QI_TRY(qi_host(e->g.num, total * 128 * sizeof(float), (void **)&angles, &ha));
  float *co = angles, *si = angles + total * 64;
  uint32_t block = 0, left = 0, end = 0, position = 0, start = 0;
  for (uint32_t i = 0; i < total; i++) {
    int32_t pos[3];
    if (p->source[i] >= 0) {
      if (left) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      pos[0] = pos[1] = pos[2] = position++;
      end = i + 1;
    } else {
      if (!left) {
        if (block >= ns) {
          status = DS4_STATUS_INVALID;
          goto cleanup;
        }
        left = lengths[block];
        start = i;
        end = i + left;
        if (end > total) {
          status = DS4_STATUS_INVALID;
          goto cleanup;
        }
        for (uint32_t j = i; j < end; j++)
          if (p->source[j] >= 0) {
            status = DS4_STATUS_INVALID;
            goto cleanup;
          }
      }
      uint32_t h = shapes[block].height, ww = shapes[block].width,
               j = i - start;
      pos[0] = position;
      pos[1] = (int32_t)(j / ww) - (int32_t)(h - h / 2);
      pos[2] = (int32_t)(j % ww) - (int32_t)(ww - ww / 2);
      if (!--left) {
        position += h > ww ? h : ww;
        block++;
      }
    }
    p->bounds[2 * i] = 0;
    p->bounds[2 * i + 1] = end;
    uint32_t at = 0;
    const uint32_t dims[] = {16, 56, 56};
    for (uint32_t axis = 0; axis < 3; axis++)
      for (uint32_t j = 0; j < dims[axis] / 2; j++) {
        if (pos[axis] < -1024 || pos[axis] >= 8192) {
          status = DS4_STATUS_UNSUPPORTED;
          goto cleanup;
        }
        float frequency = qi_dit_rope_frequency[(axis ? 8 : 0) + j];
        float angle = (float)pos[axis] * frequency;
        co[(uint64_t)i * 64 + at] = cosf(angle);
        si[(uint64_t)i * 64 + at++] = sinf(angle);
      }
  }
  QI_TRY(
      ds4_num_alloc(e->g.num, c.hidden.count, DS4_NUM_BF16, &p->conditioning));
  QI_TRY(ds4_num_copy(e->g.num, ds4_num_data(p->conditioning), c.hidden));
  if (p->reference) {
    QI_TRY(ds4_num_alloc(e->g.num, refs.count, DS4_NUM_BF16, &p->references));
    QI_TRY(ds4_num_copy(e->g.num, ds4_num_data(p->references), refs));
  }
  QI_TRY(ds4_num_alloc(e->g.num, total * 64, DS4_NUM_F32, &p->cos));
  QI_TRY(ds4_num_alloc(e->g.num, total * 64, DS4_NUM_F32, &p->sin));
  QI_TRY(ds4_num_write(e->g.num, ds4_num_data(p->cos), co, total * 64 * 4));
  QI_TRY(ds4_num_write(e->g.num, ds4_num_data(p->sin), si, total * 64 * 4));
  QI_TRY(ds4_numeric_fence(e->g.num));
  e->inputs++;
  *out = p;
  p = NULL;
cleanup:
  qi_host_free(angles, ha);
  if (p)
    input_dispose(p);
  ds4_runtime_scope_leave(&scope);
  return status;
}
/* Prefix and target are contiguous runs. Broadcasting each run preserves all
 * BF16 rounding boundaries of modulation without a full per-token parameter. */
static ds4_status modulate(qi_graph *g, ds4_num_view y, ds4_num_view params,
                           uint32_t prefix, uint32_t n) {
  ds4_status s = 0;
  if (prefix)
    s = ds4_num_point(g->num, ds4_num_slice(y, 0, (uint64_t)prefix * D),
                      ds4_num_slice(y, 0, (uint64_t)prefix * D),
                      ds4_num_slice(params, D, D), DS4_NUM_MUL, 0);
  if (!s && n > prefix)
    s = ds4_num_point(
        g->num,
        ds4_num_slice(y, (uint64_t)prefix * D, (uint64_t)(n - prefix) * D),
        ds4_num_slice(y, (uint64_t)prefix * D, (uint64_t)(n - prefix) * D),
        ds4_num_slice(params, 0, D), DS4_NUM_MUL, 0);
  return s;
}
static ds4_status timestep(qi_graph *g, float sigma, ds4_num_view temb,
                           ds4_num_view mod, ds4_num_view scale,
                           const ds4_qwen_image_observer *o) {
  qi_work w = {.num = g->num};
  ds4_status status;
  ds4_num_view t, h, act, one;
  uint16_t host[512];
  float ts[] = {qi_float(qi_bf16(sigma)), 0};
  for (uint32_t r = 0; r < 2; r++)
    for (uint32_t j = 0; j < 128; j++) {
      float a = 1000.f * ts[r] * qi_dit_time_frequency[j];
      host[r * 256 + j] = qi_bf16(cosf(a));
      host[r * 256 + 128 + j] = qi_bf16(sinf(a));
    }
  QI_TRY(qi_alloc(&w, 512, DS4_NUM_BF16, &t));
  QI_TRY(ds4_num_write(g->num, t, host, sizeof(host)));
  QI_TRY(qi_alloc(&w, 2 * D, DS4_NUM_BF16, &h));
  QI_TRY(qi_alloc(&w, 2 * D, DS4_NUM_BF16, &act));
  QI_TRY(qi_linear(g, "time_text_embed.timestep_embedder.linear_1", h, t, 2,
                   256, D, false));
  QI_TRY(ds4_num_point(g->num, h, h, QI_NONE, DS4_NUM_SILU, 0));
  QI_TRY(qi_linear(g, "time_text_embed.timestep_embedder.linear_2", temb, h, 2,
                   D, D, false));
  QI_TRY(ds4_num_point(g->num, act, temb, QI_NONE, DS4_NUM_SILU, 0));
  QI_TRY(qi_linear(g, "modulation.1", mod, act, 2, D, 4 * D, false));
  QI_TRY(ds4_qwen_image_observe(g->num, o, "modulation", 0, mod));
  QI_TRY(qi_linear(g, "norm_out.linear", scale, act, 2, D, D, false));
  QI_TRY(qi_alloc(&w, 1, DS4_NUM_BF16, &one));
  uint16_t value = qi_bf16(1);
  QI_TRY(ds4_num_write(g->num, one, &value, 2));
  QI_TRY(ds4_num_point(g->num, scale, scale, one, DS4_NUM_ADD, 0));
  for (uint32_t r = 0; r < 2; r++)
    for (uint32_t j = 0; j < 4; j++) {
      ds4_num_view slice = ds4_num_slice(mod, (uint64_t)r * 4 * D + j * D, D);
      QI_TRY(ds4_num_point(g->num, slice, slice, (j % 2) ? QI_NONE : one,
                           (j % 2) ? DS4_NUM_TANH : DS4_NUM_ADD, 0));
    }
cleanup:
  {
    ds4_status s = qi_clear(&w);
    return status ? status : s;
  }
}
static ds4_status block(qi_graph *g, ds4_qwen_image_dit_input *p,
                        uint32_t layer, ds4_num_view x, ds4_num_view mod,
                        uint32_t offset, const ds4_qwen_image_observer *o) {
  uint32_t n = p->count - offset, prefix = offset ? 0 : p->prefix;
  uint64_t size = (uint64_t)n * D, keys = (uint64_t)p->count * D;
  qi_work w = {.num = g->num};
  ds4_status status;
  ds4_num_view norm, q, k, v, rq, rk, a, h, gate, params, weight;
  char name[256];
  bool diag = getenv("DS4_SUITE_GRAPH_DIAGNOSTICS") && layer == 0;
#define DIAG(label, view)                                                      \
  do {                                                                         \
    if (diag)                                                                  \
      QI_TRY(ds4_qwen_image_observe(g->num, o, label, phase, view));           \
  } while (0)
  QI_TRY(qi_alloc(&w, size, DS4_NUM_BF16, &norm));
  QI_TRY(qi_alloc(&w, size, DS4_NUM_BF16, &q));
  QI_TRY(qi_alloc(&w, keys, DS4_NUM_BF16, &k));
  QI_TRY(qi_alloc(&w, keys, DS4_NUM_BF16, &v));
  QI_TRY(qi_alloc(&w, size, DS4_NUM_BF16, &rq));
  QI_TRY(qi_alloc(&w, size, DS4_NUM_BF16, &rk));
  QI_TRY(qi_alloc(&w, size, DS4_NUM_BF16, &a));
  QI_TRY(qi_alloc(&w, (uint64_t)n * FF, DS4_NUM_BF16, &h));
  QI_TRY(qi_alloc(&w, (uint64_t)n * FF, DS4_NUM_BF16, &gate));
  QI_TRY(qi_alloc(&w, 2 * D, DS4_NUM_BF16, &params));
  for (uint32_t phase = 0; phase < 2; phase++) {
    QI_TRY(ds4_num_norm(g->num, norm, x, QI_NONE, QI_NONE, D, 1e-6f,
                        DS4_NUM_LAYER, DS4_NUM_SCALE_NONE, false));
    DIAG("norm", norm);
    for (uint32_t r = 0; r < 2; r++)
      QI_TRY(ds4_num_copy(g->num, ds4_num_slice(params, r * D, D),
                          ds4_num_slice(mod, r * 4 * D + phase * 2 * D, D)));
    QI_TRY(modulate(g, norm, params, prefix, n));
    DIAG("modulated", norm);
    if (!phase) {
      ds4_num_view kt = ds4_num_slice(k, (uint64_t)offset * D, size),
                   vt = ds4_num_slice(v, (uint64_t)offset * D, size);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.to_q", layer);
      QI_TRY(qi_linear(g, name, q, norm, n, D, D, false));
      DIAG("qproj", q);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.to_k", layer);
      QI_TRY(qi_linear(g, name, kt, norm, n, D, D, false));
      DIAG("kproj", kt);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.to_v", layer);
      QI_TRY(qi_linear(g, name, vt, norm, n, D, D, false));
      DIAG("vproj", vt);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.norm_q.weight",
               layer);
      QI_TRY(qi_weight(g, name, &weight));
      QI_TRY(ds4_num_norm(g->num, q, q, weight, QI_NONE, 128, 1e-6f,
                          DS4_NUM_RMS, DS4_NUM_SCALE_DIRECT, true));
      DIAG("qnorm", q);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.norm_k.weight",
               layer);
      QI_TRY(qi_weight(g, name, &weight));
      QI_TRY(ds4_num_norm(g->num, kt, kt, weight, QI_NONE, 128, 1e-6f,
                          DS4_NUM_RMS, DS4_NUM_SCALE_DIRECT, true));
      DIAG("knorm", kt);
      ds4_num_view co = ds4_num_slice(ds4_num_data(p->cos),
                                      (uint64_t)offset * 64, (uint64_t)n * 64),
                   si = ds4_num_slice(ds4_num_data(p->sin),
                                      (uint64_t)offset * 64, (uint64_t)n * 64);
      QI_TRY(ds4_num_rope_complex(g->num, rq, q, co, si, 32, 128));
      QI_TRY(ds4_num_rope_complex(g->num, rk, kt, co, si, 32, 128));
      DIAG("rope_q", rq);
      DIAG("rope_k", rk);
      QI_TRY(ds4_num_copy(g->num, kt, rk));
      if (p->host_cache && p->prefix) {
        uint64_t elements = (uint64_t)p->prefix * D;
        if (!p->host_kv)
          QI_TRY(qi_host(g->num, elements * 4 * 32, (void **)&p->host_kv,
                         &p->host_account));
        uint16_t *kh = p->host_kv + elements * layer * 2, *vh = kh + elements;
        ds4_num_view kp = ds4_num_slice(k, 0, elements),
                     vp = ds4_num_slice(v, 0, elements);
        if (offset) {
          QI_TRY(ds4_num_write(g->num, kp, kh, elements * 2));
          QI_TRY(ds4_num_write(g->num, vp, vh, elements * 2));
        } else {
          QI_TRY(ds4_num_read(g->num, kp, kh, elements * 2));
          QI_TRY(ds4_num_read(g->num, vp, vh, elements * 2));
        }
      } else if (offset) {
        QI_TRY(ds4_num_copy(g->num,
                            ds4_num_slice(k, 0, (uint64_t)p->prefix * D),
                            ds4_num_data(p->k[layer])));
        QI_TRY(ds4_num_copy(g->num,
                            ds4_num_slice(v, 0, (uint64_t)p->prefix * D),
                            ds4_num_data(p->v[layer])));
      } else if (p->prefix) {
        QI_TRY(ds4_num_alloc(g->num, (uint64_t)p->prefix * D, DS4_NUM_BF16,
                             &p->k[layer]));
        QI_TRY(ds4_num_alloc(g->num, (uint64_t)p->prefix * D, DS4_NUM_BF16,
                             &p->v[layer]));
        QI_TRY(ds4_num_copy(g->num, ds4_num_data(p->k[layer]),
                            ds4_num_slice(k, 0, (uint64_t)p->prefix * D)));
        QI_TRY(ds4_num_copy(g->num, ds4_num_data(p->v[layer]),
                            ds4_num_slice(v, 0, (uint64_t)p->prefix * D)));
      }
      /* Preserve the oracle's segment GEMM geometry as well as its mask.
       * Adjacent image blocks have distinct end bounds. Text segments use
       * their causal row bounds while attending to the preceding prefix. */
      for (uint32_t first = 0; first < n;) {
        uint32_t global = first + offset, end = first + 1;
        bool causal = p->bounds[global * 2 + 1] == global + 1;
        if (causal) {
          while (end < n &&
                 p->bounds[(end + offset) * 2 + 1] == end + offset + 1)
            end++;
        } else {
          while (end < n &&
                 p->bounds[(end + offset) * 2 + 1] == p->bounds[global * 2 + 1])
            end++;
        }
        uint32_t keys = p->bounds[(end - 1 + offset) * 2 + 1];
        QI_TRY(ds4_qwen_image_attention(
            g->num,
            ds4_num_slice(a, (uint64_t)first * D, (uint64_t)(end - first) * D),
            ds4_num_slice(rq, (uint64_t)first * D, (uint64_t)(end - first) * D),
            ds4_num_slice(k, 0, (uint64_t)keys * D),
            ds4_num_slice(v, 0, (uint64_t)keys * D), end - first, keys, 32, 128,
            p->bounds + global * 2));
        first = end;
      }
      DIAG("attended", a);
      snprintf(name, sizeof(name), "transformer_blocks.%u.attn.to_out.0",
               layer);
      QI_TRY(qi_linear(g, name, norm, a, n, D, D, false));
      DIAG("attn_projected", norm);
    } else {
      snprintf(name, sizeof(name), "transformer_blocks.%u.img_mlp.gate_layer",
               layer);
      QI_TRY(qi_linear(g, name, gate, norm, n, D, FF, false));
      DIAG("gate", gate);
      snprintf(name, sizeof(name), "transformer_blocks.%u.img_mlp.proj", layer);
      QI_TRY(qi_linear(g, name, h, norm, n, D, FF, false));
      DIAG("up", h);
      QI_TRY(ds4_num_point(g->num, gate, gate, QI_NONE, DS4_NUM_SILU, 0));
      QI_TRY(ds4_num_point(g->num, h, h, gate, DS4_NUM_MUL, 0));
      snprintf(name, sizeof(name), "transformer_blocks.%u.img_mlp.out", layer);
      QI_TRY(qi_linear(g, name, norm, h, n, FF, D, false));
      DIAG("mlp", norm);
    }
    for (uint32_t r = 0; r < 2; r++)
      QI_TRY(
          ds4_num_copy(g->num, ds4_num_slice(params, r * D, D),
                       ds4_num_slice(mod, r * 4 * D + (phase * 2 + 1) * D, D)));
    QI_TRY(modulate(g, norm, params, prefix, n));
    QI_TRY(ds4_num_point(g->num, x, x, norm, DS4_NUM_ADD, 0));
    DIAG("residual", x);
  }
cleanup:
  {
    ds4_status s = qi_clear(&w);
    return status ? status : s;
  }
#undef DIAG
}
ds4_status ds4_qwen_image_dit_forward(ds4_qwen_image_dit *e,
                                      ds4_qwen_image_dit_input *p,
                                      ds4_num_view target, float sigma,
                                      const ds4_qwen_image_observer *o,
                                      ds4_num_buffer **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !p || p->dit != e || !target.data ||
      target.count != (uint64_t)p->target * 64 || target.type != DS4_NUM_F32 ||
      !isfinite(sigma) || sigma < 0 || sigma > 1)
    return DS4_STATUS_INVALID;
  qi_graph *g = &e->g;
  ds4_runtime_scope scope = {0};
  ds4_status status = ds4_runtime_model_enter(g->model, &scope);
  if (status)
    return status;
  qi_work w = {.num = g->num};
  ds4_num_buffer *result = NULL;
  uint32_t offset = p->cached ? p->prefix : 0, n = p->count - offset;
  ds4_num_view x, temb, mod, scale, images, projected, text, norm, tw;
  QI_TRY(ds4_qwen_image_observe(g->num, o, "begin", 0, QI_NONE));
  /* Resident execution: no hidden streaming or eviction between blocks. */
  QI_TRY(qi_load_all(g));
  QI_TRY(qi_alloc(&w, (uint64_t)n * D, DS4_NUM_BF16, &x));
  QI_TRY(qi_alloc(&w, 2 * D, DS4_NUM_BF16, &temb));
  QI_TRY(qi_alloc(&w, 8 * D, DS4_NUM_BF16, &mod));
  QI_TRY(qi_alloc(&w, 2 * D, DS4_NUM_BF16, &scale));
  QI_TRY(timestep(g, sigma, temb, mod, scale, o));
  uint32_t ni = p->reference + p->target;
  QI_TRY(qi_alloc(&w, (uint64_t)ni * 64, DS4_NUM_BF16, &images));
  QI_TRY(qi_alloc(&w, (uint64_t)ni * D, DS4_NUM_BF16, &projected));
  if (p->reference)
    QI_TRY(ds4_num_copy(g->num,
                        ds4_num_slice(images, 0, (uint64_t)p->reference * 64),
                        ds4_num_data(p->references)));
  QI_TRY(ds4_num_point(g->num,
                       ds4_num_slice(images, (uint64_t)(ni - p->target) * 64,
                                     (uint64_t)p->target * 64),
                       target, QI_NONE, DS4_NUM_CAST, 0));
  QI_TRY(qi_linear(g, "img_in", projected, images, ni, 64, D, false));
  if (offset)
    QI_TRY(ds4_num_copy(g->num, x,
                        ds4_num_slice(projected, (uint64_t)p->reference * D,
                                      (uint64_t)p->target * D)));
  else {
    QI_TRY(qi_alloc(&w, (uint64_t)p->text * D, DS4_NUM_BF16, &text));
    QI_TRY(qi_alloc(&w, (uint64_t)p->text * D, DS4_NUM_BF16, &norm));
    QI_TRY(qi_weight(g, "txt_in.text_norm.weight", &tw));
    QI_TRY(ds4_num_norm(g->num, norm, ds4_num_data(p->conditioning), tw,
                        QI_NONE, D, 1e-6f, DS4_NUM_RMS,
                        DS4_NUM_SCALE_ZERO_CENTERED, false));
    QI_TRY(qi_linear(g, "txt_in.in_layer", text, norm, p->text, D, D, false));
    QI_TRY(ds4_num_point(g->num, text, text, QI_NONE, DS4_NUM_GELU_TANH, 0));
    QI_TRY(qi_linear(g, "txt_in.out_layer", norm, text, p->text, D, D, false));
    for (uint32_t i = 0; i < n;) {
      uint32_t j = i + 1;
      int32_t src = p->source[i];
      while (j < n && p->source[j] == src + (src < 0 ? -(int32_t)(j - i)
                                                     : (int32_t)(j - i)))
        j++;
      uint64_t from = src < 0 ? (uint64_t)(-src - 1) * D : (uint64_t)src * D;
      QI_TRY(ds4_num_copy(
          g->num, ds4_num_slice(x, (uint64_t)i * D, (uint64_t)(j - i) * D),
          ds4_num_slice(src < 0 ? projected : norm, from,
                        (uint64_t)(j - i) * D)));
      i = j;
    }
  }
  QI_TRY(ds4_qwen_image_observe(g->num, o, "joint", 0, x));
  for (uint32_t i = 0; i < 32; i++) {
    QI_TRY(block(g, p, i, x, mod, offset, o));
    QI_TRY(ds4_qwen_image_observe(g->num, o, "block", i, x));
  }
  /* Preserve active-sequence GEMM geometry at the first step. Projecting only
   * a target view changes BF16 ties and can amplify over subsequent steps. */
  ds4_num_view prediction;
  QI_TRY(qi_alloc(&w, (uint64_t)n * 64, DS4_NUM_BF16, &prediction));
  QI_TRY(ds4_num_norm(g->num, x, x, QI_NONE, QI_NONE, D, 1e-6f, DS4_NUM_LAYER,
                      DS4_NUM_SCALE_NONE, false));
  QI_TRY(modulate(g, x, scale, offset ? 0 : p->prefix, n));
  QI_TRY(
      ds4_num_alloc(g->num, (uint64_t)p->target * 64, DS4_NUM_BF16, &result));
  QI_TRY(qi_linear(g, "proj_out", prediction, x, n, D, 64, false));
  QI_TRY(ds4_num_copy(g->num, ds4_num_data(result),
                      ds4_num_slice(prediction, (uint64_t)(n - p->target) * 64,
                                    (uint64_t)p->target * 64)));
  QI_TRY(
      ds4_qwen_image_observe(g->num, o, "velocity", 0, ds4_num_data(result)));
  QI_TRY(ds4_numeric_fence(g->num));
cleanup:
  {
    ds4_status s = qi_clear(&w);
    if (!status)
      status = s;
    if (!status)
      status = ds4_qwen_image_observe(g->num, o, "ready", 0, QI_NONE);
    if (status) {
      clear_cache(p);
      ds4_num_dispose(result);
    } else {
      p->cached = true;
      *out = result;
    }
    ds4_runtime_scope_leave(&scope);
    return status;
  }
}
void ds4_qwen_image_turbo_sigmas(float s[9]) {
  static const float recipe[] = {1.f,      .978453f, .95418f,
                                 .926626f, .89508f,  .845148f,
                                 .704534f, .414568f, 0};
  if (s)
    memcpy(s, recipe, sizeof(recipe));
}
static uint64_t random_u64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
ds4_status ds4_qwen_image_noise(float *out, size_t count, uint64_t seed) {
  if (!out || !count || count > UINT64_C(262144) * 64)
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < count; i += 2) {
    double u = ((random_u64(&seed) >> 11) + .5) * 0x1p-53,
           v = ((random_u64(&seed) >> 11) + .5) * 0x1p-53;
    double r = sqrt(-2 * log(u)), a = 6.2831853071795864769 * v;
    out[i] = (float)(r * cos(a));
    if (i + 1 < count)
      out[i + 1] = (float)(r * sin(a));
  }
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_dit_generate(ds4_qwen_image_dit *e,
                                       ds4_qwen_image_dit_input *p,
                                       const float *noise, uint64_t seed,
                                       const ds4_qwen_image_observer *o,
                                       ds4_num_buffer **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !p || p->dit != e)
    return DS4_STATUS_INVALID;
  /* The outer model scope serializes the entire loop and rejects callback
   * eviction. A failed run returns no partially advanced latent or cache. */
  ds4_runtime_scope scope = {0};
  qi_graph *g = &e->g;
  ds4_status status = ds4_runtime_model_enter(g->model, &scope);
  if (status)
    return status;
  ds4_num_buffer *latent = NULL, *velocity = NULL;
  qi_work w = {.num = g->num};
  ds4_num_view delta, rounded;
  float *host = NULL;
  ds4_resource *ha = NULL;
  uint64_t count = (uint64_t)p->target * 64;
  QI_TRY(clear_cache(p));
  if (!noise) {
    QI_TRY(qi_host(g->num, count * 4, (void **)&host, &ha));
    QI_TRY(ds4_qwen_image_noise(host, count, seed));
    noise = host;
  }
  for (uint64_t i = 0; i < count; i++)
    if (!isfinite(noise[i])) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  QI_TRY(ds4_num_alloc(g->num, count, DS4_NUM_F32, &latent));
  QI_TRY(ds4_num_write(g->num, ds4_num_data(latent), noise, count * 4));
  QI_TRY(qi_alloc(&w, count, DS4_NUM_F32, &delta));
  QI_TRY(qi_alloc(&w, count, DS4_NUM_BF16, &rounded));
  QI_TRY(ds4_num_point(g->num, rounded, ds4_num_data(latent), QI_NONE,
                       DS4_NUM_CAST, 0));
  QI_TRY(ds4_num_point(g->num, ds4_num_data(latent), rounded, QI_NONE,
                       DS4_NUM_CAST, 0));
  float sigmas[9];
  ds4_qwen_image_turbo_sigmas(sigmas);
  for (uint32_t i = 0; i < 8; i++) {
    QI_TRY(ds4_qwen_image_dit_forward(e, p, ds4_num_data(latent), sigmas[i], o,
                                      &velocity));
    /* A zero-dimensional FP32 sigma multiplies a BF16 velocity in BF16
     * under PyTorch scalar promotion; the FP32 sample addition follows. */
    QI_TRY(ds4_num_point(g->num, rounded, ds4_num_data(velocity), QI_NONE,
                         DS4_NUM_SCALE,
                         qi_float(qi_bf16(sigmas[i + 1] - sigmas[i]))));
    QI_TRY(ds4_num_point(g->num, delta, rounded, QI_NONE, DS4_NUM_CAST, 0));
    QI_TRY(ds4_num_point(g->num, ds4_num_data(latent), ds4_num_data(latent),
                         delta, DS4_NUM_ADD, 0));
    QI_TRY(ds4_num_point(g->num, rounded, ds4_num_data(latent), QI_NONE,
                         DS4_NUM_CAST, 0));
    QI_TRY(ds4_num_point(g->num, ds4_num_data(latent), rounded, QI_NONE,
                         DS4_NUM_CAST, 0));
    QI_TRY(ds4_num_free(velocity));
    velocity = NULL;
    QI_TRY(
        ds4_qwen_image_observe(g->num, o, "latent", i, ds4_num_data(latent)));
  }
  QI_TRY(ds4_numeric_fence(g->num));
cleanup:
  {
    ds4_num_dispose(velocity);
    ds4_status s = qi_clear(&w);
    if (!status)
      status = s;
    s = clear_cache(p);
    if (!status)
      status = s;
    if (!status)
      status = ds4_qwen_image_observe(g->num, o, "complete", 0, QI_NONE);
    if (status)
      ds4_num_dispose(latent);
    else
      *out = latent;
    qi_host_free(host, ha);
    ds4_runtime_scope_leave(&scope);
    return status;
  }
}
ds4_status qi_dit_load(ds4_qwen_image_dit *e) {
  ds4_runtime_scope scope = {0};
  ds4_status s = ds4_runtime_model_enter(e->g.model, &scope);
  if (!s)
    s = qi_load_all(&e->g);
  ds4_runtime_scope_leave(&scope);
  return s;
}
void qi_dit_memory(ds4_qwen_image_dit *e, uint32_t count, uint32_t target,
                   uint32_t prefix, uint32_t query_span,
                   ds4_qwen_image_dit_memory *m, uint64_t *resident) {
  /* Bound live storage rather than reserving a second unused layer:
   * block BF16 buffers <=26*N*D bytes; attention keys/scores <=24*N*D,
   * attention queries <=14*Q*D (Q is the largest segment, not all tokens);
   * forward projections <=8.032*N*D; prepared input/generation <=2.4*N*D.
   * Round the N terms up to 62 and add 64 MiB for tile/GEMM scratch. This
   * bounds cached steps too. Request-sized scratch avoids needless spills.
   * Driver allocations remain in the device sample, outside owned storage. */
  *m = (ds4_qwen_image_dit_memory){
      .weights = qi_weight_bytes(&e->g),
      .prefix_cache = (uint64_t)prefix * D * 4 * 32,
      .first_step = ((uint64_t)count * 62 + (uint64_t)query_span * 14) * D +
                    64 * 1024 * 1024,
      .cached_step = (uint64_t)target * D * 100 + (uint64_t)prefix * D * 40 +
                     64 * 1024 * 1024,
      .host = sizeof(ds4_qwen_image_dit_input) + 65536 +
              (uint64_t)count * (12 + 128 * sizeof(float)) +
              (uint64_t)target * 64 * 4};
  *resident = 0;
  for (uint32_t i = 0; i < 33; i++) {
    m->groups[i] = i;
    m->dependencies[i] = i ? i - 1 : UINT32_MAX;
    m->group_bytes[i] = qi_group_bytes(&e->g, i);
    if (ds4_weight_group_resident(e->g.groups[i]))
      *resident += m->group_bytes[i];
  }
}
ds4_status ds4_qwen_image_dit_estimate(ds4_qwen_image_dit *e,
                                       const ds4_qwen_image_dit_input *p,
                                       ds4_qwen_image_dit_memory *m) {
  if (!e || !p || p->dit != e || !m)
    return DS4_STATUS_INVALID;
  uint64_t resident;
  /* Equal end bounds identify image segments. All causal text runs together
   * contain at most p->text queries, including expanded image placeholders. */
  uint32_t span = p->text, run = 0, previous = UINT32_MAX;
  for (uint32_t i = 0; i < p->count; i++) {
    uint32_t end = p->bounds[2 * i + 1];
    run = end == previous ? run + 1 : 1;
    previous = end;
    if (run > span)
      span = run;
  }
  qi_dit_memory(e, p->count, p->target, p->prefix, span, m, &resident);
  if (p->host_cache) {
    m->host += m->prefix_cache;
    m->prefix_cache = 0;
  }
  return DS4_STATUS_OK;
}
