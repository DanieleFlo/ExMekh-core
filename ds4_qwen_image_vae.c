/* Image specialization of the frozen Qwen 2.1 VAE, Diffusers aea7b70b,
 * Apache-2.0. First-frame temporal shortcuts are preserved, not squeezed
 * away by assuming that every temporal factor is one. */
#include "ds4_qwen_image_vae.h"
#include "ds4_qwen_image_attention.h"
#include "ds4_qwen_image_graph.h"
#include "ds4_qwen_image_rgba_internal.h"
#include "ds4_qwen_image_vae_stats.h"
struct ds4_qwen_image_vae {
  qi_graph g;
};
typedef struct {
  ds4_num_buffer *b;
  uint32_t c, h, w;
} va_tensor;
static ds4_num_view va_view(va_tensor x) { return ds4_num_data(x.b); }
static ds4_status va_drop(va_tensor *x) {
  ds4_status s = ds4_num_dispose(x->b);
  *x = (va_tensor){0};
  return s;
}
static ds4_status va_new(qi_graph *g, va_tensor *x, uint32_t c, uint32_t h,
                         uint32_t w) {
  ds4_status s =
      ds4_num_alloc(g->num, (uint64_t)c * h * w, DS4_NUM_BF16, &x->b);
  if (!s) {
    x->c = c;
    x->h = h;
    x->w = w;
  }
  return s;
}
static ds4_status bind_conv(qi_graph *g, const char *name, uint32_t group,
                            uint32_t in, uint32_t out, uint32_t k) {
  char key[256];
  uint64_t shape[] = {k, k, in, out};
  snprintf(key, sizeof(key), "%s.weight", name);
  ds4_status s = qi_bind(g, key, group, 4, shape);
  if (s)
    return s;
  shape[0] = out;
  snprintf(key, sizeof(key), "%s.bias", name);
  return qi_bind(g, key, group, 1, shape);
}
static ds4_status bind_norm(qi_graph *g, const char *name, uint32_t group,
                            uint32_t c, bool image) {
  char key[256];
  snprintf(key, sizeof(key), "%s.gamma", name);
  uint64_t shape[] = {1, 1, 1, c};
  if (image) {
    shape[2] = c;
    return qi_bind(g, key, group, 3, shape);
  }
  return qi_bind(g, key, group, 4, shape);
}
static ds4_status bind_res(qi_graph *g, const char *name, uint32_t group,
                           uint32_t in, uint32_t out) {
  ds4_status status;
  char key[256];
  snprintf(key, sizeof(key), "%s.norm1", name);
  QI_TRY(bind_norm(g, key, group, in, false));
  snprintf(key, sizeof(key), "%s.norm2", name);
  QI_TRY(bind_norm(g, key, group, out, false));
  snprintf(key, sizeof(key), "%s.conv1", name);
  QI_TRY(bind_conv(g, key, group, in, out, 3));
  snprintf(key, sizeof(key), "%s.conv2", name);
  QI_TRY(bind_conv(g, key, group, out, out, 3));
  if (in != out) {
    snprintf(key, sizeof(key), "%s.conv_shortcut", name);
    QI_TRY(bind_conv(g, key, group, in, out, 1));
  }
cleanup:
  return status;
}
static ds4_status va_bind(qi_graph *g) {
  ds4_status status;
  char key[256];
  QI_TRY(bind_conv(g, "quant_conv", 0, 128, 128, 1));
  QI_TRY(bind_conv(g, "post_quant_conv", 1, 64, 64, 1));
  for (uint32_t dec = 0; dec < 2; dec++) {
    const char *side = dec ? "decoder" : "encoder";
    uint32_t base = dec ? 144 : 96, max = base * 8, group = dec;
    snprintf(key, sizeof(key), "%s.conv_in", side);
    QI_TRY(bind_conv(g, key, group, dec ? 64 : 4, dec ? max : base, 3));
    snprintf(key, sizeof(key), "%s.conv_out", side);
    QI_TRY(bind_conv(g, key, group, dec ? base : max, dec ? 4 : 128, 3));
    snprintf(key, sizeof(key), "%s.norm_out", side);
    QI_TRY(bind_norm(g, key, group, dec ? base : max, false));
    for (uint32_t r = 0; r < 2; r++) {
      snprintf(key, sizeof(key), "%s.mid_block.resnets.%u", side, r);
      QI_TRY(bind_res(g, key, group, max, max));
    }
    snprintf(key, sizeof(key), "%s.mid_block.attentions.0.norm", side);
    QI_TRY(bind_norm(g, key, group, max, true));
    snprintf(key, sizeof(key), "%s.mid_block.attentions.0.to_qkv", side);
    QI_TRY(bind_conv(g, key, group, max, 3 * max, 1));
    snprintf(key, sizeof(key), "%s.mid_block.attentions.0.proj", side);
    QI_TRY(bind_conv(g, key, group, max, max, 1));
    uint32_t mul_e[] = {1, 1, 2, 4, 8, 8}, mul_d[] = {8, 8, 8, 4, 2, 1};
    uint32_t *mul = dec ? mul_d : mul_e;
    for (uint32_t i = 0; i < 5; i++) {
      uint32_t in = base * mul[i], out = base * mul[i + 1];
      for (uint32_t r = 0; r < (dec ? 3u : 2u); r++) {
        snprintf(key, sizeof(key), "%s.%s_blocks.%u.resnets.%u", side,
                 dec ? "up" : "down", i, r);
        QI_TRY(bind_res(g, key, group, r ? out : in, out));
      }
      if (i < 4) {
        snprintf(key, sizeof(key), "%s.%s_blocks.%u.%ssampler.resample.1", side,
                 dec ? "up" : "down", i, dec ? "up" : "down");
        QI_TRY(bind_conv(g, key, group, out, out, 3));
        if (dec ? i < 3 : i > 0) {
          snprintf(key, sizeof(key), "%s.%s_blocks.%u.%ssampler.time_conv",
                   side, dec ? "up" : "down", i, dec ? "up" : "down");
          QI_TRY(bind_conv(g, key, group, out, dec ? out * 2 : out, 1));
        }
      }
    }
  }
cleanup:
  return status;
}
ds4_status ds4_qwen_image_vae_open(ds4_qwen_image_vae **out, ds4_bundle *b,
                                   ds4_numeric *n, ds4_runtime *runtime) {
  if (!out || !b || !n || !runtime)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_qwen_image_vae *e = NULL;
  ds4_resource *account = NULL;
  ds4_status status = qi_host(n, sizeof(*e), (void **)&e, &account);
  if (status)
    return status;
  e->g.account = account;
  QI_TRY(qi_init(&e->g, b, n, "vae", 2));
  QI_TRY(va_bind(&e->g));
  QI_TRY(qi_finish(&e->g, runtime, "vae"));
  *out = e;
  return DS4_STATUS_OK;
cleanup:
  qi_close(&e->g);
  qi_host_free(e, account);
  return status;
}
ds4_status ds4_qwen_image_vae_release_weights(ds4_qwen_image_vae *e) {
  return e ? qi_release(&e->g) : DS4_STATUS_INVALID;
}
ds4_status ds4_qwen_image_vae_close(ds4_qwen_image_vae *e) {
  if (!e)
    return DS4_STATUS_OK;
  ds4_status s = qi_close(&e->g);
  if (!s)
    qi_host_free(e, e->g.account);
  return s;
}
static ds4_status va_conv(qi_graph *g, const char *name, va_tensor x,
                          uint32_t out, uint32_t k, uint32_t stride,
                          uint32_t before, uint32_t after, va_tensor *y) {
  ds4_status status;
  ds4_num_view w, b;
  char key[256];
  snprintf(key, sizeof(key), "%s.weight", name);
  QI_TRY(qi_weight(g, key, &w));
  snprintf(key, sizeof(key), "%s.bias", name);
  QI_TRY(qi_weight(g, key, &b));
  uint32_t h = (x.h + before + after - k) / stride + 1,
           ww = (x.w + before + after - k) / stride + 1;
  QI_TRY(va_new(g, y, out, h, ww));
  ds4_num_convolution c = {.batch = 1,
                           .in_channels = x.c,
                           .out_channels = out,
                           .groups = 1,
                           .input = {1, x.h, x.w},
                           .kernel = {1, k, k},
                           .stride = {1, stride, stride},
                           .dilation = {1, 1, 1},
                           .pad_before = {0, before, before},
                           .pad_after = {0, after, after}};
  QI_TRY(ds4_num_conv(g->num, va_view(*y), va_view(x), w, b, &c));
cleanup:
  if (status)
    va_drop(y);
  return status;
}
static ds4_status va_norm(qi_graph *g, const char *name, va_tensor x,
                          va_tensor *y, bool silu) {
  qi_work w = {.num = g->num};
  ds4_status status;
  ds4_num_view trans, gamma;
  char key[256];
  uint64_t s = (uint64_t)x.h * x.w;
  QI_TRY(qi_alloc(&w, s * x.c, DS4_NUM_BF16, &trans));
  ds4_num_reindex ix = {.rank = 2, .shape = {s, x.c}, .stride = {1, s}};
  QI_TRY(ds4_num_reorder(g->num, trans, va_view(x), &ix));
  /* Wan/Qwen RMS uses L2 normalization rounded to BF16, then sqrt(C),
   * then gamma, each with its own rounding. RMSNorm(eps) is not equivalent. */
  QI_TRY(ds4_num_norm(g->num, trans, trans, QI_NONE, QI_NONE, x.c, 1e-12f,
                      DS4_NUM_L2, DS4_NUM_SCALE_NONE, false));
  QI_TRY(
      ds4_num_point(g->num, trans, trans, QI_NONE, DS4_NUM_SCALE, sqrtf(x.c)));
  snprintf(key, sizeof(key), "%s.gamma", name);
  QI_TRY(qi_weight(g, key, &gamma));
  QI_TRY(ds4_num_point(g->num, trans, trans, gamma, DS4_NUM_MUL, 0));
  if (silu)
    QI_TRY(ds4_num_point(g->num, trans, trans, QI_NONE, DS4_NUM_SILU, 0));
  QI_TRY(va_new(g, y, x.c, x.h, x.w));
  ix = (ds4_num_reindex){.rank = 2, .shape = {x.c, s}, .stride = {1, x.c}};
  QI_TRY(ds4_num_reorder(g->num, va_view(*y), trans, &ix));
cleanup: {
  ds4_status t = qi_clear(&w);
  if (!status)
    status = t;
  if (status)
    va_drop(y);
  return status;
}
}
static ds4_status va_res(qi_graph *g, const char *name, va_tensor *x,
                         uint32_t out) {
  ds4_status status;
  va_tensor n = {0}, a = {0}, b = {0}, shortcut = {0};
  char key[256];
  snprintf(key, sizeof(key), "%s.norm1", name);
  QI_TRY(va_norm(g, key, *x, &n, true));
  snprintf(key, sizeof(key), "%s.conv1", name);
  QI_TRY(va_conv(g, key, n, out, 3, 1, 1, 1, &a));
  QI_TRY(va_drop(&n));
  snprintf(key, sizeof(key), "%s.norm2", name);
  QI_TRY(va_norm(g, key, a, &n, true));
  QI_TRY(va_drop(&a));
  snprintf(key, sizeof(key), "%s.conv2", name);
  QI_TRY(va_conv(g, key, n, out, 3, 1, 1, 1, &b));
  QI_TRY(va_drop(&n));
  ds4_num_view skip = va_view(*x);
  if (x->c != out) {
    snprintf(key, sizeof(key), "%s.conv_shortcut", name);
    QI_TRY(va_conv(g, key, *x, out, 1, 1, 0, 0, &shortcut));
    skip = va_view(shortcut);
  }
  QI_TRY(ds4_num_point(g->num, va_view(b), va_view(b), skip, DS4_NUM_ADD, 0));
  QI_TRY(va_drop(x));
  *x = b;
  b = (va_tensor){0};
cleanup: {
  ds4_status s = va_drop(&n), t = va_drop(&a);
  if (!s)
    s = t;
  t = va_drop(&b);
  if (!s)
    s = t;
  t = va_drop(&shortcut);
  if (!s)
    s = t;
  return status ? status : s;
}
}
/* The VAE uses a single wide head, with the same FP32 SDPA intermediates as
 * the DiT. Query tiles remain bounded even at the full latent resolution. */
static ds4_status va_attention(qi_graph *g, const char *side, va_tensor *x) {
  ds4_status status;
  qi_work work = {.num = g->num};
  va_tensor n = {0}, qkv = {0}, a = {0}, y = {0};
  ds4_num_view q, k, v, attended;
  uint32_t count = x->h * x->w, c = x->c;
  uint32_t *bounds = NULL;
  ds4_resource *account = NULL;
  char key[256];
  snprintf(key, sizeof(key), "%s.mid_block.attentions.0.norm", side);
  QI_TRY(va_norm(g, key, *x, &n, false));
  snprintf(key, sizeof(key), "%s.mid_block.attentions.0.to_qkv", side);
  QI_TRY(va_conv(g, key, n, c * 3, 1, 1, 0, 0, &qkv));
  QI_TRY(va_drop(&n));
  QI_TRY(qi_alloc(&work, (uint64_t)count * c, DS4_NUM_BF16, &q));
  QI_TRY(qi_alloc(&work, (uint64_t)count * c, DS4_NUM_BF16, &k));
  QI_TRY(qi_alloc(&work, (uint64_t)count * c, DS4_NUM_BF16, &v));
  QI_TRY(qi_alloc(&work, (uint64_t)count * c, DS4_NUM_BF16, &attended));
  ds4_num_view views[] = {q, k, v};
  ds4_num_reindex ix = {.rank = 2, .shape = {count, c}, .stride = {1, count}};
  for (uint32_t i = 0; i < 3; i++)
    QI_TRY(ds4_num_reorder(g->num, views[i],
                           ds4_num_slice(va_view(qkv), (uint64_t)i * count * c,
                                         (uint64_t)count * c),
                           &ix));
  QI_TRY(qi_host(g->num, (uint64_t)count * 2 * sizeof(uint32_t),
                 (void **)&bounds, &account));
  for (uint32_t i = 0; i < count; i++)
    bounds[i * 2 + 1] = count;
  QI_TRY(ds4_qwen_image_attention(g->num, attended, q, k, v, count, count, 1, c,
                                  bounds));
  QI_TRY(va_new(g, &a, c, x->h, x->w));
  ix = (ds4_num_reindex){.rank = 2, .shape = {c, count}, .stride = {1, c}};
  QI_TRY(ds4_num_reorder(g->num, va_view(a), attended, &ix));
  snprintf(key, sizeof(key), "%s.mid_block.attentions.0.proj", side);
  QI_TRY(va_conv(g, key, a, c, 1, 1, 0, 0, &y));
  QI_TRY(ds4_num_point(g->num, va_view(y), va_view(y), va_view(*x), DS4_NUM_ADD,
                       0));
  QI_TRY(va_drop(x));
  *x = y;
  y = (va_tensor){0};
cleanup: {
  ds4_status s = qi_clear(&work), t = va_drop(&n);
  if (!s)
    s = t;
  t = va_drop(&qkv);
  if (!s)
    s = t;
  t = va_drop(&a);
  if (!s)
    s = t;
  t = va_drop(&y);
  if (!s)
    s = t;
  qi_host_free(bounds, account);
  return status ? status : s;
}
}
static ds4_status va_mid(qi_graph *g, const char *side, va_tensor *x,
                         const ds4_qwen_image_observer *o) {
  ds4_status status;
  char key[256];
  snprintf(key, sizeof(key), "%s.mid_block.resnets.0", side);
  QI_TRY(va_res(g, key, x, x->c));
  QI_TRY(va_attention(g, side, x));
  snprintf(key, sizeof(key), "%s.mid_block.resnets.1", side);
  QI_TRY(va_res(g, key, x, x->c));
  QI_TRY(ds4_qwen_image_observe(g->num, o, side, 5, va_view(*x)));
cleanup:
  return status;
}
static ds4_status va_down_shortcut(qi_graph *g, va_tensor x, uint32_t out,
                                   uint32_t fs, uint32_t ft, va_tensor *y) {
  qi_work work = {.num = g->num};
  ds4_status status;
  ds4_num_view patches;
  uint32_t h = x.h / fs, w = x.w / fs, group = x.c * ft * fs * fs / out,
           indices[8192];
  if ((uint64_t)out * group > 8192)
    return DS4_STATUS_INVALID;
  QI_TRY(qi_alloc(&work, (uint64_t)x.c * x.h * x.w, DS4_NUM_BF16, &patches));
  ds4_num_reindex ix = {
      .rank = 5,
      .shape = {x.c, fs, fs, h, w},
      .stride = {(uint64_t)x.h * x.w, x.w, 1, (uint64_t)fs * x.w, fs}};
  QI_TRY(ds4_num_reorder(g->num, patches, va_view(x), &ix));
  for (uint32_t i = 0; i < out * group; i++)
    indices[i] = (i / (fs * fs)) % ft == ft - 1
                     ? (i / (ft * fs * fs)) * fs * fs + i % (fs * fs)
                     : UINT32_MAX;
  QI_TRY(va_new(g, y, out, h, w));
  QI_TRY(ds4_num_pool(g->num, va_view(*y), patches, indices, out, group, h * w,
                      group));
cleanup: {
  ds4_status s = qi_clear(&work);
  if (!status)
    status = s;
  if (status)
    va_drop(y);
  return status;
}
}
static ds4_status va_up_shortcut(qi_graph *g, va_tensor x, uint32_t out,
                                 uint32_t ft, va_tensor *y) {
  qi_work work = {.num = g->num};
  ds4_status status;
  ds4_num_view repeated;
  uint32_t repeats = out * ft * 4 / x.c, indices[9216], rows = out * ft * 4;
  uint64_t s = (uint64_t)x.h * x.w;
  if (rows > 9216 || !repeats)
    return DS4_STATUS_INVALID;
  for (uint32_t i = 0; i < rows; i++)
    indices[i] = i / repeats;
  QI_TRY(qi_alloc(&work, (uint64_t)rows * s, DS4_NUM_BF16, &repeated));
  QI_TRY(ds4_num_pool(g->num, repeated, va_view(x), indices, rows, 1, s, 1));
  QI_TRY(va_new(g, y, out, x.h * 2, x.w * 2));
  ds4_num_reindex ix = {.rank = 5,
                        .shape = {out, x.h, 2, x.w, 2},
                        .stride = {ft * 4 * s, x.w, 2 * s, 1, s}};
  QI_TRY(ds4_num_reorder(g->num, va_view(*y),
                         ds4_num_slice(repeated, (ft - 1) * 4 * s,
                                       repeated.count - (ft - 1) * 4 * s),
                         &ix));
cleanup: {
  ds4_status t = qi_clear(&work);
  if (!status)
    status = t;
  if (status)
    va_drop(y);
  return status;
}
}
static ds4_status va_stage(qi_graph *g, va_tensor *x, uint32_t i, bool decode,
                           const ds4_qwen_image_observer *o) {
  ds4_status status;
  va_tensor shortcut = {0}, y = {0}, sample = {0};
  char key[256];
  uint32_t mul_e[] = {1, 2, 4, 8, 8}, mul_d[] = {8, 8, 4, 2, 1},
           out = decode ? 144 * mul_d[i] : 96 * mul_e[i];
  const char *side = decode ? "decoder" : "encoder",
             *direction = decode ? "up" : "down";
  if (!decode)
    QI_TRY(va_down_shortcut(g, *x, out, i < 4 ? 2 : 1, (i > 0 && i < 4) ? 2 : 1,
                            &shortcut));
  else if (i < 4)
    QI_TRY(va_up_shortcut(g, *x, out, i < 3 ? 2 : 1, &shortcut));
  for (uint32_t r = 0; r < (decode ? 3u : 2u); r++) {
    snprintf(key, sizeof(key), "%s.%s_blocks.%u.resnets.%u", side, direction, i,
             r);
    QI_TRY(va_res(g, key, x, out));
  }
  if (i < 4) {
    snprintf(key, sizeof(key), "%s.%s_blocks.%u.%ssampler.resample.1", side,
             direction, i, direction);
    if (decode) {
      QI_TRY(va_new(g, &sample, x->c, x->h * 2, x->w * 2));
      ds4_num_resample rs = {.batch = 1,
                             .channels = x->c,
                             .input = {1, x->h, x->w},
                             .output = {1, x->h * 2, x->w * 2}};
      QI_TRY(ds4_num_nearest(g->num, va_view(sample), va_view(*x), &rs));
      QI_TRY(va_conv(g, key, sample, out, 3, 1, 1, 1, &y));
    } else
      QI_TRY(va_conv(g, key, *x, out, 3, 2, 0, 1, &y));
    QI_TRY(va_drop(x));
    *x = y;
    y = (va_tensor){0};
  }
  if (shortcut.b)
    QI_TRY(ds4_num_point(g->num, va_view(*x), va_view(*x), va_view(shortcut),
                         DS4_NUM_ADD, 0));
  QI_TRY(ds4_qwen_image_observe(g->num, o, side, i, va_view(*x)));
cleanup: {
  ds4_status s = va_drop(&shortcut), t = va_drop(&y);
  if (!s)
    s = t;
  t = va_drop(&sample);
  if (!s)
    s = t;
  return status ? status : s;
}
}
static ds4_status va_head(qi_graph *g, const char *side, va_tensor *x,
                          uint32_t out) {
  ds4_status status;
  va_tensor n = {0}, y = {0};
  char key[256];
  snprintf(key, sizeof(key), "%s.norm_out", side);
  QI_TRY(va_norm(g, key, *x, &n, true));
  snprintf(key, sizeof(key), "%s.conv_out", side);
  QI_TRY(va_conv(g, key, n, out, 3, 1, 1, 1, &y));
  QI_TRY(va_drop(x));
  *x = y;
  y = (va_tensor){0};
cleanup: {
  ds4_status s = va_drop(&n), t = va_drop(&y);
  if (!s)
    s = t;
  return status ? status : s;
}
}
/* The pipeline casts the per-channel constants to BF16, then rounds each
 * subtraction/division (or multiplication/addition) separately. */
static ds4_status va_normalize(qi_graph *g, ds4_num_view x, uint32_t spatial,
                               bool encode) {
  ds4_status status;
  qi_work work = {.num = g->num};
  ds4_num_view means, stds, scale;
  uint16_t mean[64], std[64];
  for (uint32_t c = 0; c < 64; c++) {
    mean[c] = qi_bf16(qi_vae_mean[c]);
    std[c] = qi_bf16(qi_vae_std[c]);
  }
  QI_TRY(qi_alloc(&work, 64, DS4_NUM_BF16, &means));
  QI_TRY(qi_alloc(&work, 64, DS4_NUM_BF16, &stds));
  QI_TRY(qi_alloc(&work, spatial, DS4_NUM_BF16, &scale));
  QI_TRY(ds4_num_write(g->num, means, mean, sizeof(mean)));
  QI_TRY(ds4_num_write(g->num, stds, std, sizeof(std)));
  /* Apply in channel-major order; scalar SCALE uses the exact reciprocal of
   * the BF16 standard deviation, matching BF16 division's FP32 arithmetic. */
  for (uint32_t c = 0; c < 64; c++) {
    ds4_num_view channel = ds4_num_slice(x, (uint64_t)c * spatial, spatial);
    if (encode) {
      QI_TRY(ds4_num_point(g->num, scale, ds4_num_slice(means, c, 1), QI_NONE,
                           DS4_NUM_SCALE, -1));
      QI_TRY(ds4_num_point(g->num, channel, channel, ds4_num_slice(scale, 0, 1),
                           DS4_NUM_ADD, 0));
      QI_TRY(ds4_num_point(g->num, channel, channel, QI_NONE, DS4_NUM_SCALE,
                           1 / qi_float(std[c])));
    } else {
      QI_TRY(ds4_num_point(g->num, channel, channel, ds4_num_slice(stds, c, 1),
                           DS4_NUM_MUL, 0));
      QI_TRY(ds4_num_point(g->num, channel, channel, ds4_num_slice(means, c, 1),
                           DS4_NUM_ADD, 0));
    }
  }
cleanup: {
  ds4_status s = qi_clear(&work);
  return status ? status : s;
}
}
ds4_status ds4_qwen_image_vae_encode(ds4_qwen_image_vae *e,
                                     const ds4_qwen_image_reference *r,
                                     const ds4_qwen_image_observer *o,
                                     ds4_num_buffer **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !r || !r->pixels || !r->width || !r->height || r->width % 16 ||
      r->height % 16 || r->width > 16384 || r->height > 16384 ||
      (uint64_t)r->width * r->height > 64 * 1024 * 1024 ||
      (r->channels != 3 && r->channels != 4))
    return DS4_STATUS_INVALID;
  qi_graph *g = &e->g;
  ds4_runtime_scope scope = {0};
  ds4_status status = ds4_runtime_model_enter(g->model, &scope);
  if (status)
    return status;
  va_tensor x = {0}, y = {0};
  ds4_num_buffer *result = NULL;
  uint16_t *host = NULL;
  ds4_resource *account = NULL;
  uint64_t s = (uint64_t)r->width * r->height;
  QI_TRY(ds4_qwen_image_observe(g->num, o, "begin", 0, QI_NONE));
  QI_TRY(qi_host(g->num, s * 8, (void **)&host, &account));
  for (uint32_t c = 0; c < 4; c++)
    for (uint64_t i = 0; i < s; i++)
      host[c * s + i] = qi_bf16(
          c == 3 && r->channels == 3
              ? 1.f
              : (float)r->pixels[i * r->channels + c] / 255.f * 2.f - 1.f);
  QI_TRY(va_new(g, &x, 4, r->height, r->width));
  QI_TRY(ds4_num_write(g->num, va_view(x), host, s * 8));
  qi_host_free(host, account);
  host = NULL;
  account = NULL;
  QI_TRY(va_conv(g, "encoder.conv_in", x, 96, 3, 1, 1, 1, &y));
  QI_TRY(va_drop(&x));
  x = y;
  y = (va_tensor){0};
  for (uint32_t i = 0; i < 5; i++)
    QI_TRY(va_stage(g, &x, i, false, o));
  QI_TRY(va_mid(g, "encoder", &x, o));
  QI_TRY(va_head(g, "encoder", &x, 128));
  QI_TRY(va_conv(g, "quant_conv", x, 128, 1, 1, 0, 0, &y));
  ds4_num_view mode = ds4_num_slice(va_view(y), 0, (uint64_t)64 * y.h * y.w);
  QI_TRY(ds4_qwen_image_observe(g->num, o, "posterior", 0, mode));
  QI_TRY(va_normalize(g, mode, y.h * y.w, true));
  QI_TRY(ds4_num_alloc(g->num, mode.count, DS4_NUM_BF16, &result));
  ds4_num_reindex ix = {.rank = 2,
                        .shape = {(uint64_t)y.h * y.w, 64},
                        .stride = {1, (uint64_t)y.h * y.w}};
  QI_TRY(ds4_num_reorder(g->num, ds4_num_data(result), mode, &ix));
  QI_TRY(ds4_qwen_image_observe(g->num, o, "encoded", 0, ds4_num_data(result)));
cleanup: {
  ds4_status s = va_drop(&x), t = va_drop(&y);
  if (!s)
    s = t;
  if (!status)
    status = s;
  if (!status)
    status = ds4_numeric_fence(g->num);
  if (!status)
    status = ds4_qwen_image_observe(g->num, o, "complete", 0, QI_NONE);
  if (status)
    ds4_num_dispose(result);
  else
    *out = result;
  qi_host_free(host, account);
  ds4_runtime_scope_leave(&scope);
  return status;
}
}
static uint64_t va_decode_bytes(uint32_t h, uint32_t w, uint32_t core) {
  uint64_t th = core && h > core + 28 ? core + 28 : h,
           tw = core && w > core + 28 ? core + 28 : w;
  /* Includes full low-res attention and all overlapping norm/residual views,
   * conv workspace (64 MiB), plus padding and library staging margin. */
  return (uint64_t)h * w * 1152 * 64 + th * tw * 256 * 144 * 24 +
         128 * 1024 * 1024;
}
static uint64_t va_decode_host(uint32_t h, uint32_t w, uint32_t core) {
  uint64_t th = core && core + 28 < h ? core + 28 : h;
  uint64_t tw = core && core + 28 < w ? core + 28 : w;
  /* RGBA result, blend weights, BF16 tile readback (including its halo),
   * low-resolution attention bounds and the bounded per-tile range table. */
  return (uint64_t)h * w * 256 * 4 + th * tw * 256 * 8 + (uint64_t)h * w * 8 +
         65536 + ds4_qwen_image_rgba_bytes(w * 16, h * 16);
}
static ds4_status va_plan(ds4_qwen_image_vae *e, uint32_t h, uint32_t w,
                          uint32_t core, ds4_qwen_image_vae_plan *p) {
  if (!p)
    return DS4_STATUS_INVALID;
  *p = (ds4_qwen_image_vae_plan){0};
  if (!e || !h || !w || h > 1024 || w > 1024 || (uint64_t)h * w > 262144 ||
      (core && (core < 2 || core > (h > w ? h : w))))
    return DS4_STATUS_INVALID;
  *p = (ds4_qwen_image_vae_plan){.latent_height = h,
                                 .latent_width = w,
                                 .core = core,
                                 .halo = core ? 14 : 0,
                                 .overlap = core ? 1 : 0,
                                 .weights = qi_weight_bytes(&e->g),
                                 .device = va_decode_bytes(h, w, core),
                                 .host = va_decode_host(h, w, core)};
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_vae_plan_tiled(ds4_qwen_image_vae *e, uint32_t h,
                                         uint32_t w, uint32_t core,
                                         ds4_qwen_image_vae_plan *p) {
  if (!core) {
    if (p)
      *p = (ds4_qwen_image_vae_plan){0};
    return DS4_STATUS_INVALID;
  }
  return va_plan(e, h, w, core, p);
}
ds4_status ds4_qwen_image_vae_plan_decode(ds4_qwen_image_vae *e, uint32_t h,
                                          uint32_t w, uint64_t budget,
                                          ds4_qwen_image_vae_plan *p) {
  ds4_status status = va_plan(e, h, w, 0, p);
  if (status)
    return status;
  if (budget >= p->weights && p->device <= budget - p->weights)
    return DS4_STATUS_OK;
  uint32_t core = h > w ? h : w;
  while (core > 2) {
    core = (core + 1) / 2;
    va_plan(e, h, w, core, p);
    if (budget >= p->weights && p->device <= budget - p->weights)
      return DS4_STATUS_OK;
  }
  *p = (ds4_qwen_image_vae_plan){0};
  return DS4_STATUS_NOMEM;
}
static ds4_status va_decode_body(qi_graph *g, va_tensor *x,
                                 const ds4_qwen_image_observer *o) {
  ds4_status status;
  for (uint32_t i = 0; i < 5; i++)
    QI_TRY(va_stage(g, x, i, true, o));
  QI_TRY(va_head(g, "decoder", x, 4));
cleanup:
  return status;
}
ds4_status ds4_qwen_image_vae_decode(ds4_qwen_image_vae *e, ds4_num_view latent,
                                     const ds4_qwen_image_vae_plan *plan,
                                     const ds4_qwen_image_observer *o,
                                     ds4_qwen_image_rgba **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_qwen_image_vae_plan expected;
  if (!plan ||
      va_plan(e, plan->latent_height, plan->latent_width, plan->core,
              &expected) ||
      plan->halo != expected.halo || plan->overlap != expected.overlap ||
      plan->weights != expected.weights || plan->device != expected.device ||
      plan->host != expected.host || !latent.data ||
      (latent.type != DS4_NUM_F32 && latent.type != DS4_NUM_BF16) ||
      latent.count != (uint64_t)plan->latent_height * plan->latent_width * 64)
    return DS4_STATUS_INVALID;
  qi_graph *g = &e->g;
  ds4_runtime_scope scope = {0};
  ds4_status status = ds4_runtime_model_enter(g->model, &scope);
  if (status)
    return status;
  qi_work work = {.num = g->num};
  va_tensor x = {0}, y = {0}, tile = {0};
  ds4_num_view packed;
  ds4_qwen_image_rgba *image = NULL;
  ds4_resource *temporary = NULL;
  float *host = NULL, *weight = NULL, *rgba = NULL;
  ds4_resource *wa = NULL;
  uint32_t h = plan->latent_height, w = plan->latent_width, oh = h * 16,
           ow = w * 16;
  uint64_t pixels = (uint64_t)oh * ow;
  QI_TRY(ds4_qwen_image_observe(g->num, o, "begin", 0, QI_NONE));
  uint64_t owner;
  ds4_resources *resources = ds4_numeric_resources(g->num, &owner);
  QI_TRY(ds4_qwen_image_rgba_create(resources, owner, ow, oh, &image, &rgba));
  QI_TRY(qi_alloc(&work, latent.count, DS4_NUM_BF16, &packed));
  QI_TRY(ds4_num_point(g->num, packed, latent, QI_NONE, DS4_NUM_CAST, 0));
  QI_TRY(va_new(g, &x, 64, h, w));
  ds4_num_reindex ix = {
      .rank = 2, .shape = {64, (uint64_t)h * w}, .stride = {1, 64}};
  QI_TRY(ds4_num_reorder(g->num, va_view(x), packed, &ix));
  QI_TRY(va_normalize(g, va_view(x), h * w, false));
  QI_TRY(va_conv(g, "post_quant_conv", x, 64, 1, 1, 0, 0, &y));
  QI_TRY(va_drop(&x));
  x = y;
  y = (va_tensor){0};
  QI_TRY(va_conv(g, "decoder.conv_in", x, 1152, 3, 1, 1, 1, &y));
  QI_TRY(va_drop(&x));
  x = y;
  y = (va_tensor){0};
  QI_TRY(va_mid(g, "decoder", &x, o));
  uint32_t core = plan->core ? plan->core : (h > w ? h : w),
           step = plan->core ? core - plan->overlap : core;
  QI_TRY(qi_host(g->num, pixels * 4, (void **)&weight, &wa));
  for (uint32_t top = 0; top < h; top += step)
    for (uint32_t left = 0; left < w; left += step) {
      uint32_t bottom = top + core < h ? top + core : h,
               right = left + core < w ? left + core : w;
      uint32_t a = top > plan->halo ? top - plan->halo : 0,
               b = left > plan->halo ? left - plan->halo : 0;
      uint32_t z = bottom + plan->halo < h ? bottom + plan->halo : h,
               t = right + plan->halo < w ? right + plan->halo : w;
      QI_TRY(va_new(g, &tile, 1152, z - a, t - b));
      ix = (ds4_num_reindex){.rank = 3,
                             .shape = {1152, z - a, t - b},
                             .stride = {(uint64_t)h * w, w, 1}};
      QI_TRY(
          ds4_num_reorder(g->num, va_view(tile),
                          ds4_num_slice(va_view(x), (uint64_t)a * w + b,
                                        va_view(x).count - (uint64_t)a * w - b),
                          &ix));
      QI_TRY(va_decode_body(g, &tile, plan->core ? NULL : o));
      /* Read back one bounded BF16 tile, discard its halo, then blend one
       * latent of shared core to absorb convolution roundoff. */
      uint64_t ts = (uint64_t)tile.h * tile.w;
      QI_TRY(qi_host(g->num, ts * 8, (void **)&host, &temporary));
      QI_TRY(ds4_num_read(g->num, va_view(tile), host, ts * 8));
      uint16_t *raw = (uint16_t *)host;
      for (uint32_t py = top * 16; py < bottom * 16; py++)
        for (uint32_t px = left * 16; px < right * 16; px++) {
          float blend = 1;
          if (plan->core) {
            if (top && py < top * 16 + 16)
              blend *= (py - top * 16 + .5f) / 16;
            if (left && px < left * 16 + 16)
              blend *= (px - left * 16 + .5f) / 16;
            if (bottom < h && py >= (bottom - 1) * 16)
              blend *= (bottom * 16 - py - .5f) / 16;
            if (right < w && px >= (right - 1) * 16)
              blend *= (right * 16 - px - .5f) / 16;
          }
          uint64_t dest = (uint64_t)py * ow + px,
                   src = (uint64_t)(py - a * 16) * tile.w + px - b * 16;
          weight[dest] += blend;
          for (uint32_t c = 0; c < 4; c++) {
            float value = qi_float(raw[c * ts + src]);
            if (!isfinite(value)) {
              status = DS4_STATUS_INFERENCE;
              goto cleanup;
            }
            rgba[dest * 4 + c] +=
                blend * (fminf(1, fmaxf(-1, value)) * .5f + .5f);
          }
        }
      qi_host_free(host, temporary);
      host = NULL;
      temporary = NULL;
      QI_TRY(va_drop(&tile));
      QI_TRY(
          ds4_qwen_image_observe(g->num, o, "tile", top * w + left, QI_NONE));
    }
  for (uint64_t i = 0; i < pixels; i++)
    for (uint32_t c = 0; c < 4; c++)
      rgba[i * 4 + c] = fminf(1, fmaxf(0, rgba[i * 4 + c] / weight[i]));
cleanup: {
  ds4_status s = va_drop(&x), t = va_drop(&y);
  if (!s)
    s = t;
  t = va_drop(&tile);
  if (!s)
    s = t;
  t = qi_clear(&work);
  if (!s)
    s = t;
  if (!status)
    status = s;
  if (!status)
    status = ds4_numeric_fence(g->num);
  if (!status)
    status = ds4_qwen_image_observe(g->num, o, "complete", 0, QI_NONE);
  if (status)
    ds4_qwen_image_rgba_free(image);
  else
    *out = image;
  qi_host_free(host, temporary);
  qi_host_free(weight, wa);
  ds4_runtime_scope_leave(&scope);
  return status;
}
}
