#include "ds4_numeric_internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct ds4_num_buffer {
  ds4_numeric *ctx;
  ds4_num_view view;
  ds4_resource *resource;
  ds4_resource *metadata;
  ds4_num_buffer *next_deferred;
};
/* The registry checks the inherited event before entering this destructor.
 * Its accounting and the actual backend lifetime therefore cannot diverge. */
static ds4_status destroy_storage(void *opaque) {
  ds4_num_buffer *b = opaque;
  if (!b->view.data)
    return DS4_STATUS_OK;
  ds4_status status = ds4_numeric_fence(b->ctx);
  if (!status)
    status = b->ctx->ops->release(b->ctx->state, b->view.data);
  if (!status) {
    b->ctx->live_bytes -= b->view.count * ds4_num_type_bytes(b->view.type);
    b->view.data = NULL;
  }
  return status;
}
uint32_t ds4_num_type_bytes(uint32_t t) {
  return t == DS4_NUM_F32 ? 4 : t == DS4_NUM_BF16 ? 2 : 0;
}
static bool valid(ds4_num_view v) {
  return v.data && v.count && ds4_num_type_bytes(v.type) &&
         v.count <= SIZE_MAX / ds4_num_type_bytes(v.type) &&
         (uintptr_t)v.data % ds4_num_type_bytes(v.type) == 0 &&
         (uintptr_t)v.data <=
             UINTPTR_MAX - v.count * ds4_num_type_bytes(v.type);
}
static bool fits(ds4_num_view v, uint64_t n) {
  return valid(v) && n && v.count >= n;
}
static bool product(uint64_t a, uint64_t b, uint64_t *r) {
  if (!a || !b || a > UINT64_MAX / b)
    return false;
  *r = a * b;
  return true;
}
static bool overlaps(ds4_num_view a, ds4_num_view b) {
  if (!valid(a) || !valid(b))
    return false;
  uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
  return x <= y ? y - x < a.count * ds4_num_type_bytes(a.type)
                : x - y < b.count * ds4_num_type_bytes(b.type);
}
static bool partial_overlap(ds4_num_view a, ds4_num_view b) {
  return overlaps(a, b) &&
         (a.data != b.data || a.count != b.count || a.type != b.type);
}
ds4_status ds4_numeric_create(ds4_numeric **out, const ds4_numeric_ops *ops,
                              void *state, ds4_resources *r, uint64_t owner,
                              bool device) {
  if (!out || !ops || !r || !owner)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_resource_spec spec = {.owner = owner,
                            .requested = sizeof(ds4_numeric),
                            .allocated = sizeof(ds4_numeric),
                            .live = sizeof(ds4_numeric),
                            .location = DS4_MEMORY_HOST,
                            .device = -1};
  ds4_resource *metadata = NULL;
  ds4_status status = ds4_resource_register(r, &spec, &metadata);
  if (status)
    return status;
  *out = calloc(1, sizeof(**out));
  if (!*out) {
    ds4_resource_release(metadata);
    return DS4_STATUS_NOMEM;
  }
  **out = (ds4_numeric){.ops = ops,
                        .state = state,
                        .resources = r,
                        .owner = owner,
                        .device = device,
                        .metadata = metadata};
  return DS4_STATUS_OK;
}
ds4_status ds4_numeric_close(ds4_numeric *c) {
  if (!c)
    return DS4_STATUS_OK;
  ds4_status recovered = ds4_numeric_recover(c);
  if (recovered)
    return recovered;
  if (c->buffers)
    return DS4_STATUS_BUSY;
  ds4_status s = c->ops->fence(c->state);
  if (s)
    return s;
  s = ds4_resource_release(c->metadata);
  if (s)
    return s;
  if (c->ops->close)
    c->ops->close(c->state);
  free(c);
  return DS4_STATUS_OK;
}
ds4_status ds4_numeric_fence(ds4_numeric *c) {
  return c ? c->ops->fence(c->state) : DS4_STATUS_INVALID;
}
void ds4_numeric_memory(const ds4_numeric *c, uint64_t *live, uint64_t *peak) {
  if (live)
    *live = c ? c->live_bytes + c->backend_bytes : 0;
  if (peak)
    *peak = c ? c->peak_bytes : 0;
}
ds4_resources *ds4_numeric_resources(ds4_numeric *c, uint64_t *owner) {
  if (owner)
    *owner = c ? c->owner : 0;
  return c ? c->resources : NULL;
}
ds4_status ds4_num_alloc(ds4_numeric *c, uint64_t n, uint32_t type,
                         ds4_num_buffer **out) {
  uint32_t unit = ds4_num_type_bytes(type);
  if (!c || !out || !n || !unit || n > SIZE_MAX / unit)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_num_buffer *b = calloc(1, sizeof(*b));
  if (!b)
    return DS4_STATUS_NOMEM;
  b->ctx = c;
  b->view = (ds4_num_view){NULL, n, type};
  c->buffers++;
  uint64_t bytes = n * unit;
  ds4_resource_spec spec = {0};
  spec.owner = c->owner;
  spec.location = c->device ? DS4_MEMORY_DEVICE : DS4_MEMORY_HOST;
  spec.requested = spec.allocated = spec.live = bytes;
  spec.format = type;
  spec.device = c->device ? 0 : -1;
  spec.destroy = destroy_storage;
  spec.opaque = b;
  ds4_resource_spec metadata = {.owner = c->owner,
                                .requested = sizeof(*b),
                                .allocated = sizeof(*b),
                                .live = sizeof(*b),
                                .location = DS4_MEMORY_HOST,
                                .device = -1};
  ds4_status s = ds4_resource_register(c->resources, &metadata, &b->metadata);
  if (!s)
    s = ds4_resource_register(c->resources, &spec, &b->resource);
  if (!s)
    s = c->ops->allocate(c->state, bytes, &b->view.data);
  if (b->view.data) {
    c->live_bytes += bytes;
    if (c->live_bytes + c->backend_bytes > c->peak_bytes)
      c->peak_bytes = c->live_bytes + c->backend_bytes;
  }
  if (s) {
    if (b->resource && !b->view.data)
      ds4_resource_set_live(b->resource, 0);
    /* Even failed allocation metadata may inherit an in-flight lease. Keep
     * the wrapper until its records can be retired instead of losing them. */
    ds4_num_dispose(b);
    return s;
  }
  *out = b;
  return DS4_STATUS_OK;
}
ds4_status ds4_num_free(ds4_num_buffer *b) {
  if (!b)
    return DS4_STATUS_OK;
  if (b->resource) {
    ds4_status status = ds4_resource_release(b->resource);
    if (status)
      return status;
    b->resource = NULL;
  }
  if (b->metadata) {
    ds4_status status = ds4_resource_release(b->metadata);
    if (status)
      return status;
    b->metadata = NULL;
  }
  b->ctx->buffers--;
  free(b);
  return DS4_STATUS_OK;
}
ds4_num_view ds4_num_data(const ds4_num_buffer *b) {
  return b ? b->view : (ds4_num_view){0};
}

ds4_status ds4_num_dispose(ds4_num_buffer *b) {
  if (!b)
    return DS4_STATUS_OK;
  ds4_status status = ds4_num_free(b);
  if (status) {
    b->next_deferred = b->ctx->deferred;
    b->ctx->deferred = b;
  }
  return status;
}

ds4_status ds4_numeric_recover(ds4_numeric *c) {
  if (!c)
    return DS4_STATUS_INVALID;
  while (c->deferred) {
    ds4_num_buffer *b = c->deferred, *next = b->next_deferred;
    ds4_status status = ds4_num_free(b);
    if (status)
      return status;
    c->deferred = next;
  }
  return DS4_STATUS_OK;
}
ds4_num_view ds4_num_slice(ds4_num_view v, uint64_t off, uint64_t n) {
  if (!valid(v) || off > v.count || n > v.count - off || !n)
    return (ds4_num_view){0};
  v.data = (char *)v.data + off * ds4_num_type_bytes(v.type);
  v.count = n;
  return v;
}

ds4_num_view ds4_num_reinterpret(ds4_num_view storage, uint64_t off, uint64_t n,
                                 uint32_t type) {
  uint32_t unit = ds4_num_type_bytes(type);
  if (!valid(storage) || !unit || !n)
    return (ds4_num_view){0};
  uint64_t bytes = storage.count * ds4_num_type_bytes(storage.type);
  if (off > bytes || n > (bytes - off) / unit ||
      ((uintptr_t)storage.data + off) % unit)
    return (ds4_num_view){0};
  return (ds4_num_view){(char *)storage.data + off, n, type};
}
ds4_status ds4_num_buffer_write(ds4_num_buffer *b, uint64_t off,
                                const void *source, uint64_t bytes) {
  if (!b || !source)
    return DS4_STATUS_INVALID;
  uint64_t capacity = b->view.count * ds4_num_type_bytes(b->view.type);
  if (off > capacity || bytes > capacity - off)
    return DS4_STATUS_INVALID;
  return b->ctx->ops->copy(b->ctx->state, (char *)b->view.data + off, source,
                           bytes, 0);
}
ds4_status ds4_num_write(ds4_numeric *c, ds4_num_view v, const void *p,
                         uint64_t bytes) {
  if (!c || !valid(v) || !p || bytes > v.count * ds4_num_type_bytes(v.type))
    return DS4_STATUS_INVALID;
  return c->ops->copy(c->state, v.data, p, bytes, 0);
}
ds4_status ds4_num_read(ds4_numeric *c, ds4_num_view v, void *p,
                        uint64_t bytes) {
  if (!c || !valid(v) || !p || bytes > v.count * ds4_num_type_bytes(v.type))
    return DS4_STATUS_INVALID;
  return c->ops->copy(c->state, p, v.data, bytes, 1);
}
ds4_status ds4_num_copy(ds4_numeric *c, ds4_num_view y, ds4_num_view x) {
  if (!c || !valid(x) || !fits(y, x.count) || y.type != x.type)
    return DS4_STATUS_INVALID;
  return c->ops->copy(c->state, y.data, x.data,
                      x.count * ds4_num_type_bytes(x.type), 2);
}
ds4_status ds4_num_linear(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                          ds4_num_view w, ds4_num_view b, uint32_t rows,
                          uint32_t in, uint32_t out) {
  if (!c || !rows || !in || !out || rows > INT_MAX || in > INT_MAX ||
      out > INT_MAX || !fits(x, (uint64_t)rows * in) ||
      !fits(w, (uint64_t)out * in) || !fits(y, (uint64_t)rows * out) ||
      x.type != w.type || x.type != y.type ||
      (b.data && (!fits(b, out) || b.type != y.type)) || y.data == x.data ||
      y.data == w.data)
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, (uint64_t)rows * out);
  if (overlaps(y, ds4_num_slice(x, 0, (uint64_t)rows * in)) ||
      overlaps(y, ds4_num_slice(w, 0, (uint64_t)in * out)) ||
      overlaps(y, ds4_num_slice(b, 0, out)))
    return DS4_STATUS_INVALID;
  return c->ops->linear(c->state, y, x, w, b, rows, in, out);
}
ds4_status ds4_num_matmul_batch(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                                ds4_num_view w, uint32_t batch, uint32_t rows,
                                uint32_t in, uint32_t out, bool transpose_b) {
  uint64_t nx, nw, ny;
  if (!c || !batch || batch > INT_MAX || !rows || rows > INT_MAX || !in ||
      in > INT_MAX || !out || out > INT_MAX ||
      !product((uint64_t)rows * in, batch, &nx) ||
      !product((uint64_t)out * in, batch, &nw) ||
      !product((uint64_t)rows * out, batch, &ny) || !fits(x, nx) ||
      !fits(w, nw) || !fits(y, ny) || x.type != DS4_NUM_F32 ||
      w.type != DS4_NUM_F32 || y.type != DS4_NUM_F32)
    return DS4_STATUS_INVALID;
  x = ds4_num_slice(x, 0, nx);
  w = ds4_num_slice(w, 0, nw);
  y = ds4_num_slice(y, 0, ny);
  if (overlaps(y, x) || overlaps(y, w))
    return DS4_STATUS_INVALID;
  return c->ops->matmul_batch ? c->ops->matmul_batch(c->state, y, x, w, batch,
                                                     rows, in, out, transpose_b)
                              : DS4_STATUS_UNSUPPORTED;
}
ds4_status ds4_num_norm(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                        ds4_num_view w, ds4_num_view b, uint32_t width,
                        float eps, uint32_t kind, uint32_t mode, bool pre) {
  if (!c || !valid(x) || !fits(y, x.count) || x.type != y.type || !width ||
      x.count % width || !isfinite(eps) || eps <= 0 || kind > DS4_NUM_L2 ||
      mode > DS4_NUM_SCALE_ZERO_CENTERED ||
      (mode != DS4_NUM_SCALE_NONE && !fits(w, width)) ||
      (b.data && !fits(b, width)))
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, x.count);
  if (partial_overlap(y, x) ||
      (mode != DS4_NUM_SCALE_NONE && overlaps(y, ds4_num_slice(w, 0, width))) ||
      overlaps(y, ds4_num_slice(b, 0, width)))
    return DS4_STATUS_INVALID;
  return c->ops->norm(c->state, y, x, w, b, width, eps, kind, mode, pre);
}
ds4_status ds4_num_channel_l2(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                              uint32_t channels, uint32_t spatial, float eps) {
  if (!c || !channels || channels > INT_MAX || !spatial || spatial > INT_MAX ||
      !valid(x) || x.count != (uint64_t)channels * spatial ||
      !fits(y, x.count) || y.type != x.type || !isfinite(eps) || eps <= 0)
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, x.count);
  if (overlaps(y, x))
    return DS4_STATUS_INVALID;
  return c->ops->channel_l2
             ? c->ops->channel_l2(c->state, y, x, channels, spatial, eps)
             : DS4_STATUS_UNSUPPORTED;
}
ds4_status ds4_num_point(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                         ds4_num_view b, uint32_t op, float scalar) {
  if (!c || !valid(x) || !fits(y, x.count) || op > DS4_NUM_COS ||
      (op != DS4_NUM_CAST && y.type != x.type) || !isfinite(scalar) ||
      ((op == DS4_NUM_ADD || op == DS4_NUM_MUL || op == DS4_NUM_MIN ||
        op == DS4_NUM_MAX) &&
       (!valid(b) || x.count % b.count)))
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, x.count);
  bool binary = op == DS4_NUM_ADD || op == DS4_NUM_MUL || op == DS4_NUM_MIN ||
                op == DS4_NUM_MAX;
  if (partial_overlap(y, x) || (binary && partial_overlap(y, b)))
    return DS4_STATUS_INVALID;
  return c->ops->point(c->state, y, x, b, op, scalar);
}
static ds4_status num_rope(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                           ds4_num_view co, ds4_num_view si, uint32_t heads,
                           uint32_t dim, bool adj, uint32_t rounding) {
  if (!c || !valid(x) || !fits(y, x.count) || x.type != y.type || !heads ||
      !dim || dim % 2 || x.count % ((uint64_t)heads * dim) ||
      !fits(co, x.count / heads / 2) || !fits(si, x.count / heads / 2) ||
      y.data == x.data)
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, x.count);
  if (overlaps(y, x) || overlaps(y, co) || overlaps(y, si))
    return DS4_STATUS_INVALID;
  return c->ops->rope(c->state, y, x, co, si, heads, dim, adj, rounding);
}
ds4_status ds4_num_rope(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                        ds4_num_view co, ds4_num_view si, uint32_t heads,
                        uint32_t dim, bool adj, bool rounded) {
  return num_rope(c, y, x, co, si, heads, dim, adj, rounded ? 1 : 0);
}
ds4_status ds4_num_rope_complex(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                                ds4_num_view co, ds4_num_view si,
                                uint32_t heads, uint32_t dim) {
  return num_rope(c, y, x, co, si, heads, dim, true, 2);
}
ds4_status ds4_num_attend(ds4_numeric *c, ds4_num_view y, ds4_num_view q,
                          ds4_num_view k, ds4_num_view v,
                          const uint32_t *bounds, const ds4_num_attention *a) {
  uint64_t nq, nk, n;
  if (!c || !a || !bounds || !a->q_heads || !a->kv_heads ||
      a->q_heads % a->kv_heads || !a->dim || a->dim > 512 ||
      !isfinite(a->scale) || a->scale <= 0 || !isfinite(a->softcap) ||
      a->softcap < 0 || !product(a->batch, a->queries, &n) ||
      n > UINT32_MAX / 2 || !product(n, a->q_heads, &nq) ||
      !product(nq, a->dim, &nq) || !product(a->batch, a->keys, &nk) ||
      !product(nk, a->kv_heads, &nk) || !product(nk, a->dim, &nk) ||
      !fits(q, nq) || !fits(y, nq) || !fits(k, nk) || !fits(v, nk) ||
      q.type != k.type || q.type != v.type || q.type != y.type ||
      y.data == q.data || y.data == k.data || y.data == v.data)
    return DS4_STATUS_INVALID;
  if (a->keys > INT_MAX || a->queries > INT_MAX ||
      (uint64_t)a->q_heads * a->dim > INT_MAX ||
      (a->relative_keys.data &&
       (!a->relative_rows ||
        !fits(a->relative_keys,
              (uint64_t)a->relative_rows * a->q_heads * a->dim))))
    return DS4_STATUS_INVALID;
  for (uint64_t i = 0; i < n; i++)
    if (bounds[2 * i] > bounds[2 * i + 1] || bounds[2 * i + 1] > a->keys)
      return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, nq);
  if (overlaps(y, ds4_num_slice(q, 0, nq)) ||
      overlaps(y, ds4_num_slice(k, 0, nk)) ||
      overlaps(y, ds4_num_slice(v, 0, nk)) || overlaps(y, a->relative_keys))
    return DS4_STATUS_INVALID;
  return c->ops->attend(c->state, y, q, k, v, bounds, a);
}
ds4_status ds4_num_conv(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                        ds4_num_view w, ds4_num_view b,
                        const ds4_num_convolution *a) {
  uint64_t nx, nw, ny;
  if (!c || !a || !a->groups || !a->batch || !a->in_channels ||
      !a->out_channels || a->in_channels % a->groups ||
      a->out_channels % a->groups)
    return DS4_STATUS_INVALID;
  nx = (uint64_t)a->batch * a->in_channels;
  ny = (uint64_t)a->batch * a->out_channels;
  nw = (uint64_t)a->out_channels * (a->in_channels / a->groups);
  for (int i = 0; i < 3; i++) {
    if (!a->input[i] || !a->kernel[i] || !a->stride[i] || !a->dilation[i])
      return DS4_STATUS_INVALID;
    uint64_t extent = (uint64_t)a->dilation[i] * (a->kernel[i] - 1) + 1,
             padded =
                 (uint64_t)a->input[i] + a->pad_before[i] + a->pad_after[i];
    if (extent > padded || !product(nx, a->input[i], &nx) ||
        !product(nw, a->kernel[i], &nw) ||
        !product(ny, (padded - extent) / a->stride[i] + 1, &ny))
      return DS4_STATUS_INVALID;
  }
  if (!fits(x, nx) || !fits(w, nw) || !fits(y, ny) || x.type != w.type ||
      x.type != y.type ||
      (b.data && (!fits(b, a->out_channels) || b.type != y.type)) ||
      y.data == x.data)
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, ny);
  if (overlaps(y, ds4_num_slice(x, 0, nx)) ||
      overlaps(y, ds4_num_slice(w, 0, nw)) ||
      overlaps(y, ds4_num_slice(b, 0, a->out_channels)))
    return DS4_STATUS_INVALID;
  return c->ops->conv(c->state, y, x, w, b, a);
}

static float bf_get(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t bf_put(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000)
    return (uint16_t)((u >> 16) | 64);
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float get(ds4_num_view v, uint64_t i) {
  return v.type == DS4_NUM_F32 ? ((float *)v.data)[i]
                               : bf_get(((uint16_t *)v.data)[i]);
}
static float rounded(float x, uint32_t t) {
  return t == DS4_NUM_F32 ? x : bf_get(bf_put(x));
}
static void put(ds4_num_view v, uint64_t i, float f) {
  if (v.type == DS4_NUM_F32)
    ((float *)v.data)[i] = f;
  else
    ((uint16_t *)v.data)[i] = bf_put(f);
}
static ds4_status cpu_alloc(void *s, uint64_t n, void **p) {
  (void)s;
  *p = malloc(n);
  return *p ? DS4_STATUS_OK : DS4_STATUS_NOMEM;
}
static ds4_status cpu_release(void *s, void *p) {
  (void)s;
  free(p);
  return DS4_STATUS_OK;
}
static ds4_status cpu_copy(void *s, void *d, const void *p, uint64_t n,
                           int kind) {
  (void)s;
  (void)kind;
  memmove(d, p, n);
  return DS4_STATUS_OK;
}
static ds4_status cpu_fence(void *s) {
  (void)s;
  return DS4_STATUS_OK;
}
static ds4_status cpu_linear(void *s, ds4_num_view y, ds4_num_view x,
                             ds4_num_view w, ds4_num_view b, uint32_t rows,
                             uint32_t in, uint32_t out) {
  (void)s;
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t o = 0; o < out; o++) {
      float sum = 0;
      for (uint32_t i = 0; i < in; i++)
        sum += get(x, (uint64_t)r * in + i) * get(w, (uint64_t)o * in + i);
      /* Linear bias is included before the output cast, as in addmm. */
      put(y, (uint64_t)r * out + o, sum + (b.data ? get(b, o) : 0));
    }
  return DS4_STATUS_OK;
}
static ds4_status cpu_matmul_batch(void *state, ds4_num_view y, ds4_num_view x,
                                   ds4_num_view w, uint32_t batch,
                                   uint32_t rows, uint32_t in, uint32_t out,
                                   bool transpose_b) {
  (void)state;
  for (uint32_t b = 0; b < batch; b++)
    for (uint32_t row = 0; row < rows; row++)
      for (uint32_t j = 0; j < out; j++) {
        double sum = 0;
        for (uint32_t k = 0; k < in; k++)
          sum += (double)get(x, ((uint64_t)b * rows + row) * in + k) *
                 get(w, (uint64_t)b * in * out + (transpose_b
                                                      ? (uint64_t)j * in + k
                                                      : (uint64_t)k * out + j));
        put(y, ((uint64_t)b * rows + row) * out + j, (float)sum);
      }
  return DS4_STATUS_OK;
}
static ds4_status cpu_norm(void *s, ds4_num_view y, ds4_num_view x,
                           ds4_num_view w, ds4_num_view b, uint32_t width,
                           float eps, uint32_t kind, uint32_t mode, bool pre) {
  (void)s;
  for (uint64_t r = 0; r < x.count / width; r++) {
    double mean = 0, var = 0;
    if (kind == DS4_NUM_LAYER) {
      for (uint32_t j = 0; j < width; j++)
        mean += get(x, r * width + j);
      mean /= width;
    }
    for (uint32_t j = 0; j < width; j++) {
      double z = get(x, r * width + j) - mean;
      var += z * z;
    }
    float divisor = fmaxf(sqrtf((float)var), eps);
    float inv = kind == DS4_NUM_L2 ? 1 / divisor
                                   : 1 / sqrtf((float)(var / width) + eps);
    for (uint32_t j = 0; j < width; j++) {
      float z = kind == DS4_NUM_L2
                    ? get(x, r * width + j) / divisor
                    : (get(x, r * width + j) - (float)mean) * inv;
      if (pre)
        z = rounded(z, x.type);
      if (mode != DS4_NUM_SCALE_NONE)
        z *= get(w, j) + (mode == DS4_NUM_SCALE_ZERO_CENTERED ? 1 : 0);
      put(y, r * width + j, z + (b.data ? get(b, j) : 0));
    }
  }
  return DS4_STATUS_OK;
}
static ds4_status cpu_channel_l2(void *s, ds4_num_view y, ds4_num_view x,
                                 uint32_t channels, uint32_t spatial,
                                 float eps) {
  (void)s;
  for (uint32_t p = 0; p < spatial; p++) {
    double sum = 0;
    for (uint32_t c = 0; c < channels; c++) {
      double value = get(x, (uint64_t)c * spatial + p);
      sum += value * value;
    }
    float divisor = fmaxf(sqrtf((float)sum), eps);
    for (uint32_t c = 0; c < channels; c++)
      put(y, (uint64_t)p * channels + c,
          get(x, (uint64_t)c * spatial + p) / divisor);
  }
  return DS4_STATUS_OK;
}
static ds4_status cpu_point(void *s, ds4_num_view y, ds4_num_view x,
                            ds4_num_view b, uint32_t op, float scalar) {
  (void)s;
  for (uint64_t i = 0; i < x.count; i++) {
    float z = get(x, i);
    switch (op) {
    case DS4_NUM_ADD:
      z += get(b, i % b.count);
      break;
    case DS4_NUM_MUL:
      z *= get(b, i % b.count);
      break;
    case DS4_NUM_SILU:
      z = z / (1 + expf(-z));
      break;
    case DS4_NUM_SIGMOID:
      z = 1 / (1 + expf(-z));
      break;
    case DS4_NUM_GELU:
      z = 0.5f * z * (1 + erff(z * 0.7071067811865475f));
      break;
    case DS4_NUM_GELU_TANH:
      z = 0.5f * z *
          (1 + tanhf(0.7978845608028654f * (z + 0.044715f * z * z * z)));
      break;
    case DS4_NUM_SCALE:
      z *= scalar;
      break;
    case DS4_NUM_RELU:
      z = fmaxf(z, 0);
      break;
    case DS4_NUM_SOFTPLUS:
      z = z > 20 ? z : log1pf(expf(z));
      break;
    case DS4_NUM_TANH:
      z = tanhf(z);
      break;
    case DS4_NUM_MIN:
      z = fminf(z, get(b, i % b.count));
      break;
    case DS4_NUM_MAX:
      z = fmaxf(z, get(b, i % b.count));
      break;
    case DS4_NUM_SIN:
      z = sinf(z);
      break;
    case DS4_NUM_COS:
      z = cosf(z);
      break;
    default:
      break;
    }
    put(y, i, z);
  }
  return DS4_STATUS_OK;
}
static ds4_status cpu_rope(void *s, ds4_num_view y, ds4_num_view x,
                           ds4_num_view co, ds4_num_view si, uint32_t heads,
                           uint32_t dim, bool adj, uint32_t rounding) {
  (void)s;
  for (uint64_t i = 0; i < x.count / 2; i++) {
    uint32_t d = i % (dim / 2);
    uint64_t h = i / (dim / 2), a = h * dim + (adj ? 2 * d : d),
             b = a + (adj ? 1 : dim / 2), p = (h / heads) * (dim / 2) + d;
    float c = get(co, p), sn = get(si, p), u = get(x, a), v = get(x, b),
          uc = u * c, vs = v * sn, vc = v * c, us = u * sn;
    if (rounding == 1) {
      uc = rounded(uc, x.type);
      vs = rounded(vs, x.type);
      vc = rounded(vc, x.type);
      us = rounded(us, x.type);
    }
    put(y, a, rounding == 2 ? fmaf(u, c, -vs) : uc - vs);
    put(y, b, rounding == 2 ? fmaf(u, sn, vc) : vc + us);
  }
  return DS4_STATUS_OK;
}
static float score(ds4_num_view q, ds4_num_view k, uint64_t qi, uint64_t ki,
                   const ds4_num_attention *a) {
  float z = 0;
  for (uint32_t d = 0; d < a->dim; d++)
    z += get(q, qi + d) * get(k, ki + d);
  if (a->relative_keys.data) {
    int64_t index = (int64_t)((ki / a->dim / a->kv_heads) % a->keys) -
                    (int64_t)((qi / a->dim / a->q_heads) % a->queries) +
                    a->relative_offset;
    if (index >= 0 && index < a->relative_rows) {
      float relative = 0;
      uint64_t offset =
          ((uint64_t)index * a->q_heads + (qi / a->dim) % a->q_heads) * a->dim;
      for (uint32_t d = 0; d < a->dim; d++)
        relative += get(q, qi + d) * get(a->relative_keys, offset + d);
      z += relative;
    }
  }
  if (a->round_scores)
    z = rounded(z, q.type);
  z *= a->scale;
  if (a->round_scores)
    z = rounded(z, q.type);
  return a->softcap > 0 ? a->softcap * tanhf(z / a->softcap) : z;
}
static ds4_status cpu_attend(void *s, ds4_num_view y, ds4_num_view q,
                             ds4_num_view k, ds4_num_view v,
                             const uint32_t *bounds,
                             const ds4_num_attention *a) {
  (void)s;
  for (uint32_t b = 0; b < a->batch; b++)
    for (uint32_t r = 0; r < a->queries; r++)
      for (uint32_t h = 0; h < a->q_heads; h++) {
        uint64_t qi =
            (((uint64_t)b * a->queries + r) * a->q_heads + h) * a->dim;
        uint32_t lo = bounds[2 * ((uint64_t)b * a->queries + r)],
                 hi = bounds[2 * ((uint64_t)b * a->queries + r) + 1];
        float m = -INFINITY, l = 0, acc[512] = {0};
        for (uint32_t j = lo; j < hi; j++) {
          uint64_t ki = (((uint64_t)b * a->keys + j) * a->kv_heads +
                         h / (a->q_heads / a->kv_heads)) *
                        a->dim;
          float z = score(q, k, qi, ki, a), nm = fmaxf(m, z);
          l = l * expf(m - nm) + expf(z - nm);
          m = nm;
        }
        for (uint32_t j = lo; j < hi; j++) {
          uint64_t ki = (((uint64_t)b * a->keys + j) * a->kv_heads +
                         h / (a->q_heads / a->kv_heads)) *
                        a->dim;
          float p = expf(score(q, k, qi, ki, a) - m) / l;
          if (a->round_probabilities)
            p = rounded(p, q.type);
          for (uint32_t d = 0; d < a->dim; d++)
            acc[d] += p * get(v, ki + d);
        }
        for (uint32_t d = 0; d < a->dim; d++)
          put(y, qi + d, acc[d]);
      }
  return DS4_STATUS_OK;
}
static ds4_status cpu_conv(void *s, ds4_num_view y, ds4_num_view x,
                           ds4_num_view w, ds4_num_view bias,
                           const ds4_num_convolution *a) {
  (void)s;
  uint32_t o[3];
  uint64_t spatial = 1, ins = 1, ks = 1;
  for (int d = 0; d < 3; d++) {
    o[d] = (a->input[d] + a->pad_before[d] + a->pad_after[d] -
            a->dilation[d] * (a->kernel[d] - 1) - 1) /
               a->stride[d] +
           1;
    spatial *= o[d];
    ins *= a->input[d];
    ks *= a->kernel[d];
  }
  for (uint64_t i = 0; i < (uint64_t)a->batch * a->out_channels * spatial;
       i++) {
    uint32_t oc = (i / spatial) % a->out_channels,
             batch = i / (spatial * a->out_channels), p[3];
    uint64_t rem = i % spatial;
    for (int d = 2; d >= 0; d--) {
      p[d] = rem % o[d];
      rem /= o[d];
    }
    float sum = 0;
    for (uint32_t ic = 0; ic < a->in_channels / a->groups; ic++)
      for (uint64_t j = 0; j < ks; j++) {
        uint64_t r = j, xi = 0;
        bool ok = true;
        int64_t coord[3];
        for (int d = 2; d >= 0; d--) {
          coord[d] = (int64_t)p[d] * a->stride[d] +
                     (r % a->kernel[d]) * a->dilation[d] - a->pad_before[d];
          r /= a->kernel[d];
          if (coord[d] < 0 || coord[d] >= a->input[d])
            ok = false;
        }
        if (!ok)
          continue;
        for (int d = 0; d < 3; d++)
          xi = xi * a->input[d] + coord[d];
        uint32_t channel = (oc / (a->out_channels / a->groups)) *
                               (a->in_channels / a->groups) +
                           ic;
        sum +=
            get(x, ((uint64_t)batch * a->in_channels + channel) * ins + xi) *
            get(w, ((uint64_t)oc * (a->in_channels / a->groups) + ic) * ks + j);
      }
    put(y, i, sum + (bias.data ? get(bias, oc) : 0));
  }
  return DS4_STATUS_OK;
}
static ds4_status cpu_reorder(void *state, ds4_num_view y, ds4_num_view x,
                              const ds4_num_reindex *a) {
  (void)state;
  for (uint64_t i = 0; i < y.count; i++) {
    uint64_t rem = i, index = 0;
    for (uint32_t d = a->rank; d > 0; d--) {
      index += (rem % a->shape[d - 1]) * a->stride[d - 1];
      rem /= a->shape[d - 1];
    }
    put(y, i, get(x, index));
  }
  return DS4_STATUS_OK;
}
static ds4_status cpu_pool(void *state, ds4_num_view y, ds4_num_view x,
                           const uint32_t *indices, uint32_t rows,
                           uint32_t group, uint32_t width, float divisor) {
  (void)state;
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t d = 0; d < width; d++) {
      float sum = 0;
      for (uint32_t j = 0; j < group; j++) {
        uint32_t ix = indices[(uint64_t)r * group + j];
        if (ix != UINT32_MAX)
          sum += get(x, (uint64_t)ix * width + d);
      }
      put(y, (uint64_t)r * width + d, sum / divisor);
    }
  return DS4_STATUS_OK;
}
static ds4_status cpu_nearest(void *state, ds4_num_view y, ds4_num_view x,
                              const ds4_num_resample *a) {
  (void)state;
  uint64_t count = (uint64_t)a->batch * a->channels;
  for (int d = 0; d < 3; d++)
    count *= a->output[d];
  for (uint64_t i = 0; i < count; i++) {
    uint64_t rem = i, index = 0, step = 1;
    for (int d = 2; d >= 0; d--) {
      uint64_t pos = rem % a->output[d];
      rem /= a->output[d];
      uint64_t src = ((2 * pos + 1) * a->input[d]) / (2ull * a->output[d]);
      index += src * step;
      step *= a->input[d];
    }
    put(y, i, get(x, rem * step + index));
  }
  return DS4_STATUS_OK;
}
ds4_status ds4_num_reorder(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                           const ds4_num_reindex *a) {
  if (!c || !a || !a->rank || a->rank > 8 || !valid(x) || !valid(y) ||
      x.type != y.type || x.data == y.data)
    return DS4_STATUS_INVALID;
  uint64_t n = 1, maximum = 0;
  for (uint32_t d = 0; d < a->rank; d++) {
    uint64_t part;
    if (!a->shape[d] || !product(n, a->shape[d], &n) ||
        (a->stride[d] && (a->shape[d] - 1) > UINT64_MAX / a->stride[d]))
      return DS4_STATUS_INVALID;
    part = (a->shape[d] - 1) * a->stride[d];
    if (part > UINT64_MAX - maximum)
      return DS4_STATUS_INVALID;
    maximum += part;
  }
  if (!fits(y, n) || maximum >= x.count)
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, n);
  if (overlaps(y, x))
    return DS4_STATUS_INVALID;
  return c->ops->reorder(c->state, y, x, a);
}
ds4_status ds4_num_pool(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                        const uint32_t *indices, uint32_t rows, uint32_t group,
                        uint32_t width, float divisor) {
  if (!c || !valid(x) || !indices || !rows || !group || !width ||
      !fits(y, (uint64_t)rows * width) || x.count % width || x.type != y.type ||
      !isfinite(divisor) || divisor <= 0 || x.data == y.data ||
      (uint64_t)rows * group > SIZE_MAX / 4)
    return DS4_STATUS_INVALID;
  for (uint64_t i = 0; i < (uint64_t)rows * group; i++)
    if (indices[i] != UINT32_MAX && indices[i] >= x.count / width)
      return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, (uint64_t)rows * width);
  if (overlaps(y, x))
    return DS4_STATUS_INVALID;
  return c->ops->pool(c->state, y, x, indices, rows, group, width, divisor);
}
ds4_status ds4_num_nearest(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                           const ds4_num_resample *a) {
  if (!c || !a || !a->batch || !a->channels || !valid(x) || !valid(y) ||
      x.type != y.type || x.data == y.data)
    return DS4_STATUS_INVALID;
  uint64_t ni = (uint64_t)a->batch * a->channels, no = ni;
  for (int d = 0; d < 3; d++)
    if (!product(ni, a->input[d], &ni) || !product(no, a->output[d], &no))
      return DS4_STATUS_INVALID;
  if (!fits(x, ni) || !fits(y, no))
    return DS4_STATUS_INVALID;
  y = ds4_num_slice(y, 0, no);
  if (overlaps(y, ds4_num_slice(x, 0, ni)))
    return DS4_STATUS_INVALID;
  return c->ops->nearest(c->state, y, x, a);
}
static ds4_status cpu_softmax(void *state, ds4_num_view y, ds4_num_view x,
                              const uint32_t *bounds, uint32_t rows,
                              uint32_t width, float scale) {
  (void)state;
  for (uint32_t row = 0; row < rows; row++) {
    uint32_t lo = bounds[2 * row], hi = bounds[2 * row + 1];
    float maximum = -INFINITY, sum = 0;
    for (uint32_t j = lo; j < hi; j++)
      maximum = fmaxf(
          maximum, rounded(get(x, (uint64_t)row * width + j) * scale, x.type));
    for (uint32_t j = lo; j < hi; j++)
      sum += expf(rounded(get(x, (uint64_t)row * width + j) * scale, x.type) -
                  maximum);
    for (uint32_t j = 0; j < width; j++)
      put(y, (uint64_t)row * width + j,
          j < lo || j >= hi
              ? 0
              : expf(
                    rounded(get(x, (uint64_t)row * width + j) * scale, x.type) -
                    maximum) /
                    sum);
  }
  return DS4_STATUS_OK;
}
ds4_status ds4_num_softmax(ds4_numeric *c, ds4_num_view y, ds4_num_view x,
                           const uint32_t *bounds, uint32_t rows,
                           uint32_t width, float scale) {
  if (!c || !rows || !width || !bounds || !fits(y, (uint64_t)rows * width) ||
      !fits(x, (uint64_t)rows * width) || x.type != y.type ||
      !isfinite(scale) || scale <= 0)
    return DS4_STATUS_INVALID;
  for (uint32_t r = 0; r < rows; r++)
    if (bounds[2 * r] > bounds[2 * r + 1] || bounds[2 * r + 1] > width)
      return DS4_STATUS_INVALID;
  x = ds4_num_slice(x, 0, (uint64_t)rows * width);
  y = ds4_num_slice(y, 0, x.count);
  if (partial_overlap(y, x))
    return DS4_STATUS_INVALID;
  return c->ops->softmax(c->state, y, x, bounds, rows, width, scale);
}
static const ds4_numeric_ops cpu_ops = {
    cpu_alloc,        cpu_release,   cpu_copy,  cpu_fence,   NULL,
    cpu_linear,       cpu_norm,      cpu_point, cpu_rope,    cpu_attend,
    cpu_conv,         cpu_reorder,   cpu_pool,  cpu_nearest, cpu_softmax,
    cpu_matmul_batch, cpu_channel_l2};
ds4_status ds4_numeric_cpu(ds4_numeric **out, ds4_resources *r,
                           uint64_t owner) {
  return ds4_numeric_create(out, &cpu_ops, NULL, r, owner, false);
}
static float half_get(uint16_t h) {
  int e = (h >> 10) & 31;
  float f = e == 31
                ? ((h & 1023) ? NAN : INFINITY)
                : ldexpf(e ? 1 + (h & 1023) / 1024.0f : (h & 1023) / 1024.0f,
                         e ? e - 15 : -14);
  return h & 32768 ? -f : f;
}
ds4_status ds4_num_dequantize(float *out, uint64_t n, const void *p,
                              uint64_t bytes, uint32_t t) {
  uint32_t block = t == 0 || t == 30 ? 1 : 32, unit = t == 0    ? 4
                                                      : t == 30 ? 2
                                                      : t == 8  ? 34
                                                      : t == 2  ? 18
                                                                : 0;
  if (!out || !p || !n || n > SIZE_MAX / sizeof(float) || !unit || n % block ||
      n / block > bytes / unit)
    return unit ? DS4_STATUS_INVALID : DS4_STATUS_UNSUPPORTED;
  if (t == 0) {
    memcpy(out, p, n * 4);
    return DS4_STATUS_OK;
  }
  if (t == 30) {
    for (uint64_t i = 0; i < n; i++) {
      uint16_t u;
      memcpy(&u, (char *)p + 2 * i, 2);
      out[i] = bf_get(u);
    }
    return DS4_STATUS_OK;
  }
  for (uint64_t i = 0; i < n / 32; i++) {
    const uint8_t *q = (const uint8_t *)p + i * unit;
    uint16_t h;
    memcpy(&h, q, 2);
    float d = half_get(h);
    for (uint32_t j = 0; j < 32; j++)
      out[i * 32 + j] =
          d * (t == 8 ? (int8_t)q[2 + j]
                      : (int)((q[2 + j % 16] >> (j < 16 ? 0 : 4)) & 15) - 8);
  }
  return DS4_STATUS_OK;
}
