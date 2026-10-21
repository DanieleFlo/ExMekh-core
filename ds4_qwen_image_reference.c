/* Independent C implementation of the pinned Pillow 12.3 Lanczos/alpha
 * conventions. Coefficient normalization and premultiplication boundaries are
 * deliberately distinct; see T11/T12 provenance. No codec/runtime dependency.
 */
#include "ds4_qwen_image_reference.h"
#include "ds4_image.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  uint32_t *start, *count, taps;
  int32_t *weight;
} axis;
static void close_axis(axis *a) {
  free(a->start);
  free(a->count);
  free(a->weight);
  *a = (axis){0};
}
static double sinc(double x) {
  if (x == 0)
    return 1;
  x *= 3.14159265358979323846;
  return sin(x) / x;
}
static ds4_status build_axis(axis *a, uint32_t in, uint32_t out) {
  double scale = (double)in / out, filter = scale > 1 ? scale : 1,
         support = 3 * filter, inv = 1 / filter;
  a->taps = 2 * (uint32_t)ceil(support) + 1;
  a->start = calloc(out, 4);
  a->count = calloc(out, 4);
  a->weight = calloc((uint64_t)out * a->taps, 4);
  double *weights = calloc(a->taps, sizeof(double));
  if (!a->start || !a->count || !a->weight || !weights) {
    free(weights);
    close_axis(a);
    return DS4_STATUS_NOMEM;
  }
  for (uint32_t i = 0; i < out; i++) {
    double center = (i + .5) * scale;
    int64_t left = (int64_t)(center - support + .5),
            right = (int64_t)(center + support + .5);
    if (left < 0)
      left = 0;
    if (right > in)
      right = in;
    a->start[i] = left;
    a->count[i] = right - left;
    double sum = 0;
    for (uint32_t j = 0; j < a->count[i]; j++) {
      double x = (j + left - center + .5) * inv;
      weights[j] = x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
      sum += weights[j];
    }
    for (uint32_t j = 0; j < a->count[i]; j++) {
      double w = weights[j] / sum * 4194304.;
      a->weight[(uint64_t)i * a->taps + j] = (int32_t)(w < 0 ? w - .5 : w + .5);
    }
  }
  free(weights);
  return DS4_STATUS_OK;
}
static uint8_t clip(int64_t value) {
  value >>= 22;
  return value < 0 ? 0 : value > 255 ? 255 : value;
}
static uint32_t round32(double x) {
  double scaled = x / 32, base = floor(scaled), fraction = scaled - base;
  if (fraction > .5 || (fraction == .5 && fmod(base, 2) != 0))
    base++;
  return base >= 1 && base <= 64 ? (uint32_t)base * 32 : 0;
}
uint64_t qi_reference_workspace(const ds4_qwen_image_reference *r, uint32_t w,
                                uint32_t h) {
  if (r->width == w && r->height == h)
    return 0;
  /* Source premultiplication, horizontal intermediate and bounded axis
   * coefficients. Keep the coefficient margin independent of image area. */
  return (uint64_t)r->width * r->height * 4 + (uint64_t)r->height * w * 4 +
         2 * 1024 * 1024;
}
ds4_status qi_reference_plan(const ds4_qwen_image_reference *r,
                             uint32_t resolution, uint32_t *w, uint32_t *h) {
  if (!r || !w || !h || !r->pixels || !r->width || !r->height ||
      r->width > DS4_IMAGE_MAX_DIMENSION ||
      r->height > DS4_IMAGE_MAX_DIMENSION ||
      (uint64_t)r->width * r->height > DS4_IMAGE_MAX_PIXELS ||
      (r->channels != 3 && r->channels != 4) || resolution > 2048 ||
      (resolution && resolution % 32))
    return DS4_STATUS_INVALID;
  *w = r->width;
  *h = r->height;
  if (resolution) {
    double ratio = (double)r->width / r->height,
           width = sqrt((double)resolution * resolution * ratio);
    *w = round32(width);
    *h = round32(width / ratio);
  }
  uint32_t planned_w, planned_h;
  if (!*w || !*h || *w > 2048 || *h > 2048 || *w % 32 || *h % 32 ||
      !ds4_image_qwen3vl_plan(*w, *h, 64, 16384, &planned_w, &planned_h))
    return DS4_STATUS_INVALID;
  return planned_w == *w && planned_h == *h ? DS4_STATUS_OK
                                            : DS4_STATUS_UNSUPPORTED;
}
ds4_status qi_reference_resize(const ds4_qwen_image_reference *r, uint8_t *out,
                               uint32_t w, uint32_t h) {
  if (!r || !r->pixels || !r->width || !r->height ||
      r->width > DS4_IMAGE_MAX_DIMENSION ||
      r->height > DS4_IMAGE_MAX_DIMENSION ||
      (uint64_t)r->width * r->height > DS4_IMAGE_MAX_PIXELS ||
      (r->channels != 3 && r->channels != 4) || !out || !w || !h || w > 2048 ||
      h > 2048)
    return DS4_STATUS_INVALID;
  if (r->width == w && r->height == h) {
    for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
      memcpy(out + i * 4, r->pixels + i * r->channels, 3);
      out[i * 4 + 3] = r->channels == 4 ? r->pixels[i * 4 + 3] : 255;
    }
    return DS4_STATUS_OK;
  }
  uint8_t *source = malloc((uint64_t)r->width * r->height * 4),
          *intermediate = NULL;
  if (!source)
    return DS4_STATUS_NOMEM;
  axis horizontal = {0}, vertical = {0};
  ds4_status status = 0;
  /* PIL RGBA resize premultiplies RGB into uint8 RGBa, resamples all four
   * channels, then divides with integer truncation. Transparent RGB survives
   * only when geometry was unchanged (the no-resize branch above). */
  for (uint64_t i = 0; i < (uint64_t)r->width * r->height; i++) {
    uint32_t alpha = r->channels == 4 ? r->pixels[i * 4 + 3] : 255;
    source[i * 4 + 3] = alpha;
    for (uint32_t c = 0; c < 3; c++) {
      uint32_t t = r->pixels[i * r->channels + c] * alpha + 128;
      source[i * 4 + c] = ((t >> 8) + t) >> 8;
    }
  }
  const uint8_t *rows = source;
  if (r->width != w) {
    status = build_axis(&horizontal, r->width, w);
    if (status)
      goto done;
    intermediate = malloc((uint64_t)r->height * w * 4);
    if (!intermediate) {
      status = DS4_STATUS_NOMEM;
      goto done;
    }
    for (uint32_t y = 0; y < r->height; y++)
      for (uint32_t x = 0; x < w; x++)
        for (uint32_t c = 0; c < 4; c++) {
          int64_t sum = 1 << 21;
          for (uint32_t j = 0; j < horizontal.count[x]; j++)
            sum +=
                (int64_t)
                    source[((uint64_t)y * r->width + horizontal.start[x] + j) *
                               4 +
                           c] *
                horizontal.weight[(uint64_t)x * horizontal.taps + j];
          intermediate[((uint64_t)y * w + x) * 4 + c] = clip(sum);
        }
    rows = intermediate;
  }
  if (r->height != h) {
    status = build_axis(&vertical, r->height, h);
    if (status)
      goto done;
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
        for (uint32_t c = 0; c < 4; c++) {
          int64_t sum = 1 << 21;
          for (uint32_t j = 0; j < vertical.count[y]; j++)
            sum +=
                (int64_t)
                    rows[((uint64_t)(vertical.start[y] + j) * w + x) * 4 + c] *
                vertical.weight[(uint64_t)y * vertical.taps + j];
          out[((uint64_t)y * w + x) * 4 + c] = clip(sum);
        }
  } else
    memcpy(out, rows, (uint64_t)w * h * 4);
  for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
    uint32_t alpha = out[i * 4 + 3];
    if (alpha && alpha != 255)
      for (uint32_t c = 0; c < 3; c++) {
        uint32_t v = 255u * out[i * 4 + c] / alpha;
        out[i * 4 + c] = v > 255 ? 255 : v;
      }
  }
done:
  free(source);
  free(intermediate);
  close_axis(&horizontal);
  close_axis(&vertical);
  return status;
}
