/* Strict separable uint8 bicubic for the frozen torchvision processor.
 * Each axis quantizes coefficients to int16 and clamps to uint8 before the
 * next axis. A single floating point 2D convolution has different pixels.
 * Reduction conventions: PyTorch 5c488690 / Pillow Keys a=-0.5; BSD notices
 * are retained in third_party/pytorch/LICENSE. This is an independent C path.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math")
#endif
typedef struct {
  uint32_t *start, *count, length, taps, precision;
  int16_t *weights;
} suite_resize_axis;
static double suite_cubic(double x) {
  x = fabs(x);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}
static void suite_axis_free(suite_resize_axis *a) {
  free(a->start);
  free(a->count);
  free(a->weights);
  memset(a, 0, sizeof(*a));
}
static int suite_axis_build(suite_resize_axis *a, uint32_t input,
                            uint32_t output) {
  memset(a, 0, sizeof(*a));
  double scale = (double)input / output, support = 2 * fmax(1, scale),
         invscale = scale >= 1 ? 1 / scale : 1;
  uint32_t taps = 2 * (uint32_t)ceil(support) + 1;
  a->length = output;
  a->taps = taps;
  a->start = calloc(output, 4);
  a->count = calloc(output, 4);
  a->weights = calloc((size_t)output * taps, sizeof(int16_t));
  double *weights = calloc((size_t)output * taps, sizeof(double));
  if (!a->start || !a->count || !a->weights || !weights) {
    free(weights);
    suite_axis_free(a);
    return 0;
  }
  double maximum = 0;
  for (uint32_t i = 0; i < output; i++) {
    double center = scale * (i + 0.5);
    int64_t start = (int64_t)(center - support + 0.5),
            end = (int64_t)(center + support + 0.5);
    if (start < 0)
      start = 0;
    if (end > input)
      end = input;
    uint32_t count = end > start ? (uint32_t)(end - start) : 0;
    if (count > taps)
      count = taps;
    a->start[i] = start;
    a->count[i] = count;
    double sum = 0;
    for (uint32_t j = 0; j < count; j++) {
      double w = suite_cubic((j + start - center + 0.5) * invscale);
      weights[(size_t)i * taps + j] = w;
      sum += w;
    }
    for (uint32_t j = 0; j < count; j++) {
      double w = weights[(size_t)i * taps + j] / sum;
      weights[(size_t)i * taps + j] = w;
      if (w > maximum)
        maximum = w;
    }
  }
  uint32_t precision = 0;
  for (; precision < 22; precision++)
    if ((int)(0.5 + maximum * (1u << (precision + 1))) >= 32768)
      break;
  a->precision = precision;
  for (size_t i = 0; i < (size_t)output * taps; i++) {
    double v = weights[i] * (1u << precision);
    a->weights[i] = (int16_t)(int)(v < 0 ? v - 0.5 : v + 0.5);
  }
  free(weights);
  return 1;
}
static uint8_t suite_clip(int64_t value, uint32_t precision) {
  value >>= precision;
  return value < 0 ? 0 : value > 255 ? 255 : (uint8_t)value;
}
static int ds4_resize_qwen3vl_rgb(const uint8_t *src, uint32_t sw, uint32_t sh,
                                  float *dst, uint32_t dw, uint32_t dh) {
  suite_resize_axis horizontal = {0}, vertical = {0};
  uint8_t *intermediate = NULL;
  int ok = 0;
  const uint8_t *rows = src;
  if (sw != dw) {
    if (!suite_axis_build(&horizontal, sw, dw))
      goto cleanup;
    intermediate = malloc((size_t)sh * dw * 3);
    if (!intermediate)
      goto cleanup;
    for (uint32_t y = 0; y < sh; y++)
      for (uint32_t x = 0; x < dw; x++)
        for (uint32_t c = 0; c < 3; c++) {
          int64_t sum = 1ll << (horizontal.precision - 1);
          for (uint32_t j = 0; j < horizontal.count[x]; j++)
            sum += (int32_t)
                       src[((size_t)y * sw + horizontal.start[x] + j) * 3 + c] *
                   horizontal.weights[(size_t)x * horizontal.taps + j];
          intermediate[((size_t)y * dw + x) * 3 + c] =
              suite_clip(sum, horizontal.precision);
        }
    rows = intermediate;
  }
  if (sh != dh) {
    if (!suite_axis_build(&vertical, sh, dh))
      goto cleanup;
    for (uint32_t y = 0; y < dh; y++)
      for (uint32_t x = 0; x < dw; x++)
        for (uint32_t c = 0; c < 3; c++) {
          int64_t sum = 1ll << (vertical.precision - 1);
          for (uint32_t j = 0; j < vertical.count[y]; j++)
            sum +=
                (int32_t)
                    rows[((size_t)(vertical.start[y] + j) * dw + x) * 3 + c] *
                vertical.weights[(size_t)y * vertical.taps + j];
          dst[((size_t)y * dw + x) * 3 + c] =
              suite_clip(sum, vertical.precision);
        }
  } else
    for (size_t i = 0; i < (size_t)dh * dw * 3; i++)
      dst[i] = rows[i];
  ok = 1;
cleanup:
  suite_axis_free(&horizontal);
  suite_axis_free(&vertical);
  free(intermediate);
  return ok;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
