/* Real native encoder probe. Input cases are little-endian binary fixtures:
 * uint32 prompt_bytes, image_count; UTF-8 prompt; then per image width,height,
 * channels and raw pixels. Dumps complete tensors, not samples of a tensor. */
#include "ds4_qwen_image_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
  const char *dir;
  ds4_numeric *num;
} trace_context;
static ds4_status dump(void *p, const char *stage, uint32_t layer,
                       ds4_num_view v) {
  if (!v.data)
    return DS4_STATUS_OK;
  trace_context *c = p;
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s-%u.bf16", c->dir, stage, layer);
  uint64_t bytes = v.count * (v.type == 30 ? 2 : 4);
  void *host = malloc(bytes);
  if (!host)
    return DS4_STATUS_NOMEM;
  ds4_status s = ds4_num_read(c->num, v, host, bytes);
  if (!s) {
    FILE *f = fopen(path, "wb");
    if (!f)
      s = DS4_STATUS_INVALID;
    else {
      if (fwrite(host, 1, bytes, f) != bytes)
        s = DS4_STATUS_INVALID;
      fclose(f);
    }
  }
  free(host);
  return s;
}
static int write_file(const char *dir, const char *name, const void *data,
                      size_t bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  int ok = fwrite(data, 1, bytes, f) == bytes;
  fclose(f);
  return ok;
}
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
typedef struct {
  unsigned calls;
  ds4_qwen_image_encoder *encoder;
  const char *stop_stage;
} cancel_context;
static ds4_status cancel_trace(void *p, const char *stage, uint32_t layer,
                               ds4_num_view v) {
  (void)layer;
  (void)v;
  cancel_context *c = p;
  if (c->stop_stage && !strcmp(stage, c->stop_stage))
    c->calls = 3;
  else if (!c->stop_stage && !strcmp(stage, "text"))
    c->calls++;
  /* A callback may not evict weights while the forward owns a runtime lease. */
  return ds4_qwen_image_encoder_release_weights(c->encoder) == DS4_STATUS_BUSY
             ? DS4_STATUS_OK
             : DS4_STATUS_INVALID;
}
static bool cancel_requested(void *p) {
  return ((cancel_context *)p)->calls >= 3;
}
static int equal_conditioning(ds4_numeric *num, ds4_qwen_image_conditioning *a,
                              const void *expected, uint64_t bytes) {
  void *actual = malloc(bytes);
  if (!actual)
    return 0;
  ds4_status s = ds4_num_read(num, ds4_qwen_image_conditioning_data(a).hidden,
                              actual, bytes);
  int same = !s && !memcmp(actual, expected, bytes);
  free(actual);
  return same;
}
#define CHECK(expr)                                                            \
  do {                                                                         \
    ds4_status s = (expr);                                                     \
    if (s) {                                                                   \
      fprintf(stderr, "%s: status=%d\n", #expr, s);                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr,
            "index root tokenizer fixture output [fixture output ...]\n");
    return 2;
  }
  ds4_resources *resources = NULL;
  ds4_numeric *num = NULL;
  ds4_bundle *bundle = NULL;
  ds4_runtime *runtime = NULL;
  ds4_qwen_image_encoder *encoder = NULL;
  CHECK(ds4_resources_create(&resources));
  CHECK(ds4_numeric_cuda(&num, resources, 700));
  CHECK(ds4_bundle_open(&bundle, argv[1], argv[2], resources, 700));
  CHECK(ds4_runtime_create(&runtime, NULL, NULL));
  CHECK(ds4_qwen_image_encoder_open(&encoder, bundle, num, runtime, argv[3]));
  for (int arg = 4; arg + 1 < argc; arg += 2) {
    FILE *f = fopen(argv[arg], "rb");
    if (!f)
      return 2;
    uint32_t len, count;
    if (fread(&len, 4, 1, f) != 1 || fread(&count, 4, 1, f) != 1 ||
        len > 4 * 1024 * 1024 || count > 64)
      return 2;
    char *prompt = calloc(len + 1, 1);
    if (!prompt || fread(prompt, 1, len, f) != len)
      return 2;
    ds4_qwen_image_reference *refs = calloc(count ? count : 1, sizeof(*refs));
    for (uint32_t i = 0; i < count; i++) {
      if (fread(&refs[i].width, 4, 3, f) != 3 || !refs[i].width ||
          !refs[i].height || refs[i].width > 16384 || refs[i].height > 16384 ||
          (refs[i].channels != 3 && refs[i].channels != 4))
        return 2;
      size_t n = (size_t)refs[i].width * refs[i].height * refs[i].channels;
      uint8_t *pixels = malloc(n);
      if (!pixels || fread(pixels, 1, n, f) != n)
        return 2;
      refs[i].pixels = pixels;
    }
    if (fgetc(f) != EOF)
      return 2;
    fclose(f);
    mkdir(argv[arg + 1], 0755);
    ds4_qwen_image_input *input = NULL;
    if (getenv("DS4_SUITE_ENCODER_CHECKS")) {
      ds4_resource_totals before, after;
      ds4_resources_totals(resources, 0, &before);
      ds4_resource_budget original, denied;
      CHECK(ds4_resources_get_budget(resources, DS4_MEMORY_HOST, &original));
      denied = original;
      denied.limit = before.allocated[DS4_MEMORY_HOST] + 1;
      CHECK(ds4_resources_set_budget(resources, DS4_MEMORY_HOST, &denied));
      ds4_status rejected =
          ds4_qwen_image_prepare(encoder, prompt, refs, count, &input);
      CHECK(ds4_resources_set_budget(resources, DS4_MEMORY_HOST, &original));
      ds4_resources_totals(resources, 0, &after);
      if (rejected != DS4_STATUS_NOMEM || input ||
          memcmp(&before, &after, sizeof(before)))
        return 1;
    }
    uint32_t pad_to =
        getenv("DS4_SUITE_ENCODER_PAD_TO")
            ? strtoul(getenv("DS4_SUITE_ENCODER_PAD_TO"), NULL, 10)
            : 0;
    CHECK(ds4_qwen_image_prepare_padded(encoder, prompt, refs, count, pad_to,
                                        &input));
    ds4_qwen_image_input_view input_data = ds4_qwen_image_input_data(input);
    if (!write_file(argv[arg + 1], "tokens.u32", input_data.tokens,
                    input_data.count * 4) ||
        !write_file(argv[arg + 1], "positions.u32", input_data.positions,
                    input_data.count * 12) ||
        !write_file(argv[arg + 1], "image_mask.u8", input_data.image_mask,
                    input_data.count) ||
        !write_file(argv[arg + 1], "grids.u32", input_data.grids,
                    input_data.images * 8) ||
        !write_file(argv[arg + 1], "pixels.f32", input_data.pixels,
                    (size_t)input_data.patches * 1536 * 4))
      return 2;
    trace_context trace = {argv[arg + 1], num};
    ds4_qwen_image_observer observer = {&trace, NULL, dump};
    ds4_qwen_image_conditioning *conditioning = NULL;
    double start = now();
    CHECK(ds4_qwen_image_encode(encoder, input, &observer, &conditioning));
    ds4_qwen_image_conditioning_view conditioning_data =
        ds4_qwen_image_conditioning_data(conditioning);
    double elapsed = now() - start;
    printf(
        "%s: tokens=%u drop=%u images=%u patches=%u output=%u seconds=%.6f\n",
        argv[arg], input_data.count, input_data.drop, input_data.images,
        input_data.patches, conditioning_data.count, elapsed);
    fflush(stdout);
    int repeats = getenv("DS4_SUITE_ENCODER_REPEATS")
                      ? atoi(getenv("DS4_SUITE_ENCODER_REPEATS"))
                      : 0;
    if (repeats || getenv("DS4_SUITE_ENCODER_CHECKS")) {
      uint64_t bytes = (uint64_t)conditioning_data.count * 4096 * 2;
      void *expected = malloc(bytes);
      if (!expected)
        return 1;
      CHECK(ds4_num_read(num, conditioning_data.hidden, expected, bytes));
      /* The traced forward is the untimed warmup; each repetition uses the
       * public release path, with no trace transfers in the timed interval. */
      for (int rep = 0; rep < repeats; rep++) {
        ds4_qwen_image_conditioning *next = NULL;
        start = now();
        CHECK(ds4_qwen_image_encode(encoder, input, NULL, &next));
        elapsed = now() - start;
        if (!equal_conditioning(num, next, expected, bytes))
          return 1;
        uint64_t live, peak;
        ds4_numeric_memory(num, &live, &peak);
        printf("benchmark case=%s repetition=%d seconds=%.9f device_live=%llu "
               "device_peak=%llu\n",
               argv[arg], rep, elapsed, (unsigned long long)live,
               (unsigned long long)peak);
        fflush(stdout);
        CHECK(ds4_qwen_image_conditioning_free(next));
      }
      if (getenv("DS4_SUITE_ENCODER_CHECKS")) {
        ds4_resource_totals before, after;
        ds4_resources_totals(resources, 0, &before);
        cancel_context cancellation = {.encoder = encoder};
        ds4_qwen_image_observer cancelled = {&cancellation, cancel_requested,
                                             cancel_trace};
        ds4_qwen_image_conditioning *next = NULL;
        if (ds4_qwen_image_encode(encoder, input, &cancelled, &next) !=
                DS4_STATUS_CANCELLED ||
            next)
          return 1;
        ds4_resources_totals(resources, 0, &after);
        if (memcmp(&before, &after, sizeof(before)))
          return 1;
        cancellation.calls = 0;
        cancellation.stop_stage = "conditioning";
        if (ds4_qwen_image_encode(encoder, input, &cancelled, &next) !=
                DS4_STATUS_CANCELLED ||
            next)
          return 1;
        ds4_resources_totals(resources, 0, &after);
        if (memcmp(&before, &after, sizeof(before)))
          return 1;
        puts("encoder cancellation at final output callback: PASS");
        ds4_resource_budget original, denied;
        CHECK(
            ds4_resources_get_budget(resources, DS4_MEMORY_DEVICE, &original));
        denied = original;
        denied.limit = before.allocated[DS4_MEMORY_DEVICE] + 1;
        CHECK(ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &denied));
        ds4_status rejected =
            ds4_qwen_image_encode(encoder, input, NULL, &next);
        CHECK(
            ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &original));
        ds4_resources_totals(resources, 0, &after);
        if (rejected != DS4_STATUS_NOMEM || next ||
            memcmp(&before, &after, sizeof(before)))
          return 1;
        CHECK(ds4_qwen_image_encoder_release_weights(encoder));
        if (!equal_conditioning(num, conditioning, expected, bytes))
          return 1;
        CHECK(ds4_qwen_image_encode(encoder, input, NULL, &next));
        if (!equal_conditioning(num, next, expected, bytes))
          return 1;
        CHECK(ds4_qwen_image_conditioning_free(next));
        puts("encoder host/device budget, cancellation, busy callback, output "
             "lifetime, reload: PASS");
      }
      free(expected);
    }
    if (getenv("DS4_SUITE_ENCODER_CHECKS") && input_data.count < 64) {
      char directory[4096];
      snprintf(directory, sizeof(directory), "%s-padded", argv[arg + 1]);
      mkdir(directory, 0755);
      ds4_qwen_image_input *padded = NULL;
      ds4_qwen_image_conditioning *next = NULL;
      CHECK(ds4_qwen_image_prepare_padded(encoder, prompt, refs, count, 64,
                                          &padded));
      ds4_qwen_image_input_view padded_data = ds4_qwen_image_input_data(padded);
      if (padded_data.count != 64 ||
          padded_data.drop != 14 + padded_data.left_padding ||
          strcmp(padded_data.prompt, prompt))
        return 1;
      if (!write_file(directory, "tokens.u32", padded_data.tokens, 64 * 4) ||
          !write_file(directory, "positions.u32", padded_data.positions,
                      64 * 12))
        return 1;
      trace_context padded_trace = {directory, num};
      ds4_qwen_image_observer padded_observer = {&padded_trace, NULL, dump};
      CHECK(ds4_qwen_image_encode(encoder, padded, &padded_observer, &next));
      if (ds4_qwen_image_conditioning_data(next).count !=
          conditioning_data.count)
        return 1;
      CHECK(ds4_qwen_image_conditioning_free(next));
      ds4_qwen_image_input_free(padded);
    }
    CHECK(ds4_qwen_image_conditioning_free(conditioning));
    ds4_qwen_image_input_free(input);
    for (uint32_t i = 0; i < count; i++)
      free((void *)refs[i].pixels);
    free(refs);
    free(prompt);
  }
  CHECK(ds4_qwen_image_encoder_close(encoder));
  CHECK(ds4_bundle_close(bundle));
  CHECK(ds4_numeric_close(num));
  ds4_runtime_release(runtime);
  ds4_resource_totals totals;
  ds4_resources_totals(resources, 0, &totals);
  if (totals.roots || totals.views)
    return 1;
  CHECK(ds4_resources_close(resources));
  return 0;
}
