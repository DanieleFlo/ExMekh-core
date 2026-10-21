/* Native C smoke/qualification driver. No external inference process is used.
 * gemma: bundle root asset gemma outdir text [...]
 * image: bundle root asset image outdir width height seed prompt [ref.ds4i ...]
 * A .ds4i contains uint32 LE width,height,channels followed by RGB/RGBA bytes.
 */
#include "ds4_embedding_gemma2.h"
#include "ds4_qwen_image_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
extern int cudaMemGetInfo(size_t *, size_t *);
#define CHECK(x)                                                               \
  do {                                                                         \
    ds4_status s = (x);                                                        \
    if (s) {                                                                   \
      fprintf(stderr, "%s: %d\n", #x, s);                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)
typedef struct {
  ds4_numeric *num;
  const char *out;
  uint32_t step;
  bool cancel;
  ds4_embedding_gemma2 *gemma;
} trace;
static int save(const char *dir, const char *name, const void *data,
                size_t bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  bool ok = fwrite(data, 1, bytes, f) == bytes;
  return fclose(f) == 0 && ok;
}
static ds4_status dump(trace *t, const char *name, ds4_num_view v) {
  if (!v.data)
    return DS4_STATUS_OK;
  size_t bytes = v.count * (v.type == DS4_NUM_BF16 ? 2 : 4);
  void *p = malloc(bytes);
  if (!p)
    return DS4_STATUS_NOMEM;
  ds4_status s = ds4_num_read(t->num, v, p, bytes);
  if (!s && !save(t->out, name, p, bytes))
    s = DS4_STATUS_INVALID;
  free(p);
  return s;
}
static ds4_status gemma_trace(void *opaque, const char *stage, uint32_t item,
                              uint32_t layer, ds4_num_view v) {
  trace *t = opaque;
  char name[256];
  snprintf(name, sizeof(name), "item-%u-%s-%u.%s", item, stage, layer,
           v.type == DS4_NUM_BF16 ? "bf16" : "f32");
  return dump(t, name, v);
}
static bool cancelled(void *p) { return ((trace *)p)->cancel; }
static ds4_status gemma_cancel_trace(void *opaque, const char *stage,
                                     uint32_t item, uint32_t layer,
                                     ds4_num_view view) {
  (void)item;
  (void)layer;
  (void)view;
  trace *t = opaque;
  if (ds4_embedding_gemma2_release_weights(t->gemma) != DS4_STATUS_BUSY)
    return DS4_STATUS_INVALID;
  if (!strcmp(stage, "complete"))
    t->cancel = true;
  return DS4_STATUS_OK;
}
static ds4_status image_trace(void *opaque, const char *stage, uint32_t layer,
                              ds4_num_view v) {
  trace *t = opaque;
  char name[256];
  if (!strcmp(stage, "latent")) {
    snprintf(name, sizeof(name), "latent-%u.f32", layer);
    return dump(t, name, v);
  }
  if (getenv("DS4_SUITE_CANCEL_STAGE") &&
      !strcmp(getenv("DS4_SUITE_CANCEL_STAGE"), stage))
    t->cancel = true;
  return DS4_STATUS_OK;
}
int main(int argc, char **argv) {
  if (argc < 7)
    return 2;
  ds4_resources *rs = NULL;
  ds4_runtime *rt = NULL;
  ds4_numeric *n = NULL;
  ds4_bundle *b = NULL;
  CHECK(ds4_resources_create(&rs));
  CHECK(ds4_runtime_create(&rt, NULL, NULL));
  CHECK(ds4_numeric_cuda(&n, rs, 100));
  CHECK(ds4_bundle_open(&b, argv[1], argv[2], rs, 101));
  uint64_t baseline, initial_peak;
  ds4_numeric_memory(n, &baseline, &initial_peak);
  size_t free_device, total_device;
  if (cudaMemGetInfo(&free_device, &total_device))
    return 1;
  ds4_resource_budget budget = {.limit = baseline + free_device,
                                .margin = 256u * 1024u * 1024u};
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
  mkdir(argv[5], 0755);
  trace t = {.num = n, .out = argv[5]};
  if (!strcmp(argv[4], "gemma")) {
    ds4_embedding_gemma2 *e = NULL;
    CHECK(ds4_embedding_gemma2_open(&e, b, n, rt, argv[3]));
    size_t count = argc - 6;
    ds4_gemma_text *items = calloc(count, sizeof(*items));
    if (!items)
      return 1;
    for (size_t i = 0; i < count; i++) {
      items[i].text = argv[6 + i];
      items[i].task = getenv("DS4_GEMMA_TASK")
                          ? strtoul(getenv("DS4_GEMMA_TASK"), NULL, 10)
                          : 0;
    }
    ds4_gemma_input *input = NULL;
    CHECK(ds4_gemma_prepare(e, items, count, &input));
    free(items);
    for (size_t i = 0; i < count; i++) {
      const uint32_t *ids;
      uint32_t length;
      CHECK(ds4_gemma_input_tokens(input, i, &ids, &length));
      char name[128];
      snprintf(name, sizeof(name), "item-%zu-tokens.u32", i);
      if (!save(argv[5], name, ids, length * 4))
        return 1;
    }
    ds4_gemma_observer o = {.opaque = &t, .trace = gemma_trace};
    ds4_gemma_result *result = NULL;
    uint32_t dimensions =
        getenv("DS4_GEMMA_DIMENSIONS")
            ? strtoul(getenv("DS4_GEMMA_DIMENSIONS"), NULL, 10)
            : 768;
    CHECK(ds4_gemma_embed(e, input, dimensions, &o, &result));
    ds4_gemma_result_view v = ds4_gemma_result_data(result);
    if (!save(argv[5], "vectors.f32", v.vectors,
              (uint64_t)v.count * v.dimensions * 4))
      return 1;
    if (getenv("DS4_GEMMA_REPEATS")) {
      unsigned repeats = strtoul(getenv("DS4_GEMMA_REPEATS"), NULL, 10);
      if (!repeats || repeats > 100)
        return 2;
      double samples[100];
      for (unsigned i = 0; i < repeats; i++) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        ds4_gemma_result *repeat = NULL;
        CHECK(ds4_gemma_embed(e, input, dimensions, NULL, &repeat));
        clock_gettime(CLOCK_MONOTONIC, &end);
        samples[i] =
            end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
        ds4_gemma_result_view r = ds4_gemma_result_data(repeat);
        if (memcmp(r.vectors, v.vectors, (uint64_t)v.count * v.dimensions * 4))
          return 1;
        ds4_gemma_result_free(repeat);
      }
      if (!save(argv[5], "timings.f64", samples, repeats * sizeof(double)))
        return 1;
    }
    if (getenv("DS4_SUITE_FAULTS")) {
      ds4_gemma_result *rejected = NULL;
      if (ds4_gemma_embed(e, input, 64, NULL, &rejected) !=
              DS4_STATUS_INVALID ||
          rejected)
        return 1;
      ds4_resource_budget original, denied;
      CHECK(ds4_resources_get_budget(rs, DS4_MEMORY_DEVICE, &original));
      denied = original;
      denied.limit = 1;
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &denied));
      if (ds4_gemma_embed(e, input, 768, NULL, &rejected) != DS4_STATUS_NOMEM ||
          rejected)
        return 1;
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &original));
      trace cancel = {.gemma = e};
      ds4_gemma_observer cancel_observer = {.opaque = &cancel,
                                            .cancelled = cancelled,
                                            .trace = gemma_cancel_trace};
      if (ds4_gemma_embed(e, input, dimensions, &cancel_observer, &rejected) !=
              DS4_STATUS_CANCELLED ||
          rejected)
        return 1;
      CHECK(ds4_gemma_embed(e, input, dimensions, NULL, &rejected));
      ds4_gemma_result_view retry = ds4_gemma_result_data(rejected);
      if (memcmp(retry.vectors, v.vectors,
                 (uint64_t)v.count * v.dimensions * 4))
        return 1;
      ds4_gemma_result_free(rejected);
    }
    ds4_gemma_input_free(input);
    CHECK(ds4_embedding_gemma2_close(e));
    /* Results are host-owned and remain readable after adapter teardown. */
    if (!save(argv[5], "vectors-after-close.f32", v.vectors,
              (uint64_t)v.count * v.dimensions * 4))
      return 1;
    ds4_gemma_result_free(result);
  } else if (!strcmp(argv[4], "image")) {
    if (argc < 10 || argc > 74)
      return 2;
    ds4_qwen_image_pipeline *p = NULL;
    CHECK(ds4_qwen_image_pipeline_open(&p, b, n, rt, argv[3]));
    ds4_qwen_image_reference refs[64] = {{0}};
    for (int i = 10; i < argc; i++) {
      FILE *f = fopen(argv[i], "rb");
      if (!f)
        return 2;
      uint32_t shape[3];
      if (fread(shape, 4, 3, f) != 3 || !shape[0] || !shape[1] ||
          shape[0] > 2048 || shape[1] > 2048 ||
          (shape[2] != 3 && shape[2] != 4))
        return 2;
      size_t bytes = (size_t)shape[0] * shape[1] * shape[2];
      uint8_t *pixels = malloc(bytes);
      if (!pixels || fread(pixels, 1, bytes, f) != bytes || fgetc(f) != EOF ||
          fclose(f))
        return 2;
      refs[i - 10] =
          (ds4_qwen_image_reference){pixels, shape[0], shape[1], shape[2]};
    }
    ds4_qwen_image_request request = {.width = strtoul(argv[6], NULL, 10),
                                      .height = strtoul(argv[7], NULL, 10),
                                      .seed = strtoull(argv[8], NULL, 10),
                                      .prompt = argv[9],
                                      .references = refs,
                                      .reference_count = argc - 10};
    float *noise = NULL;
    if (getenv("DS4_SUITE_NOISE")) {
      uint64_t count = (uint64_t)request.width * request.height / 4;
      noise = malloc(count * 4);
      FILE *f = fopen(getenv("DS4_SUITE_NOISE"), "rb");
      if (!f || !noise || fread(noise, 4, count, f) != count ||
          fgetc(f) != EOF || fclose(f))
        return 2;
      request.noise = noise;
      request.noise_count = count;
    }
    ds4_qwen_image_observer o = {
        .opaque = &t, .trace = image_trace, .cancelled = cancelled};
    ds4_qwen_image_job *job = NULL;
    CHECK(ds4_qwen_image_job_create(&job, p, &request, NULL, 0, &o));
    free(noise);
    for (int i = 10; i < argc; i++)
      free((void *)refs[i - 10].pixels);
    ds4_execution_plan plan;
    CHECK(ds4_qwen_image_job_plan(job, &plan));
    fprintf(stdout, "plan variant=%zu victims=%u device=%llu host=%llu\n",
            plan.variant, plan.victims,
            (unsigned long long)plan.peak.bytes[DS4_MEMORY_DEVICE],
            (unsigned long long)plan.peak.bytes[DS4_MEMORY_HOST]);
    fflush(stdout);
    ds4_qwen_image_rgba *result = NULL;
    if (ds4_qwen_image_job_take(job, &result) != DS4_STATUS_NOT_READY || result)
      return 1;
    ds4_status status = ds4_qwen_image_job_run(job);
    if (getenv("DS4_SUITE_CANCEL_STAGE")) {
      if (status != DS4_STATUS_CANCELLED ||
          ds4_qwen_image_job_take(job, &result) != DS4_STATUS_NOT_READY)
        return 1;
    } else {
      CHECK(status);
      CHECK(ds4_qwen_image_job_take(job, &result));
      ds4_qwen_image_rgba_view v = ds4_qwen_image_rgba_data(result);
      if (!save(argv[5], "rgba.f32", v.pixels,
                (uint64_t)v.width * v.height * 16))
        return 1;
      char path[4096];
      snprintf(path, sizeof(path), "%s/result.png", argv[5]);
      CHECK(ds4_qwen_image_rgba_save_png(result, path));
    }
    const ds4_qwen_image_report *report = ds4_qwen_image_job_report(job);
    char json[4096];
    int len = snprintf(
        json, sizeof(json),
        "{\"width\":%u,\"height\":%u,\"seed\":%llu,\"steps\":8,\"references\":%"
        "u,\"variant\":%zu,\"victims\":%u,\"total_seconds\":%.9f,\"prepare_"
        "seconds\":%.9f,\"load_seconds\":[%.9f,%.9f,%.9f],\"phase_seconds\":[%."
        "9f,%.9f,%.9f,%.9f],\"status\":%d}\n",
        report->width, report->height, (unsigned long long)report->seed,
        report->references, report->plan.variant, report->plan.victims,
        report->total_seconds, report->prepare_seconds, report->load_seconds[0],
        report->load_seconds[1], report->load_seconds[2],
        report->phase_seconds[0], report->phase_seconds[1],
        report->phase_seconds[2], report->phase_seconds[3], status);
    if (len <= 0 || len >= (int)sizeof(json) ||
        !save(argv[5], "report.json", json, len))
      return 1;
    CHECK(ds4_qwen_image_job_close(job));
    CHECK(ds4_qwen_image_pipeline_close(p));
    ds4_qwen_image_rgba_free(result);
  } else
    return 2;
  CHECK(ds4_bundle_close(b));
  uint64_t live, peak;
  ds4_numeric_memory(n, &live, &peak);
  fprintf(stdout, "live=%llu peak=%llu\n", (unsigned long long)live,
          (unsigned long long)peak);
  CHECK(ds4_numeric_close(n));
  ds4_runtime_release(rt);
  CHECK(ds4_resources_close(rs));
  return live != baseline;
}
