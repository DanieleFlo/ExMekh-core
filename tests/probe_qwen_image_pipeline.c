/* Native multi-request and recovery qualification. Fixture: DS4PIP1\0,
 * uint32 width,height,references,prompt_bytes; uint64 seed; prompt bytes;
 * each reference: uint32 width,height,channels + pixels; packed FP32 noise.
 * All fields little-endian, qualification host is x86_64. */
#include "ds4_qwen_image_pipeline.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    ds4_status s = (x);                                                        \
    if (s) {                                                                   \
      fprintf(stderr, "%s: %d\n", #x, s);                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)
extern int cudaMemGetInfo(size_t *, size_t *);
typedef struct {
  ds4_qwen_image_request request;
  ds4_qwen_image_reference refs[64];
  char *prompt;
  float *noise;
} fixture;
static int read_fixture(const char *path, fixture *f) {
  FILE *file = fopen(path, "rb");
  if (!file)
    return 0;
  char magic[8];
  uint32_t head[4];
  uint64_t seed;
  if (fread(magic, 1, 8, file) != 8 ||
      (memcmp(magic, "DS4PIP1\0", 8) && memcmp(magic, "DS4PIP2\0", 8)) ||
      fread(head, 4, 4, file) != 4 || fread(&seed, 8, 1, file) != 1 ||
      !head[0] || !head[1] || head[0] > 2048 || head[1] > 2048 ||
      head[2] > 64 || head[3] > 4 * 1024 * 1024)
    return 0;
  uint32_t resolution = 0;
  if (!memcmp(magic, "DS4PIP2\0", 8) && fread(&resolution, 4, 1, file) != 1)
    return 0;
  f->prompt = calloc(1, (size_t)head[3] + 1);
  if (!f->prompt || fread(f->prompt, 1, head[3], file) != head[3])
    return 0;
  for (uint32_t i = 0; i < head[2]; i++) {
    uint32_t s[3];
    if (fread(s, 4, 3, file) != 3 || !s[0] || !s[1] || s[0] > 2048 ||
        s[1] > 2048 || (s[2] != 3 && s[2] != 4))
      return 0;
    uint64_t bytes = (uint64_t)s[0] * s[1] * s[2];
    uint8_t *pixels = malloc(bytes);
    if (!pixels || fread(pixels, 1, bytes, file) != bytes)
      return 0;
    f->refs[i] = (ds4_qwen_image_reference){pixels, s[0], s[1], s[2]};
  }
  uint64_t count = (uint64_t)head[0] * head[1] / 4;
  f->noise = malloc(count * 4);
  if (!f->noise || fread(f->noise, 4, count, file) != count ||
      fgetc(file) != EOF || fclose(file))
    return 0;
  f->request = (ds4_qwen_image_request){.prompt = f->prompt,
                                        .width = head[0],
                                        .height = head[1],
                                        .references = f->refs,
                                        .reference_count = head[2],
                                        .seed = seed,
                                        .noise = f->noise,
                                        .noise_count = count};
  f->request.reference_resolution = resolution;
  if (getenv("DS4_REFERENCE_RESOLUTION"))
    f->request.reference_resolution =
        strtoul(getenv("DS4_REFERENCE_RESOLUTION"), NULL, 10);
  return 1;
}
static void free_fixture(fixture *f) {
  for (size_t i = 0; i < f->request.reference_count; i++)
    free((void *)f->refs[i].pixels);
  free(f->prompt);
  free(f->noise);
}
typedef struct {
  ds4_numeric *num;
  const char *out;
  bool cancel;
  const char *cancel_stage;
  uint32_t phase;
} observer;
static int save(const char *directory, const char *name, const void *data,
                size_t bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", directory, name);
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  bool ok = fwrite(data, 1, bytes, f) == bytes;
  return fclose(f) == 0 && ok;
}
static bool cancelled(void *p) { return ((observer *)p)->cancel; }
static ds4_status trace(void *p, const char *stage, uint32_t layer,
                        ds4_num_view v) {
  observer *o = p;
  if (o->cancel_stage && !strcmp(stage, o->cancel_stage))
    o->cancel = true;
  if (!strcmp(stage, "latent") && o->out) {
    char name[128];
    snprintf(name, sizeof(name), "latent-%u.f32", layer);
    float *data = malloc(v.count * 4);
    if (!data)
      return DS4_STATUS_NOMEM;
    ds4_status s = ds4_num_read(o->num, v, data, v.count * 4);
    if (!s && !save(o->out, name, data, v.count * 4))
      s = DS4_STATUS_INVALID;
    free(data);
    return s;
  }
  if (!strcmp(stage, "conditioning") && o->out) {
    uint16_t *data = malloc(v.count * 2);
    if (!data)
      return DS4_STATUS_NOMEM;
    ds4_status s = ds4_num_read(o->num, v, data, v.count * 2);
    if (!s && !save(o->out, "conditioning.bf16", data, v.count * 2))
      s = DS4_STATUS_INVALID;
    free(data);
    return s;
  }
  if (!strcmp(stage, "encoded") && o->out) {
    char name[128];
    snprintf(name, sizeof(name), "reference-%u.bf16", o->phase++);
    uint16_t *data = malloc(v.count * 2);
    if (!data)
      return DS4_STATUS_NOMEM;
    ds4_status s = ds4_num_read(o->num, v, data, v.count * 2);
    if (!s && !save(o->out, name, data, v.count * 2))
      s = DS4_STATUS_INVALID;
    free(data);
    return s;
  }
  return DS4_STATUS_OK;
}
static int write_report(const char *out, const ds4_qwen_image_report *r,
                        ds4_numeric *n, uint64_t before,
                        uint64_t previous_peak) {
  uint64_t live, peak;
  ds4_numeric_memory(n, &live, &peak);
  char json[16384];
  int len = snprintf(
      json, sizeof(json),
      "{\"width\":%u,\"height\":%u,\"references\":%u,\"seed\":%llu,\"variant\":"
      "%zu,\"victims\":%u,\"total_seconds\":%.9f,\"prepare_seconds\":%.9f,"
      "\"load_seconds\":[%.9f,%.9f,%.9f],\"phase_seconds\":[%.9f,%.9f,%.9f,%."
      "9f],\"loads\":[%u,%u,%u],\"releases\":[%u,%u,%u],\"peak_device_owned\":%"
      "llu,\"live_device_owned\":%llu",
      r->width, r->height, r->references, (unsigned long long)r->seed,
      r->plan.variant, r->plan.victims, r->total_seconds, r->prepare_seconds,
      r->load_seconds[0], r->load_seconds[1], r->load_seconds[2],
      r->phase_seconds[0], r->phase_seconds[1], r->phase_seconds[2],
      r->phase_seconds[3], r->loads[0], r->loads[1], r->loads[2],
      r->releases[0], r->releases[1], r->releases[2], (unsigned long long)peak,
      (unsigned long long)live);
  if (len < 0 || len >= (int)sizeof(json))
    return 0;
  uint64_t predicted = before + r->plan.peak.bytes[DS4_MEMORY_DEVICE];
  /* A new high-water mark belongs to this operation. Earlier larger requests
   * make the context's cumulative peak unsuitable for a per-request bound. */
  assert(peak <= previous_peak || peak <= predicted);
  len += snprintf(
      json + len, sizeof(json) - len,
      ",\"pre_run_device_owned\":%llu,\"previous_device_peak\":%llu,"
      "\"plan_device\":%llu,\"plan_host\":%llu,\"peak_bound_checked\":%s",
      (unsigned long long)before, (unsigned long long)previous_peak,
      (unsigned long long)r->plan.peak.bytes[DS4_MEMORY_DEVICE],
      (unsigned long long)r->plan.peak.bytes[DS4_MEMORY_HOST],
      peak > previous_peak ? "true" : "false");
  len += snprintf(json + len, sizeof(json) - len,
                  ",\"recipe_version\":%u,\"rng_version\":%u,"
                  "\"model_revision\":\"%s\",\"reference_resolution\":%u",
                  r->recipe_version, r->rng_version, r->model_revision,
                  r->reference_resolution);
  const uint8_t *hashes[] = {r->prompt_sha256, r->noise_sha256};
  const char *names[] = {"prompt_sha256", "noise_sha256"};
  for (uint32_t i = 0; i < 2; i++) {
    len += snprintf(json + len, sizeof(json) - len, ",\"%s\":\"", names[i]);
    for (uint32_t j = 0; j < 32; j++)
      len += snprintf(json + len, sizeof(json) - len, "%02x", hashes[i][j]);
    len += snprintf(json + len, sizeof(json) - len, "\"");
  }
  len += snprintf(json + len, sizeof(json) - len, ",\"reference_metadata\":[");
  for (uint32_t i = 0; i < r->references; i++) {
    len += snprintf(json + len, sizeof(json) - len,
                    "%s{\"latent_width\":%u,\"latent_height\":%u,\"sha256\":\"",
                    i ? "," : "", r->reference_shapes[i].width,
                    r->reference_shapes[i].height);
    for (uint32_t j = 0; j < 32; j++)
      len += snprintf(json + len, sizeof(json) - len, "%02x",
                      r->reference_sha256[i][j]);
    len += snprintf(json + len, sizeof(json) - len, "\"}");
  }
  len += snprintf(json + len, sizeof(json) - len, "],\"sigmas\":[");
  for (uint32_t i = 0; i < 9; i++)
    len += snprintf(json + len, sizeof(json) - len, "%s%.9g", i ? "," : "",
                    r->sigmas[i]);
  len += snprintf(json + len, sizeof(json) - len, "]}\n");
  return len > 0 && len < (int)sizeof(json) &&
         save(out, "report.json", json, len);
}
typedef struct {
  ds4_resources *rs;
  ds4_resource *allocation;
  bool fail;
  uint32_t saved, restored;
} victim;
static ds4_status ready(void *p) {
  (void)p;
  return DS4_STATUS_OK;
}
static ds4_status save_victim(void *p) {
  ((victim *)p)->saved++;
  return DS4_STATUS_OK;
}
static ds4_status release_victim(void *p) {
  victim *v = p;
  ds4_status s = ds4_resource_release(v->allocation);
  if (!s)
    v->allocation = NULL;
  return s;
}
static ds4_status restore_victim(void *p) {
  victim *v = p;
  if (v->fail)
    return DS4_STATUS_RESTORE;
  v->restored++;
  ds4_resource_spec spec = {.owner = 300,
                            .location = DS4_MEMORY_DEVICE,
                            .allocated = 8ull << 30,
                            .requested = 8ull << 30,
                            .live = 8ull << 30};
  return ds4_resource_register(v->rs, &spec, &v->allocation);
}
static void discard(void *p) { (void)p; }
int main(int argc, char **argv) {
  if (argc < 6)
    return 2;
  ds4_resources *rs = NULL;
  ds4_runtime *rt = NULL;
  ds4_numeric *n = NULL;
  ds4_bundle *b = NULL;
  ds4_qwen_image_pipeline *pipeline = NULL;
  CHECK(ds4_resources_create(&rs));
  CHECK(ds4_runtime_create(&rt, NULL, NULL));
  CHECK(ds4_numeric_cuda(&n, rs, 200));
  uint64_t baseline, peak;
  ds4_numeric_memory(n, &baseline, &peak);
  size_t available, total;
  if (cudaMemGetInfo(&available, &total))
    return 1;
  ds4_resource_budget budget = {.limit = baseline + available,
                                .margin = 256u * 1024u * 1024u};
  const char *limit = getenv("DS4_SUITE_DEVICE_LIMIT");
  if (limit) {
    char *end;
    unsigned long long bytes = strtoull(limit, &end, 10);
    if (!*limit || *end || !bytes || bytes > budget.limit)
      return 2;
    budget.limit = bytes;
  }
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
  CHECK(ds4_bundle_open(&b, argv[1], argv[2], rs, 201));
  CHECK(ds4_qwen_image_pipeline_open(&pipeline, b, n, rt, argv[3]));
  mkdir(argv[4], 0755);
  fixture first = {0};
  for (int arg = 5; arg < argc; arg++) {
    fixture f = {0};
    if (!read_fixture(argv[arg], &f))
      return 2;
    char out[4096];
    snprintf(out, sizeof(out), "%s/case-%d", argv[4], arg - 5);
    mkdir(out, 0755);
    observer obs = {.num = n, .out = out};
    ds4_qwen_image_observer o = {&obs, cancelled, trace};
    ds4_qwen_image_job *j = NULL;
    ds4_qwen_image_rgba *result = NULL;
    CHECK(ds4_qwen_image_job_create(&j, pipeline, &f.request, NULL, 0, &o));
    if (getenv("DS4_SUITE_PLAN_ONLY")) {
      ds4_execution_plan plan = {0};
      ds4_status s = ds4_qwen_image_job_plan(j, &plan);
      char json[512];
      int bytes =
          snprintf(json, sizeof(json),
                   "{\"status\":%u,\"variant\":%zu,\"victims\":%u,\"device\":%"
                   "llu,\"host\":%llu}\n",
                   s, plan.variant, plan.victims,
                   (unsigned long long)plan.peak.bytes[DS4_MEMORY_DEVICE],
                   (unsigned long long)plan.peak.bytes[DS4_MEMORY_HOST]);
      if (!save(out, "plan.json", json, bytes))
        return 1;
      CHECK(ds4_qwen_image_job_close(j));
      free_fixture(&f);
      fprintf(stdout, "case-%d admission=%u\n", arg - 5, s);
      fflush(stdout);
      continue;
    }
    assert(ds4_qwen_image_pipeline_close(pipeline) == DS4_STATUS_BUSY);
    assert(ds4_qwen_image_job_take(j, &result) == DS4_STATUS_NOT_READY &&
           !result);
    uint64_t before, previous_peak;
    ds4_numeric_memory(n, &before, &previous_peak);
    CHECK(ds4_qwen_image_job_run(j));
    ds4_execution_plan completed_plan;
    CHECK(ds4_qwen_image_job_plan(j, &completed_plan));
    const ds4_qwen_image_report *completed = ds4_qwen_image_job_report(j);
    assert(completed_plan.variant == completed->plan.variant &&
           completed_plan.victims == completed->plan.victims &&
           completed_plan.peak.bytes[DS4_MEMORY_DEVICE] ==
               completed->plan.peak.bytes[DS4_MEMORY_DEVICE]);
    double completed_seconds = completed->total_seconds;
    assert(ds4_qwen_image_job_run(j) == DS4_STATUS_INVALID &&
           completed->total_seconds == completed_seconds);
    CHECK(ds4_qwen_image_job_take(j, &result));
    ds4_qwen_image_rgba_view image = ds4_qwen_image_rgba_data(result);
    if (!save(out, "rgba.f32", image.pixels,
              (uint64_t)image.width * image.height * 16) ||
        !write_report(out, ds4_qwen_image_job_report(j), n, before,
                      previous_peak))
      return 1;
    char path[8192];
    snprintf(path, sizeof(path), "%s/result.png", out);
    CHECK(ds4_qwen_image_rgba_save_png(result, path));
    CHECK(ds4_qwen_image_job_close(j));
    ds4_qwen_image_rgba_free(result);
    fprintf(stdout, "case-%d PASS\n", arg - 5);
    fflush(stdout);
    if (arg == 5) {
      first = f;
      /* The fixture owns an inline reference array. Rebase the request after
       * copying it; the loop reuses f before the warm/fault checks run. */
      first.request.references = first.refs;
    } else
      free_fixture(&f);
  }
  if (getenv("DS4_SUITE_PLAN_ONLY") || getenv("DS4_SUITE_CASES_ONLY"))
    goto finish;
  if (getenv("DS4_SUITE_REPEATS")) {
    unsigned repeats = strtoul(getenv("DS4_SUITE_REPEATS"), NULL, 10);
    if (!repeats || repeats > 100)
      return 2;
    double samples[100];
    /* One explicit warm-up, then repeated whole operations with weight loads,
     * host preparation, decode and validated publication in the measured path.
     */
    for (unsigned i = 0; i <= repeats; i++) {
      ds4_qwen_image_job *repeat = NULL;
      ds4_qwen_image_rgba *result = NULL;
      CHECK(ds4_qwen_image_job_create(&repeat, pipeline, &first.request, NULL,
                                      0, NULL));
      CHECK(ds4_qwen_image_job_run(repeat));
      CHECK(ds4_qwen_image_job_take(repeat, &result));
      if (i)
        samples[i - 1] = ds4_qwen_image_job_report(repeat)->total_seconds;
      CHECK(ds4_qwen_image_job_close(repeat));
      ds4_qwen_image_rgba_free(result);
    }
    if (!save(argv[4], "timings.f64", samples, repeats * sizeof(double)))
      return 1;
  }
  /* Invalid/unsupported requests and admission rejection cannot touch GPU
   * compute or publish a result. Restoration failure keeps the job alive. */
  ds4_qwen_image_job *j = NULL;
  ds4_qwen_image_request bad = first.request;
  bad.mask = &bad;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_UNSUPPORTED &&
         !j);
  bad = first.request;
  bad.width++;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_INVALID &&
         !j);
  bad = first.request;
  bad.reference_resolution = 17;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_INVALID &&
         !j);
  bad = first.request;
  bad.reference_count = 65;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_INVALID &&
         !j);
  bad = first.request;
  bad.steps = 9;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_UNSUPPORTED &&
         !j);
  bad = first.request;
  bad.guidance = 2;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_UNSUPPORTED &&
         !j);
  bad = first.request;
  bad.negative_prompt = "unsupported";
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_UNSUPPORTED &&
         !j);
  bad = first.request;
  bad.noise_count++;
  assert(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL) ==
             DS4_STATUS_INVALID &&
         !j);
  bad = first.request;
  bad.noise = NULL;
  bad.noise_count = 0;
  uint8_t noise_hash[32], zero[32] = {0};
  for (uint32_t i = 0; i < 2; i++) {
    CHECK(ds4_qwen_image_job_create(&j, pipeline, &bad, NULL, 0, NULL));
    const ds4_qwen_image_report *r = ds4_qwen_image_job_report(j);
    assert(r->rng_version == 1 && memcmp(r->noise_sha256, zero, 32));
    if (i)
      assert(!memcmp(noise_hash, r->noise_sha256, 32));
    memcpy(noise_hash, r->noise_sha256, 32);
    CHECK(ds4_qwen_image_job_close(j));
  }
  CHECK(ds4_qwen_image_job_create(&j, pipeline, &first.request, NULL, 0, NULL));
  ds4_resource_budget denied = budget;
  denied.limit = 1;
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &denied));
  ds4_execution_plan rejected;
  assert(ds4_qwen_image_job_plan(j, &rejected) == DS4_STATUS_NOMEM);
  CHECK(ds4_qwen_image_job_close(j));
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
  observer cancelled_obs = {.num = n, .cancel_stage = "latent"};
  ds4_qwen_image_observer cancel_observer = {&cancelled_obs, cancelled, trace};
  CHECK(ds4_qwen_image_job_create(&j, pipeline, &first.request, NULL, 0,
                                  &cancel_observer));
  assert(ds4_qwen_image_job_run(j) == DS4_STATUS_CANCELLED);
  ds4_qwen_image_rgba *result = NULL;
  assert(ds4_qwen_image_job_take(j, &result) == DS4_STATUS_NOT_READY &&
         !result);
  CHECK(ds4_qwen_image_job_close(j));
  victim v = {.rs = rs};
  CHECK(restore_victim(&v));
  v.restored = 0;
  v.fail = true;
  ds4_execution_victim descriptor = {
      .kind = DS4_RECLAIM_SESSION,
      .reclaimed.bytes = {[DS4_MEMORY_DEVICE] = 8ull << 30},
      .cost = 1,
      .context = &v,
      .quiesce = ready,
      .save = save_victim,
      .release = release_victim,
      .restore = restore_victim,
      .discard = discard};
  CHECK(ds4_qwen_image_job_create(&j, pipeline, &first.request, &descriptor, 1,
                                  NULL));
  ds4_resource_budget constrained = budget;
  constrained.limit = 20ull << 30;
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &constrained));
  assert(ds4_qwen_image_job_run(j) == DS4_STATUS_RESTORE);
  ds4_execution_recovery recovery;
  CHECK(ds4_qwen_image_job_recovery(j, &recovery));
  assert(recovery.state == DS4_EXECUTION_RECOVERABLE && v.saved == 1 &&
         !v.allocation);
  assert(ds4_qwen_image_job_take(j, &result) == DS4_STATUS_NOT_READY &&
         !result && ds4_qwen_image_job_close(j) == DS4_STATUS_BUSY);
  v.fail = false;
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
  CHECK(ds4_qwen_image_job_recover(j));
  CHECK(ds4_qwen_image_job_take(j, &result));
  CHECK(ds4_qwen_image_job_close(j));
  ds4_qwen_image_rgba_free(result);
  CHECK(release_victim(&v));
finish:
  free_fixture(&first);
  CHECK(ds4_qwen_image_pipeline_close(pipeline));
  CHECK(ds4_bundle_close(b));
  ds4_numeric_memory(n, &available, &peak);
  assert(available == baseline);
  CHECK(ds4_numeric_close(n));
  ds4_runtime_release(rt);
  CHECK(ds4_resources_close(rs));
  if (getenv("DS4_SUITE_PLAN_ONLY"))
    fprintf(stdout, "pipeline: admission recorded, inference not run\n");
  else if (getenv("DS4_SUITE_CASES_ONLY"))
    fprintf(stdout,
            "pipeline: cases/lifetime completed; fault matrix not run\n");
  else
    fprintf(
        stdout,
        "pipeline: PASS input/admission/cancel/restore/publication/lifetime "
        "peak=%llu\n",
        (unsigned long long)peak);
  return 0;
}
