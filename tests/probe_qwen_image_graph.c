/* Development gate over real frozen weights, one component process at a time.
 */
#include "ds4_qwen_image_dit.h"
#include "ds4_qwen_image_graph.h"
#include "ds4_qwen_image_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
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
  const char *dir;
  uint32_t step;
  int dit;
  uint32_t blocks, latents;
  uint64_t step_live[8];
  bool cancel;
  bool requested;
  const char *cancel_stage;
  ds4_qwen_image_dit *d;
  ds4_qwen_image_vae *v;
} trace;
static int save(const char *dir, const char *name, const void *data,
                size_t bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  int ok = fwrite(data, 1, bytes, f) == bytes;
  return fclose(f) == 0 && ok;
}
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static bool cancelled(void *p) { return ((trace *)p)->requested; }
static ds4_status dump(void *opaque, const char *stage, uint32_t layer,
                       ds4_num_view view) {
  trace *c = opaque;
  char name[256];
  if (!strcmp(stage, "block"))
    c->blocks++;
  if (c->cancel && c->blocks == 3)
    return DS4_STATUS_CANCELLED;
  if (c->cancel)
    return DS4_STATUS_OK;
  if (view.data) {
    if (c->dit)
      snprintf(name, sizeof(name), "step-%u-%s-%u.%s", c->step, stage, layer,
               view.type == 30 ? "bf16" : "f32");
    else
      snprintf(name, sizeof(name), "%s-%u.%s", stage, layer,
               view.type == 30 ? "bf16" : "f32");
    size_t bytes = view.count * (view.type == 30 ? 2 : 4);
    void *p = malloc(bytes);
    if (!p)
      return DS4_STATUS_NOMEM;
    ds4_status s = ds4_num_read(c->num, view, p, bytes);
    if (!s && !save(c->dir, name, p, bytes))
      s = DS4_STATUS_INVALID;
    free(p);
    if (s)
      return s;
  }
  if (!strcmp(stage, "latent")) {
    uint64_t peak;
    ds4_numeric_memory(c->num, &c->step_live[c->step], &peak);
    c->step++;
    c->latents++;
  }
  if (c->d && ds4_qwen_image_dit_release_weights(c->d) != DS4_STATUS_BUSY)
    return DS4_STATUS_INVALID;
  if (c->v && ds4_qwen_image_vae_release_weights(c->v) != DS4_STATUS_BUSY)
    return DS4_STATUS_INVALID;
  if (getenv("DS4_SUITE_GRAPH_DIAGNOSTICS") && !strcmp(stage, "block"))
    return DS4_STATUS_CANCELLED;
  if (c->cancel_stage && !strcmp(stage, c->cancel_stage))
    c->requested = true;
  return DS4_STATUS_OK;
}
static void *readall(const char *dir, const char *name, size_t *bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  if (n <= 0) {
    fclose(f);
    return NULL;
  }
  void *p = malloc(n);
  if (!p || fread(p, 1, n, f) != (size_t)n) {
    free(p);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *bytes = n;
  return p;
}
static int dit(ds4_bundle *bundle, ds4_numeric *num, ds4_runtime *runtime,
               const char *input, const char *out) {
  ds4_qwen_image_dit *e = NULL;
  CHECK(ds4_qwen_image_dit_open(&e, bundle, num, runtime));
  size_t bytes;
  uint32_t *fixture = readall(input, "input.ds4g", &bytes);
  if (!fixture || bytes < 16)
    return 2;
  uint32_t h = fixture[0], w = fixture[1], text = fixture[2], ns = fixture[3];
  if (ns > 65 || !ns)
    return 2;
  ds4_qwen_image_latent_shape *shapes = (void *)(fixture + 4);
  uint64_t refs = 0;
  for (uint32_t i = 0; i + 1 < ns; i++)
    refs += (uint64_t)shapes[i].height * shapes[i].width;
  uint8_t *mask = (uint8_t *)(shapes + ns);
  uint8_t *hidden = mask + text,
          *reference = hidden + (uint64_t)text * 4096 * 2,
          *noise = reference + refs * 64 * 2;
  if ((uint64_t)(noise - (uint8_t *)fixture) + (uint64_t)h * w * 64 * 4 !=
      bytes)
    return 2;
  /* The fixture is byte-packed; copy unaligned payloads through writes. */
  ds4_num_buffer *c = NULL, *r = NULL;
  CHECK(ds4_num_alloc(num, (uint64_t)text * 4096, 30, &c));
  CHECK(ds4_num_write(num, ds4_num_data(c), hidden, (uint64_t)text * 4096 * 2));
  if (refs) {
    CHECK(ds4_num_alloc(num, refs * 64, 30, &r));
    CHECK(ds4_num_write(num, ds4_num_data(r), reference, refs * 128));
  }
  float *aligned = malloc((uint64_t)h * w * 64 * 4);
  memcpy(aligned, noise, (uint64_t)h * w * 64 * 4);
  ds4_qwen_image_conditioning_view cv = {ds4_num_data(c), mask, text};
  ds4_qwen_image_dit_input *p = NULL;
  CHECK(ds4_qwen_image_dit_prepare(e, cv, ds4_num_data(r), shapes, ns, &p));
  CHECK(ds4_num_free(c));
  CHECK(ds4_num_free(r));
  free(fixture);
  ds4_qwen_image_dit_memory estimate;
  CHECK(ds4_qwen_image_dit_estimate(e, p, &estimate));
  ds4_resource_budget original, denied;
  ds4_resource_totals before, after;
  ds4_resources *resources = ds4_numeric_resources(num, NULL);
  CHECK(ds4_resources_get_budget(resources, DS4_MEMORY_DEVICE, &original));
  ds4_resources_totals(resources, 0, &before);
  denied = original;
  denied.limit = before.allocated[DS4_MEMORY_DEVICE] + 1;
  CHECK(ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &denied));
  ds4_num_buffer *rejected = NULL;
  ds4_status refusal =
      ds4_qwen_image_dit_generate(e, p, aligned, 0, NULL, &rejected);
  CHECK(ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &original));
  ds4_resources_totals(resources, 0, &after);
  if (refusal != DS4_STATUS_NOMEM || rejected ||
      memcmp(&before, &after, sizeof(before)))
    return 1;
  trace t = {.num = num, .dir = out, .dit = 1, .d = e};
  ds4_qwen_image_observer obs = {
      .opaque = &t, .trace = dump, .cancelled = cancelled};
  ds4_num_buffer *result = NULL;
  if (getenv("DS4_SUITE_GRAPH_DIAGNOSTICS")) {
    if (ds4_qwen_image_dit_generate(e, p, aligned, 0, &obs, &result) !=
            DS4_STATUS_CANCELLED ||
        result)
      return 1;
    free(aligned);
    CHECK(ds4_qwen_image_dit_input_free(p));
    CHECK(ds4_qwen_image_dit_close(e));
    return 0;
  }
  CHECK(ds4_qwen_image_dit_generate(e, p, aligned, 0, &obs, &result));
  if (t.latents != 8 || t.blocks != 32 * 8)
    return 1;
  for (uint32_t i = 1; i < 8; i++)
    if (t.step_live[i] != t.step_live[0])
      return 1;
  if (!save(out, "step-live.u64", t.step_live, sizeof(t.step_live)))
    return 1;
  CHECK(ds4_num_free(result));
  result = NULL;
  if (getenv("DS4_SUITE_GRAPH_REPEATS")) {
    unsigned repeats = strtoul(getenv("DS4_SUITE_GRAPH_REPEATS"), NULL, 10);
    if (!repeats || repeats > 100)
      return 2;
    double samples[100];
    for (unsigned i = 0; i < repeats; i++) {
      double start = now();
      CHECK(ds4_qwen_image_dit_generate(e, p, aligned, 0, NULL, &result));
      samples[i] = now() - start;
      CHECK(ds4_num_free(result));
      result = NULL;
    }
    if (!save(out, "timings.f64", samples, repeats * sizeof(double)))
      return 1;
  }
  /* Target-only/cached and complete/full recomputation use identical input. */
  CHECK(ds4_qwen_image_dit_cache_clear(p));
  ds4_num_buffer *x = NULL, *velocity = NULL, *delta = NULL, *rounded = NULL;
  uint64_t count = (uint64_t)h * w * 64;
  CHECK(ds4_num_alloc(num, count, 0, &x));
  CHECK(ds4_num_write(num, ds4_num_data(x), aligned, count * 4));
  CHECK(ds4_num_alloc(num, count, 0, &delta));
  CHECK(ds4_num_alloc(num, count, 30, &rounded));
  CHECK(ds4_num_point(num, ds4_num_data(rounded), ds4_num_data(x),
                      (ds4_num_view){0}, DS4_NUM_CAST, 0));
  CHECK(ds4_num_point(num, ds4_num_data(x), ds4_num_data(rounded),
                      (ds4_num_view){0}, DS4_NUM_CAST, 0));
  float sigma[9];
  ds4_qwen_image_turbo_sigmas(sigma);
  for (uint32_t i = 0; i < 8; i++) {
    CHECK(ds4_qwen_image_dit_cache_clear(p));
    CHECK(ds4_qwen_image_dit_forward(e, p, ds4_num_data(x), sigma[i], NULL,
                                     &velocity));
    CHECK(ds4_num_point(num, ds4_num_data(rounded), ds4_num_data(velocity),
                        (ds4_num_view){0}, DS4_NUM_SCALE,
                        qi_float(qi_bf16(sigma[i + 1] - sigma[i]))));
    CHECK(ds4_num_point(num, ds4_num_data(delta), ds4_num_data(rounded),
                        (ds4_num_view){0}, DS4_NUM_CAST, 0));
    CHECK(ds4_num_point(num, ds4_num_data(x), ds4_num_data(x),
                        ds4_num_data(delta), DS4_NUM_ADD, 0));
    CHECK(ds4_num_point(num, ds4_num_data(rounded), ds4_num_data(x),
                        (ds4_num_view){0}, DS4_NUM_CAST, 0));
    CHECK(ds4_num_point(num, ds4_num_data(x), ds4_num_data(rounded),
                        (ds4_num_view){0}, DS4_NUM_CAST, 0));
    CHECK(ds4_num_free(velocity));
    CHECK(ds4_num_read(num, ds4_num_data(x), aligned, count * 4));
    char name[128];
    snprintf(name, sizeof(name), "uncached-latent-%u.f32", i);
    if (!save(out, name, aligned, count * 4))
      return 1;
  }
  /* Isolate cache equivalence from accumulation along different BF16 Euler
   * trajectories: both forwards must receive exactly the same latent. */
  for (uint32_t i = 0; i < 8; i++) {
    char name[128];
    snprintf(name, sizeof(name), "step-%u-latent-%u.f32", i ? i - 1 : 0,
             i ? i - 1 : 0);
    size_t sample_bytes;
    float *sample = i ? readall(out, name, &sample_bytes) : NULL;
    if (i && (!sample || sample_bytes != count * 4))
      return 1;
    /* The first step is already the identical full forward checked above. */
    if (!i)
      continue;
    CHECK(ds4_num_write(num, ds4_num_data(x), sample, count * 4));
    free(sample);
    CHECK(ds4_qwen_image_dit_cache_clear(p));
    CHECK(ds4_qwen_image_dit_forward(e, p, ds4_num_data(x), sigma[i], NULL,
                                     &velocity));
    uint16_t *raw = malloc(count * 2);
    if (!raw)
      return 1;
    CHECK(ds4_num_read(num, ds4_num_data(velocity), raw, count * 2));
    snprintf(name, sizeof(name), "uncached-same-input-velocity-%u.bf16", i);
    if (!save(out, name, raw, count * 2))
      return 1;
    free(raw);
    CHECK(ds4_num_free(velocity));
  }
  CHECK(ds4_num_free(x));
  CHECK(ds4_num_free(delta));
  CHECK(ds4_num_free(rounded));
  /* A cancellation inside a block must release every cache allocation. */
  t.cancel = true;
  t.blocks = 0;
  t.step = 0;
  t.dir = out;
  CHECK(ds4_qwen_image_dit_release_weights(e));
  ds4_status cancelled =
      ds4_qwen_image_dit_generate(e, p, aligned, 0, &obs, &result);
  if (cancelled != DS4_STATUS_CANCELLED || result)
    return 1;
  free(aligned);
  CHECK(ds4_qwen_image_dit_input_free(p));
  CHECK(ds4_qwen_image_dit_release_weights(e));
  CHECK(ds4_qwen_image_dit_close(e));
  printf("dit PASS steps=8 cached/uncached dumped weights=%llu prefix=%llu\n",
         (unsigned long long)estimate.weights,
         (unsigned long long)estimate.prefix_cache);
  return 0;
}
static int vae(ds4_bundle *bundle, ds4_numeric *num, ds4_runtime *runtime,
               const char *input, const char *out) {
  ds4_qwen_image_vae *e = NULL;
  CHECK(ds4_qwen_image_vae_open(&e, bundle, num, runtime));
  size_t bytes;
  uint32_t *fixture = readall(input, "input.ds4v", &bytes);
  if (!fixture || bytes < 12)
    return 2;
  ds4_qwen_image_reference r = {.width = fixture[0],
                                .height = fixture[1],
                                .channels = fixture[2],
                                .pixels = (uint8_t *)(fixture + 3)};
  if (bytes != 12 + (uint64_t)r.width * r.height * r.channels)
    return 2;
  trace t = {.num = num, .dir = out, .v = e};
  ds4_qwen_image_observer obs = {
      .opaque = &t, .trace = dump, .cancelled = cancelled};
  ds4_num_buffer *encoded = NULL;
  t.cancel_stage = "encoded";
  if (ds4_qwen_image_vae_encode(e, &r, &obs, &encoded) !=
          DS4_STATUS_CANCELLED ||
      encoded)
    return 1;
  t.cancel_stage = NULL;
  t.requested = false;
  CHECK(ds4_qwen_image_vae_encode(e, &r, &obs, &encoded));
  CHECK(ds4_num_free(encoded));
  unsigned repeats = getenv("DS4_SUITE_GRAPH_REPEATS")
                         ? strtoul(getenv("DS4_SUITE_GRAPH_REPEATS"), NULL, 10)
                         : 0;
  if (repeats > 100)
    return 2;
  double samples[100];
  for (unsigned i = 0; i < repeats; i++) {
    double start = now();
    CHECK(ds4_qwen_image_vae_encode(e, &r, NULL, &encoded));
    samples[i] = now() - start;
    CHECK(ds4_num_free(encoded));
  }
  if (repeats &&
      !save(out, "encode-timings.f64", samples, repeats * sizeof(double)))
    return 1;
  free(fixture);
  CHECK(ds4_qwen_image_vae_release_weights(e));
  float *latent = readall(input, "decode.f32", &bytes);
  if (!latent)
    return 2;
  ds4_num_buffer *z = NULL;
  CHECK(ds4_num_alloc(num, bytes / 4, 0, &z));
  CHECK(ds4_num_write(num, ds4_num_data(z), latent, bytes));
  free(latent);
  ds4_qwen_image_vae_plan plan;
  CHECK(ds4_qwen_image_vae_plan_decode(e, r.height / 16, r.width / 16,
                                       UINT64_MAX, &plan));
  printf("vae plan full weights=%llu device=%llu host=%llu\n",
         (unsigned long long)plan.weights, (unsigned long long)plan.device,
         (unsigned long long)plan.host);
  ds4_qwen_image_vae_plan alternate;
  if (ds4_qwen_image_vae_plan_decode(e, r.height / 16, r.width / 16, 1,
                                     &alternate) != DS4_STATUS_NOMEM)
    return 1;
  if (alternate.device || alternate.weights || alternate.host || alternate.core)
    return 1;
  if (r.width >= 512) {
    CHECK(ds4_qwen_image_vae_plan_decode(e, r.height / 16, r.width / 16,
                                         plan.weights + plan.device - 1,
                                         &alternate));
    if (!alternate.core || alternate.device >= plan.device)
      return 1;
    printf(
        "vae plan bounded core=%u halo=%u overlap=%u device=%llu host=%llu\n",
        alternate.core, alternate.halo, alternate.overlap,
        (unsigned long long)alternate.device,
        (unsigned long long)alternate.host);
  }
  ds4_qwen_image_rgba *image = NULL;
  alternate = plan;
  alternate.core = UINT32_MAX;
  alternate.halo = 14;
  alternate.overlap = 1;
  if (ds4_qwen_image_vae_decode(e, ds4_num_data(z), &alternate, NULL, &image) !=
          DS4_STATUS_INVALID ||
      image)
    return 1;
  alternate = plan;
  alternate.device--;
  if (ds4_qwen_image_vae_decode(e, ds4_num_data(z), &alternate, NULL, &image) !=
          DS4_STATUS_INVALID ||
      image)
    return 1;
  t.cancel_stage = "complete";
  if (ds4_qwen_image_vae_decode(e, ds4_num_data(z), &plan, &obs, &image) !=
          DS4_STATUS_CANCELLED ||
      image)
    return 1;
  t.cancel_stage = NULL;
  t.requested = false;
  CHECK(ds4_qwen_image_vae_decode(e, ds4_num_data(z), &plan, &obs, &image));
  ds4_qwen_image_rgba_view rgba = ds4_qwen_image_rgba_data(image);
  if (!save(out, "pixels.f32", rgba.pixels,
            (size_t)rgba.width * rgba.height * 16))
    return 1;
  CHECK(ds4_qwen_image_vae_release_weights(e));
  char path[4096];
  snprintf(path, sizeof(path), "%s/result.png", out);
  CHECK(ds4_qwen_image_rgba_save_png(image, path));
  if (ds4_qwen_image_rgba_save_png(image, "/dev/full") != DS4_STATUS_INVALID)
    return 1;
  ds4_qwen_image_rgba_free(image);
  for (unsigned i = 0; i < repeats; i++) {
    /* Warmup after the intentional weight release is outside the samples. */
    if (!i) {
      CHECK(ds4_qwen_image_vae_decode(e, ds4_num_data(z), &plan, NULL, &image));
      ds4_qwen_image_rgba_free(image);
    }
    double start = now();
    CHECK(ds4_qwen_image_vae_decode(e, ds4_num_data(z), &plan, NULL, &image));
    samples[i] = now() - start;
    ds4_qwen_image_rgba_free(image);
  }
  if (repeats &&
      !save(out, "decode-timings.f64", samples, repeats * sizeof(double)))
    return 1;
  if (r.width >= 512) {
    CHECK(ds4_qwen_image_vae_plan_tiled(e, r.height / 16, r.width / 16, 8,
                                        &plan));
    char tile_json[256];
    int count =
        snprintf(tile_json, sizeof(tile_json),
                 "{\"core\":%u,\"halo\":%u,\"overlap\":%u,\"scale\":16}\n",
                 plan.core, plan.halo, plan.overlap);
    if (count < 0 || !save(out, "tile-plan.json", tile_json, (size_t)count))
      return 1;
    CHECK(ds4_qwen_image_vae_decode(e, ds4_num_data(z), &plan, NULL, &image));
    rgba = ds4_qwen_image_rgba_data(image);
    if (!save(out, "pixels-tiled.f32", rgba.pixels,
              (size_t)rgba.width * rgba.height * 16))
      return 1;
    ds4_qwen_image_rgba_free(image);
  }
  CHECK(ds4_num_free(z));
  CHECK(ds4_qwen_image_vae_release_weights(e));
  CHECK(ds4_qwen_image_vae_close(e));
  printf("vae PASS %ux%u RGBA host result after weight release\n", r.width,
         r.height);
  return 0;
}
int main(int argc, char **argv) {
  if (argc < 6 || (argc - 4) % 2 ||
      (strcmp(argv[3], "dit") && strcmp(argv[3], "vae")))
    return 2;
  ds4_resources *resources = NULL;
  ds4_numeric *num = NULL;
  ds4_runtime *runtime = NULL;
  ds4_bundle *bundle = NULL;
  CHECK(ds4_resources_create(&resources));
  CHECK(ds4_numeric_cuda(&num, resources, 900));
  CHECK(ds4_runtime_create(&runtime, NULL, NULL));
  CHECK(ds4_bundle_open(&bundle, argv[1], argv[2], resources, 900));
  for (int arg = 4; arg + 1 < argc; arg += 2) {
    mkdir(argv[arg + 1], 0755);
    uint64_t before, peak, after;
    ds4_numeric_memory(num, &before, &peak);
    int result = !strcmp(argv[3], "dit")
                     ? dit(bundle, num, runtime, argv[arg], argv[arg + 1])
                     : vae(bundle, num, runtime, argv[arg], argv[arg + 1]);
    ds4_numeric_memory(num, &after, &peak);
    printf("memory before=%llu after=%llu peak=%llu\n",
           (unsigned long long)before, (unsigned long long)after,
           (unsigned long long)peak);
    if (result)
      return result;
    if (after != before)
      return 1;
  }
  CHECK(ds4_bundle_close(bundle));
  CHECK(ds4_numeric_close(num));
  ds4_runtime_release(runtime);
  CHECK(ds4_resources_close(resources));
  return 0;
}
