#include "ds4_qwen_image_pipeline.h"
#include "ds4_qwen_image_components.h"
#include "ds4_qwen_image_reference.h"
#include "ds4_qwen_image_rgba_internal.h"
#include "ds4_sha256.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
enum { ENCODER, DIT, VAE, COMPONENT_COUNT };
enum { ENCODE_REFERENCES, ENCODE_PROMPT, DENOISE, DECODE, PHASE_COUNT };
enum { RESIDENT_VARIANTS = 4, STREAM_LEVELS = 5,
       VARIANT_COUNT = RESIDENT_VARIANTS + STREAM_LEVELS * 2,
       DIT_CACHE_MISS_COST = 7 };
static const uint32_t phase_component[PHASE_COUNT] = {
    [ENCODE_REFERENCES] = VAE, [ENCODE_PROMPT] = ENCODER,
    [DENOISE] = DIT, [DECODE] = VAE};
static uint64_t component_bit(uint32_t id) { return UINT64_C(1) << id; }
static bool host_cache(size_t variant) {
  return variant < RESIDENT_VARIANTS ? variant >= 2 : variant % 2 != 0;
}
static bool retain_components(size_t variant) {
  return variant < RESIDENT_VARIANTS && variant % 2 == 0;
}
struct ds4_qwen_image_pipeline {
  ds4_qwen_image_encoder *encoder;
  ds4_qwen_image_dit *dit;
  ds4_qwen_image_vae *vae;
  ds4_numeric *num;
  ds4_runtime *runtime;
  bool job_open;
  ds4_resource *account;
  char revision[41];
};
typedef struct {
  ds4_qwen_image_job *job;
  uint32_t id;
} component;
struct ds4_qwen_image_job {
  ds4_qwen_image_pipeline *pipeline;
  ds4_qwen_image_input *input;
  ds4_qwen_image_reference refs[64];
  ds4_qwen_image_latent_shape shapes[65];
  uint32_t images, references, text, prefix, total;
  uint16_t *conditioning, *reference_latents;
  float *noise;
  uint64_t ref_count, target_count;
  ds4_resource *account, *conditioning_account;
  ds4_num_buffer *latent;
  ds4_qwen_image_rgba *results[DS4_QWEN_IMAGE_BATCH_MAX];
  ds4_qwen_image_vae_plan decode;
  ds4_qwen_image_observer observer;
  ds4_qwen_image_report report;
  ds4_execution *execution;
  ds4_execution_request request;
  ds4_execution_group groups[COMPONENT_COUNT];
  component components[COMPONENT_COUNT];
  ds4_execution_step steps[VARIANT_COUNT][PHASE_COUNT];
  ds4_execution_variant variants[VARIANT_COUNT];
  uint64_t weight_limits[VARIANT_COUNT][COMPONENT_COUNT];
  ds4_execution_victim victims[DS4_EXECUTOR_MAX_VICTIMS];
  size_t external_victims;
  uint64_t resident[COMPONENT_COUNT];
  ds4_weight_budget weights_before[COMPONENT_COUNT];
  bool cache_fallback;
  bool committed;
};
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static ds4_status hold(ds4_numeric *n, uint64_t bytes, ds4_resource **out) {
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(n, &owner);
  return ds4_resource_register_host(rs, owner, bytes, out);
}
ds4_status ds4_qwen_image_pipeline_open(ds4_qwen_image_pipeline **out,
                                        ds4_bundle *b, ds4_numeric *n,
                                        ds4_runtime *rt,
                                        const char *tokenizer) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!b || !n || !rt || !tokenizer)
    return DS4_STATUS_INVALID;
  ds4_qwen_image_pipeline *p = calloc(1, sizeof(*p));
  if (!p)
    return DS4_STATUS_NOMEM;
  p->num = n;
  p->runtime = rt;
  snprintf(p->revision, sizeof(p->revision), "%s", ds4_bundle_revision(b));
  ds4_status status = 0;
  TRY(hold(n, sizeof(*p), &p->account));
  TRY(ds4_qwen_image_encoder_open(&p->encoder, b, n, rt, tokenizer));
  TRY(ds4_qwen_image_dit_open(&p->dit, b, n, rt));
  TRY(ds4_qwen_image_vae_open(&p->vae, b, n, rt));
  *out = p;
  p = NULL;
cleanup:
  if (p)
    ds4_qwen_image_pipeline_close(p);
  return status;
}
ds4_status ds4_qwen_image_pipeline_close(ds4_qwen_image_pipeline *p) {
  if (!p)
    return DS4_STATUS_OK;
  if (p->job_open)
    return DS4_STATUS_BUSY;
  ds4_status s = ds4_qwen_image_encoder_close(p->encoder);
  if (s)
    return s;
  p->encoder = NULL;
  s = ds4_qwen_image_dit_close(p->dit);
  if (s)
    return s;
  p->dit = NULL;
  s = ds4_qwen_image_vae_close(p->vae);
  if (s)
    return s;
  p->vae = NULL;
  ds4_resource *r = p->account;
  free(p);
  if (r)
    ds4_resource_release(r);
  return DS4_STATUS_OK;
}

ds4_status ds4_qwen_image_pipeline_release_weights(ds4_qwen_image_pipeline *p) {
  if (!p) return DS4_STATUS_INVALID;
  if (p->job_open) return DS4_STATUS_BUSY;
  ds4_status status = ds4_qwen_image_encoder_release_weights(p->encoder);
  if (!status) status = ds4_qwen_image_dit_release_weights(p->dit);
  if (!status) status = ds4_qwen_image_vae_release_weights(p->vae);
  return status;
}
static ds4_status weight_limit(const ds4_qwen_image_pipeline *p, uint32_t id,
                                uint64_t limit, ds4_weight_budget *out) {
  return id == ENCODER ? ds4_qwen_image_encoder_weights(p->encoder, limit, out)
       : id == DIT ? ds4_qwen_image_dit_weights(p->dit, limit, out)
                   : ds4_qwen_image_vae_weights(p->vae, limit, out);
}
ds4_status ds4_qwen_image_pipeline_residency(const ds4_qwen_image_pipeline *p,
                                            uint64_t resident[3], uint64_t total[3]) {
  if (!p || !resident || !total) return DS4_STATUS_INVALID;
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++) {
    ds4_weight_budget weights;
    ds4_status s = weight_limit(p, i, 0, &weights);
    if (s)
      return s;
    total[i] = weights.total;
    resident[i] = weights.resident;
  }
  return DS4_STATUS_OK;
}
static bool cancelled(void *opaque) {
  ds4_qwen_image_job *j = opaque;
  return j->observer.cancelled && j->observer.cancelled(j->observer.opaque);
}
/* This adapter declares a fixed phase-work estimate, not a time prediction.
 * Models with a different execution plan publish their own metadata. */
static void model_progress(void *opaque, const ds4_operation_progress *progress) {
  ds4_qwen_image_job *j = opaque;
  ds4_operation_progress update = *progress;
  double phases = j->references ? 5 : 4;
  double before = j->references ? 2 : 1;
  update.overall_fraction = (before + (double)progress->completed_units /
                             progress->total_units) / phases;
  update.estimated = true;
  ds4_operation_publish(&j->observer.progress, update);
}
static void pipeline_progress(ds4_qwen_image_job *j, const char *phase,
                               double completed) {
  ds4_operation_publish(&j->observer.progress, (ds4_operation_progress){
      .phase = phase, .overall_fraction = completed / (j->references ? 5 : 4),
      .estimated = true});
}
static void execution_progress(void *opaque, ds4_execution_event event,
                                size_t index, ds4_status status) {
  (void)index;
  ds4_qwen_image_job *j = opaque;
  if (event == DS4_EXEC_RESTORE)
    pipeline_progress(j, "restore", j->references ? 4 : 3);
  if (event == DS4_EXEC_LOAD)
    ds4_operation_publish(&j->observer.progress, (ds4_operation_progress){
        .phase = "loading", .overall_fraction = -1});
  (void)status;
}
static ds4_status drain(void *opaque);
static ds4_status load(void *opaque) {
  component *c = opaque;
  ds4_qwen_image_job *j = c->job;
  ds4_qwen_image_pipeline *p = j->pipeline;
  double start = now();
  ds4_status s = c->id == ENCODER
                     ? qi_encoder_load(p->encoder, j->references != 0)
                 : c->id == DIT ? qi_dit_load(p->dit)
                                : qi_vae_load(p->vae);
  j->report.load_seconds[c->id] += now() - start;
  if (!s)
    j->report.loads[c->id]++;
  return s;
}
static ds4_status release(void *opaque) {
  component *c = opaque;
  ds4_qwen_image_job *j = c->job;
  ds4_qwen_image_pipeline *p = j->pipeline;
  ds4_weight_budget before;
  ds4_status s = weight_limit(p, c->id, 0, &before);
  if (s)
    return s;
  s = c->id == ENCODER
                     ? ds4_qwen_image_encoder_release_weights(p->encoder)
                 : c->id == DIT ? ds4_qwen_image_dit_release_weights(p->dit)
                                : ds4_qwen_image_vae_release_weights(p->vae);
  if (!s && before.resident)
    j->report.releases[c->id]++;
  return s;
}
static ds4_status drain(void *opaque) {
  ds4_qwen_image_job *j = opaque;
  ds4_status s = ds4_numeric_fence(j->pipeline->num);
  if (s || !ds4_execution_get_plan(j->execution) ||
      ds4_execution_get_plan(j->execution)->variant < RESIDENT_VARIANTS) return s;
  /* Streamed groups are phase temporaries rather than executor groups. They
   * must also survive a failed drain/free so recovery can retry their release. */
  for (uint32_t i = 0; i < COMPONENT_COUNT && !s; i++) {
    s = release(j->components + i);
    if (!s) s = weight_limit(j->pipeline, i, UINT64_MAX, NULL);
  }
  return s;
}
static void abort_result(void *opaque) {
  ds4_qwen_image_job *j = opaque;
  for (uint32_t item = 0; item < j->images; item++) {
    ds4_qwen_image_rgba_free(j->results[item]);
    j->results[item] = NULL;
  }
}
static ds4_status commit(void *opaque) {
  ds4_qwen_image_job *j = opaque;
  for (uint32_t item = 0; item < j->images; item++)
    if (!j->results[item])
      return DS4_STATUS_INFERENCE;
  j->committed = true;
  return DS4_STATUS_OK;
}
static ds4_status stage(void *opaque, uint32_t id) {
  ds4_qwen_image_job *j = opaque;
  if (id >= PHASE_COUNT)
    return DS4_STATUS_INVALID;
  ds4_qwen_image_pipeline *p = j->pipeline;
  ds4_numeric *n = p->num;
  ds4_status status = 0;
  double start = now();
  ds4_num_buffer *a = NULL, *b = NULL;
  ds4_qwen_image_conditioning *conditioning = NULL;
  ds4_qwen_image_dit_input *input = NULL;
  ds4_qwen_image_observer model_observer = j->observer;
  model_observer.progress = (ds4_operation_observer){j, model_progress};
  const ds4_qwen_image_observer *o = &model_observer;
  const char *phases[] = {"references", "conditioning", "generation", "decode"};
  pipeline_progress(j, phases[id], id - (j->references ? 0 : 1));
  size_t variant = ds4_execution_get_plan(j->execution)->variant;
  uint32_t active = phase_component[id];
  if (variant >= RESIDENT_VARIANTS)
    TRY(weight_limit(p, active, j->weight_limits[variant][active], NULL));
  if (id == ENCODE_REFERENCES) {
    uint64_t offset = 0;
    for (uint32_t i = 0; i < j->references; i++) {
      TRY(ds4_qwen_image_vae_encode(p->vae, j->refs + i, o, &a));
      ds4_num_view v = ds4_num_data(a);
      TRY(ds4_num_read(n, v, j->reference_latents + offset, v.count * 2));
      offset += v.count;
      TRY(ds4_num_free(a));
      a = NULL;
    }
  } else if (id == ENCODE_PROMPT) {
    TRY(ds4_qwen_image_encode(p->encoder, j->input, o, &conditioning));
    ds4_qwen_image_conditioning_view c =
        ds4_qwen_image_conditioning_data(conditioning);
    if (c.count != j->text) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    TRY(ds4_num_read(n, c.hidden, j->conditioning, c.hidden.count * 2));
  } else if (id == DENOISE) {
    TRY(ds4_num_alloc(n, (uint64_t)j->text * 4096, DS4_NUM_BF16, &a));
    TRY(ds4_num_write(n, ds4_num_data(a), j->conditioning,
                      (uint64_t)j->text * 8192));
    if (j->ref_count) {
      TRY(ds4_num_alloc(n, j->ref_count, DS4_NUM_BF16, &b));
      TRY(ds4_num_write(n, ds4_num_data(b), j->reference_latents,
                        j->ref_count * 2));
    }
    ds4_qwen_image_input_view prepared = ds4_qwen_image_input_data(j->input);
    ds4_qwen_image_conditioning_view c = {
        ds4_num_data(a), prepared.image_mask + prepared.drop, j->text};
    TRY(ds4_qwen_image_dit_prepare(p->dit, c, ds4_num_data(b), j->shapes,
                                   j->references + 1, &input));
    if (host_cache(ds4_execution_get_plan(j->execution)->variant))
      TRY(ds4_qwen_image_dit_cache_place(input, DS4_MEMORY_HOST));
    TRY(ds4_num_free(a));
    a = NULL;
    TRY(ds4_num_free(b));
    b = NULL;
    TRY(qi_dit_generate_batch(p->dit, input, j->noise, j->report.seed, j->images,
                              o, &j->latent));
  } else if (id == DECODE) {
    for (uint32_t item = 0; item < j->images; item++) {
      if (item) model_observer.trace = NULL;
      ds4_num_view latent = ds4_num_slice(ds4_num_data(j->latent),
                                        item * j->target_count, j->target_count);
      TRY(ds4_qwen_image_vae_decode(p->vae, latent, &j->decode,
                                    o, &j->results[item]));
    }
    TRY(ds4_num_free(j->latent));
    j->latent = NULL;
  } else
    status = DS4_STATUS_INVALID;
cleanup:
  {
    ds4_status s = ds4_num_dispose(a), t = ds4_num_dispose(b);
    if (!s)
      s = t;
    t = ds4_qwen_image_conditioning_free(conditioning);
    if (!s)
      s = t;
    t = ds4_qwen_image_dit_input_free(input);
    if (!s)
      s = t;
    if (!status)
      status = s;
    if (variant >= RESIDENT_VARIANTS) {
      s = release(j->components + active);
      if (!s) s = weight_limit(p, active, UINT64_MAX, NULL);
      if (!status) status = s;
    }
    if (status && j->latent) {
      ds4_num_dispose(j->latent);
      j->latent = NULL;
    }
    j->report.phase_seconds[id] = now() - start;
    return status;
  }
}
ds4_status ds4_qwen_image_request_validate(const ds4_qwen_image_request *r) {
  if (!r || !r->prompt || (!r->prompt[0] && !r->reference_count) || r->width < 32 || r->height < 32 ||
      r->width > 2048 || r->height > 2048 || r->width % 32 || r->height % 32 ||
      r->reference_count > 64 || (r->reference_count && !r->references) ||
      r->reference_resolution > 2048 || r->reference_resolution % 32)
    return DS4_STATUS_INVALID;
  if (r->mask || (r->steps && r->steps != 8) ||
      (r->guidance && r->guidance != 1.f) ||
      (r->negative_prompt && r->negative_prompt[0]))
    return DS4_STATUS_UNSUPPORTED;
  uint64_t count = (uint64_t)r->width * r->height / 4;
  if ((r->noise && r->noise_count != count) || (!r->noise && r->noise_count))
    return DS4_STATUS_INVALID;
  if (r->noise)
    for (uint64_t i = 0; i < count; i++)
      if (!isfinite(r->noise[i]))
        return DS4_STATUS_INVALID;
  for (size_t i = 0; i < r->reference_count; i++) {
    const ds4_qwen_image_reference *ref = r->references + i;
    uint32_t w, h;
    ds4_status s = qi_reference_plan(ref, r->reference_resolution, &w, &h);
    if (s)
      return s;
  }
  return DS4_STATUS_OK;
}
static ds4_status copy_references(ds4_qwen_image_job *j,
                                  const ds4_qwen_image_request *r) {
  ds4_qwen_image_pipeline *p = j->pipeline;
  ds4_status status = DS4_STATUS_OK;
  ds4_reservation *resize_lease = NULL;
  ds4_reservation_scope resizing = {0};
  uint64_t host = sizeof(*j) + j->images * j->target_count * sizeof(float);
  for (size_t i = 0; i < r->reference_count; i++) {
    const ds4_qwen_image_reference *ref = r->references + i;
    uint32_t width, height;
    TRY(qi_reference_plan(ref, r->reference_resolution, &width, &height));
    uint64_t pixels = (uint64_t)width * height;
    host += pixels * 4;
    j->ref_count += pixels / 4;
    j->shapes[i] = (ds4_qwen_image_latent_shape){height / 16, width / 16};
  }
  j->shapes[j->references] =
      (ds4_qwen_image_latent_shape){r->height / 16, r->width / 16};
  host += j->ref_count * 2;
  TRY(hold(p->num, host, &j->account));
  uint64_t resize_peak = 0;
  for (size_t i = 0; i < r->reference_count; i++) {
    const ds4_qwen_image_reference *ref = r->references + i;
    uint64_t temporary = qi_reference_workspace(ref, j->shapes[i].width * 16,
                                                j->shapes[i].height * 16);
    if (temporary > resize_peak)
      resize_peak = temporary;
  }
  if (resize_peak) {
    uint64_t owner, bytes[DS4_MEMORY_COUNT] = {0};
    ds4_resources *rs = ds4_numeric_resources(p->num, &owner);
    bytes[DS4_MEMORY_HOST] = resize_peak;
    TRY(ds4_resources_reserve(rs, owner, bytes, &resize_lease));
    resizing = ds4_reservation_enter(resize_lease);
  }
  for (size_t i = 0; i < r->reference_count; i++) {
    uint32_t width = j->shapes[i].width * 16, height = j->shapes[i].height * 16;
    uint8_t *pixels = malloc((uint64_t)width * height * 4);
    if (!pixels) {
      status = DS4_STATUS_NOMEM;
      goto cleanup;
    }
    j->refs[i] = (ds4_qwen_image_reference){pixels, width, height, 4};
    TRY(qi_reference_resize(r->references + i, pixels, width, height));
  }
cleanup:
  ds4_reservation_leave(&resizing);
  if (resize_lease) {
    ds4_status s = ds4_reservation_cancel(resize_lease);
    if (!status)
      status = s;
  }
  return status;
}
static ds4_status plan_decode(ds4_qwen_image_job *j, uint32_t width,
                              uint32_t height, uint32_t core) {
  ds4_qwen_image_vae *vae = j->pipeline->vae;
  return core ? ds4_qwen_image_vae_plan_tiled(vae, height / 16, width / 16,
                                            core, &j->decode)
              : ds4_qwen_image_vae_plan_decode(vae, height / 16, width / 16,
                                             UINT64_MAX, &j->decode);
}
static ds4_status prepare_inputs(ds4_qwen_image_job *j,
                                 const ds4_qwen_image_request *r) {
  ds4_qwen_image_pipeline *p = j->pipeline;
  ds4_status status = DS4_STATUS_OK;
  TRY(copy_references(j, r));
  TRY(ds4_qwen_image_prepare(p->encoder, r->prompt, j->refs, r->reference_count,
                             &j->input));
  ds4_qwen_image_input_view in = ds4_qwen_image_input_data(j->input);
  j->text = in.count - in.drop;
  uint64_t prefix = (uint64_t)j->text + 3ull * in.patches / 4,
           total = prefix + j->target_count / 64;
  if (total > 262144) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  j->prefix = prefix;
  j->total = total;
  TRY(hold(p->num, (uint64_t)j->text * 8192, &j->conditioning_account));
  j->conditioning = malloc((uint64_t)j->text * 8192);
  if (j->ref_count)
    j->reference_latents = malloc(j->ref_count * 2);
  j->noise = malloc(j->images * j->target_count * sizeof(float));
  if (!j->conditioning || (j->ref_count && !j->reference_latents) ||
      !j->noise) {
    status = DS4_STATUS_NOMEM;
    goto cleanup;
  }
  for (uint32_t item = 0; item < j->images; item++) {
    float *noise = j->noise + item * j->target_count;
    if (!item && r->noise)
      memcpy(noise, r->noise, j->target_count * sizeof(float));
    else
      TRY(ds4_qwen_image_noise(noise, j->target_count, r->seed + item));
  }
  status = plan_decode(j, r->width, r->height, r->vae_core);
cleanup:
  return status;
}
static void set_stream_limit(ds4_qwen_image_job *j, size_t variant,
                             uint32_t component, uint64_t bytes) {
  uint64_t before = j->weight_limits[variant][component];
  for (size_t i = 0; i < j->variants[variant].step_count; i++) {
    ds4_execution_step *step = &j->steps[variant][i];
    if (phase_component[step->id] == component)
      step->working.bytes[DS4_MEMORY_DEVICE] =
          step->working.bytes[DS4_MEMORY_DEVICE] - before + bytes;
  }
  if (component == DIT) {
    if (bytes >= before)
      j->variants[variant].cost -= DIT_CACHE_MISS_COST * (bytes - before);
    else
      j->variants[variant].cost += DIT_CACHE_MISS_COST * (before - bytes);
  }
  j->weight_limits[variant][component] = bytes;
}
static ds4_status expand_stream_cache(ds4_qwen_image_job *j, ds4_resources *resources) {
  ds4_execution_plan initial;
  ds4_status status = ds4_execution_plan_build(resources, &j->request, &initial);
  if (status == DS4_STATUS_NOMEM)
    return DS4_STATUS_OK; /* The normal cache-victim fallback still owns admission. */
  if (status || initial.variant < RESIDENT_VARIANTS)
    return status;
  size_t variant = initial.variant;
  ds4_execution_request trial = j->request;
  trial.variants = j->variants + variant;
  trial.variant_count = 1;
  // Refine the admitted plan, preserving its victims and complete restore peak.
  // Planning is read-only: no weight allocation or eviction occurs here.
  for (uint32_t component = 0; component < COMPONENT_COUNT; component++) {
    uint64_t low = j->weight_limits[variant][component];
    uint64_t high = j->groups[component].memory.bytes[DS4_MEMORY_DEVICE];
    while (low < high) {
      uint64_t middle = high - (high - low) / 2;
      set_stream_limit(j, variant, component, middle);
      ds4_execution_plan candidate;
      status = ds4_execution_plan_build(resources, &trial, &candidate);
      if (!status && candidate.victims == initial.victims)
        low = middle;
      else if (!status || status == DS4_STATUS_NOMEM)
        high = middle - 1;
      else {
        set_stream_limit(j, variant, component, low);
        return status;
      }
    }
    set_stream_limit(j, variant, component, low);
  }
  return DS4_STATUS_OK;
}
static ds4_status refine_stream_cache(ds4_qwen_image_job *j) {
  ds4_status status = ds4_numeric_refresh_memory(j->pipeline->num);
  if (status == DS4_STATUS_UNSUPPORTED)
    return DS4_STATUS_OK;
  if (status)
    return status;
  return expand_stream_cache(j, ds4_numeric_resources(j->pipeline->num, NULL));
}
static ds4_status configure_plan(ds4_qwen_image_job *j,
                                 const ds4_execution_victim *victims,
                                 size_t count) {
  ds4_status status = DS4_STATUS_OK;
  ds4_qwen_image_pipeline *p = j->pipeline;
  uint64_t largest_ref = 0;
  ds4_runtime_estimate vae_encode = {0};
  for (uint32_t i = 0; i < j->references; i++) {
    uint64_t pixels = (uint64_t)j->refs[i].width * j->refs[i].height;
    if (pixels > largest_ref) {
      largest_ref = pixels;
      qi_vae_encode_memory(j->refs + i, &vae_encode);
    }
  }
  uint64_t weights[COMPONENT_COUNT], resident[COMPONENT_COUNT];
  qi_encoder_memory(p->encoder, j->references != 0, weights + ENCODER,
                    resident + ENCODER);
  ds4_qwen_image_dit_memory dm;
  uint32_t query_span = j->text;
  if (j->target_count / 64 > query_span)
    query_span = j->target_count / 64;
  if (largest_ref / 256 > query_span)
    query_span = largest_ref / 256;
  qi_dit_memory(p->dit, j->total, j->target_count / 64, j->prefix, query_span,
                j->images, &dm, resident + DIT);
  weights[DIT] = dm.weights;
  qi_vae_memory(p->vae, weights + VAE, resident + VAE);
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++) {
    j->components[i] = (component){j, i};
    j->groups[i] = (ds4_execution_group){.context = j->components + i,
                                         .load = load,
                                         .release = release,
                                         .load_cost = weights[i]};
    j->resident[i] = resident[i];
    j->groups[i].memory.bytes[DS4_MEMORY_DEVICE] = weights[i];
    /* Already-resident adapter weights are retained by their owner. */
    if (resident[i] == weights[i])
      j->request.resident_groups |= component_bit(i);
  }
  ds4_runtime_estimate enc;
  TRY(ds4_qwen_image_encoder_estimate(p->encoder, j->input, &enc));
  uint64_t enc_device =
      enc.device_bytes - (weights[ENCODER] - resident[ENCODER]);
  uint64_t persistent =
      j->target_count * 4 + j->ref_count * 4 + (uint64_t)j->text * 16384;
  uint32_t additional = j->images - 1;
  /* The VAE decodes one slice while retaining prior host RGBA. */
  uint64_t retained_rgba = additional * ds4_qwen_image_rgba_bytes(
      j->decode.latent_width * 16, j->decode.latent_height * 16);
  uint64_t devices[] = {vae_encode.device_bytes, enc_device,
                        dm.prefix_cache + dm.first_step + persistent,
                        j->decode.device + j->images * j->target_count * 4};
  uint64_t hosts[] = {vae_encode.host_bytes, enc.host_bytes, dm.host,
                      j->decode.host + retained_rgba};
  ds4_weight_budget budgets[COMPONENT_COUNT];
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++)
    TRY(weight_limit(p, i, 0, budgets + i));
  for (uint32_t variant = 0; variant < VARIANT_COUNT; variant++) {
    bool streamed = variant >= RESIDENT_VARIANTS;
    if (streamed) {
      uint32_t level = (variant - RESIDENT_VARIANTS) / 2;
      for (uint32_t i = 0; i < COMPONENT_COUNT; i++) {
        uint64_t cap = level == STREAM_LEVELS - 1 ? 0 : weights[i] >> (level + 1);
        j->weight_limits[variant][i] = cap < budgets[i].minimum ? budgets[i].minimum : cap;
      }
    }
    uint64_t live = 0;
    uint32_t k = 0;
    for (uint32_t i = j->references ? ENCODE_REFERENCES : ENCODE_PROMPT;
         i < PHASE_COUNT; i++) {
      uint32_t active = phase_component[i];
      uint64_t bit = component_bit(active);
      live |= bit;
      ds4_execution_step *s = &j->steps[variant][k++];
      *s =
          (ds4_execution_step){.id = i,
                               .groups = streamed ? 0 : bit,
                               .retain = retain_components(variant) ? live : 0};
      s->working.bytes[DS4_MEMORY_DEVICE] = devices[i];
      s->working.bytes[DS4_MEMORY_HOST] = hosts[i];
      if (streamed)
        s->working.bytes[DS4_MEMORY_DEVICE] += j->weight_limits[variant][active];
      if (host_cache(variant) && i == DENOISE) {
        s->working.bytes[DS4_MEMORY_DEVICE] -= dm.prefix_cache;
        s->working.bytes[DS4_MEMORY_HOST] += dm.prefix_cache;
      }
    }
    uint64_t cost = variant;
    if (host_cache(variant))
      cost += dm.prefix_cache * 8;
    if (streamed) {
      uint64_t repeated_dit_bytes =
          dm.weights - j->weight_limits[variant][DIT] + budgets[DIT].minimum;
      cost += weights[ENCODER] + weights[VAE] * (j->references + 1) + dm.weights;
      cost += DIT_CACHE_MISS_COST * repeated_dit_bytes;
    }
    j->variants[variant] = (ds4_execution_variant){
        .steps = j->steps[variant],
        .step_count = k,
        .cost = cost,
        .streamed = streamed,
        .keep = streamed ? 0 : retain_components(variant) ? live : component_bit(VAE)};
    /* Serial execution retains only the final VAE cache. Every intermediate
     * release follows admission of this complete request. */
    j->steps[variant][k - 1].retain |= j->variants[variant].keep;
  }
  if (count)
    memcpy(j->victims, victims, count * sizeof(*victims));
  j->request.groups = j->groups;
  j->request.group_count = COMPONENT_COUNT;
  j->request.variants = j->variants;
  j->request.variant_count = VARIANT_COUNT;
  j->request.victims = j->victims;
  j->request.victim_count = count;
  j->external_victims = count;
cleanup:
  return status;
}
static ds4_status create_execution(ds4_qwen_image_job *j) {
  ds4_execution_ops ops = {.context = j,
                           .execute = stage,
                           .drain = drain,
                           .commit = commit,
                           .abort = abort_result,
                           .cancelled = cancelled,
                           .journal = execution_progress};
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(j->pipeline->num, &owner);
  return ds4_execution_create(&j->execution, j->pipeline->runtime, rs, owner,
                              &j->request, &ops);
}
static void record_request(ds4_qwen_image_job *j,
                           const ds4_qwen_image_request *r,
                           double prepare_seconds) {
  ds4_qwen_image_pipeline *p = j->pipeline;
  j->report =
      (ds4_qwen_image_report){.recipe_version = 1,
                              .rng_version = r->noise ? 0 : 1,
                              .width = r->width,
                              .height = r->height,
                              .references = j->references,
                              .steps = 8,
                              .seed = r->seed,
                              .reference_resolution = r->reference_resolution,
                              .prepare_seconds = prepare_seconds};
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, r->prompt, strlen(r->prompt));
  ds4_sha256_final(&hash, j->report.prompt_sha256);
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, j->noise, j->target_count * 4);
  ds4_sha256_final(&hash, j->report.noise_sha256);
  for (size_t i = 0; i < r->reference_count; i++) {
    ds4_sha256_init(&hash);
    ds4_sha256_update(&hash, j->refs[i].pixels,
                      (uint64_t)j->refs[i].width * j->refs[i].height *
                          j->refs[i].channels);
    ds4_sha256_final(&hash, j->report.reference_sha256[i]);
    j->report.reference_shapes[i] = j->shapes[i];
  }
  memcpy(j->report.model_revision, p->revision, sizeof(p->revision));
  ds4_qwen_image_turbo_sigmas(j->report.sigmas);
}
ds4_status ds4_qwen_image_job_create_batch(ds4_qwen_image_job **out,
                                           ds4_qwen_image_pipeline *p,
                                           const ds4_qwen_image_request *r,
                                           uint32_t images,
                                           const ds4_execution_victim *victims,
                                           size_t count,
                                           const ds4_qwen_image_observer *o) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!p || !images || images > DS4_QWEN_IMAGE_BATCH_MAX ||
      count > DS4_EXECUTOR_MAX_VICTIMS || (count && !victims))
    return DS4_STATUS_INVALID;
  if (p->job_open)
    return DS4_STATUS_BUSY;
  ds4_status status = ds4_qwen_image_request_validate(r);
  if (status)
    return status;
  if (r->seed > UINT64_MAX - (images - 1))
    return DS4_STATUS_INVALID;
  double start = now();
  ds4_qwen_image_job *j = calloc(1, sizeof(*j));
  if (!j)
    return DS4_STATUS_NOMEM;
  j->pipeline = p;
  p->job_open = true;
  if (o)
    j->observer = *o;
  j->images = images;
  j->references = r->reference_count;
  j->target_count = (uint64_t)r->width * r->height / 4;
  TRY(prepare_inputs(j, r));
  TRY(configure_plan(j, victims, count));
  TRY(refine_stream_cache(j));
  TRY(create_execution(j));
  record_request(j, r, now() - start);
  *out = j;
  return DS4_STATUS_OK;
cleanup:
  ds4_qwen_image_job_close(j);
  return status;
}
ds4_status ds4_qwen_image_job_create(ds4_qwen_image_job **out,
                                     ds4_qwen_image_pipeline *p,
                                     const ds4_qwen_image_request *r,
                                     const ds4_execution_victim *victims,
                                     size_t count,
                                     const ds4_qwen_image_observer *o) {
  return ds4_qwen_image_job_create_batch(out, p, r, 1, victims, count, o);
}
static bool add_cache_victims(ds4_qwen_image_job *j) {
  size_t caches = 0;
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++)
    if (j->resident[i])
      caches++;
  if (!caches || j->request.victim_count + caches > DS4_EXECUTOR_MAX_VICTIMS)
    return false;
  /* Already-resident component weights can be rebuilt if the current
   * working set cannot fit alongside them. This only builds descriptors. */
  j->request.resident_groups = 0;
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++)
    if (j->resident[i]) {
      ds4_execution_victim *v = &j->victims[j->request.victim_count++];
      *v = (ds4_execution_victim){.kind = DS4_RECLAIM_CACHE,
                                  .cost = j->resident[i],
                                  .context = j->components + i,
                                  .release = release};
      v->reclaimed.bytes[DS4_MEMORY_DEVICE] = j->resident[i];
    }
  return true;
}
static ds4_status prepare_cache_fallback(ds4_qwen_image_job *j) {
  if (!add_cache_victims(j))
    return DS4_STATUS_NOMEM;
  j->cache_fallback = true;
  ds4_status status = ds4_execution_close(j->execution);
  if (!status)
    j->execution = NULL;
  return status;
}
ds4_status ds4_qwen_image_job_plan(ds4_qwen_image_job *j,
                                   ds4_execution_plan *out) {
  if (!j || !out)
    return DS4_STATUS_INVALID;
  if (!ds4_qwen_image_job_admission_pending(j)) {
    *out = *ds4_execution_get_plan(j->execution);
    return DS4_STATUS_OK;
  }
  ds4_resources *rs = ds4_numeric_resources(j->pipeline->num, NULL);
  ds4_status status = ds4_numeric_refresh_memory(j->pipeline->num);
  if (status && status != DS4_STATUS_UNSUPPORTED)
    return status;
  bool adaptive = status == DS4_STATUS_OK;
  if (j->execution) {
    status = ds4_execution_plan_build(rs, &j->request, out);
    if (status != DS4_STATUS_NOMEM || j->cache_fallback)
      return status;
    status = prepare_cache_fallback(j);
    if (status)
      return status;
  }
  /* Reuse the admission sample. A failed rebuild leaves the fallback
   * descriptors ready for the next plan/run call. */
  if (adaptive) {
    status = expand_stream_cache(j, rs);
    if (status)
      return status;
  }
  status = create_execution(j);
  return status ? status : ds4_execution_plan_build(rs, &j->request, out);
}
bool ds4_qwen_image_job_admission_pending(const ds4_qwen_image_job *j) {
  return j && (!j->execution ||
               ds4_execution_get_state(j->execution) == DS4_EXECUTION_NEW);
}

typedef struct { uint32_t width, height; } ds4_qwen_image_size;

static ds4_qwen_image_size scaled_size(ds4_qwen_image_size original,
                                      uint32_t long_blocks) {
  bool landscape = original.width >= original.height;
  uint32_t longer = landscape ? original.width : original.height;
  uint32_t shorter = landscape ? original.height : original.width;
  uint32_t short_blocks = (shorter * long_blocks + longer / 2) / longer;
  uint32_t long_side = long_blocks * 32;
  uint32_t short_side = (short_blocks ? short_blocks : 1) * 32;
  return landscape ? (ds4_qwen_image_size){long_side, short_side}
                   : (ds4_qwen_image_size){short_side, long_side};
}

static uint32_t admitted_blocks(const ds4_qwen_image_job *j, ds4_resources *rs,
                                 ds4_qwen_image_size original, uint32_t high,
                                 bool allow_victims) {
  uint32_t best = 0;
  /* The failed job still owns the prepared host input. Include it in the
   * budget check and rebuild only the shape-dependent plan descriptors. */
  for (uint32_t low = 1; low < high;) {
    if (j->observer.cancelled && j->observer.cancelled(j->observer.opaque))
      break;
    uint32_t middle = low + (high - low) / 2;
    ds4_qwen_image_size candidate = scaled_size(original, middle);
    ds4_qwen_image_job trial = {.pipeline = j->pipeline, .images = 1, .input = j->input,
                               .references = j->references, .text = j->text,
                               .prefix = j->prefix, .ref_count = j->ref_count,
                               .target_count = (uint64_t)candidate.width * candidate.height / 4};
    memcpy(trial.refs, j->refs, j->references * sizeof(*j->refs));
    trial.total = trial.prefix + trial.target_count / 64;
    ds4_status status = plan_decode(&trial, candidate.width, candidate.height, 0);
    if (!status)
      status = configure_plan(&trial, j->victims, allow_victims ? j->external_victims : 0);
    if (status)
      break;
    ds4_execution_plan plan;
    status = ds4_execution_plan_build(rs, &trial.request, &plan);
    if (status == DS4_STATUS_NOMEM && allow_victims && add_cache_victims(&trial))
      status = ds4_execution_plan_build(rs, &trial.request, &plan);
    if (!status) {
      best = middle;
      low = middle + 1;
    } else if (status == DS4_STATUS_NOMEM) {
      high = middle;
    } else {
      break;
    }
  }
  return best;
}

ds4_qwen_image_admission ds4_qwen_image_job_admission_hint(const ds4_qwen_image_job *j) {
  ds4_qwen_image_admission hint = {0};
  if (!j || j->images != 1 || j->decode.core)
    return hint;
  if (!ds4_qwen_image_job_admission_pending(j)) {
    ds4_execution_recovery recovery;
    if (ds4_execution_recovery_query(j->execution, &recovery) ||
        recovery.state != DS4_EXECUTION_FAILED || recovery.operation_status != DS4_STATUS_NOMEM)
      return hint;
  }
  ds4_numeric *num = j->pipeline->num;
  ds4_status status = ds4_numeric_refresh_memory(num);
  if (status && status != DS4_STATUS_UNSUPPORTED)
    return hint;
  ds4_resources *rs = ds4_numeric_resources(num, NULL);
  ds4_qwen_image_size original = {j->decode.latent_width * 16,
                                   j->decode.latent_height * 16};
  uint32_t longer = original.width >= original.height ? original.width : original.height;
  uint32_t limit_blocks = admitted_blocks(j, rs, original, longer / 32, true);
  if (limit_blocks) {
    uint32_t without_victims = admitted_blocks(j, rs, original, limit_blocks + 1, false);
    uint32_t retry_limit = without_victims ? without_victims : limit_blocks;
    /* Admission uses a memory snapshot. Leave headroom for the real retry. */
    uint32_t retry_blocks = retry_limit * 7 / 8;
    ds4_qwen_image_size limit = scaled_size(original, limit_blocks);
    ds4_qwen_image_size retry = scaled_size(original, retry_blocks ? retry_blocks : 1);
    hint = (ds4_qwen_image_admission){limit.width * limit.height, retry.width, retry.height};
  }
  return hint;
}
ds4_status ds4_qwen_image_job_run(ds4_qwen_image_job *j) {
  if (!ds4_qwen_image_job_admission_pending(j))
    return DS4_STATUS_INVALID;
  ds4_execution_plan plan;
  ds4_status planned = ds4_qwen_image_job_plan(j, &plan);
  if (planned)
    return planned;
  double start = now();
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++)
    weight_limit(j->pipeline, i, 0, j->weights_before + i);
  ds4_status s = ds4_execution_run(j->execution);
  j->report.total_seconds = now() - start + j->report.prepare_seconds;
  j->report.plan = *ds4_execution_get_plan(j->execution);
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++) {
    ds4_weight_budget *w = j->report.weights + i;
    weight_limit(j->pipeline, i, 0, w);
    w->loads -= j->weights_before[i].loads;
    w->hits -= j->weights_before[i].hits;
    w->evictions -= j->weights_before[i].evictions;
    w->transferred -= j->weights_before[i].transferred;
    if (j->report.plan.variant >= RESIDENT_VARIANTS)
      w->limit = j->weight_limits[j->report.plan.variant][i];
  }
  return s;
}
ds4_status ds4_qwen_image_job_recover(ds4_qwen_image_job *j) {
  return j ? ds4_execution_recover(j->execution) : DS4_STATUS_INVALID;
}
ds4_status ds4_qwen_image_job_recovery(ds4_qwen_image_job *j,
                                       ds4_execution_recovery *r) {
  return j ? ds4_execution_recovery_query(j->execution, r) : DS4_STATUS_INVALID;
}
ds4_status ds4_qwen_image_job_take_at(ds4_qwen_image_job *j, uint32_t index,
                                      ds4_qwen_image_rgba **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!j || index >= j->images)
    return DS4_STATUS_INVALID;
  if (!j->committed ||
      ds4_execution_get_state(j->execution) != DS4_EXECUTION_COMPLETE ||
      !j->results[index])
    return DS4_STATUS_NOT_READY;
  *out = j->results[index];
  j->results[index] = NULL;
  return DS4_STATUS_OK;
}
ds4_status ds4_qwen_image_job_take(ds4_qwen_image_job *j,
                                   ds4_qwen_image_rgba **out) {
  return ds4_qwen_image_job_take_at(j, 0, out);
}
const ds4_qwen_image_report *
ds4_qwen_image_job_report(const ds4_qwen_image_job *j) {
  return j ? &j->report : NULL;
}
ds4_status ds4_qwen_image_job_close(ds4_qwen_image_job *j) {
  if (!j)
    return DS4_STATUS_OK;
  ds4_status s = ds4_execution_close(j->execution);
  if (s)
    return s;
  j->execution = NULL;
  s = ds4_num_dispose(j->latent);
  j->latent = NULL;
  if (s)
    return s;
  abort_result(j);
  ds4_qwen_image_input_free(j->input);
  for (uint32_t i = 0; i < j->references; i++)
    free((void *)j->refs[i].pixels);
  free(j->conditioning);
  free(j->reference_latents);
  free(j->noise);
  if (j->conditioning_account)
    ds4_resource_release(j->conditioning_account);
  j->pipeline->job_open = false;
  ds4_resource *r = j->account;
  free(j);
  if (r)
    ds4_resource_release(r);
  return DS4_STATUS_OK;
}
