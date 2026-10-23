#include "ds4_qwen_image_pipeline.h"
#include "ds4_qwen_image_components.h"
#include "ds4_qwen_image_reference.h"
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
       VARIANT_COUNT = RESIDENT_VARIANTS + STREAM_LEVELS * 2 };
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
  uint32_t references, text, prefix, total;
  uint16_t *conditioning, *reference_latents;
  float *noise;
  uint64_t ref_count, target_count;
  ds4_resource *account, *conditioning_account;
  ds4_num_buffer *latent;
  ds4_qwen_image_rgba *result;
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
  ds4_qwen_image_rgba_free(j->result);
  j->result = NULL;
}
static ds4_status commit(void *opaque) {
  ds4_qwen_image_job *j = opaque;
  if (!j->result)
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
  const ds4_qwen_image_observer *o = &j->observer;
  size_t variant = ds4_execution_get_plan(j->execution)->variant;
  uint32_t active = id == ENCODE_PROMPT ? ENCODER : id == DENOISE ? DIT : VAE;
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
    TRY(ds4_qwen_image_dit_generate(p->dit, input, j->noise, j->report.seed, o,
                                    &j->latent));
  } else if (id == DECODE) {
    TRY(ds4_qwen_image_vae_decode(p->vae, ds4_num_data(j->latent), &j->decode,
                                  o, &j->result));
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
  uint64_t host = sizeof(*j) + j->target_count * 4;
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
  j->noise = malloc(j->target_count * 4);
  if (j->noise) {
    if (r->noise)
      memcpy(j->noise, r->noise, j->target_count * 4);
    else
      TRY(ds4_qwen_image_noise(j->noise, j->target_count, r->seed));
  }
  if (!j->conditioning || (j->ref_count && !j->reference_latents) ||
      !j->noise) {
    status = DS4_STATUS_NOMEM;
    goto cleanup;
  }
  if (r->vae_core)
    TRY(ds4_qwen_image_vae_plan_tiled(p->vae, r->height / 16, r->width / 16,
                                      r->vae_core, &j->decode));
  else
    TRY(ds4_qwen_image_vae_plan_decode(p->vae, r->height / 16, r->width / 16,
                                       UINT64_MAX, &j->decode));
cleanup:
  return status;
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
                &dm, resident + DIT);
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
  uint64_t devices[] = {vae_encode.device_bytes, enc_device,
                        dm.prefix_cache + dm.first_step + persistent,
                        j->decode.device + j->target_count * 4};
  uint64_t hosts[] = {vae_encode.host_bytes, enc.host_bytes, dm.host,
                      j->decode.host};
  uint64_t bits[] = {component_bit(VAE), component_bit(ENCODER),
                     component_bit(DIT), component_bit(VAE)};
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
      live |= bits[i];
      ds4_execution_step *s = &j->steps[variant][k++];
      *s =
          (ds4_execution_step){.id = i,
                               .groups = streamed ? 0 : bits[i],
                               .retain = retain_components(variant) ? live : 0};
      s->working.bytes[DS4_MEMORY_DEVICE] = devices[i];
      s->working.bytes[DS4_MEMORY_HOST] = hosts[i];
      if (streamed) {
        uint32_t active = i == ENCODE_PROMPT ? ENCODER : i == DENOISE ? DIT : VAE;
        s->working.bytes[DS4_MEMORY_DEVICE] += j->weight_limits[variant][active];
      }
      if (host_cache(variant) && i == DENOISE) {
        s->working.bytes[DS4_MEMORY_DEVICE] -= dm.prefix_cache;
        s->working.bytes[DS4_MEMORY_HOST] += dm.prefix_cache;
      }
    }
    j->variants[variant] = (ds4_execution_variant){
        .steps = j->steps[variant],
        .step_count = k,
        .cost = (host_cache(variant) ? dm.prefix_cache * 8 : 0) + variant +
                (streamed ? weights[ENCODER] + weights[VAE] * (j->references + 1) +
                    dm.weights + 7 * (dm.weights - j->weight_limits[variant][DIT] + budgets[DIT].minimum) : 0),
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
cleanup:
  return status;
}
static ds4_status create_execution(ds4_qwen_image_job *j) {
  ds4_execution_ops ops = {.context = j,
                           .execute = stage,
                           .drain = drain,
                           .commit = commit,
                           .abort = abort_result,
                           .cancelled = cancelled};
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
ds4_status ds4_qwen_image_job_create(ds4_qwen_image_job **out,
                                     ds4_qwen_image_pipeline *p,
                                     const ds4_qwen_image_request *r,
                                     const ds4_execution_victim *victims,
                                     size_t count,
                                     const ds4_qwen_image_observer *o) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!p || count > DS4_EXECUTOR_MAX_VICTIMS || (count && !victims))
    return DS4_STATUS_INVALID;
  if (p->job_open)
    return DS4_STATUS_BUSY;
  ds4_status status = ds4_qwen_image_request_validate(r);
  if (status)
    return status;
  double start = now();
  ds4_qwen_image_job *j = calloc(1, sizeof(*j));
  if (!j)
    return DS4_STATUS_NOMEM;
  j->pipeline = p;
  p->job_open = true;
  if (o)
    j->observer = *o;
  j->references = r->reference_count;
  j->target_count = (uint64_t)r->width * r->height / 4;
  TRY(prepare_inputs(j, r));
  TRY(configure_plan(j, victims, count));
  TRY(create_execution(j));
  record_request(j, r, now() - start);
  *out = j;
  return DS4_STATUS_OK;
cleanup:
  ds4_qwen_image_job_close(j);
  return status;
}
ds4_status ds4_qwen_image_job_plan(ds4_qwen_image_job *j,
                                   ds4_execution_plan *out) {
  if (!j || !out)
    return DS4_STATUS_INVALID;
  if (!j->execution)
    return DS4_STATUS_NOT_READY;
  if (ds4_execution_get_state(j->execution) != DS4_EXECUTION_NEW) {
    *out = *ds4_execution_get_plan(j->execution);
    return DS4_STATUS_OK;
  }
  ds4_resources *rs = ds4_numeric_resources(j->pipeline->num, NULL);
  ds4_status status = ds4_execution_plan_build(rs, &j->request, out);
  if (status != DS4_STATUS_NOMEM || j->cache_fallback)
    return status;
  size_t caches = 0;
  for (uint32_t i = 0; i < COMPONENT_COUNT; i++)
    if (j->resident[i])
      caches++;
  if (!caches || j->request.victim_count + caches > DS4_EXECUTOR_MAX_VICTIMS)
    return status;
  /* Existing component caches are optional owners, just like any other
   * rebuildable cache. Only consider reclaiming them after the plan which
   * keeps all of them has failed. No release happens during planning. */
  status = ds4_execution_close(j->execution);
  if (status)
    return status;
  j->execution = NULL;
  j->cache_fallback = true;
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
  status = create_execution(j);
  return status ? status : ds4_execution_plan_build(rs, &j->request, out);
}
ds4_status ds4_qwen_image_job_run(ds4_qwen_image_job *j) {
  if (!j || ds4_execution_get_state(j->execution) != DS4_EXECUTION_NEW)
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
ds4_status ds4_qwen_image_job_take(ds4_qwen_image_job *j,
                                   ds4_qwen_image_rgba **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!j)
    return DS4_STATUS_INVALID;
  if (!j->committed ||
      ds4_execution_get_state(j->execution) != DS4_EXECUTION_COMPLETE ||
      !j->result)
    return DS4_STATUS_NOT_READY;
  *out = j->result;
  j->result = NULL;
  return DS4_STATUS_OK;
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
  ds4_qwen_image_rgba_free(j->result);
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
