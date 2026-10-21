#ifndef DS4_QWEN_IMAGE_PIPELINE_H
#define DS4_QWEN_IMAGE_PIPELINE_H
#include "ds4_executor.h"
#include "ds4_qwen_image_dit.h"
#include "ds4_qwen_image_vae.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen_image_pipeline ds4_qwen_image_pipeline;
typedef struct ds4_qwen_image_job ds4_qwen_image_job;
typedef struct {
  const char *prompt;
  uint32_t width, height;
  uint64_t seed;
  const ds4_qwen_image_reference *references;
  size_t reference_count;
  /* Optional exact FP32 packed noise, copied during prepare. NULL selects
   * version-1 native RNG. Count must equal width/16*height/16*64. */
  const float *noise;
  uint64_t noise_count;
  uint32_t steps;
  float guidance; /* 0 defaults to Turbo's 8 steps / CFG 1. */
  const char *negative_prompt;
  const void *mask;  /* Unsupported parameters are rejected, never ignored. */
  uint32_t vae_core; /* 0: full decode; otherwise explicit exact tiled core. */
  uint32_t reference_resolution; /* 0: supplied aligned size; otherwise the
                                  * official square-area Lanczos recipe. */
} ds4_qwen_image_request;
typedef struct {
  uint32_t recipe_version, rng_version, width, height, references, steps;
  uint64_t seed;
  uint8_t prompt_sha256[32], noise_sha256[32], reference_sha256[64][32];
  ds4_qwen_image_latent_shape reference_shapes[64];
  char model_revision[41];
  uint32_t reference_resolution;
  float sigmas[9];
  double prepare_seconds, load_seconds[3], phase_seconds[4], total_seconds;
  uint32_t loads[3], releases[3];
  ds4_execution_plan plan;
} ds4_qwen_image_report;
/* Serial, explicit-size recipe. reference_resolution declares any resize;
 * zero preserves supplied aligned dimensions. Reference order is unchanged.
 * Bundle, numeric context and runtime outlive the pipeline. Only one job may
 * be open per pipeline; close returns BUSY until that job has been closed. */
ds4_status ds4_qwen_image_pipeline_open(ds4_qwen_image_pipeline **,
                                        ds4_bundle *, ds4_numeric *,
                                        ds4_runtime *, const char *tokenizer);
ds4_status ds4_qwen_image_pipeline_close(ds4_qwen_image_pipeline *);
/* Prepare copies inputs and builds resident/serial component plans with
 * device/host prefix cache entirely on host. Victims are explicit lifecycle
 * descriptors (e.g. LLM sessions/weights); their contexts must outlive job
 * close/recovery. */
ds4_status ds4_qwen_image_job_create(ds4_qwen_image_job **,
                                     ds4_qwen_image_pipeline *,
                                     const ds4_qwen_image_request *,
                                     const ds4_execution_victim *,
                                     size_t victims,
                                     const ds4_qwen_image_observer *);
/* Before run, compute an admission preview without releasing resources.
 * Once execution starts, return its recorded plan; never replan a past job. */
ds4_status ds4_qwen_image_job_plan(ds4_qwen_image_job *, ds4_execution_plan *);
ds4_status ds4_qwen_image_job_run(ds4_qwen_image_job *);
ds4_status ds4_qwen_image_job_recover(ds4_qwen_image_job *);
ds4_status ds4_qwen_image_job_recovery(ds4_qwen_image_job *,
                                       ds4_execution_recovery *);
/* Publication succeeds only after the executor has restored every victim.
 * take transfers host RGBA ownership; report remains usable after take.
 * The resource registry outlives the returned RGBA, even after pipeline close.
 */
ds4_status ds4_qwen_image_job_take(ds4_qwen_image_job *,
                                   ds4_qwen_image_rgba **);
const ds4_qwen_image_report *
ds4_qwen_image_job_report(const ds4_qwen_image_job *);
ds4_status ds4_qwen_image_job_close(ds4_qwen_image_job *);
#ifdef __cplusplus
}
#endif
#endif
