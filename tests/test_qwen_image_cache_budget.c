/* Small planner-only fixture: no model, mapping, numeric execution or GPU. */
#include "../ds4_qwen_image_pipeline.c"
#include "ds4_numeric_internal.h"
#include <assert.h>

static ds4_status idle(void *context) { (void)context; return DS4_STATUS_OK; }
static void make_request(ds4_qwen_image_job *job) {
  memset(job, 0, sizeof(*job));
  job->request.groups = job->groups;
  job->request.group_count = COMPONENT_COUNT;
  job->request.variants = job->variants;
  job->request.variant_count = VARIANT_COUNT;
  job->request.victims = job->victims;
  for (uint32_t component = 0; component < COMPONENT_COUNT; component++)
    job->groups[component] = (ds4_execution_group){
        .load = idle, .release = idle,
        .memory.bytes[DS4_MEMORY_DEVICE] = 4000};
  const uint64_t scratch[PHASE_COUNT] = {700, 400, 500, 1000};
  for (size_t variant = 0; variant < VARIANT_COUNT; variant++) {
    job->variants[variant] = (ds4_execution_variant){
        .steps = job->steps[variant], .step_count = PHASE_COUNT,
        .cost = 100000 + variant, .streamed = variant >= RESIDENT_VARIANTS};
    for (uint32_t component = 0; component < COMPONENT_COUNT; component++)
      job->weight_limits[variant][component] = 1000;
    for (uint32_t phase = 0; phase < PHASE_COUNT; phase++)
      job->steps[variant][phase] = (ds4_execution_step){
          .id = phase, .groups = variant < RESIDENT_VARIANTS ? 7 : 0,
          .working.bytes = {[DS4_MEMORY_DEVICE] = 1000 + scratch[phase],
                            [DS4_MEMORY_HOST] = 100}};
  }
}
static void check_capacity(uint64_t host_limit, uint64_t device_peak) {
  ds4_resources *resources = NULL;
  ds4_resource *cache = NULL;
  assert(!ds4_resources_create(&resources));
  assert(!ds4_resources_set_unified_memory(resources, true));
  ds4_resource_budget host = {.limit = host_limit, .margin = 200};
  ds4_resource_budget device = {.limit = 3500, .margin = 100};
  assert(!ds4_resources_set_budget(resources, DS4_MEMORY_HOST, &host));
  assert(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &device));
  ds4_resource_spec spec = {.owner = 1, .location = DS4_MEMORY_HOST, .device = -1,
                            .requested = 1000, .allocated = 1000, .live = 1000};
  assert(!ds4_resource_register(resources, &spec, &cache));
  ds4_qwen_image_job job;
  make_request(&job);
  ds4_execution_victim victim = {.kind = DS4_RECLAIM_CACHE, .cost = 100000,
                                  .release = idle,
                                  .reclaimed.bytes[DS4_MEMORY_HOST] = 1000};
  job.request.victims = &victim;
  job.request.victim_count = 1;
  ds4_execution_plan before, after;
  assert(!ds4_execution_plan_build(resources, &job.request, &before));
  assert(before.variant == RESIDENT_VARIANTS && !before.victims);
  assert(!expand_stream_cache(&job, resources));
  ds4_resource_info cached;
  assert(!ds4_resource_inspect(cache, &cached));
  assert(cached.spec.live == 1000 && cached.spec.allocated == 1000);
  assert(!ds4_execution_plan_build(resources, &job.request, &after));
  assert(after.variant == before.variant && after.victims == before.victims);
  assert(after.peak.bytes[DS4_MEMORY_DEVICE] == device_peak);
  assert(job.weight_limits[after.variant][ENCODER] == device_peak - 400);
  assert(job.weight_limits[after.variant][DIT] == device_peak - 500);
  assert(job.weight_limits[after.variant][VAE] == device_peak - 1000);
  for (size_t variant = RESIDENT_VARIANTS + 1; variant < VARIANT_COUNT; variant++)
    for (uint32_t component = 0; component < COMPONENT_COUNT; component++)
      assert(job.weight_limits[variant][component] == 1000);
  // One more byte needs either extra eviction or a larger explicit cap.
  ds4_execution_request trial = job.request;
  size_t variant = after.variant;
  trial.variants = job.variants + variant;
  trial.variant_count = 1;
  for (uint32_t component = 0; component < COMPONENT_COUNT; component++) {
    uint64_t limit = job.weight_limits[variant][component];
    set_stream_limit(&job, variant, component, limit + 1);
    ds4_status status = ds4_execution_plan_build(resources, &trial, &after);
    assert(status == DS4_STATUS_NOMEM || (!status && after.victims));
    set_stream_limit(&job, variant, component, limit);
  }
  ds4_resource_budget unchanged;
  assert(!ds4_resources_get_budget(resources, DS4_MEMORY_HOST, &unchanged));
  assert(unchanged.limit == host.limit && unchanged.margin == host.margin);
  assert(!ds4_resource_release(cache));
  assert(!ds4_resources_close(resources));
}
static void check_refusal(void) {
  ds4_resources *resources = NULL;
  assert(!ds4_resources_create(&resources));
  ds4_resource_budget device = {.limit = 999};
  assert(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &device));
  ds4_qwen_image_job job;
  make_request(&job);
  assert(!expand_stream_cache(&job, resources));
  ds4_execution_plan plan;
  assert(ds4_execution_plan_build(resources, &job.request, &plan) == DS4_STATUS_NOMEM);
  for (size_t variant = 0; variant < VARIANT_COUNT; variant++)
    for (uint32_t component = 0; component < COMPONENT_COUNT; component++)
      assert(job.weight_limits[variant][component] == 1000);
  assert(!ds4_resources_close(resources));
}
static ds4_status refresh_once_per_plan(void *context) {
  unsigned *samples = context;
  return ++*samples == 2 ? DS4_STATUS_NOT_READY : DS4_STATUS_OK;
}
static void check_fallback_retry(void) {
  ds4_resources *resources = NULL;
  ds4_runtime *runtime = NULL;
  ds4_numeric *numeric = NULL;
  unsigned samples = 0;
  const ds4_numeric_ops ops = {.fence = idle, .refresh_memory = refresh_once_per_plan};
  assert(!ds4_resources_create(&resources));
  assert(!ds4_runtime_create(&runtime, NULL, NULL));
  assert(!ds4_numeric_create(&numeric, &ops, &samples, resources, 701, true));
  ds4_resource_budget device = {.limit = 2300};
  assert(!ds4_resources_set_budget(resources, DS4_MEMORY_DEVICE, &device));
  ds4_resource *cache = NULL;
  ds4_resource_spec spec = {.owner = 701, .location = DS4_MEMORY_DEVICE,
                            .requested = 800, .allocated = 800, .live = 800};
  assert(!ds4_resource_register(resources, &spec, &cache));
  ds4_qwen_image_pipeline pipeline = {.num = numeric, .runtime = runtime};
  ds4_qwen_image_job job;
  make_request(&job);
  job.pipeline = &pipeline;
  job.resident[VAE] = 800;
  assert(!create_execution(&job));
  ds4_execution_plan plan;
  assert(!ds4_qwen_image_job_plan(&job, &plan) && samples == 1);
  assert(plan.victims == 1 && job.request.victim_count == 1);
  assert(ds4_qwen_image_job_plan(&job, &plan) == DS4_STATUS_NOT_READY);
  assert(!ds4_qwen_image_job_plan(&job, &plan) && samples == 3);
  assert(plan.victims == 1 && job.request.victim_count == 1);
  ds4_resource_info cached;
  assert(!ds4_resource_inspect(cache, &cached) && cached.spec.live == 800);
  assert(!ds4_execution_close(job.execution));
  assert(!ds4_resource_release(cache));
  assert(!ds4_numeric_close(numeric));
  ds4_runtime_release(runtime);
  assert(!ds4_resources_close(resources));
}
int main(void) {
  assert(ds4_numeric_refresh_memory(NULL) == DS4_STATUS_INVALID);
  ds4_resources *resources = NULL;
  ds4_numeric *numeric = NULL;
  assert(!ds4_resources_create(&resources));
  assert(!ds4_numeric_cpu(&numeric, resources, 701));
  assert(ds4_numeric_refresh_memory(numeric) == DS4_STATUS_UNSUPPORTED);
  assert(!ds4_numeric_close(numeric));
  assert(!ds4_resources_close(resources));
  check_capacity(4000, 2700); /* Unified physical host budget is limiting. */
  check_capacity(5000, 3400); /* Explicit Metal/device cap is limiting. */
  check_refusal();
  check_fallback_retry();
  puts("Image cache budget: PASS byte limits, unified reserve, victim planning and retry");
}
