/* Fixture-only CUDA qualification. All inference uses native suite APIs. */
#include "ds4_gemma_multimodal.h"
#include "ds4_gemma_vision.h"
#include "ds4_gemma_visual.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
extern int cudaMemGetInfo(size_t *, size_t *);
#define CHECK(x)                                                               \
  do {                                                                         \
    ds4_status s = (x);                                                        \
    if (s) {                                                                   \
      fprintf(stderr, "%s: %u\n", #x, s);                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)
static bool read_bytes(FILE *f, void *data, size_t n) {
  return fread(data, 1, n, f) == n;
}
static bool read32(FILE *f, uint32_t *v) {
  uint8_t b[4];
  if (!read_bytes(f, b, 4))
    return false;
  *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 |
       (uint32_t)b[3] << 24;
  return true;
}
static bool save(const char *dir, const char *name, const void *data,
                 size_t bytes) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f)
    return false;
  bool ok = fwrite(data, 1, bytes, f) == bytes;
  return !fclose(f) && ok;
}
static double seconds(void) {
  struct timespec time;
  assert(!clock_gettime(CLOCK_MONOTONIC, &time));
  return time.tv_sec + time.tv_nsec * 1e-9;
}
typedef struct {
  ds4_numeric *num;
  const char *out;
  uint32_t frame, next;
  uint32_t clip, next_clip;
  bool cancel;
  ds4_gemma_visual *adapter;
  ds4_gemma_visual_input *input;
  ds4_gemma_residency resident;
} trace;
static ds4_status residency(trace *t, const char *stage) {
  ds4_gemma_residency now;
  ds4_status s = ds4_gemma_visual_resident_weights(t->adapter, &now);
  if (s || (!memcmp(&now, &t->resident, sizeof(now)) &&
            strcmp(stage, "prepared") && strcmp(stage, "complete")))
    return s;
  t->resident = now;
  char path[4096];
  snprintf(path, sizeof(path), "%s/residency.jsonl", t->out);
  FILE *f = fopen(path, "a");
  if (!f)
    return DS4_STATUS_INVALID;
  fprintf(f,
          "{\"stage\":\"%s\",\"text\":%llu,\"vision\":%llu,\"audio\":%llu}\n",
          stage, (unsigned long long)now.text_bytes,
          (unsigned long long)now.vision_bytes,
          (unsigned long long)now.audio_bytes);
  return fclose(f) ? DS4_STATUS_INVALID : DS4_STATUS_OK;
}
static bool cancelled(void *p) { return ((trace *)p)->cancel; }
static ds4_status observe(void *opaque, const char *stage, uint32_t item,
                          uint32_t layer, ds4_num_view v) {
  trace *t = opaque;
  ds4_status rs = residency(t, stage);
  if (rs)
    return rs;
  if (!strcmp(stage, "audio-sscp-conv") && !layer)
    t->clip = t->next_clip++;
  bool skip_trace =
      getenv("DS4_GEMMA_AUDIO_TRACE_ONLY") &&
      (strncmp(stage, "audio-", 6) ||
       (strcmp(getenv("DS4_GEMMA_AUDIO_TRACE_ONLY"), "2") &&
        strcmp(stage, "audio-layer") && strcmp(stage, "audio-input") &&
        strcmp(stage, "audio-output") && strcmp(stage, "audio-projected") &&
        strcmp(stage, "audio-subsample") &&
        strcmp(stage, "audio-probabilities")));
  if (!strcmp(stage, "vision-patches")) {
    if (!item)
      t->frame = t->next;
    t->next++;
  }
  if (t->adapter &&
      ds4_gemma_visual_release_vision(t->adapter) != DS4_STATUS_BUSY)
    return DS4_STATUS_INVALID;
  if (t->input && ds4_gemma_visual_input_free(t->input) != DS4_STATUS_BUSY)
    return DS4_STATUS_INVALID;
  const char *cancel = getenv("DS4_GEMMA_VISUAL_CANCEL");
  if (cancel && !strcmp(cancel, stage))
    t->cancel = true;
  if (!v.data || skip_trace)
    return DS4_STATUS_OK;
  size_t bytes = v.count * (v.type == DS4_NUM_BF16 ? 2 : 4);
  void *data = malloc(bytes);
  if (!data)
    return DS4_STATUS_NOMEM;
  ds4_status s = ds4_num_read(t->num, v, data, bytes);
  char name[256];
  bool vision = !strncmp(stage, "vision-", 7);
  bool audio = !strncmp(stage, "audio-", 6);
  snprintf(name, sizeof(name), "%s-%u-%s-%u.%s",
           vision  ? "frame"
           : audio ? "clip"
                   : "item",
           vision  ? t->frame + item
           : audio ? t->clip
                   : item,
           stage, layer, v.type == DS4_NUM_BF16 ? "bf16" : "f32");
  if (!s && !save(t->out, name, data, bytes))
    s = DS4_STATUS_INVALID;
  free(data);
  return s;
}
/* Exercise the typed public entry point with the exact same fixture text.
 * Test serialization contains markers; product callers supply typed parts. */
static ds4_status prepare_typed(ds4_gemma_visual *e,
                                const ds4_gemma_visual_request *r, size_t count,
                                ds4_gemma_visual_input **out) {
  ds4_gemma_multimodal_request *requests = calloc(count, sizeof(*requests));
  ds4_gemma_content **parts = calloc(count, sizeof(*parts));
  char **texts = calloc(count, sizeof(*texts));
  assert(requests && parts && texts);
  for (size_t i = 0; i < count; i++) {
    parts[i] = calloc(2 * r[i].media_count + 3, sizeof(**parts));
    assert(parts[i]);
    requests[i].content = parts[i];
    requests[i].task = r[i].task;
    requests[i].add_timestamps = r[i].add_timestamps;
    if (!r[i].prompt) {
      for (size_t j = 0; j < r[i].media_count; j++) {
        if (j)
          parts[i][requests[i].count++] = (ds4_gemma_content){.text = " "};
        parts[i][requests[i].count++] =
            (ds4_gemma_content){.media = r[i].media[j]};
      }
    } else {
      texts[i] = strdup(r[i].prompt);
      assert(texts[i]);
      char *text = texts[i];
      size_t m = 0;
      while (*text) {
        char *next = NULL;
        size_t len = 0;
        const char *markers[] = {"<|image|>", "<|video|>", "<|audio|>"};
        for (unsigned k = 0; k < 3; k++) {
          char *p = strstr(text, markers[k]);
          if (p && (!next || p < next)) {
            next = p;
            len = strlen(markers[k]);
          }
        }
        if (!next) {
          parts[i][requests[i].count++] = (ds4_gemma_content){.text = text};
          break;
        }
        if (next > text) {
          *next = 0;
          parts[i][requests[i].count++] = (ds4_gemma_content){.text = text};
        }
        assert(m < r[i].media_count);
        parts[i][requests[i].count++] =
            (ds4_gemma_content){.media = r[i].media[m++]};
        text = next + len;
      }
      if (!requests[i].count)
        parts[i][requests[i].count++] = (ds4_gemma_content){.text = ""};
      assert(m == r[i].media_count);
    }
  }
  ds4_status s = ds4_gemma_multimodal_prepare(e, requests, count, out);
  if (!s)
    for (size_t i = 0; i < count; i++) {
      uint8_t key[32], again[32], other[32];
      CHECK(ds4_gemma_multimodal_key(*out, i, 768, key));
      CHECK(ds4_gemma_multimodal_key(*out, i, 768, again));
      CHECK(ds4_gemma_multimodal_key(*out, i, 128, other));
      assert(!memcmp(key, again, 32) && memcmp(key, other, 32));
      ds4_gemma_segment segment;
      CHECK(ds4_gemma_multimodal_segment(*out, i, &segment));
      assert(!segment.start_seconds && !segment.end_seconds);
    }
  for (size_t i = 0; i < count; i++) {
    free(parts[i]);
    free(texts[i]);
  }
  free(requests);
  free(parts);
  free(texts);
  return s;
}
int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  ds4_resources *rs = NULL;
  ds4_runtime *runtime = NULL;
  ds4_numeric *num = NULL;
  ds4_bundle *bundle = NULL;
  ds4_gemma_processor *processor = NULL;
  ds4_gemma_visual *adapter = NULL;
  CHECK(ds4_resources_create(&rs));
  CHECK(ds4_runtime_create(&runtime, NULL, NULL));
  CHECK(ds4_numeric_cuda(&num, rs, 301));
  uint64_t baseline, peak;
  ds4_numeric_memory(num, &baseline, &peak);
  size_t free_device, total;
  assert(!cudaMemGetInfo(&free_device, &total));
  ds4_resource_budget budget = {.limit = baseline + free_device,
                                .margin = 256u << 20};
  const char *limit = getenv("DS4_GEMMA_VISUAL_DEVICE_LIMIT");
  if (limit) {
    budget.limit = strtoull(limit, NULL, 10);
    budget.margin = 0;
  }
  CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
  CHECK(ds4_bundle_open(&bundle, argv[1], argv[2], rs, 302));
  CHECK(ds4_gemma_processor_open(&processor, bundle, rs, 303, argv[4],
                                 64ull << 20));
  CHECK(ds4_gemma_visual_open(&adapter, bundle, num, runtime, argv[3],
                              processor));
  FILE *fixture = fopen(argv[5], "rb");
  assert(fixture);
  uint8_t magic[8];
  uint32_t count;
  assert(read_bytes(fixture, magic, 8) && !memcmp(magic, "DS4VIS1\0", 8) &&
         read32(fixture, &count) && count && count <= 1024);
  ds4_gemma_visual_request *requests = calloc(count, sizeof(*requests));
  assert(requests);
  ds4_media_limits limits = ds4_media_default_limits();
  mkdir(argv[6], 0755);
  for (uint32_t i = 0; i < count; i++) {
    uint32_t prompt_size, task, nmedia, timestamps;
    assert(read32(fixture, &prompt_size) && read32(fixture, &task) &&
           read32(fixture, &nmedia) && read32(fixture, &timestamps));
    assert(prompt_size < 1048576 && nmedia <= 64 && timestamps <= 1);
    char *prompt = calloc(prompt_size + 1, 1);
    assert(prompt && read_bytes(fixture, prompt, prompt_size));
    ds4_gemma_media **media = calloc(nmedia, sizeof(*media));
    assert(media || !nmedia);
    requests[i] = (ds4_gemma_visual_request){prompt_size ? prompt : NULL, task,
                                             media, nmedia, timestamps};
    if (!prompt_size)
      free(prompt);
    for (uint32_t j = 0; j < nmedia; j++) {
      uint32_t type, frames, w, h, soft;
      assert(read32(fixture, &type) && read32(fixture, &frames) &&
             read32(fixture, &w) && read32(fixture, &h) &&
             read32(fixture, &soft));
      if (type == 4) {
        assert(frames && frames < 64 * 1024 * 1024 && !w && !h);
        void *encoded = malloc(frames);
        assert(encoded && read_bytes(fixture, encoded, frames));
        ds4_media *decoded = NULL;
        ds4_media_decode_options options = {
            .limits = limits, .video_audio = DS4_MEDIA_AUDIO_FIRST};
        CHECK(ds4_media_decode(&decoded, rs, 304, DS4_MEDIA_VIDEO, encoded,
                               frames, &options));
        ds4_gemma_media_recipe recipe = {soft};
        CHECK(ds4_gemma_media_prepare(processor, decoded, &recipe, media + j));
        ds4_media_free(decoded);
        free(encoded);
        continue;
      }
      if (type == 3) {
        assert(frames && frames <= 480000 && w == 16000 && h == 1 && !soft);
        float *wave = malloc((size_t)frames * 4);
        assert(wave && read_bytes(fixture, wave, (size_t)frames * 4));
        ds4_media *decoded = NULL;
        CHECK(ds4_media_audio_samples(&decoded, rs, 304, wave, frames, 1, w,
                                      &limits));
        CHECK(ds4_gemma_media_prepare(processor, decoded, NULL, media + j));
        ds4_gemma_media_view v = ds4_gemma_media_data(media[j]);
        char name[128];
        snprintf(name, sizeof(name), "item-%u-media-%u-features.f32", i, j);
        assert(save(argv[6], name, v.audio_features,
                    (uint64_t)v.audio_frames * 128 * 4));
        snprintf(name, sizeof(name), "item-%u-media-%u-mask.u8", i, j);
        assert(save(argv[6], name, v.audio_mask, v.audio_frames));
        ds4_media_free(decoded);
        free(wave);
        continue;
      }
      assert((type == 1 || type == 2) && frames && frames <= 32 && w && h &&
             (uint64_t)w * h <= limits.pixels_per_frame);
      size_t bytes = (size_t)w * h * 3;
      uint8_t *pixels = malloc(bytes * frames);
      assert(pixels);
      ds4_media_frame_input input[32];
      for (uint32_t k = 0; k < frames; k++) {
        double time;
        assert(read_bytes(fixture, &time, 8) &&
               read_bytes(fixture, pixels + bytes * k, bytes));
        input[k] = (ds4_media_frame_input){pixels + bytes * k, w, h, 3, time};
      }
      ds4_media *decoded = NULL;
      if (type == 1) {
        assert(frames == 1);
        CHECK(ds4_media_image_pixels(&decoded, rs, 304, input, &limits));
      } else {
        ds4_media_video_recipe recipe = {0, 0, DS4_MEDIA_UNIFORM};
        CHECK(ds4_media_video_frames(&decoded, rs, 304, input, frames, 0,
                                     frames, &recipe, &limits));
      }
      ds4_gemma_media_recipe recipe = {soft};
      CHECK(ds4_gemma_media_prepare(processor, decoded, &recipe, media + j));
      ds4_gemma_media_view v = ds4_gemma_media_data(media[j]);
      char name[128];
      snprintf(name, sizeof(name), "item-%u-media-%u-pixels.f32", i, j);
      assert(save(argv[6], name, v.patches,
                  (uint64_t)v.frames * v.max_patches * 768 * 4));
      snprintf(name, sizeof(name), "item-%u-media-%u-positions.i32", i, j);
      assert(save(argv[6], name, v.positions,
                  (uint64_t)v.frames * v.max_patches * 2 * 4));
      ds4_media_free(decoded);
      free(pixels);
    }
  }
  assert(fgetc(fixture) == EOF && !fclose(fixture));
  ds4_gemma_visual_input *input = NULL;
  CHECK(prepare_typed(adapter, requests, count, &input));
  for (uint32_t i = 0; i < count; i++) {
    const uint32_t *ids;
    uint32_t n;
    CHECK(ds4_gemma_visual_input_tokens(input, i, &ids, &n));
    char name[128];
    snprintf(name, sizeof(name), "item-%u-tokens.u32", i);
    assert(save(argv[6], name, ids, n * 4));
    for (size_t j = 0; j < requests[i].media_count; j++)
      ds4_gemma_media_free(requests[i].media[j]);
    free((void *)requests[i].media);
    free((void *)requests[i].prompt);
  }
  free(requests);
  ds4_gemma_processor_close(processor);
  processor =
      NULL; /* Input owns preprocessing; encoder copied its RoPE data. */
  assert(ds4_gemma_visual_close(adapter) == DS4_STATUS_BUSY);
  trace t = {.num = num,
             .out = argv[6],
             .frame = UINT32_MAX,
             .adapter = adapter,
             .input = input};
  ds4_gemma_observer observer = {&t, cancelled, observe};
  CHECK(residency(&t, "prepared"));
  assert(!t.resident.text_bytes && !t.resident.vision_bytes &&
         !t.resident.audio_bytes);
  ds4_gemma_result *result = NULL;
  ds4_gemma_plan plan;
  CHECK(ds4_gemma_visual_plan(adapter, input, &plan));
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/plan.json", argv[6]);
    FILE *f = fopen(path, "w");
    assert(f);
    fprintf(f,
            "{\"device_bytes\":%llu,\"host_bytes\":%llu,\"reclaim_device_"
            "bytes\":%llu,\"modalities\":%u,\"release_encoders\":%s}\n",
            (unsigned long long)plan.peak.device_bytes,
            (unsigned long long)plan.peak.host_bytes,
            (unsigned long long)plan.reclaim_device_bytes, plan.modalities,
            plan.release_encoders ? "true" : "false");
    assert(!fclose(f));
  }
  uint32_t dimensions = 768;
  const char *dim = getenv("DS4_GEMMA_VISUAL_DIMENSIONS");
  if (dim)
    dimensions = strtoul(dim, NULL, 10);
  double cold_start = seconds();
  ds4_status status = ds4_gemma_visual_embed(
      adapter, input, dimensions,
      getenv("DS4_GEMMA_VISUAL_NO_TRACE") ? NULL : &observer, &result);
  double cold_seconds = seconds() - cold_start;
  CHECK(residency(&t, "complete"));
  if (getenv("DS4_GEMMA_VISUAL_CANCEL")) {
    assert(status == DS4_STATUS_CANCELLED && !result);
  } else {
    if (status) {
      fprintf(stderr, "embed status=%u\n", status);
      return 1;
    }
    ds4_gemma_result_view view = ds4_gemma_result_data(result);
    assert(view.count == count && view.dimensions == dimensions &&
           save(argv[6], "vectors.f32", view.vectors,
                (uint64_t)view.count * view.dimensions * 4));
    size_t vector_bytes = (uint64_t)view.count * view.dimensions * 4;
    void *original = malloc(vector_bytes);
    assert(original);
    memcpy(original, view.vectors, vector_bytes);
    ds4_gemma_result_free(result);
    result = NULL;
    if (getenv("DS4_GEMMA_VISUAL_BENCH")) {
      char path[4096];
      snprintf(path, sizeof(path), "%s/timing.json", argv[6]);
      FILE *timing = fopen(path, "w");
      assert(timing);
      fprintf(timing, "{\"cold_seconds\":%.9f,\"samples_seconds\":[",
              cold_seconds);
      for (unsigned repeat = 0; repeat < 5; repeat++) {
        double start = seconds();
        CHECK(
            ds4_gemma_visual_embed(adapter, input, dimensions, NULL, &result));
        fprintf(timing, "%s%.9f", repeat ? "," : "", seconds() - start);
        assert(!memcmp(original, ds4_gemma_result_data(result).vectors,
                       vector_bytes));
        ds4_gemma_result_free(result);
        result = NULL;
      }
      fprintf(timing, "]}\n");
      assert(!fclose(timing));
    }
    if (getenv("DS4_GEMMA_VISUAL_FAULTS")) {
      assert(ds4_gemma_visual_embed(adapter, input, 64, NULL, &result) ==
                 DS4_STATUS_INVALID &&
             !result);
      ds4_resource_budget denied = budget;
      denied.limit = 1;
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &denied));
      assert(ds4_gemma_visual_embed(adapter, input, 768, NULL, &result) ==
                 DS4_STATUS_NOMEM &&
             !result);
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
      CHECK(ds4_gemma_visual_release_vision(adapter));
      CHECK(ds4_gemma_visual_release_text(adapter));
      CHECK(ds4_gemma_visual_release_audio(adapter));
      CHECK(ds4_gemma_visual_embed(adapter, input, dimensions, NULL, &result));
      assert(!memcmp(original, ds4_gemma_result_data(result).vectors,
                     vector_bytes));
      /* A text-only request must reclaim idle towers from an earlier mixed
       * request when its budget shrinks; co-resident execution is the oracle.
       */
      ds4_gemma_content text_part = {.text = "solo testo"};
      ds4_gemma_multimodal_request text_request = {.content = &text_part,
                                                   .count = 1};
      ds4_gemma_multimodal_input *text_input = NULL;
      ds4_gemma_result *text_result = NULL;
      CHECK(
          ds4_gemma_multimodal_prepare(adapter, &text_request, 1, &text_input));
      CHECK(ds4_gemma_multimodal_embed(adapter, text_input, 768, NULL,
                                       &text_result));
      float text_vector[768];
      memcpy(text_vector, ds4_gemma_result_data(text_result).vectors,
             sizeof(text_vector));
      ds4_gemma_result_free(text_result);
      text_result = NULL;
      ds4_resource_budget small = {.limit = 500000000};
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &small));
      ds4_gemma_plan text_plan;
      CHECK(ds4_gemma_multimodal_plan(adapter, text_input, &text_plan));
      ds4_gemma_residency prior;
      CHECK(ds4_gemma_multimodal_resident_weights(adapter, &prior));
      if (prior.audio_bytes || prior.vision_bytes)
        assert(text_plan.release_encoders && text_plan.reclaim_device_bytes);
      CHECK(ds4_gemma_multimodal_embed(adapter, text_input, 768, NULL,
                                       &text_result));
      assert(!memcmp(text_vector, ds4_gemma_result_data(text_result).vectors,
                     sizeof(text_vector)));
      ds4_gemma_residency after;
      CHECK(ds4_gemma_multimodal_resident_weights(adapter, &after));
      assert(!after.audio_bytes && !after.vision_bytes);
      char path[4096];
      snprintf(path, sizeof(path), "%s/warm-text-plan.json", argv[6]);
      FILE *f = fopen(path, "w");
      assert(f);
      fprintf(f,
              "{\"limit\":500000000,\"peak_bound\":%llu,\"reclaim\":%llu,"
              "\"release\":%s,\"bit_exact\":true}\n",
              (unsigned long long)text_plan.peak.device_bytes,
              (unsigned long long)text_plan.reclaim_device_bytes,
              text_plan.release_encoders ? "true" : "false");
      assert(!fclose(f));
      ds4_gemma_result_free(text_result);
      CHECK(ds4_gemma_multimodal_input_free(text_input));
      CHECK(ds4_resources_set_budget(rs, DS4_MEMORY_DEVICE, &budget));
    }
    free(original);
  }
  CHECK(ds4_gemma_visual_input_free(input));
  CHECK(ds4_gemma_visual_close(adapter));
  ds4_gemma_processor_close(processor);
  CHECK(ds4_bundle_close(bundle));
  ds4_numeric_memory(num, &free_device, &peak);
  assert(free_device == baseline);
  CHECK(ds4_numeric_close(num));
  ds4_runtime_release(runtime);
  if (result) {
    ds4_gemma_result_view view = ds4_gemma_result_data(result);
    assert(view.count == count);
    ds4_gemma_result_free(result);
  }
  CHECK(ds4_resources_close(rs));
  printf("vision/backbone/lifetime: PASS peak=%llu\n",
         (unsigned long long)peak);
  return 0;
}
