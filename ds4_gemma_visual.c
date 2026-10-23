#include "ds4_gemma_visual.h"
#include "ds4_embedding_gemma2_internal.h"
#include "ds4_gemma_audio.h"
#include "ds4_gemma_multimodal.h"
#include "ds4_gemma_vision.h"
#include "ds4_host_memory.h"
#include "ds4_sha256.h"
#include <math.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
struct ds4_gemma_visual {
  ds4_embedding_gemma2 *text;
  ds4_gemma_vision *vision;
  ds4_gemma_audio *audio;
  ds4_numeric *num;
  ds4_runtime *runtime;
  ds4_resources *resources;
  uint64_t owner;
  size_t inputs;
  bool running;
  ds4_host_buffer account;
  uint8_t identity[32];
};
typedef struct {
  size_t start, count;
  uint32_t images, videos, audio;
  uint32_t task;
  ds4_gemma_segment segment;
} visual_item;
struct ds4_gemma_visual_input {
  ds4_gemma_visual *adapter;
  ds4_gemma_input *backbone;
  size_t count, media_count;
  ds4_host_buffer account, items, media;
};
static bool dimensions_valid(uint32_t n) {
  return n == 128 || n == 256 || n == 512 || n == 768;
}
ds4_status ds4_gemma_visual_resident_weights(const ds4_gemma_visual *e,
                                             ds4_gemma_residency *out) {
  if (!e || !out)
    return DS4_STATUS_INVALID;
  *out = (ds4_gemma_residency){ds4_embedding_gemma2_resident_weights(e->text),
                               ds4_gemma_vision_resident_weights(e->vision),
                               ds4_gemma_audio_resident_weights(e->audio)};
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_visual_open(ds4_gemma_visual **out, ds4_bundle *bundle,
                                 ds4_numeric *num, ds4_runtime *runtime,
                                 const char *asset,
                                 const ds4_gemma_processor *processor) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!bundle || !num || !runtime || !asset || !processor)
    return DS4_STATUS_INVALID;
  ds4_gemma_visual *e = NULL;
  ds4_host_buffer account = {0};
  ds4_status status;
  uint64_t owner;
  ds4_resources *rs = ds4_numeric_resources(num, &owner);
  TRY(ds4_host_alloc(rs, owner, sizeof(*e), &account));
  e = account.data;
  e->account = account;
  e->num = num;
  e->runtime = runtime;
  e->resources = rs;
  e->owner = owner;
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, ds4_bundle_model(bundle),
                    strlen(ds4_bundle_model(bundle)) + 1);
  ds4_sha256_update(&hash, ds4_bundle_revision(bundle),
                    strlen(ds4_bundle_revision(bundle)) + 1);
  ds4_sha256_update(&hash, ds4_bundle_profile(bundle),
                    strlen(ds4_bundle_profile(bundle)) + 1);
  ds4_sha256_final(&hash, e->identity);
  TRY(ds4_embedding_gemma2_open(&e->text, bundle, num, runtime, asset));
  TRY(ds4_gemma_vision_open(&e->vision, bundle, num, runtime, processor));
  TRY(ds4_gemma_audio_open(&e->audio, bundle, num, runtime));
  *out = e;
  return DS4_STATUS_OK;
cleanup:
  ds4_gemma_visual_close(e);
  return status;
}
ds4_status ds4_gemma_visual_close(ds4_gemma_visual *e) {
  if (!e)
    return DS4_STATUS_OK;
  if (e->inputs || e->running)
    return DS4_STATUS_BUSY;
  ds4_status s = ds4_gemma_audio_close(e->audio);
  if (s)
    return s;
  e->audio = NULL;
  s = ds4_gemma_vision_close(e->vision);
  if (s)
    return s;
  e->vision = NULL;
  s = ds4_embedding_gemma2_close(e->text);
  if (s)
    return s;
  e->text = NULL;
  ds4_host_buffer account = e->account;
  ds4_host_free(&account);
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_visual_release_vision(ds4_gemma_visual *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  return e->running ? DS4_STATUS_BUSY
                    : ds4_gemma_vision_release_weights(e->vision);
}
ds4_status ds4_gemma_visual_release_text(ds4_gemma_visual *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  return e->running ? DS4_STATUS_BUSY
                    : ds4_embedding_gemma2_release_weights(e->text);
}
ds4_status ds4_gemma_visual_release_audio(ds4_gemma_visual *e) {
  if (!e)
    return DS4_STATUS_INVALID;
  return e->running ? DS4_STATUS_BUSY
                    : ds4_gemma_audio_release_weights(e->audio);
}
ds4_status ds4_gemma_visual_input_free(ds4_gemma_visual_input *in) {
  if (!in)
    return DS4_STATUS_OK;
  if (in->adapter->running)
    return DS4_STATUS_BUSY;
  ds4_gemma_input_free(in->backbone);
  if (in->media.data)
    for (size_t i = 0; i < in->media_count; i++)
      ds4_gemma_media_free(((ds4_gemma_media **)in->media.data)[i]);
  ds4_host_free(&in->items);
  ds4_host_free(&in->media);
  in->adapter->inputs--;
  ds4_host_buffer account = in->account;
  ds4_host_free(&account);
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_visual_input_tokens(const ds4_gemma_visual_input *in,
                                         size_t item, const uint32_t **tokens,
                                         uint32_t *count) {
  return ds4_gemma_input_tokens(in ? in->backbone : NULL, item, tokens, count);
}
typedef struct {
  char *data;
  size_t count, capacity;
} text_builder;
ds4_status ds4_gemma_multimodal_prepare_tokens(ds4_gemma_multimodal *e,
    const uint32_t *const *ids, const uint32_t *lengths, size_t count,
    ds4_gemma_multimodal_input **out) {
  if (!out) return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !count || count > 1024) return DS4_STATUS_INVALID;
  if (e->running) return DS4_STATUS_BUSY;
  ds4_host_buffer account = {0};
  ds4_status s = ds4_host_alloc(e->resources, e->owner,
                                sizeof(ds4_gemma_visual_input), &account);
  if (s) return s;
  ds4_gemma_visual_input *in = account.data;
  in->account = account;
  in->adapter = e;
  in->count = count;
  e->inputs++;
  s = ds4_host_alloc(e->resources, e->owner, count * sizeof(visual_item), &in->items);
  if (!s) s = ds4_gemma_prepare_tokens(e->text, ids, lengths, count, &in->backbone);
  if (s) ds4_gemma_visual_input_free(in);
  else *out = in;
  return s;
}
static bool append(text_builder *b, const char *s, size_t n) {
  if (n >= SIZE_MAX - b->count)
    return false;
  if (b->data) {
    if (b->count >= b->capacity || n >= b->capacity - b->count)
      return false;
    memcpy(b->data + b->count, s, n);
    b->data[b->count + n] = 0;
  }
  b->count += n;
  return true;
}
/* The same builder measures and writes, so admission cannot drift from the
 * marker/timestamp expansion. bytes includes the terminating NUL. */
static ds4_status expand_prompt(const ds4_gemma_visual_request *r, char *out,
                                size_t capacity, size_t *bytes) {
  static const char image[] = "<|image|>", video[] = "<|video|>",
                    audio[] = "<|audio|>";
  text_builder b = {out, 0, capacity};
  size_t next = 0;
  const char *prompt = r->prompt ? r->prompt : "";
  for (size_t i = 0; r->prompt ? prompt[i] != 0 : next < r->media_count;) {
    ds4_gemma_media_view v = {0};
    bool marker = !r->prompt;
    ds4_media_type type = 0;
    size_t length = 0;
    if (r->prompt) {
      if (!strncmp(prompt + i, image, sizeof(image) - 1)) {
        marker = true;
        type = DS4_MEDIA_IMAGE;
        length = sizeof(image) - 1;
      } else if (!strncmp(prompt + i, video, sizeof(video) - 1)) {
        marker = true;
        type = DS4_MEDIA_VIDEO;
        length = sizeof(video) - 1;
      } else if (!strncmp(prompt + i, audio, sizeof(audio) - 1)) {
        marker = true;
        type = DS4_MEDIA_AUDIO;
        length = sizeof(audio) - 1;
      }
    }
    if (!marker) {
      if (!append(&b, prompt + i, 1))
        return DS4_STATUS_INVALID;
      i++;
      continue;
    }
    if (next == r->media_count)
      return DS4_STATUS_INVALID;
    v = ds4_gemma_media_data(r->media[next]);
    if (type && v.type != type)
      return DS4_STATUS_INVALID;
    if (!r->prompt && next && !append(&b, " ", 1))
      return DS4_STATUS_INVALID;
    for (uint32_t frame = 0; frame < v.frames; frame++) {
      if (r->add_timestamps && v.type == DS4_MEDIA_VIDEO) {
        double time = v.processor_timestamps[frame];
        if (!isfinite(time) || time < 0 || time >= 600000)
          return DS4_STATUS_INVALID;
        char stamp[24];
        int n = snprintf(stamp, sizeof(stamp), "%s%02u:%02u ", frame ? " " : "",
                         (unsigned)time / 60, (unsigned)time % 60);
        if (n <= 0 || !append(&b, stamp, n))
          return DS4_STATUS_INVALID;
      }
      if (!append(&b, "<|image>", 8))
        return DS4_STATUS_INVALID;
      const char *token = v.type == DS4_MEDIA_IMAGE ? image : video;
      size_t n =
          v.type == DS4_MEDIA_IMAGE ? sizeof(image) - 1 : sizeof(video) - 1;
      for (uint32_t t = 0; t < v.frame_soft_tokens[frame]; t++)
        if (!append(&b, token, n))
          return DS4_STATUS_INVALID;
      if (!append(&b, "<image|>", 8))
        return DS4_STATUS_INVALID;
    }
    /* A video carrying an explicitly selected audio track contributes its
     * frame sequence followed by the complete track. Track duration never
     * controls the number/order of visual frames. */
    if (v.audio_frames) {
      if (!append(&b, "<|audio>", 8))
        return DS4_STATUS_INVALID;
      for (uint32_t t = 0; t < v.audio_soft_tokens; t++)
        if (!append(&b, audio, sizeof(audio) - 1))
          return DS4_STATUS_INVALID;
      if (!append(&b, "<audio|>", 8))
        return DS4_STATUS_INVALID;
    }
    next++;
    i += length;
  }
  if (next != r->media_count || b.count > 1048576)
    return DS4_STATUS_INVALID;
  *bytes = b.count + 1;
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_visual_prepare(ds4_gemma_visual *e,
                                    const ds4_gemma_visual_request *requests,
                                    size_t count,
                                    ds4_gemma_visual_input **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !requests || !count || count > 1024)
    return DS4_STATUS_INVALID;
  if (e->running)
    return DS4_STATUS_BUSY;
  uint64_t media_count = 0, total_text = 0;
  for (size_t i = 0; i < count; i++) {
    const ds4_gemma_visual_request *r = requests + i;
    if ((r->media_count && !r->media) || (!r->media_count && !r->prompt) ||
        r->media_count > 64 || r->task > DS4_GEMMA_SIMILARITY)
      return DS4_STATUS_INVALID;
    media_count += r->media_count;
    size_t text = r->prompt ? strnlen(r->prompt, 1048577) : 0;
    if (text > 1048576)
      return DS4_STATUS_INVALID;
    uint64_t soft = 0, frames = 0;
    for (size_t j = 0; j < r->media_count; j++) {
      ds4_gemma_media_view v = ds4_gemma_media_data(r->media[j]);
      if (v.type != DS4_MEDIA_IMAGE && v.type != DS4_MEDIA_VIDEO &&
          v.type != DS4_MEDIA_AUDIO)
        return DS4_STATUS_UNSUPPORTED;
      if (!v.frames && !v.audio_soft_tokens)
        return DS4_STATUS_INVALID;
      soft += v.soft_tokens;
      frames += v.frames + (v.audio_soft_tokens != 0);
    }
    if (soft + 2 * frames + 2 > 8192)
      return DS4_STATUS_INVALID;
    size_t expanded_bytes;
    ds4_status s = expand_prompt(r, NULL, 0, &expanded_bytes);
    if (s)
      return s;
    total_text += expanded_bytes;
  }
  ds4_status status;
  ds4_gemma_visual_input *in = NULL;
  ds4_host_buffer account = {0}, texts = {0}, strings = {0};
  TRY(ds4_host_alloc(e->resources, e->owner, sizeof(*in), &account));
  in = account.data;
  in->account = account;
  in->adapter = e;
  in->count = count;
  e->inputs++;
  TRY(ds4_host_alloc(e->resources, e->owner, count * sizeof(visual_item),
                     &in->items));
  TRY(ds4_host_alloc(e->resources, e->owner,
                     media_count * sizeof(ds4_gemma_media *), &in->media));
  TRY(ds4_host_alloc(e->resources, e->owner, count * sizeof(ds4_gemma_text),
                     &texts));
  TRY(ds4_host_alloc(e->resources, e->owner, total_text, &strings));
  uint64_t offset = 0;
  for (size_t i = 0; i < count; i++) {
    const ds4_gemma_visual_request *r = requests + i;
    visual_item *it = (visual_item *)in->items.data + i;
    *it = (visual_item){
        .start = in->media_count, .count = r->media_count, .task = r->task};
    for (size_t j = 0; j < r->media_count; j++) {
      ds4_gemma_media *m = r->media[j];
      ds4_gemma_media_view v = ds4_gemma_media_data(m);
      TRY(ds4_gemma_media_retain(m));
      ((ds4_gemma_media **)in->media.data)[in->media_count++] = m;
      if (v.type == DS4_MEDIA_IMAGE)
        it->images += v.soft_tokens - v.audio_soft_tokens;
      else if (v.type == DS4_MEDIA_VIDEO)
        it->videos += v.soft_tokens - v.audio_soft_tokens;
      it->audio += v.audio_soft_tokens;
    }
    size_t expanded_bytes;
    char *text = (char *)strings.data + offset;
    TRY(expand_prompt(r, text, total_text - offset, &expanded_bytes));
    ((ds4_gemma_text *)texts.data)[i] = (ds4_gemma_text){text, r->task};
    offset += expanded_bytes;
  }
  TRY(ds4_gemma_prepare(e->text, texts.data, count, &in->backbone));
  for (size_t i = 0; i < count; i++) {
    visual_item *it = (visual_item *)in->items.data + i;
    TRY(gemma_input_visual_validate(in->backbone, i, it->images, it->videos,
                                    it->audio));
  }
  *out = in;
  in = NULL;
cleanup:
  ds4_host_free(&texts);
  ds4_host_free(&strings);
  ds4_gemma_visual_input_free(in);
  return status;
}
enum { TEXT_BRANCH, VISION_BRANCH, AUDIO_BRANCH, BRANCH_COUNT };
static ds4_status branch_weights(ds4_gemma_visual *e, unsigned branch,
                                  uint64_t limit, ds4_weight_budget *out) {
  return branch == TEXT_BRANCH ? ds4_embedding_gemma2_weights(e->text, limit, out)
       : branch == VISION_BRANCH ? ds4_gemma_vision_weights(e->vision, limit, out)
                                 : ds4_gemma_audio_weights(e->audio, limit, out);
}
typedef struct {
  ds4_runtime_estimate phase[BRANCH_COUNT];
  uint64_t detached_host_bytes;
  uint32_t modalities;
} request_cost;

static bool add_bytes(uint64_t *total, uint64_t bytes) {
  if (bytes > UINT64_MAX - *total)
    return false;
  *total += bytes;
  return true;
}
static void max_phase(ds4_runtime_estimate *peak, ds4_runtime_estimate phase) {
  if (peak->device_bytes < phase.device_bytes)
    peak->device_bytes = phase.device_bytes;
  if (peak->host_bytes < phase.host_bytes)
    peak->host_bytes = phase.host_bytes;
  peak->known = true;
}
/* Collect each tower's own estimate once. The orchestrator combines phase
 * costs; it never reconstructs an encoder's weights, graph or workspace. */
static ds4_status collect_cost(ds4_gemma_visual *e,
                               const ds4_gemma_visual_input *in,
                               request_cost *cost) {
  if (!e || !in || in->adapter != e)
    return DS4_STATUS_INVALID;
  *cost = (request_cost){.modalities = DS4_MODALITY_TEXT};
  ds4_status status =
      ds4_gemma_estimate(e->text, in->backbone, &cost->phase[TEXT_BRANCH]);
  if (status)
    return status;
  for (size_t i = 0; i < in->media_count; i++) {
    ds4_gemma_media *media = ((ds4_gemma_media **)in->media.data)[i];
    ds4_gemma_media_view m = ds4_gemma_media_data(media);
    ds4_runtime_estimate phase;
    if (m.frames) {
      status = ds4_gemma_vision_estimate(e->vision, media, &phase);
      if (status)
        return status;
      max_phase(&cost->phase[VISION_BRANCH], phase);
      cost->modalities |=
          m.type == DS4_MEDIA_IMAGE ? DS4_MODALITY_IMAGE : DS4_MODALITY_VIDEO;
    }
    if (m.audio_frames) {
      status = ds4_gemma_audio_estimate(e->audio, media, &phase);
      if (status)
        return status;
      max_phase(&cost->phase[AUDIO_BRANCH], phase);
      cost->modalities |= DS4_MODALITY_AUDIO;
    }
    /* Detached rows coexist with the backbone's copied feature storage. */
    if (!add_bytes(&cost->detached_host_bytes, (uint64_t)m.soft_tokens * 2048))
      return DS4_STATUS_INVALID;
  }
  return DS4_STATUS_OK;
}
static ds4_status co_resident_cost(const request_cost *cost,
                                   ds4_runtime_estimate *out) {
  *out = (ds4_runtime_estimate){.known = true};
  for (unsigned i = 0; i < BRANCH_COUNT; i++)
    if (!add_bytes(&out->device_bytes, cost->phase[i].device_bytes) ||
        !add_bytes(&out->host_bytes, cost->phase[i].host_bytes))
      return DS4_STATUS_INVALID;
  return add_bytes(&out->host_bytes, cost->detached_host_bytes)
             ? DS4_STATUS_OK
             : DS4_STATUS_INVALID;
}

ds4_status ds4_gemma_visual_estimate(ds4_gemma_visual *e,
                                     const ds4_gemma_visual_input *in,
                                     ds4_runtime_estimate *out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = (ds4_runtime_estimate){0};
  request_cost cost;
  ds4_status status = collect_cost(e, in, &cost);
  return status ? status : co_resident_cost(&cost, out);
}

static ds4_status plan_full(ds4_gemma_visual *e,
                                 const ds4_gemma_visual_input *in,
                                 ds4_gemma_plan *out, uint64_t weight_limits[3],
                                 bool *streamed) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = (ds4_gemma_plan){0};
  memset(weight_limits, 0, 3 * sizeof(*weight_limits));
  *streamed = false;
  request_cost cost;
  ds4_status status = collect_cost(e, in, &cost);
  if (!status)
    status = co_resident_cost(&cost, &out->peak);
  if (status)
    return status;
  out->modalities = cost.modalities;
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE] =
                                          out->peak.device_bytes,
                                      [DS4_MEMORY_HOST] = out->peak.host_bytes};
  status = ds4_resources_can_reserve(e->resources, bytes, NULL);
  if (status != DS4_STATUS_NOMEM)
    return status;

  const uint64_t resident[BRANCH_COUNT] = {
      ds4_embedding_gemma2_resident_weights(e->text),
      ds4_gemma_vision_resident_weights(e->vision),
      ds4_gemma_audio_resident_weights(e->audio)};
  uint64_t peak = 0;
  for (unsigned i = 0; i < BRANCH_COUNT; i++) {
    if (!add_bytes(&out->reclaim_device_bytes, resident[i]))
      return DS4_STATUS_INVALID;
    if (cost.phase[i].known) {
      uint64_t demand = cost.phase[i].device_bytes;
      if (!add_bytes(&demand, resident[i]))
        return DS4_STATUS_INVALID;
      if (peak < demand)
        peak = demand;
    }
  }
  out->release_encoders = true;
  out->peak.device_bytes = peak;
  uint64_t reclaim[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE] =
                                            out->reclaim_device_bytes};
  bytes[DS4_MEMORY_DEVICE] = peak;
  status = ds4_resources_can_reserve(e->resources, bytes, reclaim);
  if (status != DS4_STATUS_NOMEM) return status;
  uint64_t available;
  status = ds4_resources_working_capacity(e->resources, DS4_MEMORY_DEVICE, &available);
  if (status || !add_bytes(&available, out->reclaim_device_bytes))
    return status ? status : DS4_STATUS_INVALID;
  peak = 0;
  for (unsigned i = 0; i < BRANCH_COUNT; i++) {
    if (!cost.phase[i].known) continue;
    ds4_weight_budget weights;
    status = branch_weights(e, i, 0, &weights);
    if (status) return status;
    uint64_t scratch = cost.phase[i].device_bytes + resident[i] - weights.total;
    if (scratch > available || weights.minimum > available - scratch)
      return DS4_STATUS_NOMEM;
    uint64_t cap = available - scratch;
    if (cap > weights.total) cap = weights.total;
    weight_limits[i] = cap;
    if (peak < scratch + cap) peak = scratch + cap;
  }
  *streamed = true;
  out->peak.device_bytes = bytes[DS4_MEMORY_DEVICE] = peak;
  return ds4_resources_can_reserve(e->resources, bytes, reclaim);
}
ds4_status ds4_gemma_visual_plan(ds4_gemma_visual *e,
                                 const ds4_gemma_visual_input *in,
                                 ds4_gemma_plan *out) {
  uint64_t limits[3]; bool streamed;
  return plan_full(e, in, out, limits, &streamed);
}

/* Encoder outputs are detached host rows. This scope owns staging until
 * the backbone copies them; the operation owns only its
 * reservation/transaction. */
static ds4_status encode_item(ds4_gemma_visual *e, ds4_gemma_visual_input *in,
                              size_t i, bool release_encoders,
                              const ds4_gemma_observer *o) {
  ds4_status status;
  ds4_host_buffer image_tokens = {0}, video_tokens = {0}, audio_tokens = {0};
  ds4_gemma_visual_features *features = NULL;
  ds4_gemma_audio_features *audio_features = NULL;
  visual_item *it = (visual_item *)in->items.data + i;
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)it->images * 1024,
                     &image_tokens));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)it->videos * 1024,
                     &video_tokens));
  TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)it->audio * 1024,
                     &audio_tokens));
  uint32_t images = 0, videos = 0, audio = 0;
  for (size_t j = 0; j < it->count; j++) {
    ds4_gemma_media *media =
        ((ds4_gemma_media **)in->media.data)[it->start + j];
    ds4_gemma_media_view v = ds4_gemma_media_data(media);
    if (v.frames) {
      TRY(ds4_gemma_vision_encode(e->vision, media, o, &features));
      ds4_gemma_visual_features_view f =
          ds4_gemma_visual_features_data(features);
      uint16_t *dest =
          v.type == DS4_MEDIA_IMAGE ? image_tokens.data : video_tokens.data;
      uint32_t *offset = v.type == DS4_MEDIA_IMAGE ? &images : &videos;
      memcpy(dest + (uint64_t)*offset * 512, f.tokens,
             (uint64_t)f.soft_tokens * 1024);
      *offset += f.soft_tokens;
      ds4_gemma_visual_features_free(features);
      features = NULL;
      if (release_encoders)
        TRY(ds4_gemma_vision_release_weights(e->vision));
    }
    if (v.audio_frames) {
      TRY(ds4_gemma_audio_encode(e->audio, media, o, &audio_features));
      ds4_gemma_audio_features_view f =
          ds4_gemma_audio_features_data(audio_features);
      memcpy((uint16_t *)audio_tokens.data + (uint64_t)audio * 512, f.tokens,
             (uint64_t)f.soft_tokens * 1024);
      audio += f.soft_tokens;
      ds4_gemma_audio_features_free(audio_features);
      audio_features = NULL;
      if (release_encoders)
        TRY(ds4_gemma_audio_release_weights(e->audio));
    }
  }
  TRY(gemma_input_visual(in->backbone, i,
                         (gemma_visual_tokens){.images = image_tokens.data,
                                               .videos = video_tokens.data,
                                               .audio = audio_tokens.data,
                                               .image_tokens = images,
                                               .video_tokens = videos,
                                               .audio_tokens = audio}));
cleanup:
  ds4_gemma_visual_features_free(features);
  ds4_gemma_audio_features_free(audio_features);
  ds4_host_free(&image_tokens);
  ds4_host_free(&video_tokens);
  ds4_host_free(&audio_tokens);
  return status;
}

ds4_status ds4_gemma_visual_embed(ds4_gemma_visual *e,
                                  ds4_gemma_visual_input *in,
                                  uint32_t dimensions,
                                  const ds4_gemma_observer *o,
                                  ds4_gemma_result **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !in || in->adapter != e || !dimensions_valid(dimensions))
    return DS4_STATUS_INVALID;
  if (e->running)
    return DS4_STATUS_BUSY;
  ds4_status status;
  ds4_runtime_transaction transaction = {0};
  ds4_reservation *lease = NULL;
  ds4_reservation_scope working = {0};
  bool started = false;
  bool stream_active = false;
  uint64_t weight_limits[3];
  bool streamed;
  ds4_gemma_plan plan = {0};
  if (!ds4_runtime_transaction_held(e->runtime))
    TRY(ds4_runtime_transaction_begin(e->runtime, &transaction));
  e->running = true;
  started = true;
  TRY(plan_full(e, in, &plan, weight_limits, &streamed));
  stream_active = streamed;
  ds4_runtime_estimate estimate = plan.peak;
  if (plan.release_encoders) {
    TRY(ds4_embedding_gemma2_release_weights(e->text));
    TRY(ds4_gemma_vision_release_weights(e->vision));
    TRY(ds4_gemma_audio_release_weights(e->audio));
  }
  if (streamed)
    for (unsigned i = 0; i < BRANCH_COUNT; i++)
      if (weight_limits[i]) TRY(branch_weights(e, i, weight_limits[i], NULL));
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE] =
                                          estimate.device_bytes,
                                      [DS4_MEMORY_HOST] = estimate.host_bytes};
  TRY(ds4_resources_reserve(e->resources, e->owner, bytes, &lease));
  working = ds4_reservation_enter(lease);
  for (size_t i = 0; i < in->count; i++)
    TRY(encode_item(e, in, i, plan.release_encoders, o));
  TRY(ds4_gemma_embed(e->text, in->backbone, dimensions, o, out));
cleanup:
  if (stream_active) {
    ds4_status s = ds4_embedding_gemma2_release_weights(e->text);
    if (!s) s = ds4_gemma_vision_release_weights(e->vision);
    if (!s) s = ds4_gemma_audio_release_weights(e->audio);
    for (unsigned i = 0; i < BRANCH_COUNT && !s; i++)
      s = branch_weights(e, i, UINT64_MAX, NULL);
    if (!status) status = s;
  }
  if (in)
    gemma_input_visual_clear(in->backbone);
  ds4_reservation_leave(&working);
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status && *out) {
    ds4_gemma_result_free(*out);
    *out = NULL;
  }
  if (started)
    e->running = false;
  ds4_runtime_transaction_end(&transaction);
  return status;
}

static void hash_u64(ds4_sha256 *hash, uint64_t value) {
  uint8_t bytes[8];
  for (unsigned i = 0; i < 8; i++)
    bytes[i] = (uint8_t)(value >> (8 * i));
  ds4_sha256_update(hash, bytes, 8);
}
ds4_status ds4_gemma_multimodal_key(const ds4_gemma_multimodal_input *in,
                                    size_t item, uint32_t dimensions,
                                    uint8_t key[32]) {
  if (!in || !key || item >= in->count || !dimensions_valid(dimensions))
    return DS4_STATUS_INVALID;
  const visual_item *it = (const visual_item *)in->items.data + item;
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  static const char recipe[] =
      "ds4-gemma-composition-v2/bf16/mean-all-tokens/matryoshka-l2";
  ds4_sha256_update(&hash, recipe, sizeof(recipe));
  ds4_sha256_update(&hash, in->adapter->identity, 32);
  hash_u64(&hash, it->task);
  hash_u64(&hash, dimensions);
  hash_u64(&hash, it->segment.id);
  uint64_t bits;
  memcpy(&bits, &it->segment.start_seconds, 8);
  hash_u64(&hash, bits);
  memcpy(&bits, &it->segment.end_seconds, 8);
  hash_u64(&hash, bits);
  const uint32_t *ids;
  uint32_t count;
  ds4_status s = ds4_gemma_input_tokens(in->backbone, item, &ids, &count);
  if (s)
    return s;
  hash_u64(&hash, count);
  for (uint32_t i = 0; i < count; i++)
    hash_u64(&hash, ids[i]);
  hash_u64(&hash, it->count);
  for (size_t i = 0; i < it->count; i++) {
    ds4_gemma_media_view m = ds4_gemma_media_data(
        ((ds4_gemma_media **)in->media.data)[it->start + i]);
    ds4_sha256_update(&hash, m.key, 32);
  }
  ds4_sha256_final(&hash, key);
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_multimodal_segment(const ds4_gemma_multimodal_input *in,
                                        size_t item, ds4_gemma_segment *out) {
  if (!in || item >= in->count || !out)
    return DS4_STATUS_INVALID;
  *out = ((const visual_item *)in->items.data)[item].segment;
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_multimodal_prepare(ds4_gemma_multimodal *e,
                                        const ds4_gemma_multimodal_request *r,
                                        size_t count,
                                        ds4_gemma_multimodal_input **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!e || !r || !count || count > 1024)
    return DS4_STATUS_INVALID;
  if (e->running)
    return DS4_STATUS_BUSY;
  uint64_t bytes = 0, media_count = 0;
  for (size_t i = 0; i < count; i++) {
    if (!r[i].content || !r[i].count || r[i].count > 1024 ||
        r[i].task > DS4_GEMMA_SIMILARITY ||
        !isfinite(r[i].segment.start_seconds) ||
        !isfinite(r[i].segment.end_seconds) || r[i].segment.start_seconds < 0 ||
        r[i].segment.end_seconds < 0 ||
        (r[i].segment.end_seconds &&
         r[i].segment.end_seconds <= r[i].segment.start_seconds))
      return DS4_STATUS_INVALID;
    uint64_t length = 1;
    for (size_t j = 0; j < r[i].count; j++) {
      const ds4_gemma_content *c = r[i].content + j;
      if ((c->text != NULL) == (c->media != NULL))
        return DS4_STATUS_INVALID;
      if (c->text) {
        size_t n = strnlen(c->text, 1048577);
        if (n > 1048576 || strstr(c->text, "<|image") ||
            strstr(c->text, "<image|") || strstr(c->text, "<|video") ||
            strstr(c->text, "<|audio") || strstr(c->text, "<audio|"))
          return DS4_STATUS_INVALID;
        length += n;
      } else {
        length += 9;
        media_count++;
      }
    }
    if (length > 1048576)
      return DS4_STATUS_INVALID;
    bytes += length;
  }
  ds4_status status;
  ds4_host_buffer requests = {0}, strings = {0}, media = {0};
  TRY(ds4_host_alloc(e->resources, e->owner,
                     count * sizeof(ds4_gemma_visual_request), &requests));
  TRY(ds4_host_alloc(e->resources, e->owner, bytes, &strings));
  TRY(ds4_host_alloc(e->resources, e->owner,
                     media_count * sizeof(ds4_gemma_media *), &media));
  uint64_t at = 0, mi = 0;
  for (size_t i = 0; i < count; i++) {
    ds4_gemma_visual_request *request =
        (ds4_gemma_visual_request *)requests.data + i;
    request->prompt = (char *)strings.data + at;
    request->media = media_count ? (ds4_gemma_media **)media.data + mi : NULL;
    request->task = r[i].task;
    request->add_timestamps = r[i].add_timestamps;
    text_builder b = {(char *)strings.data + at, 0, bytes - at};
    for (size_t j = 0; j < r[i].count; j++) {
      const ds4_gemma_content *c = r[i].content + j;
      const char *text = c->text;
      if (c->media) {
        ds4_gemma_media_view m = ds4_gemma_media_data(c->media);
        if (m.type == DS4_MEDIA_IMAGE)
          text = "<|image|>";
        else if (m.type == DS4_MEDIA_VIDEO)
          text = "<|video|>";
        else if (m.type == DS4_MEDIA_AUDIO)
          text = "<|audio|>";
        else {
          status = DS4_STATUS_INVALID;
          goto cleanup;
        }
        ((ds4_gemma_media **)media.data)[mi++] = c->media;
        request->media_count++;
      }
      if (!append(&b, text, strlen(text))) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
    }
    at += b.count + 1;
  }
  TRY(ds4_gemma_visual_prepare(e, requests.data, count, out));
  for (size_t i = 0; i < count; i++)
    ((visual_item *)(*out)->items.data)[i].segment = r[i].segment;
cleanup:
  ds4_host_free(&requests);
  ds4_host_free(&strings);
  ds4_host_free(&media);
  return status;
}
