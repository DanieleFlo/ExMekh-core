#include "ds4_gemma_visual.h"
#include "ds4_embedding_gemma2_internal.h"
#include "ds4_gemma_vision.h"
#include "ds4_host_memory.h"
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
  ds4_numeric *num;
  ds4_runtime *runtime;
  ds4_resources *resources;
  uint64_t owner;
  size_t inputs;
  bool running;
  ds4_host_buffer account;
};
typedef struct {
  size_t start, count;
  uint32_t images, videos;
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
  TRY(ds4_embedding_gemma2_open(&e->text, bundle, num, runtime, asset));
  TRY(ds4_gemma_vision_open(&e->vision, bundle, num, runtime, processor));
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
  ds4_status s = ds4_gemma_vision_close(e->vision);
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
  return in ? ds4_gemma_input_tokens(in->backbone, item, tokens, count)
            : DS4_STATUS_INVALID;
}
typedef struct {
  char *data;
  size_t count, capacity;
} text_builder;
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
  static const char image[] = "<|image|>", video[] = "<|video|>";
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
      } else if (!strncmp(prompt + i, "<|audio|>", 9))
        return DS4_STATUS_UNSUPPORTED;
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
    if (!r->media || !r->media_count || r->media_count > 64 ||
        r->task > DS4_GEMMA_SIMILARITY)
      return DS4_STATUS_INVALID;
    media_count += r->media_count;
    size_t text = r->prompt ? strnlen(r->prompt, 1048577) : 0;
    if (text > 1048576)
      return DS4_STATUS_INVALID;
    uint64_t soft = 0, frames = 0;
    for (size_t j = 0; j < r->media_count; j++) {
      ds4_gemma_media_view v = ds4_gemma_media_data(r->media[j]);
      if (v.type != DS4_MEDIA_IMAGE && v.type != DS4_MEDIA_VIDEO)
        return DS4_STATUS_UNSUPPORTED;
      if (!v.frames || v.audio_frames)
        return DS4_STATUS_UNSUPPORTED;
      soft += v.soft_tokens;
      frames += v.frames;
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
    *it = (visual_item){.start = in->media_count, .count = r->media_count};
    for (size_t j = 0; j < r->media_count; j++) {
      ds4_gemma_media *m = r->media[j];
      ds4_gemma_media_view v = ds4_gemma_media_data(m);
      TRY(ds4_gemma_media_retain(m));
      ((ds4_gemma_media **)in->media.data)[in->media_count++] = m;
      if (v.type == DS4_MEDIA_IMAGE)
        it->images += v.soft_tokens;
      else
        it->videos += v.soft_tokens;
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
    TRY(gemma_input_visual_validate(in->backbone, i, it->images, it->videos));
  }
  *out = in;
  in = NULL;
cleanup:
  ds4_host_free(&texts);
  ds4_host_free(&strings);
  ds4_gemma_visual_input_free(in);
  return status;
}
ds4_status ds4_gemma_visual_estimate(ds4_gemma_visual *e,
                                     const ds4_gemma_visual_input *in,
                                     ds4_runtime_estimate *out) {
  if (!e || !in || in->adapter != e || !out)
    return DS4_STATUS_INVALID;
  ds4_status s = ds4_gemma_estimate(e->text, in->backbone, out);
  if (s)
    return s;
  uint64_t vision_device = 0, vision_host = 0, detached = 0;
  for (size_t i = 0; i < in->media_count; i++) {
    ds4_runtime_estimate v;
    s = ds4_gemma_vision_estimate(e->vision,
                                  ((ds4_gemma_media **)in->media.data)[i], &v);
    if (s)
      return s;
    /* Both towers may retain their weights. Summing phase demands is a safe
     * complete-operation bound, while frame activations stay serial. */
    if (v.device_bytes > vision_device)
      vision_device = v.device_bytes;
    if (v.host_bytes > vision_host)
      vision_host = v.host_bytes;
    ds4_gemma_media_view m =
        ds4_gemma_media_data(((ds4_gemma_media **)in->media.data)[i]);
    detached += (uint64_t)m.soft_tokens * 2048;
  }
  if (vision_device > UINT64_MAX - out->device_bytes ||
      vision_host > UINT64_MAX - out->host_bytes ||
      detached > UINT64_MAX - out->host_bytes - vision_host)
    return DS4_STATUS_INVALID;
  out->device_bytes += vision_device;
  out->host_bytes += vision_host + detached;
  return DS4_STATUS_OK;
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
  ds4_host_buffer image_tokens = {0}, video_tokens = {0};
  ds4_gemma_visual_features *features = NULL;
  TRY(ds4_runtime_transaction_begin(e->runtime, &transaction));
  e->running = true;
  ds4_runtime_estimate estimate;
  TRY(ds4_gemma_visual_estimate(e, in, &estimate));
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_DEVICE] =
                                          estimate.device_bytes,
                                      [DS4_MEMORY_HOST] = estimate.host_bytes};
  TRY(ds4_resources_reserve(e->resources, e->owner, bytes, &lease));
  working = ds4_reservation_enter(lease);
  for (size_t i = 0; i < in->count; i++) {
    visual_item *it = (visual_item *)in->items.data + i;
    TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)it->images * 1024,
                       &image_tokens));
    TRY(ds4_host_alloc(e->resources, e->owner, (uint64_t)it->videos * 1024,
                       &video_tokens));
    uint32_t images = 0, videos = 0;
    for (size_t j = 0; j < it->count; j++) {
      ds4_gemma_media *media =
          ((ds4_gemma_media **)in->media.data)[it->start + j];
      ds4_gemma_media_view v = ds4_gemma_media_data(media);
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
    }
    TRY(gemma_input_visual(in->backbone, i,
                           (gemma_visual_tokens){image_tokens.data,
                                                 video_tokens.data, images,
                                                 videos}));
    ds4_host_free(&image_tokens);
    ds4_host_free(&video_tokens);
  }
  TRY(ds4_gemma_embed(e->text, in->backbone, dimensions, o, out));
cleanup:
  if (in)
    gemma_input_visual_clear(in->backbone);
  ds4_gemma_visual_features_free(features);
  ds4_host_free(&image_tokens);
  ds4_host_free(&video_tokens);
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
  if (transaction.runtime)
    e->running = false;
  ds4_runtime_transaction_end(&transaction);
  return status;
}
