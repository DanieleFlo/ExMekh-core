#include "ds4_image.h"
#include "ds4_media_internal.h"
#include "ds4_sha256.h"
#include <math.h>
#include <soxr.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)

ds4_media_limits ds4_media_default_limits(void) {
  return (ds4_media_limits){64ull << 20, 512ull << 20, 4u << 20, 16384, 30.0};
}
static bool valid_limits(const ds4_media_limits *l) {
  return l && l->encoded_bytes && l->encoded_bytes <= (1ull << 30) &&
         l->retained_bytes >= sizeof(ds4_media) &&
         l->retained_bytes <= SIZE_MAX && l->pixels_per_frame &&
         l->pixels_per_frame <= DS4_IMAGE_MAX_PIXELS && l->decoded_frames &&
         l->decoded_frames <= 1048576 && isfinite(l->audio_seconds) &&
         l->audio_seconds > 0 && l->audio_seconds <= 3600;
}
ds4_status media_new(ds4_media **out, ds4_resources *rs, uint64_t owner,
                     ds4_media_type type, const ds4_media_limits *l) {
  *out = NULL;
  if (!rs || !owner || !valid_limits(l) || type < DS4_MEDIA_IMAGE ||
      type > DS4_MEDIA_AUDIO)
    return DS4_STATUS_INVALID;
  ds4_host_buffer allocation;
  ds4_status s = ds4_host_alloc(rs, owner, sizeof(ds4_media), &allocation);
  if (s)
    return s;
  ds4_media *m = allocation.data;
  m->account = allocation;
  m->resources = rs;
  m->owner = owner;
  m->limits = *l;
  m->bytes = sizeof(*m);
  m->references = 1;
  m->view.type = type;
  *out = m;
  return DS4_STATUS_OK;
}
ds4_status media_alloc(ds4_media *m, uint64_t bytes, ds4_host_buffer *out) {
  if (bytes > m->limits.retained_bytes - m->bytes)
    return DS4_STATUS_NOMEM;
  ds4_status s = ds4_host_alloc(m->resources, m->owner, bytes, out);
  if (!s)
    m->bytes += bytes;
  return s;
}
ds4_status ds4_media_retain(ds4_media *m) {
  if (!m || m->references == SIZE_MAX)
    return DS4_STATUS_INVALID;
  m->references++;
  return DS4_STATUS_OK;
}
void ds4_media_free(ds4_media *m) {
  if (!m || --m->references)
    return;
  media_frame *frames = m->frames.data;
  if (frames)
    for (uint32_t i = 0; i < m->view.frames; i++)
      ds4_host_free(&frames[i].storage);
  ds4_host_free(&m->frames);
  ds4_host_free(&m->audio);
  ds4_host_free(&m->original);
  ds4_host_buffer account = m->account;
  ds4_host_free(&account);
}
ds4_media_view ds4_media_data(const ds4_media *m) {
  return m ? m->view : (ds4_media_view){0};
}
ds4_status ds4_media_frame(const ds4_media *m, uint32_t i,
                           ds4_media_frame_view *out) {
  if (!m || !out || i >= m->view.frames)
    return DS4_STATUS_INVALID;
  *out = ((media_frame *)m->frames.data)[i].view;
  return DS4_STATUS_OK;
}
ds4_status ds4_media_original(const ds4_media *m, const uint8_t **bytes,
                              size_t *n) {
  if (!m || !bytes || !n)
    return DS4_STATUS_INVALID;
  *bytes = m->original.data;
  *n = m->original_bytes;
  return DS4_STATUS_OK;
}
ds4_status media_video_plan(uint32_t count, double fps, double duration,
                            const ds4_media_video_recipe *r, uint32_t limit,
                            media_video_selection *out) {
  if (!count || !r || !out || !isfinite(fps) || fps < 0 ||
      !isfinite(duration) || duration < 0 || !isfinite(r->fps) || r->fps < 0 ||
      r->overflow < DS4_MEDIA_UNIFORM || r->overflow > DS4_MEDIA_TRUNCATE ||
      (r->fps && (!fps || !duration)))
    return DS4_STATUS_INVALID;
  double samples = r->fps ? floor(duration * r->fps) : count;
  if (samples < 1)
    samples = 1;
  /* Matches the processor's index-based sampling, including repeated indices
   * on upsampling. Bound the recipe before enumerating any indices. */
  if (samples > 1048576)
    return DS4_STATUS_INVALID;
  uint32_t n = (uint32_t)samples;
  uint32_t keep = r->max_frames && n > r->max_frames ? r->max_frames : n;
  if (keep > limit)
    return DS4_STATUS_INVALID;
  *out = (media_video_selection){count, n, keep, r->fps ? fps / r->fps : 1,
                                 r->overflow};
  return DS4_STATUS_OK;
}
void media_video_indices(const media_video_selection *p, uint32_t *indices) {
  for (uint32_t i = 0; i < p->selected; i++) {
    uint32_t source = i;
    if (p->selected < p->samples && p->overflow == DS4_MEDIA_UNIFORM)
      source = p->selected == 1
                   ? 0
                   : (uint64_t)i * (p->samples - 1) / (p->selected - 1);
    /* An extremely small target rate can make step infinite. The first
     * sample is still index zero; never cast 0*infinity (NaN) to an integer. */
    double index = source ? floor(source * p->step) : 0;
    indices[i] =
        index >= p->source_count ? p->source_count - 1 : (uint32_t)index;
  }
}
ds4_status media_frame_allocate(ds4_media *m, uint32_t count) {
  if (!count || count > m->limits.decoded_frames)
    return DS4_STATUS_INVALID;
  ds4_status s =
      media_alloc(m, (uint64_t)count * sizeof(media_frame), &m->frames);
  if (!s)
    m->view.frames = count;
  return s;
}
ds4_status media_frame_copy(ds4_media *m, uint32_t slot,
                            const ds4_media_frame_input *f, uint32_t index,
                            double processor_time) {
  if (!f || !f->pixels || !f->width || !f->height ||
      f->width > DS4_IMAGE_MAX_DIMENSION ||
      f->height > DS4_IMAGE_MAX_DIMENSION ||
      (uint64_t)f->width * f->height > m->limits.pixels_per_frame ||
      (f->channels != 3 && f->channels != 4) || !isfinite(f->timestamp) ||
      !isfinite(processor_time) || slot >= m->view.frames)
    return DS4_STATUS_INVALID;
  media_frame *dest = (media_frame *)m->frames.data + slot;
  ds4_status s =
      media_alloc(m, (uint64_t)f->width * f->height * 3, &dest->storage);
  if (s)
    return s;
  uint8_t *rgb = dest->storage.data;
  /* The image processor drops alpha, including hidden RGB at alpha=0. */
  for (uint64_t i = 0; i < (uint64_t)f->width * f->height; i++)
    memcpy(rgb + i * 3, f->pixels + i * f->channels, 3);
  dest->view = (ds4_media_frame_view){rgb,   f->width,     f->height,
                                      index, f->timestamp, processor_time};
  return DS4_STATUS_OK;
}
void media_finish(ds4_media *m) {
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, &m->view.type, sizeof(m->view.type));
  for (uint32_t i = 0; i < m->view.frames; i++) {
    const ds4_media_frame_view *f = &((media_frame *)m->frames.data)[i].view;
    ds4_sha256_update(&hash, &f->width, sizeof(f->width));
    ds4_sha256_update(&hash, &f->height, sizeof(f->height));
    ds4_sha256_update(&hash, &f->source_index, sizeof(f->source_index));
    ds4_sha256_update(&hash, &f->timestamp, sizeof(f->timestamp));
    ds4_sha256_update(&hash, f->rgb, (uint64_t)f->width * f->height * 3);
  }
  if (m->view.audio_samples)
    ds4_sha256_update(&hash, m->view.audio,
                      m->view.audio_samples * sizeof(float));
  ds4_sha256_final(&hash, m->view.decoded_sha256);
  if (!m->original_bytes)
    memcpy(m->view.source_sha256, m->view.decoded_sha256, 32);
}
ds4_status ds4_media_image_pixels(ds4_media **out, ds4_resources *rs,
                                  uint64_t owner,
                                  const ds4_media_frame_input *f,
                                  const ds4_media_limits *limits) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_media *m = NULL;
  ds4_status status;
  TRY(media_new(&m, rs, owner, DS4_MEDIA_IMAGE, limits));
  TRY(media_frame_allocate(m, 1));
  TRY(media_frame_copy(m, 0, f, 0, 0));
  m->view.source_frames = 1;
  media_finish(m);
  *out = m;
  return DS4_STATUS_OK;
cleanup:
  ds4_media_free(m);
  return status;
}
ds4_status ds4_media_video_frames(ds4_media **out, ds4_resources *rs,
                                  uint64_t owner,
                                  const ds4_media_frame_input *frames, size_t n,
                                  double fps, double duration,
                                  const ds4_media_video_recipe *recipe,
                                  const ds4_media_limits *limits) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!frames || !n || !valid_limits(limits) || n > limits->decoded_frames)
    return DS4_STATUS_INVALID;
  for (size_t i = 0; i < n; i++)
    if (!isfinite(frames[i].timestamp) ||
        (i && frames[i].timestamp < frames[i - 1].timestamp))
      return DS4_STATUS_INVALID;
  ds4_media *m = NULL;
  ds4_host_buffer selected = {0};
  ds4_status status;
  TRY(media_new(&m, rs, owner, DS4_MEDIA_VIDEO, limits));
  media_video_selection selection;
  TRY(media_video_plan(n, fps, duration, recipe, limits->decoded_frames,
                       &selection));
  uint32_t count = selection.selected;
  TRY(ds4_host_alloc(rs, owner, (uint64_t)count * sizeof(uint32_t), &selected));
  media_video_indices(&selection, selected.data);
  TRY(media_frame_allocate(m, count));
  for (uint32_t i = 0; i < count; i++) {
    uint32_t index = ((uint32_t *)selected.data)[i];
    TRY(media_frame_copy(m, i, frames + index, index,
                         fps ? index / fps : frames[index].timestamp));
  }
  m->view.source_frames = n;
  m->view.source_fps = fps;
  m->view.duration = duration;
  m->view.video_recipe = *recipe;
  m->view.audio_policy = DS4_MEDIA_AUDIO_EXCLUDE;
  media_finish(m);
  *out = m;
  m = NULL;
cleanup:
  ds4_host_free(&selected);
  ds4_media_free(m);
  return status;
}
ds4_status media_audio_convert(ds4_media *m, const float *samples, uint64_t n,
                               uint32_t channels, uint32_t rate) {
  if (!samples || !n || !channels || channels > 8 || !rate || rate > 384000 ||
      (double)n / rate > m->limits.audio_seconds ||
      n > SIZE_MAX / (channels * 4u))
    return DS4_STATUS_INVALID;
  ds4_status status;
  ds4_host_buffer mono = {0};
  TRY(ds4_host_alloc(m->resources, m->owner, n * sizeof(float), &mono));
  float *x = mono.data;
  for (uint64_t i = 0; i < n; i++) {
    float sum = 0;
    for (uint32_t c = 0; c < channels; c++) {
      float v = samples[i * channels + c];
      if (!isfinite(v)) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      sum += v;
    }
    if (!isfinite(sum)) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    x[i] = sum / channels;
  }
  uint64_t count = (n * 16000 + rate - 1) / rate;
  TRY(media_alloc(m, count * sizeof(float), &m->audio));
  if (rate == 16000)
    memcpy(m->audio.data, x, count * sizeof(float));
  else {
    soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    soxr_runtime_spec_t runtime = soxr_runtime_spec(1);
    size_t consumed = 0, produced = 0;
    soxr_error_t error =
        soxr_oneshot(rate, 16000, 1, x, n, &consumed, m->audio.data, count,
                     &produced, &io, &quality, &runtime);
    if (error || consumed != n || !produced || produced > count) {
      status = DS4_STATUS_INFERENCE;
      goto cleanup;
    }
    /* T00 uses soxr.resample directly. Keep its actual output length rather
     * than appending a silent sample when the ratio rounds below ceil. */
    count = produced;
  }
  m->view.audio = m->audio.data;
  m->view.audio_samples = count;
  m->view.duration = (double)n / rate;
  status = DS4_STATUS_OK;
cleanup:
  ds4_host_free(&mono);
  return status;
}
ds4_status ds4_media_audio_samples(ds4_media **out, ds4_resources *rs,
                                   uint64_t owner, const float *samples,
                                   uint64_t n, uint32_t channels, uint32_t rate,
                                   const ds4_media_limits *limits) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_media *m = NULL;
  ds4_status status;
  TRY(media_new(&m, rs, owner, DS4_MEDIA_AUDIO, limits));
  TRY(media_audio_convert(m, samples, n, channels, rate));
  media_finish(m);
  *out = m;
  return DS4_STATUS_OK;
cleanup:
  ds4_media_free(m);
  return status;
}
ds4_status ds4_media_decode(ds4_media **out, ds4_resources *rs, uint64_t owner,
                            ds4_media_type type, const void *bytes, size_t n,
                            const ds4_media_decode_options *options) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!bytes || !n || !options || !valid_limits(&options->limits) ||
      n > options->limits.encoded_bytes || !isfinite(options->start_seconds) ||
      !isfinite(options->end_seconds) || options->start_seconds < 0 ||
      options->start_seconds > 1e9 || options->end_seconds > 1e9 ||
      (options->end_seconds &&
       options->end_seconds <= options->start_seconds) ||
      (type == DS4_MEDIA_IMAGE &&
       (options->start_seconds || options->end_seconds)) ||
      (type == DS4_MEDIA_VIDEO &&
       options->video_audio != DS4_MEDIA_AUDIO_EXCLUDE &&
       options->video_audio != DS4_MEDIA_AUDIO_FIRST))
    return DS4_STATUS_INVALID;
  ds4_media *m = NULL;
  ds4_image image = {0};
  ds4_reservation *lease = NULL;
  ds4_reservation_scope scope = {0};
  ds4_status status;
  TRY(media_new(&m, rs, owner, type, &options->limits));
  TRY(media_alloc(m, n, &m->original));
  memcpy(m->original.data, bytes, n);
  m->original_bytes = n;
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, bytes, n);
  ds4_sha256_final(&hash, m->view.source_sha256);
  uint64_t demand[DS4_MEMORY_COUNT] = {0};
  demand[DS4_MEMORY_HOST] =
      type == DS4_MEDIA_AUDIO
          ? (uint64_t)ceil(m->limits.audio_seconds * 384000 * 8) + (32ull << 20)
          : (uint64_t)m->limits.pixels_per_frame *
                    (type == DS4_MEDIA_IMAGE ? 16 : 96) +
                (32ull << 20);
  if (type == DS4_MEDIA_VIDEO && options->video_audio == DS4_MEDIA_AUDIO_FIRST)
    demand[DS4_MEMORY_HOST] +=
        (uint64_t)ceil(m->limits.audio_seconds * 384000 * 8) + (32ull << 20);
  /* Cover bounded demux metadata, packets and extradata copies in addition
   * to the authoritative encoded bytes already registered by this object. */
  demand[DS4_MEMORY_HOST] += (uint64_t)n * 2;
  TRY(ds4_resources_reserve(rs, owner, demand, &lease));
  scope = ds4_reservation_enter(lease);
  if (type == DS4_MEDIA_IMAGE) {
    uint32_t w, h;
    if (!ds4_image_encoded_size(bytes, n, &w, &h) ||
        (uint64_t)w * h > m->limits.pixels_per_frame ||
        !ds4_image_decode_memory(&image, bytes, n, NULL, 0)) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    TRY(media_frame_allocate(m, 1));
    ds4_media_frame_input input = {image.rgb, image.width, image.height, 3, 0};
    TRY(media_frame_copy(m, 0, &input, 0, 0));
    m->view.source_frames = 1;
  } else
    TRY(media_decode_av(m, options));
cleanup:
  ds4_image_free(&image);
  ds4_reservation_leave(&scope);
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status)
    ds4_media_free(m);
  else {
    media_finish(m);
    *out = m;
  }
  return status;
}
