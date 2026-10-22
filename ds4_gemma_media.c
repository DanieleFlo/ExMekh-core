/* Frozen Transformers processor equations, Apache-2.0; see provenance.
 * CPU preprocessing only, separate from model weights and numeric contexts. */
#include "ds4_gemma_media.h"
#include "ds4_host_memory.h"
#include "ds4_image.h"
#include "ds4_sha256.h"
#include <math.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
struct ds4_gemma_media {
  size_t references;
  ds4_gemma_media_view view;
  ds4_host_buffer account, patches, positions, counts, times, audio, mask;
};
typedef struct {
  ds4_gemma_media *media;
  uint64_t visit;
} cache_entry;
struct ds4_gemma_processor {
  ds4_resources *resources;
  uint64_t owner, clock;
  ds4_host_buffer account, asset;
  const float *window, *rope;
  const double *mel, *twiddles;
  uint8_t identity[32];
  cache_entry cache[64];
  ds4_gemma_cache_stats stats;
};
static uint32_t read32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static bool budget_valid(uint32_t n) {
  return n == 70 || n == 140 || n == 280 || n == 560 || n == 1120;
}
ds4_status ds4_gemma_media_retain(ds4_gemma_media *m) {
  if (!m || m->references == SIZE_MAX)
    return DS4_STATUS_INVALID;
  m->references++;
  return DS4_STATUS_OK;
}
void ds4_gemma_media_free(ds4_gemma_media *m) {
  if (!m || --m->references)
    return;
  ds4_host_free(&m->patches);
  ds4_host_free(&m->positions);
  ds4_host_free(&m->counts);
  ds4_host_free(&m->times);
  ds4_host_free(&m->audio);
  ds4_host_free(&m->mask);
  ds4_host_buffer account = m->account;
  ds4_host_free(&account);
}
ds4_gemma_media_view ds4_gemma_media_data(const ds4_gemma_media *m) {
  return m ? m->view : (ds4_gemma_media_view){0};
}
void ds4_gemma_processor_clear(ds4_gemma_processor *p) {
  if (!p)
    return;
  for (unsigned i = 0; i < 64; i++) {
    ds4_gemma_media_free(p->cache[i].media);
    p->cache[i] = (cache_entry){0};
  }
  p->stats.bytes = 0;
  p->stats.entries = 0;
}
void ds4_gemma_processor_close(ds4_gemma_processor *p) {
  if (!p)
    return;
  ds4_gemma_processor_clear(p);
  ds4_host_free(&p->asset);
  ds4_host_buffer account = p->account;
  ds4_host_free(&account);
}
ds4_gemma_cache_stats ds4_gemma_processor_cache(const ds4_gemma_processor *p) {
  return p ? p->stats : (ds4_gemma_cache_stats){0};
}
ds4_media_video_recipe
ds4_gemma_processor_video_recipe(const ds4_gemma_processor *p) {
  return p ? (ds4_media_video_recipe){1, 32, DS4_MEDIA_UNIFORM}
           : (ds4_media_video_recipe){0};
}
ds4_status ds4_gemma_processor_vision_rope(const ds4_gemma_processor *p,
                                           float f[16]) {
  if (!p || !f)
    return DS4_STATUS_INVALID;
  memcpy(f, p->rope, 16 * sizeof(float));
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_processor_open(ds4_gemma_processor **out,
                                    ds4_bundle *bundle, ds4_resources *rs,
                                    uint64_t owner, const char *path,
                                    uint64_t cache_bytes) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  if (!bundle || !rs || !owner || !path)
    return DS4_STATUS_INVALID;
  if (strcmp(ds4_bundle_model(bundle), "google/embeddinggemma-2") ||
      strcmp(ds4_bundle_revision(bundle),
             "914f7f89142e33e77833254d9c9b90c3cef7303b") ||
      strcmp(ds4_bundle_profile(bundle), "bf16-native-v1"))
    return DS4_STATUS_UNSUPPORTED;
  ds4_gemma_processor *p = NULL;
  ds4_host_buffer account = {0};
  FILE *file = NULL;
  ds4_status status;
  TRY(ds4_host_alloc(rs, owner, sizeof(*p), &account));
  p = account.data;
  p->account = account;
  p->resources = rs;
  p->owner = owner;
  p->stats.limit = cache_bytes;
  file = fopen(path, "rb");
  if (!file || fseek(file, 0, SEEK_END) || ftell(file) != 268808 ||
      fseek(file, 0, SEEK_SET)) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  TRY(ds4_host_alloc(rs, owner, 268808, &p->asset));
  if (fread(p->asset.data, 1, 268808, file) != 268808) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  if (fclose(file)) {
    file = NULL;
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  file = NULL;
  const uint8_t *data = p->asset.data;
  const char *names[] = {"google/embeddinggemma-2/config.json",
                         "google/embeddinggemma-2/processor_config.json",
                         "google/embeddinggemma-2/preprocessor_config.json"};
  uint32_t found = 0;
  for (size_t i = 0; i < ds4_bundle_file_count(bundle); i++) {
    const ds4_bundle_file *f = ds4_bundle_file_at(bundle, i);
    for (unsigned j = 0; j < 3; j++)
      if (!strcmp(f->path, names[j]) &&
          !memcmp(f->sha256, data + 8 + j * 32, 32))
        found |= 1u << j;
  }
  ds4_sha256 hash;
  uint8_t digest[32];
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, data + 136, 268808 - 136);
  ds4_sha256_final(&hash, digest);
  if (found != 7 || memcmp(data, "DS4GMP1\0", 8) ||
      memcmp(data + 104, digest, 32)) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  uint32_t expected[] = {16,  3,   280, 140, 32,  16000, 8192,
                         320, 160, 512, 128, 128, 480000};
  for (unsigned i = 0; i < 13; i++)
    if (read32(data + 136 + i * 4) != expected[i]) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  double floor;
  memcpy(&floor, data + 136 + 56, 8);
  if (floor != 0.001) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  p->window = (const float *)(data + 200);
  p->mel = (const double *)(data + 200 + 320 * 4);
  p->rope = (const float *)(data + 200 + 320 * 4 + 257 * 128 * 8);
  p->twiddles = (const double *)(data + 200 + 320 * 4 + 257 * 128 * 8 + 64);
  for (unsigned i = 0; i < 320; i++)
    if (!isfinite(p->window[i]) || p->window[i] < 0 || p->window[i] > 1) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  for (unsigned i = 0; i < 257 * 128; i++)
    if (!isfinite(p->mel[i]) || p->mel[i] < 0) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  for (unsigned i = 0; i < 16; i++)
    if (!isfinite(p->rope[i]) || p->rope[i] <= 0 || p->rope[i] > 1) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  for (unsigned i = 0; i < 512; i++)
    if (!isfinite(p->twiddles[i]) || fabs(p->twiddles[i]) > 1) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, data, 268808);
  ds4_sha256_final(&hash, p->identity);
  *out = p;
  p = NULL;
cleanup:
  if (file)
    fclose(file);
  ds4_gemma_processor_close(p);
  return status;
}
static bool geometry(uint32_t w, uint32_t h, uint32_t budget, uint32_t *ow,
                     uint32_t *oh) {
  double factor = sqrt((double)budget * 9 * 256 / ((double)w * h));
  *ow = (uint32_t)floor(factor * w / 48) * 48;
  *oh = (uint32_t)floor(factor * h / 48) * 48;
  if (!*oh && !*ow)
    return false;
  if (!*oh) {
    *oh = 48;
    *ow = (w / h < budget ? w / h : budget) * 48;
  }
  if (!*ow) {
    *ow = 48;
    *oh = (h / w < budget ? h / w : budget) * 48;
  }
  return *ow && *oh && (uint64_t)*ow * *oh <= (uint64_t)budget * 9 * 256;
}
static ds4_status plan(ds4_gemma_processor *p, const ds4_media *source,
                       const ds4_gemma_media_recipe *recipe,
                       ds4_gemma_media_view *v,
                       ds4_runtime_estimate *estimate) {
  if (!p || !source || !v || !estimate)
    return DS4_STATUS_INVALID;
  ds4_media_view media = ds4_media_data(source);
  uint32_t budget = recipe && recipe->soft_tokens   ? recipe->soft_tokens
                    : media.type == DS4_MEDIA_IMAGE ? 280
                                                    : 140;
  if (!budget_valid(budget) ||
      (media.type == DS4_MEDIA_AUDIO && recipe && recipe->soft_tokens))
    return DS4_STATUS_INVALID;
  *v = (ds4_gemma_media_view){.type = media.type,
                              .frames = media.frames,
                              .soft_token_budget = media.frames ? budget : 0,
                              .max_patches = media.frames ? budget * 9 : 0};
  uint64_t scratch = 0;
  for (uint32_t i = 0; i < media.frames; i++) {
    ds4_media_frame_view frame;
    uint32_t w, h;
    ds4_status s = ds4_media_frame(source, i, &frame);
    if (s)
      return s;
    if (!geometry(frame.width, frame.height, budget, &w, &h))
      return DS4_STATUS_INVALID;
    v->soft_tokens += w * h / (256 * 9);
    uint64_t bytes =
        (uint64_t)w * h * 12 + (uint64_t)frame.height * w * 3 + (2u << 20);
    if (bytes > scratch)
      scratch = bytes;
  }
  if (media.audio_samples) {
    if (media.audio_samples > 480000 || media.audio_samples <= 128)
      return DS4_STATUS_INVALID;
    uint32_t padded = (media.audio_samples + 127) / 128 * 128;
    v->audio_frames = (padded + 160 - 321) / 160 + 1;
    uint32_t valid =
        media.audio_samples > 160 ? (media.audio_samples - 161) / 160 + 1 : 0;
    v->audio_soft_tokens = (valid + 3) / 4;
    if (v->audio_soft_tokens > 280)
      v->audio_soft_tokens = 280;
    v->soft_tokens += v->audio_soft_tokens;
  }
  if ((uint64_t)v->soft_tokens + media.frames * 2ull + 2 > 8192)
    return DS4_STATUS_INVALID;
  v->host_bytes = sizeof(ds4_gemma_media) +
                  (uint64_t)media.frames * v->max_patches * (768 * 4 + 2 * 4) +
                  (uint64_t)media.frames * (4 + 16) +
                  (uint64_t)v->audio_frames * (128 * 4 + 1);
  *estimate = (ds4_runtime_estimate){
      .known = true, .host_bytes = v->host_bytes + scratch + 16384};
  return DS4_STATUS_OK;
}
ds4_status ds4_gemma_media_plan(ds4_gemma_processor *p, const ds4_media *source,
                                const ds4_gemma_media_recipe *r,
                                ds4_runtime_estimate *e,
                                uint32_t *soft_tokens) {
  if (!e || !soft_tokens)
    return DS4_STATUS_INVALID;
  *e = (ds4_runtime_estimate){0};
  *soft_tokens = 0;
  ds4_gemma_media_view v;
  ds4_status s = plan(p, source, r, &v, e);
  if (!s)
    *soft_tokens = v.soft_tokens;
  return s;
}
static void hash_u64(ds4_sha256 *h, uint64_t value) {
  uint8_t bytes[8];
  for (unsigned i = 0; i < 8; i++)
    bytes[i] = value >> (8 * i);
  ds4_sha256_update(h, bytes, 8);
}
static void media_key(ds4_gemma_processor *p, const ds4_media *source,
                      ds4_gemma_media_view *v) {
  ds4_media_view media = ds4_media_data(source);
  ds4_sha256 hash;
  ds4_sha256_init(&hash);
  ds4_sha256_update(&hash, p->identity, 32);
  ds4_sha256_update(&hash, media.source_sha256, 32);
  ds4_sha256_update(&hash, media.decoded_sha256, 32);
  hash_u64(&hash, 1); /* FP32 preprocessing / BF16 model profile, recipe v1. */
  hash_u64(&hash, media.type);
  hash_u64(&hash, v->soft_token_budget);
  hash_u64(&hash, media.audio_policy);
  hash_u64(&hash, media.frames);
  hash_u64(&hash, media.video_recipe.max_frames);
  hash_u64(&hash, media.video_recipe.overflow);
  const double values[] = {media.source_fps, media.duration,
                           media.video_recipe.fps, media.segment_start,
                           media.segment_end};
  for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
    uint64_t bits;
    memcpy(&bits, values + i, 8);
    hash_u64(&hash, bits);
  }
  ds4_sha256_final(&hash, v->key);
}
static ds4_status prepare_visual(ds4_gemma_processor *p,
                                 const ds4_media *source, ds4_gemma_media *m) {
  ds4_status status = DS4_STATUS_OK;
  ds4_host_buffer resized = {0};
  float *patches = m->patches.data;
  int32_t *positions = m->positions.data;
  uint32_t max = m->view.max_patches;
  for (uint32_t i = 0; i < m->view.frames; i++) {
    ds4_media_frame_view f;
    uint32_t w, h;
    TRY(ds4_media_frame(source, i, &f));
    if (!geometry(f.width, f.height, m->view.soft_token_budget, &w, &h)) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    TRY(ds4_host_alloc(p->resources, p->owner, (uint64_t)w * h * 12, &resized));
    ds4_image image = {
        .width = f.width, .height = f.height, .rgb = (uint8_t *)f.rgb};
    if (!ds4_image_resize_rgb_u8(&image, w, h, resized.data,
                                 (uint64_t)w * h * 3)) {
      status = DS4_STATUS_NOMEM;
      goto cleanup;
    }
    uint32_t grid_w = w / 16, grid_h = h / 16;
    ((uint32_t *)m->counts.data)[i] = grid_w * grid_h / 9;
    ((double *)m->times.data)[i] = f.timestamp;
    ((double *)m->times.data)[m->view.frames + i] = f.processor_timestamp;
    for (uint32_t patch = 0; patch < max; patch++) {
      uint64_t index = (uint64_t)i * max + patch;
      positions[index * 2] =
          patch < grid_w * grid_h ? (int32_t)(patch % grid_w) : -1;
      positions[index * 2 + 1] =
          patch < grid_w * grid_h ? (int32_t)(patch / grid_w) : -1;
      if (patch >= grid_w * grid_h)
        continue;
      for (uint32_t y = 0; y < 16; y++)
        for (uint32_t x = 0; x < 16; x++)
          for (uint32_t c = 0; c < 3; c++) {
            uint64_t pixel = ((uint64_t)(patch / grid_w * 16 + y) * w +
                              patch % grid_w * 16 + x) *
                                 3 +
                             c;
            patches[index * 768 + (y * 16 + x) * 3 + c] =
                ((float *)resized.data)[pixel] * (float)(1.0 / 255.0);
          }
    }
    ds4_host_free(&resized);
  }
cleanup:
  ds4_host_free(&resized);
  return status;
}
static void spectrum(ds4_gemma_processor *p, const float *wave, uint64_t n,
                     uint32_t frame, float magnitude[257]) {
  double re[512] = {0}, im[512] = {0};
  for (uint32_t i = 0; i < 320; i++) {
    int64_t sample = (int64_t)frame * 160 + i - 160;
    float value = sample >= 0 && (uint64_t)sample < n ? wave[sample] : 0;
    float windowed = value * p->window[i];
    uint32_t reverse = 0;
    for (unsigned k = 0; k < 9; k++)
      reverse = reverse * 2 + ((i >> k) & 1);
    re[reverse] = windowed;
  }
  for (uint32_t width = 2; width <= 512; width *= 2)
    for (uint32_t block = 0; block < 512; block += width)
      for (uint32_t j = 0; j < width / 2; j++) {
        uint32_t a = block + j, b = a + width / 2, k = j * (512 / width);
        double wr = p->twiddles[2 * k], wi = p->twiddles[2 * k + 1];
        double tr = wr * re[b] - wi * im[b], ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
  /* NumPy 2.5 keeps rfft(float32) in complex64. Round FFT components before
   * magnitude, then multiply the frozen float64 mel bank, then round output. */
  for (uint32_t k = 0; k < 257; k++)
    magnitude[k] = hypotf((float)re[k], (float)im[k]);
}
static void prepare_audio(ds4_gemma_processor *p, const ds4_media *source,
                          ds4_gemma_media *m) {
  ds4_media_view media = ds4_media_data(source);
  float *features = m->audio.data;
  uint8_t *mask = m->mask.data;
  for (uint32_t t = 0; t < m->view.audio_frames; t++) {
    mask[t] = (uint64_t)t * 160 + 160 < media.audio_samples;
    if (!mask[t])
      continue;
    float magnitude[257];
    spectrum(p, media.audio, media.audio_samples, t, magnitude);
    for (uint32_t mel = 0; mel < 128; mel++) {
      double sum = 0;
      for (uint32_t k = 0; k < 257; k++)
        sum += magnitude[k] * p->mel[k * 128 + mel];
      features[(uint64_t)t * 128 + mel] = (float)log(sum + 0.001);
    }
  }
}
static void cache_insert(ds4_gemma_processor *p, ds4_gemma_media *m) {
  if (!p->stats.limit || m->view.host_bytes > p->stats.limit)
    return;
  while (p->stats.entries == 64 ||
         m->view.host_bytes > p->stats.limit - p->stats.bytes) {
    uint32_t oldest = 64;
    for (uint32_t i = 0; i < 64; i++)
      if (p->cache[i].media &&
          (oldest == 64 || p->cache[i].visit < p->cache[oldest].visit))
        oldest = i;
    if (oldest == 64)
      return;
    p->stats.bytes -= p->cache[oldest].media->view.host_bytes;
    p->stats.entries--;
    ds4_gemma_media_free(p->cache[oldest].media);
    p->cache[oldest] = (cache_entry){0};
  }
  for (uint32_t i = 0; i < 64; i++)
    if (!p->cache[i].media) {
      if (ds4_gemma_media_retain(m))
        return;
      p->cache[i] = (cache_entry){m, ++p->clock};
      p->stats.bytes += m->view.host_bytes;
      p->stats.entries++;
      return;
    }
}
ds4_status ds4_gemma_media_prepare(ds4_gemma_processor *p,
                                   const ds4_media *source,
                                   const ds4_gemma_media_recipe *recipe,
                                   ds4_gemma_media **out) {
  if (!out)
    return DS4_STATUS_INVALID;
  *out = NULL;
  ds4_gemma_media_view v;
  ds4_runtime_estimate estimate;
  ds4_status status = plan(p, source, recipe, &v, &estimate);
  if (status)
    return status;
  media_key(p, source, &v);
  for (unsigned i = 0; i < 64; i++)
    if (p->cache[i].media && !memcmp(p->cache[i].media->view.key, v.key, 32)) {
      status = ds4_gemma_media_retain(p->cache[i].media);
      if (status)
        return status;
      p->cache[i].visit = ++p->clock;
      p->stats.hits++;
      *out = p->cache[i].media;
      return DS4_STATUS_OK;
    }
  p->stats.misses++;
  ds4_gemma_media *m = NULL;
  ds4_host_buffer account = {0};
  ds4_reservation *lease = NULL;
  ds4_reservation_scope scope = {0};
  uint64_t bytes[DS4_MEMORY_COUNT] = {[DS4_MEMORY_HOST] = estimate.host_bytes};
  TRY(ds4_resources_reserve(p->resources, p->owner, bytes, &lease));
  scope = ds4_reservation_enter(lease);
  TRY(ds4_host_alloc(p->resources, p->owner, sizeof(*m), &account));
  m = account.data;
  m->account = account;
  m->references = 1;
  m->view = v;
  TRY(ds4_host_alloc(p->resources, p->owner,
                     (uint64_t)v.frames * v.max_patches * 768 * 4,
                     &m->patches));
  TRY(ds4_host_alloc(p->resources, p->owner,
                     (uint64_t)v.frames * v.max_patches * 2 * 4,
                     &m->positions));
  TRY(ds4_host_alloc(p->resources, p->owner, (uint64_t)v.frames * 4,
                     &m->counts));
  TRY(ds4_host_alloc(p->resources, p->owner, (uint64_t)v.frames * 16,
                     &m->times));
  TRY(ds4_host_alloc(p->resources, p->owner, (uint64_t)v.audio_frames * 128 * 4,
                     &m->audio));
  TRY(ds4_host_alloc(p->resources, p->owner, v.audio_frames, &m->mask));
  m->view.patches = m->patches.data;
  m->view.positions = m->positions.data;
  m->view.frame_soft_tokens = m->counts.data;
  m->view.timestamps = m->times.data;
  m->view.processor_timestamps =
      v.frames ? (double *)m->times.data + v.frames : NULL;
  m->view.audio_features = m->audio.data;
  m->view.audio_mask = m->mask.data;
  TRY(prepare_visual(p, source, m));
  if (v.audio_frames)
    prepare_audio(p, source, m);
cleanup:
  ds4_reservation_leave(&scope);
  if (lease) {
    ds4_status s = ds4_reservation_cancel(lease);
    if (!status)
      status = s;
  }
  if (status)
    ds4_gemma_media_free(m);
  else {
    cache_insert(p, m);
    *out = m;
  }
  return status;
}
