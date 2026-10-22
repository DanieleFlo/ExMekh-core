#ifndef DS4_MEDIA_H
#define DS4_MEDIA_H
#include "ds4_resources.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_media ds4_media;
typedef enum {
  DS4_MEDIA_IMAGE = 1,
  DS4_MEDIA_VIDEO,
  DS4_MEDIA_AUDIO
} ds4_media_type;
typedef enum { DS4_MEDIA_UNIFORM, DS4_MEDIA_TRUNCATE } ds4_media_overflow;
typedef enum {
  DS4_MEDIA_AUDIO_EXCLUDE = 1,
  DS4_MEDIA_AUDIO_FIRST
} ds4_media_audio_policy;
typedef struct {
  /* retained_bytes bounds original + decoded storage + object metadata.
   * Decoder/resampler working memory has a separate registry reservation. */
  uint64_t encoded_bytes, retained_bytes;
  uint32_t pixels_per_frame, decoded_frames;
  double audio_seconds;
} ds4_media_limits;
typedef struct {
  /* fps=0 preserves frame indices. A positive rate uses declared source fps,
   * never guesses it. The cap is applied after rate sampling. Zero cap keeps
   * all selected frames, subject to decoded byte/frame admission. */
  double fps;
  uint32_t max_frames;
  ds4_media_overflow overflow;
} ds4_media_video_recipe;
typedef struct {
  ds4_media_limits limits;
  ds4_media_video_recipe video;
  ds4_media_audio_policy
      video_audio; /* Required, even for visual-only video. */
  double start_seconds,
      end_seconds; /* end=0: end of media; no hidden truncation. */
} ds4_media_decode_options;
typedef struct {
  const uint8_t *pixels;
  uint32_t width, height, channels;
  double timestamp; /* Original presentation time, finite and nondecreasing. */
} ds4_media_frame_input;
typedef struct {
  const uint8_t *rgb;
  uint32_t width, height, source_index;
  double timestamp, processor_timestamp;
} ds4_media_frame_view;
typedef struct {
  ds4_media_type type;
  uint32_t frames, source_frames;
  double source_fps, duration;
  uint64_t audio_samples;
  const float *audio; /* Mono FP32 at exactly 16 kHz. */
  bool has_audio_track;
  ds4_media_audio_policy audio_policy;
  ds4_media_video_recipe video_recipe;
  uint8_t source_sha256[32], decoded_sha256[32];
  double segment_start, segment_end;
} ds4_media_view;
ds4_media_limits ds4_media_default_limits(void);
/* All constructors copy caller storage. The original encoded bytes, decoded
 * frames/waveform and later processor tensors have distinct ownership.
 * Reference-count mutations on the same handle are externally serialized.
 * Errors publish NULL and never enter the model runtime or initialize a GPU.
 * Only memory input is decoded: libav cannot fetch URLs or nested resources. */
ds4_status ds4_media_decode(ds4_media **, ds4_resources *, uint64_t owner,
                            ds4_media_type, const void *, size_t,
                            const ds4_media_decode_options *);
ds4_status ds4_media_image_pixels(ds4_media **, ds4_resources *, uint64_t owner,
                                  const ds4_media_frame_input *,
                                  const ds4_media_limits *);
ds4_status ds4_media_video_frames(ds4_media **, ds4_resources *, uint64_t owner,
                                  const ds4_media_frame_input *, size_t count,
                                  double source_fps, double duration,
                                  const ds4_media_video_recipe *,
                                  const ds4_media_limits *);
ds4_status ds4_media_audio_samples(ds4_media **, ds4_resources *,
                                   uint64_t owner, const float *interleaved,
                                   uint64_t samples, uint32_t channels,
                                   uint32_t sample_rate,
                                   const ds4_media_limits *);
ds4_status ds4_media_retain(ds4_media *);
void ds4_media_free(ds4_media *);
ds4_media_view ds4_media_data(const ds4_media *);
ds4_status ds4_media_frame(const ds4_media *, uint32_t, ds4_media_frame_view *);
ds4_status ds4_media_original(const ds4_media *, const uint8_t **, size_t *);
#ifdef __cplusplus
}
#endif
#endif
