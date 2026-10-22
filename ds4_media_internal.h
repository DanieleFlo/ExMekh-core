#ifndef DS4_MEDIA_INTERNAL_H
#define DS4_MEDIA_INTERNAL_H
#include "ds4_host_memory.h"
#include "ds4_media.h"
typedef struct {
  ds4_host_buffer storage;
  ds4_media_frame_view view;
} media_frame;
struct ds4_media {
  ds4_resources *resources;
  uint64_t owner, bytes;
  size_t references, original_bytes;
  ds4_media_limits limits;
  ds4_media_view view;
  ds4_host_buffer account, original, frames, audio;
};
ds4_status media_new(ds4_media **, ds4_resources *, uint64_t, ds4_media_type,
                     const ds4_media_limits *);
ds4_status media_alloc(ds4_media *, uint64_t, ds4_host_buffer *);
typedef struct {
  uint32_t source_count, samples, selected;
  double step;
  ds4_media_overflow overflow;
} media_video_selection;
/* Shared admission/index policy for raw frames and decoded containers. */
ds4_status media_video_plan(uint32_t count, double fps, double duration,
                            const ds4_media_video_recipe *, uint32_t limit,
                            media_video_selection *);
void media_video_indices(const media_video_selection *, uint32_t *indices);
ds4_status media_frame_allocate(ds4_media *, uint32_t count);
ds4_status media_frame_copy(ds4_media *, uint32_t slot,
                            const ds4_media_frame_input *, uint32_t index,
                            double processor_time);
ds4_status media_audio_convert(ds4_media *, const float *, uint64_t samples,
                               uint32_t channels, uint32_t sample_rate);
void media_finish(ds4_media *);
ds4_status media_decode_av(ds4_media *, const ds4_media_decode_options *);
#endif
