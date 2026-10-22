/* Software libav decode only. Custom memory IO has no URL/file fallback. */
#include "ds4_media_internal.h"
#include <errno.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <math.h>
#include <string.h>
#define TRY(x)                                                                 \
  do {                                                                         \
    status = (x);                                                              \
    if (status)                                                                \
      goto cleanup;                                                            \
  } while (0)
typedef struct {
  ds4_media *media;
  size_t cursor;
  AVFormatContext *format;
  AVIOContext *io;
  AVCodecContext *codec;
  AVPacket *packet;
  AVFrame *frame;
  int stream;
  bool flushed;
} reader;
static int read_memory(void *opaque, uint8_t *out, int n) {
  reader *r = opaque;
  size_t remaining = r->media->original_bytes - r->cursor;
  if (!remaining)
    return AVERROR_EOF;
  if ((size_t)n > remaining)
    n = remaining;
  memcpy(out, (uint8_t *)r->media->original.data + r->cursor, n);
  r->cursor += n;
  return n;
}
static int64_t seek_memory(void *opaque, int64_t offset, int whence) {
  reader *r = opaque;
  if (whence == AVSEEK_SIZE)
    return r->media->original_bytes;
  whence &= ~AVSEEK_FORCE;
  int64_t base = whence == SEEK_SET   ? 0
                 : whence == SEEK_CUR ? (int64_t)r->cursor
                 : whence == SEEK_END ? (int64_t)r->media->original_bytes
                                      : -1;
  if (base < 0 || offset < -base ||
      offset > (int64_t)r->media->original_bytes - base)
    return AVERROR(EINVAL);
  r->cursor = base + offset;
  return r->cursor;
}
static int reject_external_io(AVFormatContext *f, AVIOContext **pb,
                              const char *url, int flags,
                              AVDictionary **options) {
  (void)f;
  (void)pb;
  (void)url;
  (void)flags;
  (void)options;
  return AVERROR(EACCES);
}
static ds4_status av_status(int result) {
  return result >= 0                 ? DS4_STATUS_OK
         : result == AVERROR(ENOMEM) ? DS4_STATUS_NOMEM
                                     : DS4_STATUS_INVALID;
}
static void close_reader(reader *r) {
  av_frame_free(&r->frame);
  av_packet_free(&r->packet);
  avcodec_free_context(&r->codec);
  avformat_close_input(&r->format);
  if (r->io)
    av_freep(&r->io->buffer);
  avio_context_free(&r->io);
}
static ds4_status open_reader(reader *r, ds4_media *m, enum AVMediaType type) {
  *r = (reader){.media = m};
  r->format = avformat_alloc_context();
  uint8_t *buffer = av_malloc(32768);
  if (!r->format || !buffer) {
    av_free(buffer);
    return DS4_STATUS_NOMEM;
  }
  r->io =
      avio_alloc_context(buffer, 32768, 0, r, read_memory, NULL, seek_memory);
  if (!r->io) {
    av_free(buffer);
    return DS4_STATUS_NOMEM;
  }
  r->format->pb = r->io;
  r->format->flags |= AVFMT_FLAG_CUSTOM_IO;
  r->format->io_open = reject_external_io;
  r->format->max_streams = 16;
  r->format->probesize = 5u << 20;
  r->format->max_analyze_duration = AV_TIME_BASE;
  AVDictionary *options = NULL;
  av_dict_set(&options, "format_whitelist",
              "mov,matroska,webm,avi,nut,wav,flac,mp3,ogg,mpegts", 0);
  int result = avformat_open_input(&r->format, NULL, NULL, &options);
  av_dict_free(&options);
  if (result < 0)
    return av_status(result);
  if (!r->format->nb_streams || r->format->nb_streams > 16)
    return DS4_STATUS_INVALID;
  AVDictionary *stream_options[16] = {0};
  for (unsigned i = 0; i < r->format->nb_streams; i++) {
    av_dict_set_int(&stream_options[i], "max_pixels",
                    m->limits.pixels_per_frame, 0);
    av_dict_set(&stream_options[i], "threads", "1", 0);
  }
  result = avformat_find_stream_info(r->format, stream_options);
  for (unsigned i = 0; i < 16; i++)
    av_dict_free(stream_options + i);
  if (result < 0)
    return av_status(result);
  r->stream = -1;
  if (type == AVMEDIA_TYPE_AUDIO) {
    for (unsigned i = 0; i < r->format->nb_streams; i++)
      if (r->format->streams[i]->codecpar->codec_type == type) {
        r->stream = i;
        break;
      }
  } else
    r->stream = av_find_best_stream(r->format, type, -1, -1, NULL, 0);
  if (r->stream == AVERROR_DECODER_NOT_FOUND)
    return DS4_STATUS_UNSUPPORTED;
  if (r->stream < 0)
    return DS4_STATUS_INVALID;
  AVCodecParameters *p = r->format->streams[r->stream]->codecpar;
  if (type == AVMEDIA_TYPE_VIDEO &&
      (p->width <= 0 || p->height <= 0 ||
       (uint64_t)p->width * p->height > m->limits.pixels_per_frame))
    return DS4_STATUS_INVALID;
  const AVCodec *codec = avcodec_find_decoder(p->codec_id);
  if (!codec)
    return DS4_STATUS_UNSUPPORTED;
  r->codec = avcodec_alloc_context3(codec);
  r->packet = av_packet_alloc();
  r->frame = av_frame_alloc();
  if (!r->codec || !r->packet || !r->frame)
    return DS4_STATUS_NOMEM;
  result = avcodec_parameters_to_context(r->codec, p);
  if (result < 0)
    return av_status(result);
  r->codec->thread_count = 1;
  r->codec->max_pixels = m->limits.pixels_per_frame;
  r->codec->err_recognition = AV_EF_CAREFUL | AV_EF_EXPLODE;
  return av_status(avcodec_open2(r->codec, codec, NULL));
}
/* 1: frame, 0: clean EOF, negative libav error. No packet/frame queue is kept.
 */
static int next_frame(reader *r) {
  av_frame_unref(r->frame);
  for (;;) {
    int result = avcodec_receive_frame(r->codec, r->frame);
    if (result == 0)
      return 1;
    if (result == AVERROR_EOF)
      return 0;
    if (result != AVERROR(EAGAIN) || r->flushed)
      return result;
    do {
      av_packet_unref(r->packet);
      result = av_read_frame(r->format, r->packet);
    } while (result >= 0 && r->packet->stream_index != r->stream);
    if (result < 0 && result != AVERROR_EOF)
      return result;
    if (result == AVERROR_EOF) {
      r->flushed = true;
      result = avcodec_send_packet(r->codec, NULL);
    } else {
      if (r->packet->size < 0 ||
          (size_t)r->packet->size > r->media->original_bytes)
        return AVERROR_INVALIDDATA;
      result = avcodec_send_packet(r->codec, r->packet);
    }
    if (result < 0)
      return result;
  }
}
static ds4_status copy_rgb(reader *r, uint32_t slot, uint32_t source,
                           double timestamp, double processor_timestamp) {
  AVFrame *f = r->frame;
  ds4_media *m = r->media;
  if (f->color_trc == AVCOL_TRC_SMPTE2084 ||
      f->color_trc == AVCOL_TRC_ARIB_STD_B67 ||
      f->colorspace == AVCOL_SPC_BT2020_CL)
    return DS4_STATUS_UNSUPPORTED;
  if (f->width <= 0 || f->height <= 0 ||
      (uint64_t)f->width * f->height > m->limits.pixels_per_frame)
    return DS4_STATUS_INVALID;
  ds4_host_buffer pixels = {0};
  ds4_status status = ds4_host_alloc(
      m->resources, m->owner, (uint64_t)f->width * f->height * 3, &pixels);
  if (status)
    return status;
  struct SwsContext *sws = sws_getContext(
      f->width, f->height, f->format, f->width, f->height, AV_PIX_FMT_RGB24,
      SWS_BILINEAR | SWS_BITEXACT, NULL, NULL, NULL);
  if (!sws) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  const AVPixFmtDescriptor *format = av_pix_fmt_desc_get(f->format);
  if (!format) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  if (!(format->flags & AV_PIX_FMT_FLAG_RGB)) {
    /* sws_getContext alone defaults to BT.601 limited range. Honor the
     * decoded matrix/range, matching the explicit reference reformatter. */
    int space =
        f->colorspace == AVCOL_SPC_UNSPECIFIED ? SWS_CS_DEFAULT : f->colorspace;
    const int *coefficients = sws_getCoefficients(space);
    if (sws_setColorspaceDetails(sws, coefficients,
                                 f->color_range == AVCOL_RANGE_JPEG,
                                 coefficients, 1, 0, 1 << 16, 1 << 16) < 0) {
      status = DS4_STATUS_UNSUPPORTED;
      goto cleanup;
    }
  }
  uint8_t *dest[4] = {pixels.data, NULL, NULL, NULL};
  int stride[4] = {f->width * 3, 0, 0, 0};
  if (sws_scale(sws, (const uint8_t *const *)f->data, f->linesize, 0, f->height,
                dest, stride) != f->height) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  ds4_media_frame_input input = {pixels.data, f->width, f->height, 3,
                                 timestamp};
  status = media_frame_copy(m, slot, &input, source, processor_timestamp);
cleanup:
  sws_freeContext(sws);
  ds4_host_free(&pixels);
  return status;
}
static ds4_status decode_video(ds4_media *m,
                               const ds4_media_decode_options *o) {
  reader r = {0};
  ds4_host_buffer positions = {0}, times = {0}, selected = {0};
  ds4_status status;
  TRY(open_reader(&r, m, AVMEDIA_TYPE_VIDEO));
  AVStream *stream = r.format->streams[r.stream];
  double fps = av_q2d(stream->avg_frame_rate);
  if (!isfinite(fps) || fps < 0) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  double timebase = av_q2d(stream->time_base);
  for (unsigned i = 0; i < r.format->nb_streams; i++)
    m->view.has_audio_track |=
        r.format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
  uint32_t cap = m->limits.decoded_frames, count = 0, source_count = 0;
  TRY(ds4_host_alloc(m->resources, m->owner, (uint64_t)cap * 4, &positions));
  TRY(ds4_host_alloc(m->resources, m->owner, (uint64_t)cap * 8, &times));
  double previous = -INFINITY;
  int result;
  while ((result = next_frame(&r)) > 0) {
    AVFrame *f = r.frame;
    if (source_count == cap || f->best_effort_timestamp == AV_NOPTS_VALUE ||
        f->width <= 0 || f->height <= 0 ||
        (uint64_t)f->width * f->height > m->limits.pixels_per_frame) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    double timestamp = f->best_effort_timestamp * timebase;
    if (!isfinite(timestamp) || timestamp < previous) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    previous = timestamp;
    if (timestamp >= o->start_seconds &&
        (!o->end_seconds || timestamp < o->end_seconds)) {
      ((uint32_t *)positions.data)[count] = source_count;
      ((double *)times.data)[count++] = timestamp;
    }
    source_count++;
  }
  TRY(av_status(result));
  if (!count) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  /* The pinned processor samples by native frame index and average fps.
   * Keep actual PTS separately, including for variable-rate containers. */
  double duration = fps ? count / fps : previous - ((double *)times.data)[0];
  media_video_selection selection;
  TRY(media_video_plan(count, fps, duration, &o->video, cap, &selection));
  uint32_t keep = selection.selected;
  TRY(ds4_host_alloc(m->resources, m->owner, (uint64_t)keep * sizeof(uint32_t),
                     &selected));
  media_video_indices(&selection, selected.data);
  TRY(media_frame_allocate(m, keep));
  m->view.source_frames = source_count;
  m->view.source_fps = fps;
  m->view.duration = duration;
  m->view.video_recipe = o->video;
  m->view.audio_policy = o->video_audio;
  m->view.segment_start = ((double *)times.data)[0];
  m->view.segment_end =
      o->end_seconds ? o->end_seconds : previous + (fps ? 1 / fps : 0);
  close_reader(&r);
  TRY(open_reader(&r, m, AVMEDIA_TYPE_VIDEO));
  uint32_t decoded = 0, copied = 0;
  while (copied < keep && (result = next_frame(&r)) > 0) {
    while (copied < keep) {
      uint32_t segment_index = ((uint32_t *)selected.data)[copied];
      uint32_t original_index = ((uint32_t *)positions.data)[segment_index];
      if (original_index != decoded)
        break;
      TRY(copy_rgb(
          &r, copied, original_index, ((double *)times.data)[segment_index],
          fps ? original_index / fps : ((double *)times.data)[segment_index]));
      copied++;
    }
    decoded++;
  }
  if (copied != keep)
    status = DS4_STATUS_INVALID;
cleanup:
  close_reader(&r);
  ds4_host_free(&positions);
  ds4_host_free(&times);
  ds4_host_free(&selected);
  return status;
}
static ds4_status decode_audio(ds4_media *m,
                               const ds4_media_decode_options *o) {
  reader r = {0};
  SwrContext *swr = NULL;
  ds4_host_buffer samples = {0}, converted = {0};
  ds4_status status;
  TRY(open_reader(&r, m, AVMEDIA_TYPE_AUDIO));
  uint32_t rate = r.codec->sample_rate,
           channels = r.codec->ch_layout.nb_channels;
  if (!rate || rate > 384000 || !channels || channels > 8) {
    status = DS4_STATUS_UNSUPPORTED;
    goto cleanup;
  }
  uint64_t maximum = (uint64_t)ceil(m->limits.audio_seconds * rate);
  TRY(ds4_host_alloc(m->resources, m->owner, maximum * 4, &samples));
  int64_t cursor = 0;
  uint64_t used = 0, first_sample = 0;
  int64_t begin = (int64_t)floor(o->start_seconds * rate);
  int64_t end =
      o->end_seconds ? (int64_t)floor(o->end_seconds * rate) : INT64_MAX;
  AVRational time_base = r.format->streams[r.stream]->time_base;
  double tick = av_q2d(time_base) * rate;
  if (!isfinite(tick) || tick <= 0) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  bool started = false;
  int result;
  enum AVSampleFormat format = AV_SAMPLE_FMT_NONE;
  while ((result = next_frame(&r)) > 0) {
    AVFrame *f = r.frame;
    if ((uint32_t)f->sample_rate != rate ||
        (uint32_t)f->ch_layout.nb_channels != channels || f->nb_samples <= 0 ||
        (uint32_t)f->nb_samples > rate) {
      status = DS4_STATUS_UNSUPPORTED;
      goto cleanup;
    }
    if (f->best_effort_timestamp != AV_NOPTS_VALUE) {
      int64_t timestamp = av_rescale_q(f->best_effort_timestamp, time_base,
                                       (AVRational){1, rate});
      if (timestamp > 1e15 || timestamp < -1e15) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      if (!started)
        cursor = timestamp;
      else if (fabs((double)timestamp - cursor) > ceil(tick)) {
        /* Gaps need an explicit policy. Do not silently concatenate an audio
         * discontinuity and claim it belongs to a continuous video segment. */
        status = DS4_STATUS_UNSUPPORTED;
        goto cleanup;
      }
    }
    started = true;
    if (!swr) {
      format = f->format;
      TRY(av_status(swr_alloc_set_opts2(&swr, &f->ch_layout, AV_SAMPLE_FMT_FLT,
                                        rate, &f->ch_layout, format, rate, 0,
                                        NULL)));
      TRY(av_status(swr_init(swr)));
    }
    if (f->format != format) {
      status = DS4_STATUS_UNSUPPORTED;
      goto cleanup;
    }
    ds4_host_free(&converted);
    TRY(ds4_host_alloc(m->resources, m->owner,
                       (uint64_t)f->nb_samples * channels * 4, &converted));
    uint8_t *output = converted.data;
    int n = swr_convert(swr, &output, f->nb_samples,
                        (const uint8_t **)f->extended_data, f->nb_samples);
    if (n != f->nb_samples) {
      status = DS4_STATUS_INVALID;
      goto cleanup;
    }
    for (int i = 0; i < n; i++, cursor++) {
      if (cursor < begin)
        continue;
      if (cursor >= end)
        break;
      if (used == maximum) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      float sum = 0;
      for (uint32_t c = 0; c < channels; c++) {
        float value = ((float *)converted.data)[(uint64_t)i * channels + c];
        if (!isfinite(value)) {
          status = DS4_STATUS_INVALID;
          goto cleanup;
        }
        sum += value;
      }
      if (!isfinite(sum)) {
        status = DS4_STATUS_INVALID;
        goto cleanup;
      }
      if (!used)
        first_sample = cursor;
      ((float *)samples.data)[used++] = sum / channels;
    }
    if (cursor >= end)
      break;
  }
  TRY(av_status(result));
  if (!used) {
    status = DS4_STATUS_INVALID;
    goto cleanup;
  }
  double video_duration = m->view.duration;
  TRY(media_audio_convert(m, samples.data, used, 1, rate));
  if (m->view.type == DS4_MEDIA_VIDEO)
    m->view.duration = video_duration;
  else {
    m->view.segment_start = (double)first_sample / rate;
    m->view.segment_end = (double)(first_sample + used) / rate;
  }
  m->view.has_audio_track = true;
cleanup:
  swr_free(&swr);
  close_reader(&r);
  ds4_host_free(&samples);
  ds4_host_free(&converted);
  return status;
}
ds4_status media_decode_av(ds4_media *m, const ds4_media_decode_options *o) {
  ds4_status status = DS4_STATUS_OK;
  if (m->view.type == DS4_MEDIA_VIDEO) {
    status = decode_video(m, o);
    if (!status && o->video_audio == DS4_MEDIA_AUDIO_FIRST)
      status = decode_audio(m, o);
  } else if (m->view.type == DS4_MEDIA_AUDIO)
    status = decode_audio(m, o);
  else
    status = DS4_STATUS_INVALID;
  return status;
}
