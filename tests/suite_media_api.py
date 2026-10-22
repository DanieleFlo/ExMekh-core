"""ctypes bindings for the host-only media/preprocessor qualification tools."""

import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Limits(C.Structure):
    _fields_ = [
        ("encoded_bytes", C.c_uint64),
        ("retained_bytes", C.c_uint64),
        ("pixels_per_frame", C.c_uint32),
        ("decoded_frames", C.c_uint32),
        ("audio_seconds", C.c_double),
    ]


class VideoRecipe(C.Structure):
    _fields_ = [("fps", C.c_double), ("max_frames", C.c_uint32), ("overflow", C.c_int)]


class DecodeOptions(C.Structure):
    _fields_ = [
        ("limits", Limits),
        ("video", VideoRecipe),
        ("video_audio", C.c_int),
        ("start_seconds", C.c_double),
        ("end_seconds", C.c_double),
    ]


class FrameInput(C.Structure):
    _fields_ = [
        ("pixels", C.c_void_p),
        ("width", C.c_uint32),
        ("height", C.c_uint32),
        ("channels", C.c_uint32),
        ("timestamp", C.c_double),
    ]


class FrameView(C.Structure):
    _fields_ = [
        ("rgb", C.POINTER(C.c_uint8)),
        ("width", C.c_uint32),
        ("height", C.c_uint32),
        ("source_index", C.c_uint32),
        ("timestamp", C.c_double),
        ("processor_timestamp", C.c_double),
    ]


class MediaView(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("frames", C.c_uint32),
        ("source_frames", C.c_uint32),
        ("source_fps", C.c_double),
        ("duration", C.c_double),
        ("audio_samples", C.c_uint64),
        ("audio", C.POINTER(C.c_float)),
        ("has_audio_track", C.c_bool),
        ("audio_policy", C.c_int),
        ("video_recipe", VideoRecipe),
        ("source_sha256", C.c_uint8 * 32),
        ("decoded_sha256", C.c_uint8 * 32),
        ("segment_start", C.c_double),
        ("segment_end", C.c_double),
    ]


class PreparedView(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("frames", C.c_uint32),
        ("max_patches", C.c_uint32),
        ("soft_token_budget", C.c_uint32),
        ("soft_tokens", C.c_uint32),
        ("audio_frames", C.c_uint32),
        ("audio_soft_tokens", C.c_uint32),
        ("patches", C.POINTER(C.c_float)),
        ("positions", C.POINTER(C.c_int32)),
        ("frame_soft_tokens", C.POINTER(C.c_uint32)),
        ("audio_features", C.POINTER(C.c_float)),
        ("audio_mask", C.POINTER(C.c_uint8)),
        ("timestamps", C.POINTER(C.c_double)),
        ("processor_timestamps", C.POINTER(C.c_double)),
        ("host_bytes", C.c_uint64),
        ("key", C.c_uint8 * 32),
    ]


class MediaRecipe(C.Structure):
    _fields_ = [("soft_tokens", C.c_uint32)]


class CacheStats(C.Structure):
    _fields_ = [
        ("hits", C.c_uint64),
        ("misses", C.c_uint64),
        ("bytes", C.c_uint64),
        ("limit", C.c_uint64),
        ("entries", C.c_uint32),
    ]


class Estimate(C.Structure):
    _fields_ = [
        ("device_bytes", C.c_uint64),
        ("host_bytes", C.c_uint64),
        ("pinned_bytes", C.c_uint64),
        ("checkpoint_bytes", C.c_uint64),
        ("known", C.c_bool),
    ]


class VisualRequest(C.Structure):
    _fields_ = [
        ("prompt", C.c_char_p),
        ("task", C.c_uint32),
        ("media", C.POINTER(C.c_void_p)),
        ("media_count", C.c_size_t),
        ("add_timestamps", C.c_bool),
    ]


class Budget(C.Structure):
    _fields_ = [
        ("limit", C.c_uint64),
        ("margin", C.c_uint64),
        ("global_free", C.c_uint64),
        ("global_known", C.c_bool),
    ]


class Content(C.Structure):
    _fields_ = [("text", C.c_char_p), ("media", C.c_void_p)]


class Segment(C.Structure):
    _fields_ = [
        ("id", C.c_uint64),
        ("start_seconds", C.c_double),
        ("end_seconds", C.c_double),
    ]


class MultimodalRequest(C.Structure):
    _fields_ = [
        ("content", C.POINTER(Content)),
        ("count", C.c_size_t),
        ("task", C.c_uint32),
        ("add_timestamps", C.c_bool),
        ("segment", Segment),
    ]


def load_library():
    lib = C.CDLL(
        os.environ.get("DS4_SUITE_CPU_LIBRARY", str(ROOT / "tests/libds4_suite_cpu.so"))
    )
    ptr, out = C.c_void_p, C.POINTER(C.c_void_p)
    signatures = {
        "ds4_resources_create": ([out], C.c_int),
        "ds4_resources_close": ([ptr], C.c_int),
        "ds4_bundle_open": ([out, C.c_char_p, C.c_char_p, ptr, C.c_uint64], C.c_int),
        "ds4_bundle_close": ([ptr], C.c_int),
        "ds4_media_default_limits": ([], Limits),
        "ds4_media_image_pixels": (
            [out, ptr, C.c_uint64, C.POINTER(FrameInput), C.POINTER(Limits)],
            C.c_int,
        ),
        "ds4_media_video_frames": (
            [
                out,
                ptr,
                C.c_uint64,
                C.POINTER(FrameInput),
                C.c_size_t,
                C.c_double,
                C.c_double,
                C.POINTER(VideoRecipe),
                C.POINTER(Limits),
            ],
            C.c_int,
        ),
        "ds4_media_audio_samples": (
            [
                out,
                ptr,
                C.c_uint64,
                ptr,
                C.c_uint64,
                C.c_uint32,
                C.c_uint32,
                C.POINTER(Limits),
            ],
            C.c_int,
        ),
        "ds4_media_decode": (
            [out, ptr, C.c_uint64, C.c_int, ptr, C.c_size_t, C.POINTER(DecodeOptions)],
            C.c_int,
        ),
        "ds4_media_data": ([ptr], MediaView),
        "ds4_media_frame": ([ptr, C.c_uint32, C.POINTER(FrameView)], C.c_int),
        "ds4_media_free": ([ptr], None),
        "ds4_gemma_processor_open": (
            [out, ptr, ptr, C.c_uint64, C.c_char_p, C.c_uint64],
            C.c_int,
        ),
        "ds4_gemma_processor_close": ([ptr], None),
        "ds4_gemma_processor_clear": ([ptr], None),
        "ds4_gemma_processor_cache": ([ptr], CacheStats),
        "ds4_gemma_media_prepare": ([ptr, ptr, C.POINTER(MediaRecipe), out], C.c_int),
        "ds4_gemma_media_plan": (
            [
                ptr,
                ptr,
                C.POINTER(MediaRecipe),
                C.POINTER(Estimate),
                C.POINTER(C.c_uint32),
            ],
            C.c_int,
        ),
        "ds4_gemma_media_data": ([ptr], PreparedView),
        "ds4_gemma_media_free": ([ptr], None),
        "ds4_runtime_create": ([out, ptr, ptr], C.c_int),
        "ds4_runtime_release": ([ptr], None),
        "ds4_numeric_cpu": ([out, ptr, C.c_uint64], C.c_int),
        "ds4_numeric_close": ([ptr], C.c_int),
        "ds4_numeric_memory": (
            [ptr, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)],
            None,
        ),
        "ds4_resources_set_budget": ([ptr, C.c_int, C.POINTER(Budget)], C.c_int),
        "ds4_resources_get_budget": ([ptr, C.c_int, C.POINTER(Budget)], C.c_int),
        "ds4_gemma_visual_open": ([out, ptr, ptr, ptr, C.c_char_p, ptr], C.c_int),
        "ds4_gemma_visual_close": ([ptr], C.c_int),
        "ds4_gemma_visual_prepare": (
            [ptr, C.POINTER(VisualRequest), C.c_size_t, out],
            C.c_int,
        ),
        "ds4_gemma_visual_input_free": ([ptr], C.c_int),
        "ds4_gemma_multimodal_prepare": (
            [ptr, C.POINTER(MultimodalRequest), C.c_size_t, out],
            C.c_int,
        ),
        "ds4_gemma_multimodal_key": ([ptr, C.c_size_t, C.c_uint32, ptr], C.c_int),
        "ds4_gemma_multimodal_segment": (
            [ptr, C.c_size_t, C.POINTER(Segment)],
            C.c_int,
        ),
        "ds4_gemma_visual_input_tokens": (
            [ptr, C.c_size_t, C.POINTER(C.POINTER(C.c_uint32)), C.POINTER(C.c_uint32)],
            C.c_int,
        ),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def codec_rgb_reference(yuv, width, height, space, color_range):
    """Independent wrapper around the pinned codec's public conversion API.

    PyAV is only a fixture writer: its bundled FFmpeg 8 SIMD conversion has
    different integer rounding from the qualified FFmpeg 7.1.1 C build.
    """
    import numpy as np

    prefix = Path(
        os.environ.get(
            "DS4_MEDIA_CODEC_PREFIX",
            ROOT
            / "performance-results/native-suite-t13-t14-20261021/dependencies/installed",
        )
    )
    lib = C.CDLL(str(prefix / "lib/libswscale.so"))
    ptr = C.c_void_p
    lib.sws_getContext.argtypes = [C.c_int] * 7 + [ptr] * 3
    lib.sws_getContext.restype = ptr
    lib.sws_getCoefficients.argtypes = [C.c_int]
    lib.sws_getCoefficients.restype = C.POINTER(C.c_int)
    lib.sws_setColorspaceDetails.argtypes = [
        ptr,
        C.POINTER(C.c_int),
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
    ]
    lib.sws_scale.argtypes = [
        ptr,
        C.POINTER(ptr),
        C.POINTER(C.c_int),
        C.c_int,
        C.c_int,
        C.POINTER(ptr),
        C.POINTER(C.c_int),
    ]
    lib.sws_freeContext.argtypes = [ptr]
    context = lib.sws_getContext(
        width, height, 0, width, height, 2, 2 | 0x80000, None, None, None
    )
    if not context:
        raise RuntimeError("reference swscale allocation")
    try:
        coeff = lib.sws_getCoefficients(space)
        assert (
            lib.sws_setColorspaceDetails(
                context, coeff, int(color_range == 2), coeff, 1, 0, 1 << 16, 1 << 16
            )
            == 0
        )
        result = np.empty((height, width, 3), np.uint8)
        size = width * height
        planes = (ptr * 4)(
            yuv.ctypes.data,
            yuv.ctypes.data + size,
            yuv.ctypes.data + size + size // 4,
            None,
        )
        strides = (C.c_int * 4)(width, width // 2, width // 2, 0)
        dest = (ptr * 4)(result.ctypes.data, None, None, None)
        output_strides = (C.c_int * 4)(width * 3, 0, 0, 0)
        assert (
            lib.sws_scale(context, planes, strides, 0, height, dest, output_strides)
            == height
        )
        return result
    finally:
        lib.sws_freeContext(context)
