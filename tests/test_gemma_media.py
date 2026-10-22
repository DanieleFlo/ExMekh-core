"""Native decode, sampling and processor boundaries against frozen references."""

import atexit
import ctypes as C
import io
import json
import os
import sys
import tempfile
import unittest
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tests.suite_media_api import (
    ROOT,
    Budget,
    Content,
    DecodeOptions,
    Estimate,
    FrameInput,
    FrameView,
    MediaRecipe,
    MultimodalRequest,
    Segment,
    VideoRecipe,
    VisualRequest,
    codec_rgb_reference,
    load_library,
)

BASE = ROOT / "performance-results/native-suite-t13-t14-20261021"
PREPARED = ROOT / "performance-results/native-suite-oracles-20261020"
METRICS = []


def record_metrics(name, actual, expected, **parameters):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    delta = a - b
    metrics = {
        "name": name,
        **parameters,
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(b.ravel()), 1e-12)
        ),
    }
    assert metrics["max_abs"] <= 2e-5 and metrics["relative_l2"] <= 1e-5, metrics
    METRICS.append(metrics)


@atexit.register
def save_metrics():
    path = os.environ.get("DS4_MEDIA_METRICS")
    if path:
        Path(path).write_text(json.dumps(METRICS, indent=2) + "\n")


def reference_processor():
    lock = json.loads((PREPARED / "prepared.json").read_text())
    source = next(
        s for s in lock["sources"] if s["repository"] == "huggingface/transformers"
    )
    sys.path.insert(
        0, str(PREPARED / "sources" / f"transformers-{source['revision']}" / "src")
    )
    sys.path.insert(0, str(PREPARED / "deps"))
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        PREPARED / "models/google/embeddinggemma-2", local_files_only=True
    )


class _MediaCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def setUp(self):
        self.resources = C.c_void_p()
        self.assertEqual(self.lib.ds4_resources_create(C.byref(self.resources)), 0)
        self.limits = self.lib.ds4_media_default_limits()
        self.handles = []

    def tearDown(self):
        for handle in reversed(self.handles):
            self.lib.ds4_media_free(handle)
        self.assertEqual(self.lib.ds4_resources_close(self.resources), 0)

    def pixels(self, image):
        h, w, c = image.shape
        frame = FrameInput(image.ctypes.data, w, h, c, 0)
        out = C.c_void_p()
        self.assertEqual(
            self.lib.ds4_media_image_pixels(
                C.byref(out), self.resources, 81, C.byref(frame), C.byref(self.limits)
            ),
            0,
        )
        self.handles.append(out)
        return out

    def decode(self, data, kind, options=None, expected=0):
        out = C.c_void_p(1)
        options = options or DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 1, 0, 0)
        status = self.lib.ds4_media_decode(
            C.byref(out),
            self.resources,
            81,
            kind,
            C.c_char_p(data),
            len(data),
            C.byref(options),
        )
        self.assertEqual(status, expected)
        if status:
            self.assertIsNone(out.value)
        else:
            self.handles.append(out)
        return out

    def frame(self, media, index=0):
        f = FrameView()
        self.assertEqual(self.lib.ds4_media_frame(media, index, C.byref(f)), 0)
        rgb = (
            np.ctypeslib.as_array(f.rgb, shape=(f.height * f.width * 3,))
            .reshape(f.height, f.width, 3)
            .copy()
        )
        return f, rgb

    def audio(self, x, rate):
        out = C.c_void_p()
        channels = 1 if x.ndim == 1 else x.shape[1]
        self.assertEqual(
            self.lib.ds4_media_audio_samples(
                C.byref(out),
                self.resources,
                81,
                x.ctypes.data,
                len(x),
                channels,
                rate,
                C.byref(self.limits),
            ),
            0,
        )
        self.handles.append(out)
        return out


class Media(_MediaCase):
    def test_alpha_copy_and_encoded_exif(self):
        rng = np.random.default_rng(15)
        rgba = rng.integers(0, 256, (39, 57, 4), np.uint8)
        rgba[..., 3] = 0
        raw = self.pixels(rgba)
        np.testing.assert_array_equal(self.frame(raw)[1], rgba[..., :3])
        for mode in ("PNG", "JPEG"):
            for orientation in range(1, 9):
                image = Image.fromarray(rgba[..., :3])
                exif = Image.Exif()
                exif[274] = orientation
                encoded = io.BytesIO()
                image.save(
                    encoded,
                    format=mode,
                    exif=exif,
                    **({"quality": 95, "subsampling": 0} if mode == "JPEG" else {}),
                )
                media = self.decode(encoded.getvalue(), 1)
                expected = np.array(
                    ImageOps.exif_transpose(
                        Image.open(io.BytesIO(encoded.getvalue()))
                    ).convert("RGB")
                )
                with self.subTest(codec=mode, orientation=orientation):
                    np.testing.assert_array_equal(self.frame(media)[1], expected)

    def test_resampling_channels_and_finite_validation(self):
        sys.path.insert(0, str(PREPARED / "deps"))
        import soxr

        for rate in (8000, 16000, 22050, 44100, 48000):
            t = np.arange(rate + 17, dtype=np.float32) / rate
            x = np.stack(
                (0.2 * np.sin(t * 440 * 2 * np.pi), 0.1 * np.cos(t * 370 * 2 * np.pi)),
                -1,
            ).astype(np.float32)
            mono = x.mean(axis=-1, dtype=np.float32)
            reference = (
                mono
                if rate == 16000
                else soxr.resample(mono, rate, 16000, quality="HQ")
            )
            count = len(reference)
            media = self.audio(x, rate)
            view = self.lib.ds4_media_data(media)
            actual = np.ctypeslib.as_array(view.audio, shape=(view.audio_samples,))
            with self.subTest(rate=rate):
                self.assertEqual(len(actual), count)
                np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
                record_metrics("resample", actual, reference, rate=rate, samples=len(x))
        x = np.array([0, np.nan], np.float32)
        out = C.c_void_p(1)
        self.assertNotEqual(
            self.lib.ds4_media_audio_samples(
                C.byref(out),
                self.resources,
                81,
                x.ctypes.data,
                len(x),
                1,
                16000,
                C.byref(self.limits),
            ),
            0,
        )
        self.assertIsNone(out.value)

    def test_encoded_audio_segment_and_duration_limits(self):
        rate = 16000
        x = (np.sin(np.arange(rate) * 0.03) * 16384).astype("<i2")
        stereo = np.stack((x, -x // 2), -1)
        encoded = io.BytesIO()
        with wave.open(encoded, "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(stereo.tobytes())
        options = DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 1, 0.2, 0.6)
        media = self.decode(encoded.getvalue(), 3, options)
        v = self.lib.ds4_media_data(media)
        actual = np.ctypeslib.as_array(v.audio, shape=(v.audio_samples,))
        expected = (stereo.astype(np.float32) / 32768).mean(-1)[3200:9600]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(v.segment_start, 0.2)
        self.assertEqual(v.segment_end, 0.6)
        options.limits.audio_seconds = 0.1
        self.decode(encoded.getvalue(), 3, options, expected=1)

    def test_video_color_matrix_and_range(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        for space in (1, 5, 9):
            for color_range in (1, 2):
                encoded = io.BytesIO()
                yuv = np.full((72, 96), 128, np.uint8)
                yuv[:48, :48] = 64
                yuv[:48, 48:] = 190
                yuv[48:60] = 90
                yuv[60:] = 220
                with av.open(encoded, "w", format="matroska") as container:
                    stream = container.add_stream("ffv1", rate=1)
                    stream.width, stream.height, stream.pix_fmt = 96, 48, "yuv420p"
                    stream.codec_context.colorspace = space
                    stream.codec_context.color_range = color_range
                    frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
                    frame.colorspace, frame.color_range = space, color_range
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                data = encoded.getvalue()
                expected = codec_rgb_reference(yuv, 96, 48, space, color_range)
                media = self.decode(data, 2)
                with self.subTest(space=space, color_range=color_range):
                    np.testing.assert_array_equal(self.frame(media)[1], expected)

    def test_video_audio_policy_and_explicit_segment(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        encoded = io.BytesIO()
        wave_data = (np.sin(np.arange(16000) * 0.03) * 16000).astype(np.int16)
        with av.open(encoded, "w", format="matroska") as container:
            video = container.add_stream("ffv1", rate=10)
            video.width, video.height, video.pix_fmt = 48, 48, "bgr0"
            audio = container.add_stream("pcm_s16le", rate=16000)
            audio.layout = "mono"
            for i in range(10):
                frame = av.VideoFrame.from_ndarray(
                    np.full((48, 48, 3), i * 20, np.uint8), format="rgb24"
                )
                frame.pts = i
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            frame = av.AudioFrame.from_ndarray(
                wave_data[None], format="s16", layout="mono"
            )
            frame.sample_rate, frame.pts = 16000, 0
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
        options = DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 1, 0.2, 0.6)
        excluded = self.lib.ds4_media_data(self.decode(encoded.getvalue(), 2, options))
        self.assertTrue(excluded.has_audio_track)
        self.assertEqual(excluded.audio_samples, 0)
        options.video_audio = 2
        selected = self.decode(encoded.getvalue(), 2, options)
        v = self.lib.ds4_media_data(selected)
        self.assertEqual(v.frames, 4)
        self.assertEqual(v.audio_samples, 6400)
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(v.audio, shape=(6400,)),
            wave_data[3200:9600].astype(np.float32) / 32768,
        )
        for i in range(4):
            f, rgb = self.frame(selected, i)
            self.assertAlmostEqual(f.timestamp, (i + 2) / 10)
            np.testing.assert_array_equal(
                rgb, np.full((48, 48, 3), (i + 2) * 20, np.uint8)
            )

    def test_frame_recipe_order_and_limits(self):
        frames = [np.full((48, 96, 3), i * 7, np.uint8) for i in range(15)]
        inputs = (FrameInput * len(frames))(
            *(FrameInput(f.ctypes.data, 96, 48, 3, i / 4) for i, f in enumerate(frames))
        )
        out = C.c_void_p()
        recipe = VideoRecipe(2, 4, 0)
        self.assertEqual(
            self.lib.ds4_media_video_frames(
                C.byref(out),
                self.resources,
                81,
                inputs,
                len(frames),
                4,
                len(frames) / 4,
                C.byref(recipe),
                C.byref(self.limits),
            ),
            0,
        )
        self.handles.append(out)
        expected = [0, 4, 8, 12]
        for i, index in enumerate(expected):
            view, pixels = self.frame(out, i)
            self.assertEqual(view.source_index, index)
            self.assertEqual(view.timestamp, index / 4)
            np.testing.assert_array_equal(pixels, frames[index])
        self.limits.pixels_per_frame = 4
        rejected = C.c_void_p(1)
        self.assertNotEqual(
            self.lib.ds4_media_image_pixels(
                C.byref(rejected),
                self.resources,
                81,
                C.byref(inputs[0]),
                C.byref(self.limits),
            ),
            0,
        )
        self.assertIsNone(rejected.value)

    def test_raw_and_encoded_video_share_sampling_admission(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        pixels = np.full((2, 48, 48, 3), 37, np.uint8)
        inputs = (FrameInput * 2)(
            *(FrameInput(f.ctypes.data, 48, 48, 3, i) for i, f in enumerate(pixels))
        )
        encoded = io.BytesIO()
        with av.open(encoded, "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=1)
            stream.width, stream.height, stream.pix_fmt = 48, 48, "bgr0"
            for i, rgb in enumerate(pixels):
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = i
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.limits.decoded_frames = 8
        for fps, cap, expected in (
            (2, 0, [0, 0, 1, 1]),
            (1.2, 0, [0, 0]),
            (0, 100, [0, 1]),
            (float.fromhex("0x0.0000000000001p-1022"), 0, [0]),
            (float("nan"), 0, None),
            (float("inf"), 0, None),
            (-1, 0, None),
            (1e308, 8, None),
        ):
            with self.subTest(fps=fps, cap=cap):
                recipe = VideoRecipe(fps, cap, 0)
                raw = C.c_void_p(1)
                status = 0 if expected is not None else 1
                self.assertEqual(
                    self.lib.ds4_media_video_frames(
                        C.byref(raw),
                        self.resources,
                        81,
                        inputs,
                        2,
                        1,
                        2,
                        C.byref(recipe),
                        C.byref(self.limits),
                    ),
                    status,
                )
                if not status:
                    self.handles.append(raw)
                else:
                    self.assertIsNone(raw.value)
                decoded = self.decode(
                    encoded.getvalue(),
                    2,
                    DecodeOptions(self.limits, recipe, 1, 0, 0),
                    expected=status,
                )
                if status:
                    continue
                for media in (raw, decoded):
                    self.assertEqual(
                        self.lib.ds4_media_data(media).frames, len(expected)
                    )
                    self.assertEqual(
                        [
                            self.frame(media, i)[0].source_index
                            for i in range(len(expected))
                        ],
                        expected,
                    )

    def test_audio_timestamp_offset_and_discontinuity(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        x = (np.sin(np.arange(16000) * 0.03) * 16000).astype(np.int16)

        def encode(parts):
            data = io.BytesIO()
            with av.open(data, "w", format="matroska") as container:
                stream = container.add_stream("pcm_s16le", rate=16000)
                stream.layout = "mono"
                for samples, pts in parts:
                    frame = av.AudioFrame.from_ndarray(
                        samples[None], format="s16", layout="mono"
                    )
                    frame.sample_rate, frame.pts = 16000, pts
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            return data.getvalue()

        options = DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 1, 0.2, 0.6)
        media = self.decode(encode([(x, 1600)]), 3, options)
        v = self.lib.ds4_media_data(media)
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(v.audio, shape=(v.audio_samples,)),
            x[1600:8000].astype(np.float32) / 32768,
        )
        self.assertEqual(v.segment_start, 0.2)
        self.assertEqual(v.segment_end, 0.6)
        options.start_seconds = options.end_seconds = 0
        self.decode(encode([(x[:8000], 0), (x[8000:], 12000)]), 3, options, expected=2)

    def test_decode_host_admission_releases_on_failure(self):
        encoded = io.BytesIO()
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(encoded, format="PNG")
        denied = Budget(1024, 0, 0, False)
        original = Budget()
        self.assertEqual(
            self.lib.ds4_resources_get_budget(self.resources, 3, C.byref(original)), 0
        )
        self.assertEqual(
            self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(denied)), 0
        )
        self.decode(encoded.getvalue(), 1, expected=6)
        self.assertEqual(
            self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(original)), 0
        )
        self.decode(encoded.getvalue(), 1)

    def test_malformed_container_and_expansion_limits(self):
        for kind in (1, 2, 3):
            self.decode(b"invalid input", kind, expected=1)
        encoded = io.BytesIO()
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(encoded, format="PNG")
        opts = DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 1, 0, 0)
        opts.limits.pixels_per_frame = 16
        self.decode(encoded.getvalue(), 1, opts, expected=1)

    def test_video_container_vfr_and_audio_selection(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        times = [0, 250, 800, 1400, 1600, 2700, 4000, 4500]
        source = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vfr.mkv"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("ffv1", rate=2)
                stream.width, stream.height, stream.pix_fmt = 96, 48, "bgr0"
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.time_base = Fraction(1, 1000)
                for i, timestamp in enumerate(times):
                    pixels = np.zeros((48, 96, 3), np.uint8)
                    pixels[..., i % 3] = i * 29
                    source.append(pixels)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            data = path.read_bytes()
        opts = DecodeOptions(self.limits, VideoRecipe(1, 32, 0), 1, 0, 0)
        media = self.decode(data, 2, opts)
        view = self.lib.ds4_media_data(media)
        self.assertEqual(view.source_frames, len(source))
        expected = [0, 2, 4, 6]
        self.assertEqual(view.frames, len(expected))
        for i, index in enumerate(expected):
            frame, pixels = self.frame(media, i)
            self.assertAlmostEqual(frame.timestamp, times[index] / 1000)
            self.assertAlmostEqual(frame.processor_timestamp, index / 2)
            np.testing.assert_array_equal(pixels, source[index])
        opts.video_audio = 0
        self.decode(data, 2, opts, expected=1)
        opts.video_audio = 2
        self.decode(data, 2, opts, expected=1)


class Processor(_MediaCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not (BASE / "gemma-media.ds4p").exists():
            raise unittest.SkipTest(
                "compile the frozen Gemma media processor asset first"
            )
        cls.reference = reference_processor()

    def setUp(self):
        super().setUp()
        self.bundle, self.processor = C.c_void_p(), C.c_void_p()
        self.assertEqual(
            self.lib.ds4_bundle_open(
                C.byref(self.bundle),
                os.fsencode(
                    ROOT
                    / "performance-results/native-suite-t05-t06-20261020/bundles/embeddinggemma-2-bf16/bundle.ds4b"
                ),
                os.fsencode(PREPARED / "models"),
                self.resources,
                82,
            ),
            0,
        )
        self.assertEqual(
            self.lib.ds4_gemma_processor_open(
                C.byref(self.processor),
                self.bundle,
                self.resources,
                83,
                os.fsencode(BASE / "gemma-media.ds4p"),
                24 << 20,
            ),
            0,
        )
        self.prepared = []

    def tearDown(self):
        for p in self.prepared:
            self.lib.ds4_gemma_media_free(p)
        self.lib.ds4_gemma_processor_close(self.processor)
        self.assertEqual(self.lib.ds4_bundle_close(self.bundle), 0)
        super().tearDown()

    def prepare(self, media, budget=0):
        p = C.c_void_p()
        recipe = MediaRecipe(budget)
        estimate, tokens = Estimate(), C.c_uint32()
        self.assertEqual(
            self.lib.ds4_gemma_media_plan(
                self.processor,
                media,
                C.byref(recipe),
                C.byref(estimate),
                C.byref(tokens),
            ),
            0,
        )
        self.assertEqual(
            self.lib.ds4_gemma_media_prepare(
                self.processor, media, C.byref(recipe), C.byref(p)
            ),
            0,
        )
        self.prepared.append(p)
        view = self.lib.ds4_gemma_media_data(p)
        self.assertEqual(tokens.value, view.soft_tokens)
        self.assertGreaterEqual(estimate.host_bytes, view.host_bytes)
        self.assertEqual(estimate.device_bytes, 0)
        return view

    def test_patch_pixels_positions_and_budgets(self):
        rng = np.random.default_rng(18)
        for h, w in ((96, 144), (64, 64), (255, 321), (1, 1024), (1024, 1), (48, 5376)):
            pixels = rng.integers(0, 256, (h, w, 3), np.uint8)
            media = self.pixels(pixels)
            for budget in (70, 140, 280, 560, 1120):
                v = self.prepare(media, budget)
                expected = self.reference.image_processor(
                    pixels,
                    max_soft_tokens=budget,
                    input_data_format="channels_last",
                    return_tensors="np",
                )
                actual = np.ctypeslib.as_array(
                    v.patches, shape=(v.frames * v.max_patches * 768,)
                ).reshape(1, v.max_patches, 768)
                positions = np.ctypeslib.as_array(
                    v.positions, shape=(v.frames * v.max_patches * 2,)
                ).reshape(1, v.max_patches, 2)
                with self.subTest(shape=(h, w), budget=budget):
                    np.testing.assert_array_equal(actual, expected["pixel_values"])
                    np.testing.assert_array_equal(
                        positions, expected["image_position_ids"]
                    )
                    self.assertEqual(
                        v.soft_tokens, expected["num_soft_tokens_per_image"][0]
                    )

    def test_audio_spectrogram_masks_and_token_counts(self):
        for n in (160, 161, 319, 16000, 16017, 32000, 480000):
            cases = [
                np.zeros(n, np.float32),
                (0.2 * np.sin(np.arange(n) * 2 * np.pi * 440 / 16000)).astype(
                    np.float32
                ),
            ]
            impulse = np.zeros(n, np.float32)
            impulse[n // 2] = 1
            cases.append(impulse)
            for x in cases:
                v = self.prepare(self.audio(x, 16000))
                expected = self.reference.feature_extractor([x], return_tensors="np")
                actual = np.ctypeslib.as_array(
                    v.audio_features, shape=(v.audio_frames * 128,)
                ).reshape(v.audio_frames, 128)
                mask = np.ctypeslib.as_array(v.audio_mask, shape=(v.audio_frames,))
                with self.subTest(samples=n, signal=int(np.count_nonzero(x))):
                    np.testing.assert_array_equal(
                        mask.astype(bool), expected["input_features_mask"][0]
                    )
                    np.testing.assert_allclose(
                        actual, expected["input_features"][0], atol=2e-5, rtol=1e-5
                    )
                    record_metrics(
                        "audio-features",
                        actual,
                        expected["input_features"][0],
                        samples=n,
                        nonzero=int(np.count_nonzero(x)),
                    )
                    self.assertEqual(
                        v.audio_soft_tokens,
                        int(expected["input_features_mask"][0][::4].sum()),
                    )

    def test_video_audio_plan_checks_complete_context(self):
        sys.path.insert(0, str(BASE / "development"))
        import av

        encoded = io.BytesIO()
        with av.open(encoded, "w", format="matroska") as container:
            video = container.add_stream("ffv1", rate=1)
            video.width, video.height, video.pix_fmt = 48, 48, "bgr0"
            audio = container.add_stream("pcm_s16le", rate=16000)
            audio.layout = "mono"
            for i in range(31):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((48, 48, 3), np.uint8), format="rgb24"
                )
                frame.pts = i
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            for i in range(30):
                frame = av.AudioFrame.from_ndarray(
                    np.zeros((1, 16000), np.int16), format="s16", layout="mono"
                )
                frame.sample_rate, frame.pts = 16000, i * 16000
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
        options = DecodeOptions(self.limits, VideoRecipe(0, 0, 0), 2, 0, 0)
        media = self.decode(encoded.getvalue(), 2, options)
        recipe = MediaRecipe(280)
        estimate, tokens = Estimate(), C.c_uint32(99)
        self.assertEqual(
            self.lib.ds4_gemma_media_plan(
                self.processor,
                media,
                C.byref(recipe),
                C.byref(estimate),
                C.byref(tokens),
            ),
            1,
        )
        self.assertFalse(estimate.known)
        self.assertEqual(tokens.value, 0)
        prepared = C.c_void_p(1)
        self.assertEqual(
            self.lib.ds4_gemma_media_prepare(
                self.processor, media, C.byref(recipe), C.byref(prepared)
            ),
            1,
        )
        self.assertIsNone(prepared.value)

    def test_cache_keys_and_borrowed_lifetime(self):
        source = np.zeros((96, 144, 4), np.uint8)
        a = self.prepare(self.pixels(source))
        b = self.prepare(self.pixels(source))
        self.assertEqual(bytes(a.key), bytes(b.key))
        self.assertEqual(self.lib.ds4_gemma_processor_cache(self.processor).hits, 1)
        c = self.prepare(self.handles[-1], 140)
        self.assertNotEqual(bytes(a.key), bytes(c.key))
        source[0, 0, 0] = 255
        d = self.prepare(self.pixels(source))
        self.assertNotEqual(bytes(a.key), bytes(d.key))
        self.lib.ds4_gemma_processor_clear(self.processor)
        self.assertEqual(self.lib.ds4_gemma_processor_cache(self.processor).bytes, 0)
        self.assertEqual(a.frames, 1)
        self.assertTrue(
            np.isfinite(
                np.ctypeslib.as_array(a.patches, shape=(a.max_patches * 768,))
            ).all()
        )

    def test_cache_eviction_and_asset_validation(self):
        for i in range(9):
            self.prepare(self.pixels(np.full((96, 144, 3), i, np.uint8)))
            stats = self.lib.ds4_gemma_processor_cache(self.processor)
            self.assertLessEqual(stats.bytes, stats.limit)
            self.assertLessEqual(stats.entries, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.ds4p"
            data = bytearray((BASE / "gemma-media.ds4p").read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            out = C.c_void_p(1)
            self.assertNotEqual(
                self.lib.ds4_gemma_processor_open(
                    C.byref(out), self.bundle, self.resources, 83, os.fsencode(path), 0
                ),
                0,
            )
            self.assertIsNone(out.value)

    def test_typed_multimodal_order_identity_segments_and_admission(self):
        self.prepare(self.pixels(np.zeros((96, 144, 3), np.uint8)))
        image = self.prepared[-1]
        self.prepare(self.audio(np.zeros(16017, np.float32), 16000))
        sound = self.prepared[-1]
        runtime, numeric, adapter = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self.assertEqual(self.lib.ds4_runtime_create(C.byref(runtime), None, None), 0)
        self.assertEqual(
            self.lib.ds4_numeric_cpu(C.byref(numeric), self.resources, 84), 0
        )
        self.assertEqual(
            self.lib.ds4_gemma_visual_open(
                C.byref(adapter),
                self.bundle,
                numeric,
                runtime,
                os.fsencode(
                    ROOT
                    / "performance-results/native-suite-t11-t12-20261021/gemma-f32-rope.ds4g"
                ),
                self.processor,
            ),
            0,
        )

        def prepare(parts, task=0, segment=None, expected=0):
            content = (Content * len(parts))(*parts)
            request = MultimodalRequest(
                content, len(parts), task, False, segment or Segment()
            )
            out = C.c_void_p(1)
            self.assertEqual(
                self.lib.ds4_gemma_multimodal_prepare(
                    adapter, C.byref(request), 1, C.byref(out)
                ),
                expected,
            )
            if expected:
                self.assertIsNone(out.value)
            return out

        def key(input, dim=768):
            value = (C.c_ubyte * 32)()
            self.assertEqual(self.lib.ds4_gemma_multimodal_key(input, 0, dim, value), 0)
            return bytes(value)

        order = [
            Content(b"Prima ", None),
            Content(None, image),
            Content(b" poi ", None),
            Content(None, sound),
        ]
        original = prepare(order)
        self.assertEqual(key(original), key(original))
        self.assertNotEqual(key(original), key(original, 128))
        for parts, task, segment in (
            (order, 1, None),
            (list(reversed(order)), 0, None),
            (order, 0, Segment(7, 1.25, 2.5)),
        ):
            value = prepare(parts, task, segment)
            self.assertNotEqual(key(original), key(value))
            meta = Segment()
            self.assertEqual(
                self.lib.ds4_gemma_multimodal_segment(value, 0, C.byref(meta)), 0
            )
            self.assertEqual(meta.id, segment.id if segment else 0)
            self.assertEqual(self.lib.ds4_gemma_visual_input_free(value), 0)
        for parts, segment in (
            ([Content(b"<|audio|>", None)], None),
            ([Content(b"both", image)], None),
            ([Content(None, None)], None),
            ([Content(b" x" * 8192, None)], None),
            (order, Segment(1, 2, 1)),
            (order, Segment(1, float("nan"), 3)),
        ):
            prepare(parts, segment=segment, expected=1)
        text = prepare([Content(b"solo testo", None)])
        self.assertEqual(self.lib.ds4_gemma_visual_input_free(text), 0)
        source = (
            ROOT / "tests/fixtures/native-suite-v1/media/audio-0.wav"
        ).read_bytes()
        segments = [Segment(11, 0, 0.5), Segment(12, 0.5, 1.4)]
        requests = (MultimodalRequest * 2)()
        contents = []
        soft = []
        for i, segment in enumerate(segments):
            options = DecodeOptions(
                self.limits,
                VideoRecipe(1, 32, 0),
                0,
                segment.start_seconds,
                segment.end_seconds,
            )
            decoded = self.decode(source, 3, options)
            view = self.prepare(decoded)
            soft.append(view.audio_soft_tokens)
            contents.append((Content * 1)(Content(None, self.prepared[-1])))
            requests[i] = MultimodalRequest(contents[-1], 1, 0, False, segment)
        combined = C.c_void_p()
        self.assertEqual(
            self.lib.ds4_gemma_multimodal_prepare(
                adapter, requests, 2, C.byref(combined)
            ),
            0,
        )
        lengths = []
        for i, expected in enumerate(segments):
            meta, ids, length = Segment(), C.POINTER(C.c_uint32)(), C.c_uint32()
            self.assertEqual(
                self.lib.ds4_gemma_multimodal_segment(combined, i, C.byref(meta)), 0
            )
            self.assertEqual(
                (meta.id, meta.start_seconds, meta.end_seconds),
                (expected.id, expected.start_seconds, expected.end_seconds),
            )
            self.assertEqual(
                self.lib.ds4_gemma_visual_input_tokens(
                    combined, i, C.byref(ids), C.byref(length)
                ),
                0,
            )
            lengths.append(length.value)
        self.assertEqual(lengths[1] - lengths[0], soft[1] - soft[0])
        self.assertNotEqual(lengths[0], lengths[1])
        self.assertEqual(self.lib.ds4_gemma_visual_input_free(combined), 0)
        live, peak = C.c_uint64(), C.c_uint64()
        self.lib.ds4_numeric_memory(numeric, C.byref(live), C.byref(peak))
        self.assertEqual((live.value, peak.value), (0, 0))
        self.assertEqual(self.lib.ds4_gemma_visual_close(adapter), 5)
        self.assertEqual(self.lib.ds4_gemma_visual_input_free(original), 0)
        self.assertEqual(self.lib.ds4_gemma_visual_close(adapter), 0)
        self.assertEqual(self.lib.ds4_numeric_close(numeric), 0)
        self.lib.ds4_runtime_release(runtime)

    def test_visual_placeholder_admission_and_ownership(self):
        self.prepare(self.pixels(np.zeros((96, 144, 3), np.uint8)))
        media = (C.c_void_p * 1)(self.prepared[-1])
        runtime, numeric, adapter = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self.assertEqual(self.lib.ds4_runtime_create(C.byref(runtime), None, None), 0)
        self.assertEqual(
            self.lib.ds4_numeric_cpu(C.byref(numeric), self.resources, 84), 0
        )
        self.assertEqual(
            self.lib.ds4_gemma_visual_open(
                C.byref(adapter),
                self.bundle,
                numeric,
                runtime,
                os.fsencode(
                    ROOT
                    / "performance-results/native-suite-t11-t12-20261021/gemma-f32-rope.ds4g"
                ),
                self.processor,
            ),
            0,
        )
        for prompt in (
            b"",
            b"<|video|>",
            b"<|image|> <|image|>",
            b"<|audio|>",
            b"<|image|><|audio|>",
            b"<|image|>" + b" x" * 8192,
        ):
            out = C.c_void_p(1)
            request = VisualRequest(prompt, 0, media, 1, False)
            self.assertNotEqual(
                self.lib.ds4_gemma_visual_prepare(
                    adapter, C.byref(request), 1, C.byref(out)
                ),
                0,
            )
            self.assertIsNone(out.value)
        request = VisualRequest(b"Trova <|image|>", 1, media, 1, False)
        out = C.c_void_p()
        self.assertEqual(
            self.lib.ds4_gemma_visual_prepare(
                adapter, C.byref(request), 1, C.byref(out)
            ),
            0,
        )
        self.assertEqual(
            self.lib.ds4_gemma_visual_close(adapter), 5
        )  # BUSY, retained input
        ids, count = C.POINTER(C.c_uint32)(), C.c_uint32()
        self.assertEqual(
            self.lib.ds4_gemma_visual_input_tokens(
                out, 0, C.byref(ids), C.byref(count)
            ),
            0,
        )
        expected = self.reference(
            text="task: search result | query: Trova <|image|>",
            images=np.zeros((96, 144, 3), np.uint8),
            return_tensors="np",
        )["input_ids"][0]
        np.testing.assert_array_equal(ids[: count.value], expected)
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((3, 96, 144, 3), np.uint8)
        times = [0, 61.9, 124.2]
        raw = (FrameInput * 3)(
            *(FrameInput(f.ctypes.data, 144, 96, 3, t) for f, t in zip(frames, times))
        )
        video = C.c_void_p()
        recipe = VideoRecipe(0, 0, 0)
        self.assertEqual(
            self.lib.ds4_media_video_frames(
                C.byref(video),
                self.resources,
                81,
                raw,
                3,
                0,
                124.2,
                C.byref(recipe),
                C.byref(self.limits),
            ),
            0,
        )
        self.handles.append(video)
        self.prepare(video, 70)
        media_video = (C.c_void_p * 1)(self.prepared[-1])
        video_request = VisualRequest(b"Scene <|video|> fine.", 0, media_video, 1, True)
        video_input = C.c_void_p()
        self.assertEqual(
            self.lib.ds4_gemma_visual_prepare(
                adapter, C.byref(video_request), 1, C.byref(video_input)
            ),
            0,
        )
        self.assertEqual(
            self.lib.ds4_gemma_visual_input_tokens(
                video_input, 0, C.byref(ids), C.byref(count)
            ),
            0,
        )
        expected = self.reference(
            text="Scene <|video|> fine.",
            videos=frames,
            video_metadata=[
                VideoMetadata(
                    total_num_frames=1243, fps=10, frames_indices=[0, 619, 1242]
                )
            ],
            videos_kwargs={
                "do_sample_frames": False,
                "max_soft_tokens": 70,
                "add_timestamps": True,
            },
            return_tensors="np",
        )["input_ids"][0]
        np.testing.assert_array_equal(ids[: count.value], expected)
        self.assertEqual(self.lib.ds4_gemma_visual_input_free(video_input), 0)
        # A small request fits in 400 KiB of available host space. Admission
        # must charge its expanded prompt, rather than a fixed 256 KiB slab.
        original = Budget()
        self.assertEqual(
            self.lib.ds4_resources_get_budget(self.resources, 3, C.byref(original)), 0
        )
        bounded = Budget(original.limit, original.margin, 400 << 10, True)
        self.assertEqual(
            self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(bounded)), 0
        )
        compact = C.c_void_p()
        try:
            self.assertEqual(
                self.lib.ds4_gemma_visual_prepare(
                    adapter, C.byref(request), 1, C.byref(compact)
                ),
                0,
            )
        finally:
            self.assertEqual(self.lib.ds4_gemma_visual_input_free(compact), 0)
            self.assertEqual(
                self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(original)),
                0,
            )
        live, peak = C.c_uint64(), C.c_uint64()
        self.lib.ds4_numeric_memory(numeric, C.byref(live), C.byref(peak))
        self.assertEqual(live.value, 0)
        self.assertEqual(peak.value, 0)
        denied = Budget(1, 0, 0, False)
        original = Budget()
        self.assertEqual(
            self.lib.ds4_resources_get_budget(self.resources, 3, C.byref(original)), 0
        )
        self.assertEqual(
            self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(denied)), 0
        )
        rejected = C.c_void_p(1)
        self.assertEqual(
            self.lib.ds4_gemma_visual_prepare(
                adapter, C.byref(request), 1, C.byref(rejected)
            ),
            6,
        )
        self.assertIsNone(rejected.value)
        self.assertEqual(
            self.lib.ds4_resources_set_budget(self.resources, 3, C.byref(original)), 0
        )
        self.lib.ds4_gemma_processor_clear(self.processor)
        for handle in self.prepared:
            self.lib.ds4_gemma_media_free(handle)
        self.prepared.clear()
        self.assertEqual(self.lib.ds4_gemma_visual_input_free(out), 0)
        self.assertEqual(self.lib.ds4_gemma_visual_close(adapter), 0)
        self.assertEqual(self.lib.ds4_numeric_close(numeric), 0)
        self.lib.ds4_runtime_release(runtime)


if __name__ == "__main__":
    unittest.main()
