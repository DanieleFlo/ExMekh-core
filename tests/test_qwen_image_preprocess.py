"""Pixel/grid/patch identity against the exact image bundle's fast processor."""

import ctypes as C
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Image(C.Structure):
    _fields_ = [
        ("width", C.c_uint32),
        ("height", C.c_uint32),
        ("rgb", C.c_void_p),
        ("fingerprint", C.c_uint8 * 32),
    ]


class Patches(C.Structure):
    _fields_ = [
        (n, C.c_uint32)
        for n in (
            "content_width",
            "content_height",
            "padded_width",
            "padded_height",
            "grid_width",
            "grid_height",
            "patch_count",
            "image_token_count",
        )
    ] + [("patches", C.POINTER(C.c_float))]


class Preprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = os.environ.get("DS4_SUITE_PREPARED")
        if not prepared:
            raise unittest.SkipTest(
                "set DS4_SUITE_PREPARED to the verified oracle snapshot"
            )
        prepared = Path(prepared)
        sys.path.insert(
            0, str(next((prepared / "sources").glob("transformers-*")) / "src")
        )
        sys.path.insert(0, str(prepared / "deps"))
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            prepared / "models/Qwen/Qwen-Image-2.1-Turbo/processor",
            local_files_only=True,
        ).image_processor
        cls.lib = C.CDLL(str(ROOT / "tests/libds4_image_suite.so"))
        cls.lib.ds4_image_preprocess_qwen3vl_raw.argtypes = [
            C.POINTER(Patches),
            C.POINTER(Image),
            C.c_uint32,
            C.c_uint32,
            C.c_char_p,
            C.c_size_t,
        ]
        cls.lib.ds4_image_preprocess_qwen3vl_raw.restype = C.c_int
        cls.lib.ds4_image_patches_free.argtypes = [C.POINTER(Patches)]

    def test_aspect_ratios_resize_ties_patch_order_and_normalization(self):
        from PIL import Image as PILImage

        for height, width in (
            (256, 256),
            (96, 144),
            (257, 389),
            (80, 48),
            (512, 768),
            (1025, 257),
        ):
            y, x = np.mgrid[:height, :width]
            rgb = np.stack(
                ((x * 3 + y) % 256, (y * 5) % 256, (x + y * 2) % 256), -1
            ).astype(np.uint8)
            oracle = self.processor(
                images=[PILImage.fromarray(rgb)], return_tensors="np"
            )
            image = Image(width, height, rgb.ctypes.data, (C.c_uint8 * 32)())
            patches = Patches()
            error = C.create_string_buffer(256)
            self.assertEqual(
                self.lib.ds4_image_preprocess_qwen3vl_raw(
                    C.byref(patches), C.byref(image), 64, 16384, error, 256
                ),
                1,
                error.value,
            )
            try:
                raw = (
                    np.ctypeslib.as_array(
                        patches.patches, shape=(patches.patch_count * 768,)
                    )
                    .copy()
                    .reshape(-1, 3, 1, 16, 16)
                )
                pixels = np.repeat(
                    (raw - np.float32(127.5)) / np.float32(127.5), 2, axis=2
                ).reshape(-1, 1536)
                np.testing.assert_array_equal(
                    np.array([1, patches.grid_height, patches.grid_width]),
                    oracle["image_grid_thw"][0],
                )
                np.testing.assert_array_equal(
                    pixels, oracle["pixel_values"], err_msg=f"{height}x{width}"
                )
            finally:
                self.lib.ds4_image_patches_free(C.byref(patches))


if __name__ == "__main__":
    unittest.main()
