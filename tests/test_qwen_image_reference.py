"""Exact native Lanczos/alpha pixels against the pinned Pillow processor."""

import ctypes as C
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class Reference(C.Structure):
    _fields_ = [
        ("pixels", C.c_void_p),
        ("width", C.c_uint32),
        ("height", C.c_uint32),
        ("channels", C.c_uint32),
    ]


class Resize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(
            os.environ.get(
                "DS4_SUITE_CPU_LIBRARY",
                str(Path(__file__).resolve().parent / "libds4_suite_cpu.so"),
            )
        )
        cls.lib.qi_reference_resize.argtypes = [
            C.POINTER(Reference),
            C.c_void_p,
            C.c_uint32,
            C.c_uint32,
        ]
        cls.lib.qi_reference_plan.argtypes = [
            C.POINTER(Reference),
            C.c_uint32,
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_uint32),
        ]

    def test_lanczos_rgba_premultiplied_and_rgb(self):
        rng = np.random.default_rng(20261021)
        for h, w, oh, ow in [
            (96, 144, 224, 320),
            (144, 192, 224, 288),
            (137, 97, 320, 224),
            (256, 256, 64, 96),
            (40, 35, 40, 35),
            (96, 144, 96, 192),
            (96, 144, 160, 144),
            (1024, 128, 128, 32),
        ]:
            for channels in (3, 4):
                source = rng.integers(0, 256, (h, w, channels), np.uint8)
                if channels == 4:
                    source[::5, :, 3] = 0
                ref = Reference(source.ctypes.data, w, h, channels)
                out = np.empty((oh, ow, 4), np.uint8)
                self.assertEqual(
                    self.lib.qi_reference_resize(C.byref(ref), out.ctypes.data, ow, oh),
                    0,
                )
                expected = np.array(
                    Image.fromarray(source)
                    .convert("RGBA")
                    .resize((ow, oh), Image.Resampling.LANCZOS)
                )
                with self.subTest(shape=(h, w, channels), target=(oh, ow)):
                    np.testing.assert_array_equal(out, expected)

    def test_area_rounding_and_bounds(self):
        source = np.zeros((96, 144, 3), np.uint8)
        ref = Reference(source.ctypes.data, 144, 96, 3)
        w, h = C.c_uint32(), C.c_uint32()
        self.assertEqual(
            self.lib.qi_reference_plan(C.byref(ref), 256, C.byref(w), C.byref(h)), 0
        )
        self.assertEqual((w.value, h.value), (320, 224))
        self.assertNotEqual(
            self.lib.qi_reference_plan(C.byref(ref), 0, C.byref(w), C.byref(h)), 0
        )
        for resolution in (1, 255, 4096):
            self.assertNotEqual(
                self.lib.qi_reference_plan(
                    C.byref(ref), resolution, C.byref(w), C.byref(h)
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
