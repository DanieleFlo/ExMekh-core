"""Small independent gates; full frozen model gates use probe_qwen_image_graph."""

import ctypes as C
import math
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("count", C.c_uint64), ("type", C.c_uint32)]


class Graph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(str(ROOT / "tests/libds4_qwen_image_graph_cpu.so"))
        ptr = C.c_void_p
        signatures = {
            "ds4_resources_create": [C.POINTER(ptr)],
            "ds4_resources_close": [ptr],
            "ds4_numeric_cpu": [C.POINTER(ptr), ptr, C.c_uint64],
            "ds4_numeric_close": [ptr],
            "ds4_numeric_memory": [ptr, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)],
            "ds4_num_alloc": [ptr, C.c_uint64, C.c_uint32, C.POINTER(ptr)],
            "ds4_num_data": [ptr],
            "ds4_num_free": [ptr],
            "ds4_num_write": [ptr, View, ptr, C.c_uint64],
            "ds4_num_read": [ptr, View, ptr, C.c_uint64],
            "ds4_qwen_image_attention": [
                ptr,
                View,
                View,
                View,
                View,
                *([C.c_uint32] * 4),
                C.POINTER(C.c_uint32),
            ],
            "ds4_qwen_image_noise": [C.POINTER(C.c_float), C.c_size_t, C.c_uint64],
            "ds4_qwen_image_turbo_sigmas": [C.POINTER(C.c_float)],
        }
        for name, args in signatures.items():
            fn = getattr(cls.lib, name)
            fn.argtypes = args
            fn.restype = (
                View
                if name == "ds4_num_data"
                else None
                if name in ["ds4_numeric_memory", "ds4_qwen_image_turbo_sigmas"]
                else C.c_int
            )

    def test_turbo_grid(self):
        out = (C.c_float * 9)()
        self.lib.ds4_qwen_image_turbo_sigmas(out)
        np.testing.assert_array_equal(
            np.array(out),
            np.array(
                [
                    1,
                    0.978453,
                    0.95418,
                    0.926626,
                    0.89508,
                    0.845148,
                    0.704534,
                    0.414568,
                    0,
                ],
                np.float32,
            ),
        )
        self.assertEqual(len(out) - 1, 8)
        self.assertTrue(np.all(np.diff(out) < 0))

    def test_rng_independent_reference(self):
        mask = (1 << 64) - 1
        state = 20261020

        def uniform():
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            return ((z >> 11) + 0.5) * 2**-53

        expected = []
        for _ in range(9):
            r = math.sqrt(-2 * math.log(uniform()))
            angle = 2 * math.pi * uniform()
            expected.extend([r * math.cos(angle), r * math.sin(angle)])
        out = (C.c_float * 17)()
        self.assertEqual(self.lib.ds4_qwen_image_noise(out, 17, 20261020), 0)
        np.testing.assert_array_equal(
            np.array(out), np.array(expected[:17], np.float32)
        )
        self.assertEqual(self.lib.ds4_qwen_image_noise(None, 17, 0), 1)
        self.assertEqual(self.lib.ds4_qwen_image_noise(out, 0, 0), 1)

    def test_attention_block_causal_and_wide_vae(self):
        torch.set_num_threads(2)
        for nq, nk, heads, dim in [(9, 9, 2, 128), (5, 11, 1, 768), (3, 7, 1, 1152)]:
            with self.subTest(dim=dim):
                resources, num = C.c_void_p(), C.c_void_p()
                self.assertEqual(self.lib.ds4_resources_create(C.byref(resources)), 0)
                self.assertEqual(
                    self.lib.ds4_numeric_cpu(C.byref(num), resources, 915), 0
                )
                rng = np.random.default_rng(221)
                buffers, tensors = [], []
                for shape in [
                    (nq, heads, dim),
                    (nk, heads, dim),
                    (nk, heads, dim),
                    (nq, heads, dim),
                ]:
                    t = torch.tensor(rng.standard_normal(shape), dtype=torch.bfloat16)
                    tensors.append(t)
                    b = C.c_void_p()
                    self.assertEqual(
                        self.lib.ds4_num_alloc(num, t.numel(), 30, C.byref(b)), 0
                    )
                    buffers.append(b)
                    self.assertEqual(
                        self.lib.ds4_num_write(
                            num, self.lib.ds4_num_data(b), t.data_ptr(), t.numel() * 2
                        ),
                        0,
                    )
                bounds = np.zeros((nq, 2), np.uint32)
                bounds[:, 1] = nk
                if nq == nk:
                    bounds[:3, 1] = [1, 2, 3]
                views = [self.lib.ds4_num_data(b) for b in buffers]
                self.assertEqual(
                    self.lib.ds4_qwen_image_attention(
                        num,
                        views[3],
                        *views[:3],
                        nq,
                        nk,
                        heads,
                        dim,
                        bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
                    ),
                    0,
                )
                result = torch.empty_like(tensors[0])
                self.assertEqual(
                    self.lib.ds4_num_read(
                        num, views[3], result.data_ptr(), result.numel() * 2
                    ),
                    0,
                )
                valid = (
                    torch.arange(nk)[None]
                    < torch.tensor(bounds[:, 1].astype(np.int64))[:, None]
                )
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    expected = (
                        F.scaled_dot_product_attention(
                            *(t.permute(1, 0, 2)[None] for t in tensors[:3]),
                            attn_mask=valid,
                        )
                        .squeeze(0)
                        .permute(1, 0, 2)
                    )
                delta = result.float() - expected.float()
                self.assertLessEqual(float(delta.abs().max()), 0.04)
                self.assertLessEqual(
                    float(delta.norm() / expected.float().norm()), 0.02
                )
                for b in buffers:
                    self.assertEqual(self.lib.ds4_num_free(b), 0)
                self.assertEqual(self.lib.ds4_numeric_close(num), 0)
                self.assertEqual(self.lib.ds4_resources_close(resources), 0)


if __name__ == "__main__":
    unittest.main()
