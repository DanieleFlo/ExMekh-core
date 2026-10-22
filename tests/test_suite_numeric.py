"""Independent PyTorch oracle for the C/CUDA suite kernels (no model server).

Set DS4_SUITE_DUMP to retain every full input/output plus numeric metrics.
The frozen T00 ceilings apply per case, never to averages across cases.
"""

import ctypes as C
import json
import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("count", C.c_uint64), ("type", C.c_uint32)]


class Attention(C.Structure):
    _fields_ = [
        (k, C.c_uint32)
        for k in ("batch", "queries", "keys", "q_heads", "kv_heads", "dim")
    ] + [
        ("scale", C.c_float),
        ("softcap", C.c_float),
        ("round_scores", C.c_bool),
        ("round_probabilities", C.c_bool),
        ("relative_keys", View),
        ("relative_rows", C.c_uint32),
        ("relative_offset", C.c_int32),
    ]


class Reindex(C.Structure):
    _fields_ = [
        ("rank", C.c_uint32),
        ("shape", C.c_uint64 * 8),
        ("stride", C.c_uint64 * 8),
    ]


class Resample(C.Structure):
    _fields_ = [
        ("batch", C.c_uint32),
        ("channels", C.c_uint32),
        ("input", C.c_uint32 * 3),
        ("output", C.c_uint32 * 3),
    ]


class Conv(C.Structure):
    _fields_ = [
        (k, C.c_uint32) for k in ("batch", "in_channels", "out_channels", "groups")
    ] + [
        (k, C.c_uint32 * 3)
        for k in ("input", "kernel", "stride", "dilation", "pad_before", "pad_after")
    ]


class Numeric:
    def __init__(self, cuda=False):
        self.lib = C.CDLL(
            str(ROOT / f"tests/libds4_numeric_{'cuda' if cuda else 'cpu'}.so")
        )
        lib = self.lib
        ptr = C.c_void_p
        signatures = {
            "ds4_resources_create": [C.POINTER(ptr)],
            "ds4_resources_close": [ptr],
            "ds4_numeric_cpu": [C.POINTER(ptr), ptr, C.c_uint64],
            "ds4_numeric_close": [ptr],
            "ds4_numeric_fence": [ptr],
            "ds4_num_alloc": [ptr, C.c_uint64, C.c_uint32, C.POINTER(ptr)],
            "ds4_num_free": [ptr],
            "ds4_num_data": [ptr],
            "ds4_num_write": [ptr, View, ptr, C.c_uint64],
            "ds4_num_read": [ptr, View, ptr, C.c_uint64],
            "ds4_num_linear": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
            ],
            "ds4_num_matmul_batch": [
                ptr,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
            ],
            "ds4_num_matmul_batch_layout": [
                ptr,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
                C.c_bool,
            ],
            "ds4_num_channel_l2": [ptr, View, View, C.c_uint32, C.c_uint32, C.c_float],
            "ds4_num_norm": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_float,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
            ],
            "ds4_num_point": [ptr, View, View, View, C.c_uint32, C.c_float],
            "ds4_num_rope": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
                C.c_bool,
            ],
            "ds4_num_rope_complex": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
            ],
            "ds4_num_attend": [
                ptr,
                View,
                View,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.POINTER(Attention),
            ],
            "ds4_num_conv": [ptr, View, View, View, View, C.POINTER(Conv)],
            "ds4_num_dequantize": [ptr, C.c_uint64, ptr, C.c_uint64, C.c_uint32],
            "ds4_num_reorder": [ptr, View, View, C.POINTER(Reindex)],
            "ds4_num_pool": [
                ptr,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_float,
            ],
            "ds4_num_nearest": [ptr, View, View, C.POINTER(Resample)],
            "ds4_num_softmax": [
                ptr,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.c_uint32,
                C.c_uint32,
                C.c_float,
            ],
        }
        if cuda:
            signatures["ds4_numeric_cuda"] = signatures["ds4_numeric_cpu"]
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = View if name == "ds4_num_data" else C.c_int
        self.resources, self.ctx = ptr(), ptr()
        self.ok(lib.ds4_resources_create(C.byref(self.resources)))
        self.ok(
            getattr(lib, "ds4_numeric_cuda" if cuda else "ds4_numeric_cpu")(
                C.byref(self.ctx), self.resources, 100
            )
        )
        self.buffers = []

    @staticmethod
    def ok(status):
        if status:
            raise AssertionError(f"native status {status}")

    def tensor(self, tensor):
        tensor = tensor.detach().cpu().contiguous()
        dtype = 30 if tensor.dtype == torch.bfloat16 else 0
        p = C.c_void_p()
        self.ok(self.lib.ds4_num_alloc(self.ctx, tensor.numel(), dtype, C.byref(p)))
        self.buffers.append(p)
        v = self.lib.ds4_num_data(p)
        array = (
            tensor.view(torch.uint16).numpy() if dtype == 30 else tensor.float().numpy()
        )
        self.ok(self.lib.ds4_num_write(self.ctx, v, array.ctypes.data, array.nbytes))
        return v

    def read(self, v, shape):
        arr = np.empty(v.count, dtype=np.uint16 if v.type == 30 else np.float32)
        self.ok(self.lib.ds4_num_read(self.ctx, v, arr.ctypes.data, arr.nbytes))
        result = torch.from_numpy(arr)
        return (
            (result.view(torch.bfloat16) if v.type == 30 else result)
            .float()
            .reshape(shape)
        )

    def call(self, name, *args):
        self.ok(getattr(self.lib, "ds4_num_" + name)(self.ctx, *args))

    def clear(self):
        while self.buffers:
            self.ok(self.lib.ds4_num_free(self.buffers.pop()))

    def close(self):
        self.clear()
        self.ok(self.lib.ds4_numeric_close(self.ctx))
        self.ok(self.lib.ds4_resources_close(self.resources))


class SuiteNumeric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = Numeric(bool(os.environ.get("DS4_SUITE_NUMERIC_CUDA")))
        cls.metrics = []
        cls.dump = (
            Path(os.environ["DS4_SUITE_DUMP"])
            if os.environ.get("DS4_SUITE_DUMP")
            else None
        )
        if cls.dump:
            cls.dump.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.n.close()
        if cls.dump:
            (cls.dump / "metrics.json").write_text(
                json.dumps(cls.metrics, indent=2) + "\n"
            )

    def tearDown(self):
        self.n.clear()

    def compare(self, name, view, expected, **inputs):
        actual = self.n.read(view, expected.shape)
        delta = actual.double() - expected.double()
        absolute = delta.abs().max().item()
        relative = (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
        ).item()
        self.metrics.append(
            {"case": name, "max_abs": absolute, "relative_l2": relative}
        )
        if self.dump:
            np.savez(
                self.dump / f"{name}.npz",
                actual=actual.numpy(),
                expected=expected.float().numpy(),
                **{k: v.float().numpy() for k, v in inputs.items()},
            )
        self.assertTrue(torch.isfinite(actual).all(), name)
        self.assertLessEqual(absolute, 0.04 if view.type == 30 else 0.00002, name)
        self.assertLessEqual(relative, 0.02 if view.type == 30 else 0.00001, name)

    def test_matmul_batch(self):
        torch.manual_seed(20261021)
        x = torch.randn(3, 7, 65)
        w = torch.randn(3, 13, 65) / math.sqrt(65)
        y = self.n.tensor(torch.zeros(3, 7, 13))
        self.n.call(
            "matmul_batch", y, self.n.tensor(x), self.n.tensor(w), 3, 7, 65, 13, True
        )
        self.compare("linear-batch", y, x @ w.transpose(-2, -1), x=x, w=w)
        self.n.call(
            "matmul_batch",
            y,
            self.n.tensor(x),
            self.n.tensor(w.transpose(-2, -1).contiguous()),
            3,
            7,
            65,
            13,
            False,
        )
        self.compare("matmul-batch", y, x @ w.transpose(-2, -1), x=x, w=w)
        for left in (False, True):
            for right in (False, True):
                self.n.call(
                    "matmul_batch_layout",
                    y,
                    self.n.tensor(x.transpose(-2, -1).contiguous() if left else x),
                    self.n.tensor(w if right else w.transpose(-2, -1).contiguous()),
                    3,
                    7,
                    65,
                    13,
                    left,
                    right,
                )
                self.compare(
                    f"matmul-layout-{left}-{right}",
                    y,
                    x @ w.transpose(-2, -1),
                    x=x,
                    w=w,
                )

    def test_linear(self):
        for dtype in (torch.float32, torch.bfloat16):
            for rows, in_dim, out_dim in (
                (3, 65, 127),
                (2, 4096, 1024),
                (2, 512, 2048),
                (1, 1152, 4304),
            ):
                torch.manual_seed(7)
                x = torch.randn(rows, in_dim).to(dtype) * 0.2
                w = (torch.randn(out_dim, in_dim) / in_dim**0.5).to(dtype)
                b = torch.randn(out_dim).to(dtype) * 0.1
                y = self.n.tensor(torch.zeros(rows, out_dim, dtype=dtype))
                self.n.call(
                    "linear",
                    y,
                    self.n.tensor(x),
                    self.n.tensor(w),
                    self.n.tensor(b),
                    rows,
                    in_dim,
                    out_dim,
                )
                self.compare(
                    f"linear-{dtype}-{in_dim}-{out_dim}",
                    y,
                    F.linear(x, w, b),
                    x=x,
                    w=w,
                    b=b,
                )
                self.n.clear()

    def test_norms_and_points(self):
        torch.manual_seed(11)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(3, 512).to(dtype)
            w = torch.randn(512).to(dtype)
            b = torch.randn(512).to(dtype)
            xv, wv, bv = (self.n.tensor(v) for v in (x, w, b))
            y = self.n.tensor(torch.zeros_like(x))
            for mode in range(3):
                for pre in (False, True):
                    z = x.float() * torch.rsqrt(
                        x.float().square().mean(-1, keepdim=True) + 1e-6
                    )
                    if pre:
                        z = z.to(dtype).float()
                    if mode:
                        z *= w.float() + (1 if mode == 2 else 0)
                    self.n.call("norm", y, xv, wv, View(), 512, 1e-6, 0, mode, pre)
                    self.compare(f"rms-{dtype}-{mode}-{pre}", y, z.to(dtype), x=x, w=w)
            self.n.call("norm", y, xv, wv, bv, 512, 1e-6, 1, 1, False)
            self.compare(
                f"layernorm-{dtype}",
                y,
                F.layer_norm(x, (512,), w, b, 1e-6),
                x=x,
                w=w,
                b=b,
            )
            self.n.call("norm", y, xv, View(), View(), 512, 1e-12, 2, 0, True)
            self.compare(
                f"l2norm-{dtype}", y, F.normalize(x.float(), dim=-1).to(dtype), x=x
            )
            for op, expected in (
                (0, x + b),
                (1, x * b),
                (2, F.silu(x)),
                (3, F.gelu(x)),
                (4, F.gelu(x, approximate="tanh")),
                (5, x.sigmoid()),
                (6, x * 0.25),
                (7, x),
                (8, F.relu(x)),
                (9, F.softplus(x)),
                (10, x.tanh()),
                (11, torch.minimum(x, b)),
                (12, torch.maximum(x, b)),
                (13, x.sin()),
                (14, x.cos()),
            ):
                self.n.call("point", y, xv, bv, op, 0.25)
                self.compare(f"point-{dtype}-{op}", y, expected, x=x, b=b)

    def test_complex_rope(self):
        torch.manual_seed(20261021)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(31, 3, 128).to(dtype)
            angles = torch.randn(31, 1, 64)
            co, si = angles.cos(), angles.sin()
            u, v = x[..., ::2].float(), x[..., 1::2].float()
            # Each float32 product is exact in float64. Round only the
            # unfused term, then one FMA result, independently of native code.
            real = (u.double() * co.double() - (v * si).double()).float()
            imag = (u.double() * si.double() + (v * co).double()).float()
            expected = torch.stack((real, imag), -1).flatten(-2).to(dtype)
            y = self.n.tensor(torch.zeros_like(x))
            self.n.call(
                "rope_complex",
                y,
                self.n.tensor(x),
                self.n.tensor(co),
                self.n.tensor(si),
                3,
                128,
            )
            self.compare(f"complex-rope-{dtype}", y, expected)

    def test_rope(self):
        torch.manual_seed(17)
        for dtype, dim in (
            (dtype, dim)
            for dtype in (torch.float32, torch.bfloat16)
            for dim in (128, 256, 512)
        ):
            for adj in (False, True):
                for rounded in (False, True):
                    x = torch.randn(7, 3, dim).to(dtype)
                    angles = torch.randn(7, 1, dim // 2)
                    co, si = angles.cos().to(dtype), angles.sin().to(dtype)
                    u, v = (x[..., ::2], x[..., 1::2]) if adj else x.chunk(2, -1)
                    a, b, c, d = (
                        u.float() * co.float(),
                        v.float() * si.float(),
                        v.float() * co.float(),
                        u.float() * si.float(),
                    )
                    if rounded:
                        a, b, c, d = (t.to(dtype).float() for t in (a, b, c, d))
                    expected = (
                        torch.stack((a - b, c + d), -1).flatten(-2)
                        if adj
                        else torch.cat((a - b, c + d), -1)
                    )
                    y = self.n.tensor(torch.zeros_like(x))
                    self.n.call(
                        "rope",
                        y,
                        self.n.tensor(x),
                        self.n.tensor(co),
                        self.n.tensor(si),
                        3,
                        dim,
                        adj,
                        rounded,
                    )
                    self.compare(
                        f"rope-{dtype}-{dim}-{adj}-{rounded}",
                        y,
                        expected.to(dtype),
                        x=x,
                        cos=co,
                        sin=si,
                    )

    def test_short_rms_reduction_order(self):
        torch.manual_seed(37)
        cuda = bool(os.environ.get("DS4_SUITE_NUMERIC_CUDA"))
        device = "cuda" if cuda else "cpu"
        for rows in (1, 2, 4, 8, 12, 15, 16, 17):
            for width in (128, 256, 512, 768, 4096):
                x = (torch.randn(rows, width) * 0.3).bfloat16()
                weight = torch.randn(width).bfloat16()
                value = x.to(device).float()
                normalized = (
                    value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                )
                for pre in (False, True):
                    expected = (
                        (
                            (normalized.bfloat16().float() if pre else normalized)
                            * weight.to(device).float()
                        )
                        .bfloat16()
                        .cpu()
                    )
                    y = self.n.tensor(torch.zeros_like(x))
                    self.n.call(
                        "norm",
                        y,
                        self.n.tensor(x),
                        self.n.tensor(weight),
                        View(),
                        width,
                        1e-6,
                        0,
                        1,
                        pre,
                    )
                    actual = self.n.read(y, x.shape)
                    if cuda:
                        self.assertTrue(
                            torch.equal(actual, expected.float()),
                            f"short RMS {rows}x{width} pre={pre}",
                        )
                    self.compare(
                        f"short-rms-{rows}-{width}-{pre}",
                        y,
                        expected,
                        x=x,
                        weight=weight,
                    )
                    self.n.clear()

    def test_channel_l2_layout_and_rounding(self):
        torch.manual_seed(43)
        cuda = bool(os.environ.get("DS4_SUITE_NUMERIC_CUDA"))
        device = "cuda" if cuda else "cpu"
        for dtype in (torch.float32, torch.bfloat16):
            for channels, spatial in (
                (96, 4),
                (96, 24),
                (96, 65536),
                (192, 28),
                (384, 7),
                (768, 24),
                (768, 4144),
                (1152, 128),
            ):
                x = torch.randn(channels, spatial).to(dtype)
                x[:, 0] = 0
                expected = (
                    F.normalize(x.to(device).float(), dim=0, eps=1e-12)
                    .to(dtype)
                    .T.contiguous()
                    .cpu()
                )
                source = self.n.tensor(x)
                y = self.n.tensor(torch.zeros(spatial, channels, dtype=dtype))
                self.n.call("channel_l2", y, source, channels, spatial, 1e-12)
                actual = self.n.read(y, expected.shape)
                if cuda and dtype == torch.bfloat16:
                    self.assertTrue(
                        torch.equal(actual, expected.float()),
                        f"channel L2 {channels}x{spatial}",
                    )
                self.compare(
                    f"channel-l2-{dtype}-{channels}-{spatial}", y, expected, x=x
                )
                self.assertEqual(
                    self.n.lib.ds4_num_channel_l2(
                        self.n.ctx, source, source, channels, spatial, 1e-12
                    ),
                    1,
                )
                self.n.clear()

    def test_attention(self):
        torch.manual_seed(23)
        for dtype in (torch.float32, torch.bfloat16):
            for dim, heads, kv in (
                (64, 12, 12),
                (72, 16, 16),
                (128, 32, 8),
                (128, 32, 32),
                (256, 4, 2),
                (512, 4, 1),
                (128, 8, 8),
            ):
                for mode in ("causal", "local", "segmented", "block", "full"):
                    batch, nq, nk = 2, 33, 37
                    q = (torch.randn(batch, nq, heads, dim) * 0.3).to(dtype)
                    k, v = (
                        (torch.randn(batch, nk, kv, dim) * 0.3).to(dtype)
                        for _ in range(2)
                    )
                    bounds = np.array(
                        [
                            [
                                (0, min(nk, r + 1))
                                if mode == "causal"
                                else (max(0, r - 5), min(nk, r + 6))
                                if mode == "local"
                                else (0, 17)
                                if mode == "segmented" and r < 17
                                else (17, nk)
                                if mode == "segmented"
                                else (0, 17 if r < 17 else nk)
                                if mode == "block"
                                else (0, nk)
                                for r in range(nq)
                            ]
                            for _ in range(batch)
                        ],
                        dtype=np.uint32,
                    )
                    bounds[0, 0] = [0, 0]
                    a = Attention(
                        batch, nq, nk, heads, kv, dim, dim**-0.5, 0, True, True
                    )
                    keys = k.repeat_interleave(heads // kv, dim=2).transpose(1, 2)
                    vals = v.repeat_interleave(heads // kv, dim=2).transpose(1, 2)
                    scores = (q.transpose(1, 2) @ keys.transpose(-1, -2)) * a.scale
                    for b in range(batch):
                        for r in range(nq):
                            lo, hi = bounds[b, r]
                            scores[b, :, r, :lo] = -float("inf")
                            scores[b, :, r, hi:] = -float("inf")
                    probs = scores.float().softmax(-1).nan_to_num().to(dtype)
                    expected = (probs @ vals).transpose(1, 2)
                    y = self.n.tensor(torch.zeros_like(q))
                    self.n.call(
                        "attend",
                        y,
                        self.n.tensor(q),
                        self.n.tensor(k),
                        self.n.tensor(v),
                        bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
                        C.byref(a),
                    )
                    self.compare(
                        f"attention-{dtype}-{heads}-{kv}-{dim}-{mode}",
                        y,
                        expected,
                        q=q,
                        k=k,
                        v=v,
                    )
                    self.n.clear()

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_CUDA"), "CUDA eager GEMM geometry"
    )
    def test_attention_dense_video_batch(self):
        torch.manual_seed(37)
        batch, rows, heads, dim = 4, 1260, 12, 64
        q, k, v = [
            (torch.randn(batch, rows, heads, dim) * 0.3).bfloat16() for _ in range(3)
        ]
        bounds = np.array(
            [[(0, valid)] * rows for valid in (1134, 1080, 1170, 1134)], np.uint32
        )
        scores = q.cuda().transpose(1, 2) @ k.cuda().transpose(1, 2).transpose(-2, -1)
        for b in range(batch):
            scores[b, :, :, int(bounds[b, 0, 1]) :] = -float("inf")
        expected = (
            (scores.float().softmax(-1).bfloat16() @ v.cuda().transpose(1, 2))
            .transpose(1, 2)
            .cpu()
        )
        y = self.n.tensor(torch.zeros_like(q))
        a = Attention(batch, rows, rows, heads, heads, dim, 1, 0, True, True)
        self.n.call(
            "attend",
            y,
            self.n.tensor(q),
            self.n.tensor(k),
            self.n.tensor(v),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            C.byref(a),
        )
        self.compare("dense-video-batch", y, expected, q=q, k=k, v=v)
        self.assertTrue(torch.equal(self.n.read(y, expected.shape), expected))

    def test_attention_long_bf16_softmax(self):
        # Vision and image editing cross both eager softmax reduction regimes.
        # Include an unaligned width and an empty row, beyond a short unit case.
        torch.manual_seed(29)
        device = "cuda" if os.environ.get("DS4_SUITE_NUMERIC_CUDA") else "cpu"
        for nk in (2049, 4144, 9217):
            nq, heads, dim = 9, 4, 72
            q = (torch.randn(1, nq, heads, dim) * 0.3).bfloat16()
            k, v = ((torch.randn(1, nk, heads, dim) * 0.3).bfloat16() for _ in range(2))
            bounds = np.array(
                [(0, 0), (0, nk), *[(r, nk - r) for r in range(2, nq)]],
                dtype=np.uint32,
            )
            scores = (
                q.to(device).transpose(1, 2)
                @ k.to(device).transpose(1, 2).transpose(-1, -2)
            ) * dim**-0.5
            for r, (lo, hi) in enumerate(bounds):
                scores[:, :, r, :lo] = -float("inf")
                scores[:, :, r, hi:] = -float("inf")
            expected = (
                (
                    scores.float().softmax(-1).nan_to_num().bfloat16()
                    @ v.to(device).transpose(1, 2)
                )
                .transpose(1, 2)
                .cpu()
            )
            y = self.n.tensor(torch.zeros_like(q))
            a = Attention(1, nq, nk, heads, heads, dim, dim**-0.5, 0, True, True)
            self.n.call(
                "attend",
                y,
                self.n.tensor(q),
                self.n.tensor(k),
                self.n.tensor(v),
                bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
                C.byref(a),
            )
            self.compare(f"attention-long-bf16-{nk}", y, expected, q=q, k=k, v=v)
            self.n.clear()

    def test_attention_tile_window_edges(self):
        torch.manual_seed(24)
        for dtype in (torch.float32, torch.bfloat16):
            for dim, heads, kv in ((128, 4, 1), (256, 2, 1), (512, 2, 2)):
                nq, nk = 257, 513
                q = (torch.randn(1, nq, heads, dim) * 0.1).to(dtype)
                k, v = ((torch.randn(1, nk, kv, dim) * 0.1).to(dtype) for _ in range(2))
                bounds = np.array(
                    [(max(0, r + 256 - 511), r + 257) for r in range(nq)],
                    dtype=np.uint32,
                )
                scores = (
                    q.transpose(1, 2)
                    @ k.repeat_interleave(heads // kv, 2)
                    .transpose(1, 2)
                    .transpose(-1, -2)
                ) * dim**-0.5
                for r, (lo, hi) in enumerate(bounds):
                    scores[:, :, r, :lo] = -float("inf")
                    scores[:, :, r, hi:] = -float("inf")
                expected = (
                    scores.float().softmax(-1).to(dtype)
                    @ v.repeat_interleave(heads // kv, 2).transpose(1, 2)
                ).transpose(1, 2)
                y = self.n.tensor(torch.zeros_like(q))
                a = Attention(1, nq, nk, heads, kv, dim, dim**-0.5, 0, True, True)
                self.n.call(
                    "attend",
                    y,
                    self.n.tensor(q),
                    self.n.tensor(k),
                    self.n.tensor(v),
                    bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
                    C.byref(a),
                )
                self.compare(
                    f"attention-window512-tile257-{dtype}-{dim}",
                    y,
                    expected,
                    q=q,
                    k=k,
                    v=v,
                )
                self.n.clear()

    @unittest.skipUnless(os.environ.get("DS4_SUITE_NUMERIC_CUDA"), "CUDA GEMM geometry")
    def test_attention_full_query_gqa_geometry(self):
        # A real single-reference encoder has 1072 rows. Compare the complete
        # eager GQA operation so query tiling cannot silently change BF16 ties.
        torch.manual_seed(39)
        rows, heads, kv, dim = 1072, 32, 8, 128
        q = (torch.randn(1, rows, heads, dim, device="cuda") * 0.3).bfloat16()
        k, v = (
            (torch.randn(1, rows, kv, dim, device="cuda") * 0.3).bfloat16()
            for _ in range(2)
        )
        scores = (
            q.transpose(1, 2)
            @ k.repeat_interleave(heads // kv, 2).transpose(1, 2).transpose(-1, -2)
        ) * dim**-0.5
        scores.masked_fill_(
            torch.ones(rows, rows, device="cuda", dtype=torch.bool).triu(1),
            -float("inf"),
        )
        expected = (
            (
                scores.float().softmax(-1).bfloat16()
                @ v.repeat_interleave(heads // kv, 2).transpose(1, 2)
            )
            .transpose(1, 2)
            .cpu()
        )
        bounds = np.array([(0, i + 1) for i in range(rows)], dtype=np.uint32)
        y = self.n.tensor(torch.zeros_like(expected))
        a = Attention(1, rows, rows, heads, kv, dim, dim**-0.5, 0, True, True)
        self.n.call(
            "attend",
            y,
            self.n.tensor(q),
            self.n.tensor(k),
            self.n.tensor(v),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            C.byref(a),
        )
        actual = self.n.read(y, expected.shape)
        self.assertTrue(torch.equal(actual, expected.float()))

    def test_convolution(self):
        torch.manual_seed(31)
        for dtype in (torch.float32, torch.bfloat16):
            for groups in (1, 2):
                a = Conv(
                    1,
                    2,
                    4,
                    groups,
                    (3, 7, 9),
                    (2, 3, 3),
                    (1, 2, 1),
                    (1, 1, 2),
                    (1, 1, 2),
                    (0, 2, 1),
                )
                x = torch.randn(1, 2, 3, 7, 9).to(dtype) * 0.2
                w = torch.randn(4, 2 // groups, 2, 3, 3).to(dtype) * 0.1
                b = torch.randn(4).to(dtype) * 0.1
                padded = F.pad(x, (2, 1, 1, 2, 1, 0))
                expected = F.conv3d(
                    padded, w, b, stride=(1, 2, 1), dilation=(1, 1, 2), groups=groups
                )
                y = self.n.tensor(torch.zeros_like(expected))
                self.n.call(
                    "conv",
                    y,
                    self.n.tensor(x),
                    self.n.tensor(w),
                    self.n.tensor(b),
                    C.byref(a),
                )
                self.compare(f"conv-{dtype}-{groups}", y, expected, x=x, w=w, b=b)

    def test_audio_relative_attention(self):
        torch.manual_seed(45)
        q, k, v = (torch.randn(1, 37, 8, 128) * 0.1 for _ in range(3))
        relative = torch.randn(13, 8, 128) * 0.1
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)
        for r in range(37):
            for j in range(37):
                idx = j - r + 12
                if 0 <= idx < 13:
                    scores[0, :, r, j] += (q[0, r] * relative[idx]).sum(-1)
        scores = 50 * torch.tanh(scores / 50)
        bounds = np.array(
            [
                [max(0, (r // 12) * 12 - 12), min(37, (r // 12 + 1) * 12)]
                for r in range(37)
            ],
            dtype=np.uint32,
        )
        for r, (lo, hi) in enumerate(bounds):
            scores[:, :, r, :lo] = -float("inf")
            scores[:, :, r, hi:] = -float("inf")
        expected = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
        y = self.n.tensor(torch.zeros_like(q))
        a = Attention(
            1, 37, 37, 8, 8, 128, 1, 50, False, False, self.n.tensor(relative), 13, 12
        )
        self.n.call(
            "attend",
            y,
            self.n.tensor(q),
            self.n.tensor(k),
            self.n.tensor(v),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            C.byref(a),
        )
        self.compare(
            "audio-relative-softcap", y, expected, q=q, k=k, v=v, relative=relative
        )

    def test_layout_pool_resample(self):
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).to(dtype)
            expected = x.permute(0, 2, 3, 1).contiguous()
            y = self.n.tensor(torch.zeros_like(expected))
            a = Reindex(4, (2, 5, 7, 3), (105, 7, 1, 35))
            self.n.call("reorder", y, self.n.tensor(x), C.byref(a))
            self.compare(f"reorder-{dtype}", y, expected, x=x)
            x = torch.randn(18, 768).to(dtype)
            ix = np.array(
                [
                    [0, 2, 4, 6, 8, 10, 12, 14, 16],
                    [1, 3, 5, 7, 9, 11, 13, 15, 17],
                    [
                        0,
                        1,
                        2,
                        3,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                    ],
                ],
                dtype=np.uint32,
            )
            expected = torch.stack(
                (
                    x[::2].float().sum(0) / 9,
                    x[1::2].float().sum(0) / 9,
                    x[:4].float().sum(0) / 9,
                )
            ).to(dtype)
            y = self.n.tensor(torch.zeros_like(expected))
            self.n.call(
                "pool",
                y,
                self.n.tensor(x),
                ix.ctypes.data_as(C.POINTER(C.c_uint32)),
                3,
                9,
                768,
                9,
            )
            self.compare(f"pool-{dtype}", y, expected, x=x)
            x = torch.randn(1, 4, 2, 3, 5).to(dtype)
            expected = F.interpolate(x, size=(3, 7, 9), mode="nearest-exact")
            y = self.n.tensor(torch.zeros_like(expected))
            a = Resample(1, 4, (2, 3, 5), (3, 7, 9))
            self.n.call("nearest", y, self.n.tensor(x), C.byref(a))
            self.compare(f"nearest-{dtype}", y, expected, x=x)

    def test_long_softmax_reduction_and_masks(self):
        rng = torch.Generator().manual_seed(20261021)
        cuda = bool(os.environ.get("DS4_SUITE_NUMERIC_CUDA"))
        for width in (2048, 2049, 4096, 4119, 8192, 9216, 9217, 12288, 14461):
            x = torch.randn(7, width, generator=rng)
            ranges = np.array(
                [
                    [0, width],
                    [1, width - 1],
                    [7, 8],
                    [0, 0],
                    [width // 2, width],
                    [0, 31],
                    [3, width],
                ],
                dtype=np.uint32,
            )
            masked = x.clone()
            for row, (lo, hi) in enumerate(ranges):
                masked[row, :lo] = -torch.inf
                masked[row, hi:] = -torch.inf
            expected = (
                masked.to("cuda" if cuda else "cpu").softmax(-1).nan_to_num().cpu()
            )
            y = self.n.tensor(torch.empty_like(x))
            self.n.call(
                "softmax",
                y,
                self.n.tensor(x),
                ranges.ctypes.data_as(C.POINTER(C.c_uint32)),
                7,
                width,
                1.0,
            )
            actual = self.n.read(y, x.shape)
            if cuda and width > 2048:
                self.assertTrue(torch.equal(actual, expected), f"wide softmax {width}")
            self.compare(f"long-softmax-{width}", y, expected, x=x)

    def test_invalid_dimensions_and_lifetime(self):
        x = self.n.tensor(torch.ones(64))
        self.assertEqual(
            self.n.lib.ds4_num_linear(self.n.ctx, x, x, x, View(), 0, 8, 8), 1
        )
        self.assertEqual(
            self.n.lib.ds4_num_norm(
                self.n.ctx, x, x, View(), View(), 7, 1e-6, 0, 0, False
            ),
            1,
        )
        self.assertEqual(self.n.lib.ds4_numeric_close(self.n.ctx), 5)
        data = np.array([0x3C00] + [0] * 16, dtype=np.uint16)
        out = np.zeros(32, dtype=np.float32)
        self.assertEqual(
            self.n.lib.ds4_num_dequantize(out.ctypes.data, 32, data.ctypes.data, 1, 8),
            1,
        )
        self.assertEqual(
            self.n.lib.ds4_num_dequantize(
                out.ctypes.data, 32, data.ctypes.data, data.nbytes, 8
            ),
            0,
        )
        np.testing.assert_array_equal(out, 0)
        self.assertEqual(
            self.n.lib.ds4_num_dequantize(
                out.ctypes.data, 32, data.ctypes.data, data.nbytes, 12
            ),
            2,
        )

    def test_output_capacity_preserves_unused_tail(self):
        x = torch.tensor([-2.0, 0.0, 1.0, 2.0, 99.0, 98.0, 97.0, 96.0])
        y = self.n.tensor(torch.full((8,), -123.0))
        bounds = np.array([[0, 4]], dtype=np.uint32)
        self.n.call(
            "softmax",
            y,
            self.n.tensor(x),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            1,
            4,
            1.0,
        )
        expected = torch.cat((x[:4].softmax(-1), torch.full((4,), -123.0)))
        self.compare("softmax-output-tail", y, expected, x=x)
        # An oversized source must also work with an exactly sized destination.
        small = self.n.tensor(torch.zeros(4))
        self.n.call(
            "softmax",
            small,
            self.n.tensor(x),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            1,
            4,
            1.0,
        )
        self.compare("softmax-prefix", small, x[:4].softmax(-1), x=x)
        x = torch.arange(6).reshape(2, 3).float()
        y = self.n.tensor(torch.full((10,), -123.0))
        a = Reindex(2, (3, 2), (1, 3))
        self.n.call("reorder", y, self.n.tensor(x), C.byref(a))
        self.compare(
            "reorder-output-tail",
            y,
            torch.cat((x.T.flatten(), torch.full((4,), -123.0))),
            x=x,
        )
        a = Resample(1, 1, (1, 1, 2), (1, 1, 3))
        y = self.n.tensor(torch.full((6,), -123.0))
        self.n.call("nearest", y, self.n.tensor(torch.tensor([2.0, 5.0])), C.byref(a))
        self.compare(
            "nearest-output-tail",
            y,
            torch.tensor([2.0, 5.0, 5.0, -123.0, -123.0, -123.0]),
        )

    def test_overlapping_views_are_rejected(self):
        backing = self.n.tensor(torch.arange(24).float())
        x = View(backing.data, 8, 0)
        shifted = View(backing.data + 4, 8, 0)
        self.assertEqual(
            self.n.lib.ds4_num_norm(
                self.n.ctx, shifted, x, View(), View(), 4, 1e-6, 0, 0, False
            ),
            1,
        )
        y = self.n.tensor(torch.zeros(8))
        broadcast_from_output = View(y.data, 2, 0)
        self.assertEqual(
            self.n.lib.ds4_num_point(self.n.ctx, y, x, broadcast_from_output, 0, 0), 1
        )
        # Exact in-place operations stay supported.
        self.n.call("point", x, x, View(), 6, 0.25)

    def test_dtype_conversion_is_explicit(self):
        x = self.n.tensor(torch.ones(8))
        y = self.n.tensor(torch.zeros(8, dtype=torch.bfloat16))
        self.assertEqual(self.n.lib.ds4_num_point(self.n.ctx, y, x, View(), 2, 0), 1)
        self.n.call("point", y, x, View(), 7, 0)
        ix = np.arange(8, dtype=np.uint32)
        self.assertEqual(
            self.n.lib.ds4_num_pool(
                self.n.ctx, y, x, ix.ctypes.data_as(C.POINTER(C.c_uint32)), 8, 1, 1, 1
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
