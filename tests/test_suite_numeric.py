"""Independent PyTorch oracle for the C/CUDA/Metal suite kernels (no model server).

Set DS4_SUITE_DUMP to retain every full input/output plus numeric metrics.
The frozen T00 ceilings apply per case, never to averages across cases.
"""

import ctypes as C
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tests.suite_numeric_api import (
    Attention,
    Budget,
    Conv,
    NativeError,
    Numeric,
    Reindex,
    Resample,
    View,
)

ROOT = Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)


class SuiteNumeric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = "cpu"
        if os.environ.get("DS4_SUITE_NUMERIC_METAL"):
            backend = "metal"
        elif os.environ.get("DS4_SUITE_NUMERIC_CUDA"):
            backend = "cuda"
        cls.n = Numeric(backend)
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

    @unittest.skipUnless(os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal file upload")
    def test_file_upload_exact_bytes_chunk_tail_and_short_read(self):
        for count, dtype in ((37, torch.bfloat16), ((8 << 20) // 4 + 5, torch.float32)):
            with (
                self.subTest(count=count, dtype=dtype),
                tempfile.TemporaryDirectory() as directory,
            ):
                expected = torch.arange(count).to(dtype)
                payload = (
                    (
                        expected.view(torch.uint16)
                        if dtype == torch.bfloat16
                        else expected
                    )
                    .numpy()
                    .tobytes()
                )
                path = Path(directory) / "weights.bin"
                path.write_bytes(b"unaligned-prefix!" + payload)
                view = self.n.file_tensor(path, count, dtype, offset=17)
                self.assertTrue(
                    torch.equal(self.n.read(view, (count,)), expected.float())
                )
                with self.assertRaises(NativeError):
                    self.n.file_tensor(
                        path, 1, torch.float32, offset=path.stat().st_size
                    )

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

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal workspace admission"
    )
    def test_linear_workspace_bounds_owned_peak(self):
        original = self.n
        self.n = Numeric("metal")
        self.addCleanup(setattr, self, "n", original)
        self.addCleanup(self.n.close)
        torch.manual_seed(20261029)
        # A separate context makes this call's peak independent of the
        # earlier attention tests' much larger high-water marks.
        x = torch.randn(3, 4096, dtype=torch.bfloat16)
        w = torch.randn(1024, 4096, dtype=torch.bfloat16) / 64
        estimated = self.n.linear_workspace(3, 4096, 1024)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(output_dtype=dtype):
                xv, wv = self.n.tensor(x.to(dtype)), self.n.tensor(w.to(dtype))
                y = self.n.tensor(torch.zeros(3, 1024, dtype=dtype))
                before = self.n.memory_usage().live_bytes
                self.n.call("linear", y, xv, wv, View(), 3, 4096, 1024)
                usage = self.n.memory_usage()
                self.assertEqual(usage.live_bytes, before)
                # A workspace bound permits a direct operation with no staging.
                self.assertLessEqual(usage.peak_bytes - before, estimated)
                expected = (
                    F.linear(x, w)
                    if dtype == torch.bfloat16
                    else F.linear(x.float(), w.float())
                )
                self.compare(
                    "linear-owned-workspace-" + str(dtype), y, expected, x=x, w=w
                )
                self.n.clear()
        self.assertEqual(self.n.linear_workspace(0, 4096, 1024), 2**64 - 1)
        self.assertEqual(
            self.n.linear_workspace(2**32 - 1, 2**32 - 1, 2**32 - 1),
            2**64 - 1,
        )

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal BF16 LayerNorm boundaries"
    )
    def test_bf16_layernorm_matches_cpu_with_bias_and_in_place(self):
        torch.manual_seed(20261101)
        for width in (32, 128, 1152, 4608):
            x = torch.randn(7, width).bfloat16()
            weight = torch.randn(width).bfloat16()
            bias = torch.randn(width).bfloat16()
            expected = F.layer_norm(x, (width,), weight, bias, 1e-6)
            for in_place in (False, True):
                with self.subTest(width=width, in_place=in_place):
                    xv = self.n.tensor(x)
                    y = xv if in_place else self.n.tensor(torch.zeros_like(x))
                    self.n.call(
                        "norm",
                        y,
                        xv,
                        self.n.tensor(weight),
                        self.n.tensor(bias),
                        width,
                        1e-6,
                        1,
                        1,
                        False,
                    )
                    actual = self.n.read(y, x.shape).bfloat16()
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int16), expected.view(torch.int16)
                        )
                    )
                    self.n.clear()

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal BF16 channel-major RMS"
    )
    def test_bf16_channel_rms_preserves_cpu_outer_geometry(self):
        torch.manual_seed(20261103)
        for channels, spatial in ((31, 5), (32, 33), (128, 68), (1024, 69)):
            with self.subTest(channels=channels, spatial=spatial):
                source = torch.randn(channels, spatial).bfloat16()
                scale = torch.randn(channels).bfloat16()
                x = source.transpose(0, 1).float()
                expected = (
                    x
                    * torch.pow(x.square().mean(-1, keepdim=True) + 1e-6, -0.5)
                    * scale.float()
                ).bfloat16()
                y = self.n.tensor(torch.zeros(spatial, channels).bfloat16())
                self.n.call(
                    "channel_rms",
                    y,
                    self.n.tensor(source),
                    self.n.tensor(scale),
                    channels,
                    spatial,
                    1e-6,
                )
                actual = self.n.read(y, expected.shape).bfloat16()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.n.clear()

    def test_matmul_bf16_stored_layout(self):
        torch.manual_seed(41)
        device = "cuda" if os.environ.get("DS4_SUITE_NUMERIC_CUDA") else "cpu"
        for batch, rows, inner, out in ((1, 63, 1024, 1024), (3, 7, 65, 13)):
            for left in (False, True):
                for right in (False, True):
                    x = torch.randn(batch, rows, inner).bfloat16().to(device)
                    w = (
                        (torch.randn(batch, inner, out) / math.sqrt(inner))
                        .bfloat16()
                        .to(device)
                    )
                    stored_x = x.transpose(-2, -1).contiguous() if left else x
                    stored_w = w.transpose(-2, -1).contiguous() if right else w
                    logical_x = stored_x.transpose(-2, -1) if left else stored_x
                    logical_w = stored_w.transpose(-2, -1) if right else stored_w
                    expected = (
                        (logical_x[0] @ logical_w[0]).unsqueeze(0)
                        if batch == 1
                        else logical_x @ logical_w
                    )
                    result = self.n.tensor(
                        torch.zeros(batch, rows, out, dtype=torch.bfloat16)
                    )
                    self.n.call(
                        "matmul_batch_layout",
                        result,
                        self.n.tensor(stored_x.cpu()),
                        self.n.tensor(stored_w.cpu()),
                        batch,
                        rows,
                        inner,
                        out,
                        left,
                        right,
                    )
                    self.compare(
                        f"bf16-layout-{batch}-{left}-{right}", result, expected.cpu()
                    )
                    self.n.clear()

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

    def test_bf16_linear_tile_preserves_rounding_and_partial_rows(self):
        if self.n.backend != "metal":
            self.skipTest("Metal SIMD projection")
        rng = torch.Generator().manual_seed(20261103)
        rows, inner, columns = 65, 1024, 1030
        x = (torch.randn(rows, inner, generator=rng) / 16).bfloat16()
        w = (torch.randn(columns, inner, generator=rng) / 16).bfloat16()
        b = torch.randn(columns, generator=rng).bfloat16()
        output = self.n.tensor(torch.zeros(rows, columns, dtype=torch.bfloat16))
        self.n.call(
            "linear",
            output,
            self.n.tensor(x),
            self.n.tensor(w),
            self.n.tensor(b),
            rows,
            inner,
            columns,
        )
        actual = self.n.read(output, (rows, columns)).bfloat16()
        expected = F.linear(x, w, b)
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )

    def test_bf16_linear_tile_uses_only_active_ranges_of_borrowed_views(self):
        if self.n.backend != "metal":
            self.skipTest("Metal SIMD projection")
        rng = torch.Generator().manual_seed(20261104)
        rows, inner, columns = 65, 1024, 1030
        padding = 16
        x = (torch.randn(rows, inner, generator=rng) / 16).bfloat16()
        w = (torch.randn(columns, inner, generator=rng) / 16).bfloat16()
        bias = torch.randn(columns, generator=rng).bfloat16()
        output_start = padding + x.numel() + padding
        storage = torch.full(
            (output_start + rows * columns + padding,),
            -123.0,
            dtype=torch.bfloat16,
        )
        storage[padding : padding + x.numel()] = x.flatten()
        backing = self.n.tensor(storage)
        # The declared input capacity includes the disjoint output region.
        # Only the matrix prefix participates in computation and alias checks.
        inputs = self.n.slice(backing, padding, storage.numel() - padding)
        output = self.n.slice(backing, output_start, rows * columns + padding)
        weights = self.n.tensor(F.pad(w.flatten(), (padding, padding), value=-123))
        weights = self.n.slice(weights, padding, w.numel() + padding)
        biases = self.n.tensor(F.pad(bias, (padding, padding), value=-123))
        biases = self.n.slice(biases, padding, columns + padding)
        self.n.call("linear", output, inputs, weights, biases, rows, inner, columns)

        overlapping = self.n.slice(backing, padding + 1, rows * columns)
        self.assertEqual(
            self.n.lib.ds4_num_linear(
                self.n.ctx,
                overlapping,
                inputs,
                weights,
                biases,
                rows,
                inner,
                columns,
            ),
            1,
        )
        storage[output_start : output_start + rows * columns] = F.linear(
            x, w, bias
        ).flatten()
        actual = self.n.read(backing, storage.shape).bfloat16()
        torch.testing.assert_close(
            actual.view(torch.int16), storage.view(torch.int16), atol=0, rtol=0
        )

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

    def test_activation_finite_tails(self):
        # Real Gemma PLE gates exceed the random unit fixture's range. A GPU
        # tanh overflow must not turn a saturated positive GELU into NaN.
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.tensor(
                [-1e10, -100, -30, -20, -10, -8, -5, 0, 5, 8, 10, 20, 30, 100, 1e10]
            ).to(dtype)
            source = self.n.tensor(x)
            y = self.n.tensor(torch.zeros_like(x))
            for op, expected in (
                (2, F.silu(x)),
                (3, F.gelu(x)),
                (4, F.gelu(x, approximate="tanh")),
                (10, x.tanh()),
            ):
                self.n.call("point", y, source, View(), op, 0)
                self.compare(f"activation-tail-{dtype}-{op}", y, expected, x=x)

    def test_bf16_erf_gelu_boundaries(self):
        if self.n.backend != "metal":
            self.skipTest("Pinned CPU/Metal BF16 erf operation order")
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        values = values.view(torch.bfloat16)
        values = values[torch.isfinite(values)]
        generator = torch.Generator().manual_seed(20261101)
        shuffled = values[torch.randperm(values.numel(), generator=generator)]
        for name, inputs in (("ordered", values), ("shuffled", shuffled)):
            with self.subTest(order=name):
                output = self.n.tensor(torch.zeros_like(inputs))
                self.n.call("point", output, self.n.tensor(inputs), View(), 3, 0)
                actual = self.n.read(output, inputs.shape).bfloat16()
                expected = F.gelu(inputs)
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )

    def test_vision_trigonometric_boundaries(self):
        if self.n.backend != "metal":
            self.skipTest("Pinned CPU/Metal trigonometric operation order")
        angles = torch.arange(16).float()[:, None] * (
            1 / (10000 ** (torch.arange(0, 36, 2).float() / 36))
        )
        values = torch.cat(
            (
                angles.flatten(),
                torch.linspace(-124, 124, 8192),
                torch.tensor([0.0, -0.0, 1e-38, -1e-38]),
            )
        )
        for op, expected in ((13, values.sin()), (14, values.cos())):
            with self.subTest(op=op):
                output = self.n.tensor(torch.zeros_like(values))
                self.n.call("point", output, self.n.tensor(values), View(), op, 0)
                actual = self.n.read(output, expected.shape)
                self.assertTrue(
                    torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                )

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

    def test_channel_rms_audio_layout(self):
        torch.manual_seed(47)
        device = "cuda" if os.environ.get("DS4_SUITE_NUMERIC_CUDA") else "cpu"
        for spatial in (1, 7, 12, 63, 64, 127, 280, 750):
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(1024, spatial).to(dtype)
                weight = torch.randn(1024).to(dtype)
                transposed = x.to(device).T
                value = transposed.float()
                expected = (
                    (
                        value
                        * (value.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
                        * weight.to(device).float()
                    )
                    .to(dtype)
                    .cpu()
                )
                source = self.n.tensor(x)
                scale = self.n.tensor(weight)
                y = self.n.tensor(torch.zeros(spatial, 1024, dtype=dtype))
                self.n.call("channel_rms", y, source, scale, 1024, spatial, 1e-6)
                self.compare(
                    f"channel-rms-{dtype}-{spatial}", y, expected, x=x, weight=weight
                )
                self.assertEqual(
                    self.n.lib.ds4_num_channel_rms(
                        self.n.ctx, source, source, scale, 1024, spatial, 1e-6
                    ),
                    1,
                )
                self.n.clear()

    def test_channel_rms_ignores_unused_capacity_for_aliasing(self):
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
            tail = torch.tensor([2.0, 3.0, 99.0, 98.0], dtype=dtype)
            backing = self.n.tensor(torch.cat((torch.zeros(6, dtype=dtype), tail)))
            scale = self.n.slice(backing, 6, 4)
            value = source.T.float()
            expected = (
                (
                    value
                    * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    * tail[:2].float()
                )
                .to(dtype)
                .flatten()
            )
            self.n.call(
                "channel_rms", backing, self.n.tensor(source), scale, 2, 3, 1e-6
            )
            self.compare(
                f"channel-rms-capacity-{dtype}",
                backing,
                torch.cat((expected, tail)).float(),
            )
            torch.testing.assert_close(
                self.n.read(backing, (10,))[6:], tail.float(), rtol=0, atol=0
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
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal BF16 batched QK geometry"
    )
    def test_bf16_attention_preserves_cpu_layout_and_tail(self):
        torch.manual_seed(20261102)
        length, heads = 73, 3
        for batch, dim in ((2, 64), (1, 72), (2, 72)):
            with self.subTest(batch=batch, head_dim=dim):
                q, k, v = [
                    torch.randn(batch, length, heads, dim).bfloat16() for _ in range(3)
                ]
                scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * (
                    dim**-0.5
                )
                probabilities = scores.float().softmax(-1).bfloat16()
                expected = (
                    (probabilities @ v.transpose(1, 2)).transpose(1, 2).contiguous()
                )
                bounds = np.tile(
                    np.array([[0, length]], dtype=np.uint32), (batch * length, 1)
                )
                parameters = Attention(
                    batch, length, length, heads, heads, dim, dim**-0.5, 0, True, True
                )
                y = self.n.tensor(torch.zeros_like(q))
                self.n.call(
                    "attend",
                    y,
                    self.n.tensor(q),
                    self.n.tensor(k),
                    self.n.tensor(v),
                    bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
                    C.byref(parameters),
                )
                actual = self.n.read(y, expected.shape).bfloat16()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
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

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal ordered convolution"
    )
    def test_bf16_image_convolution_views(self):
        rng = torch.Generator().manual_seed(20261104)
        # Include a spatial tail, batches/groups, asymmetric padding, stride,
        # dilation, and unaligned-channel fallback through the public API.
        for channels, groups, kernel, stride, dilation, before, after, biased in (
            (8, 1, 1, 1, 1, (0, 0), (0, 0), False),
            (8, 2, 3, 1, 1, (1, 1), (1, 1), True),
            (4, 1, 3, 2, 2, (2, 1), (0, 3), False),
            (5, 1, 3, 1, 1, (1, 1), (1, 1), True),
        ):
            with self.subTest(
                channels=channels, groups=groups, kernel=kernel, stride=stride
            ):
                x = (torch.randn((2, channels, 5, 7), generator=rng) / 8).bfloat16()
                w = (
                    torch.randn((6, channels // groups, kernel, kernel), generator=rng)
                    / 8
                ).bfloat16()
                b = torch.randn(6, generator=rng).bfloat16() if biased else None
                padded = F.pad(x, (before[1], after[1], before[0], after[0]))
                expected = F.conv2d(
                    padded, w, b, stride=stride, dilation=dilation, groups=groups
                )
                padding = 16
                backing = torch.full(
                    (padding + expected.numel() + padding,), 19, dtype=torch.bfloat16
                )
                storage = self.n.tensor(backing)
                y = self.n.slice(storage, padding, expected.numel() + padding)

                def operand(value, padding=padding):
                    data = torch.cat(
                        (
                            torch.full((padding,), 23, dtype=torch.bfloat16),
                            value.flatten(),
                        )
                    )
                    return self.n.slice(self.n.tensor(data), padding, value.numel())

                a = Conv(
                    2,
                    channels,
                    6,
                    groups,
                    (1, 5, 7),
                    (1, kernel, kernel),
                    (1, stride, stride),
                    (1, dilation, dilation),
                    (0, *before),
                    (0, *after),
                )
                self.n.call(
                    "conv",
                    y,
                    operand(x),
                    operand(w),
                    operand(b) if biased else View(),
                    C.byref(a),
                )
                backing[padding : padding + expected.numel()] = expected.flatten()
                actual = self.n.read(storage, backing.shape).bfloat16()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), backing.view(torch.int16))
                )
                self.n.clear()

    def test_convolution_wide_padding(self):
        if self.n.backend == "cuda":
            self.skipTest("cuDNN descriptors use signed int dimensions and strides")
        # Two output positions fit even though the padded width exceeds uint32.
        # The first lies in zero padding and the second lands on the input.
        width = (1 << 32) - 1
        a = Conv(
            1,
            4,
            1,
            1,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, width),
            (1, 1, 1),
            (0, 0, width),
            (0, 0, 1),
        )
        x = self.n.tensor(torch.ones(4, dtype=torch.bfloat16))
        w = self.n.tensor(torch.ones(4, dtype=torch.bfloat16))
        y = self.n.tensor(torch.full((2,), -1, dtype=torch.bfloat16))
        self.n.call("conv", y, x, w, View(), C.byref(a))
        self.assertTrue(torch.equal(self.n.read(y, (2,)), torch.tensor([0.0, 4.0])))

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

    def test_constructor_clears_failure_output(self):
        name = f"ds4_numeric_{self.n.backend}"
        create = getattr(self.n.lib, name)
        for resources, owner in ((None, 100), (self.n.resources, 0)):
            output = C.c_void_p(1)
            self.assertNotEqual(create(C.byref(output), resources, owner), 0)
            self.assertIsNone(output.value, name)

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_CUDA"), "CUDA allocation accounting"
    )
    def test_native_free_refreshes_shared_device_budget(self):
        lib = self.n.lib
        lib.ds4_resources_reconcile.argtypes = [
            C.c_void_p,
            C.c_uint32,
            C.c_uint64,
            C.c_bool,
        ]
        lib.ds4_resources_get_budget.argtypes = [
            C.c_void_p,
            C.c_uint32,
            C.POINTER(Budget),
        ]
        original = Budget()
        self.n.ok(lib.ds4_resources_get_budget(self.n.resources, 0, C.byref(original)))
        self.n.tensor(torch.ones(16))
        self.n.ok(lib.ds4_resources_reconcile(self.n.resources, 0, 0, True))
        self.n.clear()
        current = Budget()
        self.n.ok(lib.ds4_resources_get_budget(self.n.resources, 0, C.byref(current)))
        self.assertTrue(current.global_known)
        self.assertGreater(current.global_free, 0)
        self.assertEqual(
            (current.limit, current.margin), (original.limit, original.margin)
        )
        self.n.tensor(torch.ones(16))

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
        x = self.n.slice(backing, 0, 8)
        shifted = self.n.slice(backing, 1, 8)
        self.assertEqual(
            self.n.lib.ds4_num_norm(
                self.n.ctx, shifted, x, View(), View(), 4, 1e-6, 0, 0, False
            ),
            1,
        )
        y = self.n.tensor(torch.zeros(8))
        broadcast_from_output = self.n.slice(y, 0, 2)
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
