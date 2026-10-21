#!/usr/bin/env python3
"""Development benchmark: synchronized native primitives, independent outputs.

No model weights or Python code enter the production C library. Complete
operands and outputs accompany each timing; this is a baseline, not a speedup.
"""

import argparse
import ctypes as C
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_suite_numeric import Attention, Conv, Numeric, Reindex, Resample, View


def statistics(samples):
    a = np.asarray(samples)
    return {
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "stddev": float(a.std()),
        "cv": float(a.std() / a.mean()),
        "verdict": "NEED_MORE_DATA" if a.std() / a.mean() > 0.05 else "MEASURED",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=5)
    args = p.parse_args()
    if args.repetitions < 5:
        p.error("at least five repetitions after warmup")
    args.output.mkdir(parents=True, exist_ok=False)
    n = Numeric(cuda=True)
    records = []
    torch.manual_seed(20261020)

    def run(name, operation, y, expected, **inputs):
        operation()
        n.ok(n.lib.ds4_numeric_fence(n.ctx))
        actual = n.read(y, expected.shape)
        delta = actual.double() - expected.double()
        absolute = float(delta.abs().max())
        relative = float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
        )
        assert torch.isfinite(actual).all()
        assert absolute <= (0.04 if y.type == 30 else 0.00002), (name, absolute)
        assert relative <= (0.02 if y.type == 30 else 0.00001), (name, relative)
        samples = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            operation()
            n.ok(n.lib.ds4_numeric_fence(n.ctx))
            samples.append(time.perf_counter() - start)
        np.savez(
            args.output / (name + ".npz"),
            actual=actual.numpy(),
            expected=expected.float().numpy(),
            **{k: v.float().numpy() for k, v in inputs.items()},
        )
        records.append(
            {
                "case": name,
                "shape": list(expected.shape),
                "dtype": y.type,
                "correctness": "PASS",
                "max_abs": absolute,
                "relative_l2": relative,
                "seconds": samples,
                "statistics": statistics(samples),
            }
        )
        (args.output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
        print(name, records[-1]["statistics"], flush=True)
        n.clear()

    for rows, width, output in (
        (104, 4096, 12288),
        (257, 1536, 6144),
        (513, 4096, 4096),
    ):
        x = (torch.randn(rows, width) * 0.1).bfloat16()
        w = (torch.randn(output, width) / width**0.5).bfloat16()
        xv, wv = n.tensor(x), n.tensor(w)
        y = n.tensor(torch.zeros(rows, output, dtype=torch.bfloat16))
        run(
            f"gemm-{rows}-{width}-{output}",
            partial(n.call, "linear", y, xv, wv, View(), rows, width, output),
            y,
            F.linear(x, w),
            x=x,
            w=w,
        )
    x = torch.randn(257, 4096).bfloat16()
    scale = torch.ones(4096).bfloat16()
    xv, sv, y = n.tensor(x), n.tensor(scale), n.tensor(torch.zeros_like(x))
    run(
        "rms-257-4096",
        lambda: n.call("norm", y, xv, sv, View(), 4096, 1e-6, 0, 1, True),
        y,
        (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16(),
        x=x,
        scale=scale,
    )
    x = torch.randn(257, 12288).bfloat16()
    xv, y = n.tensor(x), n.tensor(torch.zeros_like(x))
    run(
        "gelu-257-12288",
        lambda: n.call("point", y, xv, View(), 3, 0),
        y,
        F.gelu(x),
        x=x,
    )
    q, k, v = ((torch.randn(1, 513, 4, 128) * 0.1).bfloat16() for _ in range(3))
    bounds = np.array([(max(0, r - 511), r + 1) for r in range(513)], dtype=np.uint32)
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * 128**-0.5
    for r, (lo, hi) in enumerate(bounds):
        scores[:, :, r, :lo], scores[:, :, r, hi:] = -float("inf"), -float("inf")
    expected = (scores.float().softmax(-1).bfloat16() @ v.transpose(1, 2)).transpose(
        1, 2
    )
    qv, kv, vv, y = n.tensor(q), n.tensor(k), n.tensor(v), n.tensor(torch.zeros_like(q))
    a = Attention(1, 513, 513, 4, 4, 128, 128**-0.5, 0, True, True)
    run(
        "attention-window512-513",
        lambda: n.call(
            "attend",
            y,
            qv,
            kv,
            vv,
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            C.byref(a),
        ),
        y,
        expected,
        q=q,
        k=k,
        v=v,
    )
    q = torch.randn(1, 257, 4, 512) * 0.1
    k, v = (torch.randn(1, 513, 1, 512) * 0.1 for _ in range(2))
    bounds = np.array([(0, r + 257) for r in range(257)], dtype=np.uint32)
    scores = (
        q.transpose(1, 2) @ k.repeat_interleave(4, 2).transpose(1, 2).transpose(-1, -2)
    ) * 512**-0.5
    for r, (_, hi) in enumerate(bounds):
        scores[:, :, r, hi:] = -float("inf")
    expected = (
        scores.softmax(-1) @ v.repeat_interleave(4, 2).transpose(1, 2)
    ).transpose(1, 2)
    qv, kv, vv, y = n.tensor(q), n.tensor(k), n.tensor(v), n.tensor(torch.zeros_like(q))
    a = Attention(1, 257, 513, 4, 1, 512, 512**-0.5, 0, False, False)
    run(
        "attention-online-mqa-512",
        lambda: n.call(
            "attend",
            y,
            qv,
            kv,
            vv,
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            C.byref(a),
        ),
        y,
        expected,
        q=q,
        k=k,
        v=v,
    )
    # Patch Conv3d is the real Qwen vision geometry; no im2col reference is used.
    x = (torch.randn(1, 3, 2, 256, 256) * 0.1).bfloat16()
    w = (torch.randn(1152, 3, 2, 16, 16) * 0.01).bfloat16()
    expected = F.conv3d(x, w, stride=(2, 16, 16))
    xv, wv, y = n.tensor(x), n.tensor(w), n.tensor(torch.zeros_like(expected))
    a = Conv(
        1,
        3,
        1152,
        1,
        (2, 256, 256),
        (2, 16, 16),
        (2, 16, 16),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
    )
    run(
        "patch-conv3d-256",
        lambda: n.call("conv", y, xv, wv, View(), C.byref(a)),
        y,
        expected,
        x=x,
        w=w,
    )
    for kind in (1, 2):
        x = torch.randn(257, 1152)
        xv, y = n.tensor(x), n.tensor(torch.zeros_like(x))
        expected = (
            F.layer_norm(x, (1152,), eps=1e-6)
            if kind == 1
            else F.normalize(x, dim=-1, eps=1e-6)
        )
        run(
            f"norm-{kind}-257-1152",
            partial(n.call, "norm", y, xv, View(), View(), 1152, 1e-6, kind, 0, False),
            y,
            expected,
            x=x,
        )
    x = torch.randn(257, 4, 128).bfloat16()
    angles = torch.randn(257, 64)
    co, si = angles.cos().bfloat16(), angles.sin().bfloat16()
    left, right = x.chunk(2, -1)
    expected = torch.cat(
        (
            left * co[:, None] - right * si[:, None],
            right * co[:, None] + left * si[:, None],
        ),
        -1,
    )
    xv, cv, sv, y = (
        n.tensor(x),
        n.tensor(co),
        n.tensor(si),
        n.tensor(torch.zeros_like(x)),
    )
    run(
        "rope-257-4-128",
        lambda: n.call("rope", y, xv, cv, sv, 4, 128, False, True),
        y,
        expected,
        x=x,
        cos=co,
        sin=si,
    )
    x = torch.randn(1, 32, 2, 32, 32).bfloat16()
    xv = n.tensor(x)
    expected = x.permute(0, 2, 3, 4, 1).contiguous()
    y = n.tensor(torch.zeros_like(expected))
    a = Reindex(5, (1, 2, 32, 32, 32), (65536, 1024, 32, 1, 2048))
    run("reorder-vae", lambda: n.call("reorder", y, xv, C.byref(a)), y, expected, x=x)
    x = torch.randn(1024, 768).bfloat16()
    ix = np.arange(1024, dtype=np.uint32).reshape(256, 4)
    expected = x.reshape(256, 4, 768).float().mean(1).bfloat16()
    xv, y = n.tensor(x), n.tensor(torch.zeros_like(expected))
    run(
        "pool-vision-256-4-768",
        lambda: n.call(
            "pool", y, xv, ix.ctypes.data_as(C.POINTER(C.c_uint32)), 256, 4, 768, 4
        ),
        y,
        expected,
        x=x,
    )
    x = torch.randn(1, 32, 2, 32, 32).bfloat16()
    expected = F.interpolate(x, size=(3, 65, 65), mode="nearest-exact")
    xv, y = n.tensor(x), n.tensor(torch.zeros_like(expected))
    a = Resample(1, 32, (2, 32, 32), (3, 65, 65))
    run("nearest-vae", lambda: n.call("nearest", y, xv, C.byref(a)), y, expected, x=x)
    n.close()


if __name__ == "__main__":
    main()
