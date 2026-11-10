#!/usr/bin/env python3
"""Replay the frozen I-P02 BF16 Conv2d shapes against PyTorch CPU.

Each native library runs in its own process. Timings include the synchronous
Metal call, with operands already resident; they do not qualify a whole model.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.suite_numeric_api import Conv, Numeric

SHAPES = (
    (8, 12, 7, 9, 1),
    (8, 12, 7, 9, 3),
    (64, 1152, 16, 16, 3),
    (1152, 1152, 16, 16, 3),
    (384, 384, 64, 64, 3),
    (192, 96, 128, 128, 3),
    (96, 4, 128, 128, 3),
    (1152, 1152, 16, 16, 1),
)


def replay(numeric, shape, generator, warmup, repetitions):
    channels, outputs, height, width, kernel = shape
    x = (torch.randn((1, channels, height, width), generator=generator) / 16).bfloat16()
    w = (torch.randn((outputs, channels, kernel, kernel), generator=generator) / 16).bfloat16()
    b = torch.randn(outputs, generator=generator).bfloat16()
    expected = F.conv2d(x, w, b, padding=kernel // 2).float()
    arguments = Conv(
        1, channels, outputs, 1, (1, height, width), (1, kernel, kernel),
        (1, 1, 1), (1, 1, 1), (0, kernel // 2, kernel // 2), (0, kernel // 2, kernel // 2),
    )
    views = (
        numeric.tensor(torch.zeros_like(expected).bfloat16()),
        numeric.tensor(x), numeric.tensor(w), numeric.tensor(b),
    )
    samples = []
    for iteration in range(warmup + repetitions):
        start = time.perf_counter_ns()
        numeric.call("conv", *views, C.byref(arguments))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if iteration >= warmup:
            samples.append(elapsed)
    actual = numeric.read(views[0], expected.shape)
    record = {
        "shape": shape,
        "ms": samples,
        "mismatch": torch.count_nonzero(actual.view(torch.int32) != expected.view(torch.int32)).item(),
        "output_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
    }
    numeric.clear()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("warmup and repetitions must be positive")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261104)
    records = []
    with Numeric("metal", library_path=args.numeric_library.resolve()) as numeric:
        numeric.configure_metal_budget(reserve_bytes=2 << 30, storage_limit_bytes=1 << 30)
        for shape in SHAPES[:2] if args.small_only else SHAPES:
            record = replay(numeric, shape, generator, args.warmup, args.repetitions)
            records.append(record)
            print(record, flush=True)
    report = {
        "seed": 20261104,
        "library_sha256": hashlib.sha256(args.numeric_library.read_bytes()).hexdigest(),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "scope": "Isolated resident operands; independent CPU oracle; no full-model speedup claim",
        "cases": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(record["mismatch"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
