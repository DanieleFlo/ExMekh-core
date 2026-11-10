#!/usr/bin/env python3
"""Replay Gemma's first PLE projection and RMSNorm on identical BF16 inputs.

Development diagnostic, never a serving path or a replacement qualification.
Uses existing short text traces and only the two PLE tensors from safetensors;
does not load the model. FP64 witnesses distinguish accumulation differences
from storage rounding without assuming either backend is the exact oracle.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LIMIT = 1024**3
HIDDEN_WIDTH = 512
PLE_WIDTH = 12288
PLE_SCALE = 0.04419417382415922


def bf16_values(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def bf16_bits(values):
    """Round F32 storage to BF16, ties to even, preserving NaNs as NaNs."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return np.where(nan, (bits >> 16) | 0x40, rounded >> 16).astype(np.uint16)


def compare_bits(reference, candidate):
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("shape mismatch or empty boundary")
    a, b = (
        bf16_values(reference).astype(np.float64),
        bf16_values(candidate).astype(np.float64),
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite boundary")
    return {
        "elements": int(reference.size),
        "mismatches": int(np.count_nonzero(reference != candidate)),
        "bit_exact": bool(np.array_equal(reference, candidate)),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def dot_witnesses(x, weight, reference, candidate, limit=32):
    """Only recompute mismatching dots; no full F64 weight expansion."""
    indices = np.argwhere(reference != candidate)
    result = []
    for row, column in indices[:limit]:
        products = x[row].astype(np.float64) * weight[column].astype(np.float64)
        value = float(np.sum(products, dtype=np.float64))
        cpu, metal = (
            float(bf16_values(reference[row, column])),
            float(bf16_values(candidate[row, column])),
        )
        result.append(
            {
                "row": int(row),
                "column": int(column),
                "fp64_sum": value,
                "cpu": cpu,
                "metal": metal,
                "cpu_absolute_error": abs(cpu - value),
                "metal_absolute_error": abs(metal - value),
                "fp64_then_f32_then_bf16": float(bf16_values(bf16_bits(value))),
            }
        )
    return result


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def read_boundary(path, width, max_rows):
    size = path.stat().st_size
    if not size or size % (width * 2) or size > max_rows * width * 2:
        raise ValueError(f"invalid or oversized boundary: {path}")
    return np.fromfile(path, dtype="<u2").reshape(-1, width)


@dataclass(frozen=True)
class PLETrace:
    item: int
    embeddings: np.ndarray
    native_ple: np.ndarray
    oracle_ple: np.ndarray
    files: tuple[Path, ...]

    @classmethod
    def load(cls, native, oracle, item):
        files = tuple(
            directory / f"item-{item}-{stage}-0.bf16"
            for directory in (native, oracle)
            for stage in ("embeddings", "ple")
        )
        native_x, native_ple, oracle_x, oracle_ple = [
            read_boundary(path, width, 128)
            for path, width in zip(files, (HIDDEN_WIDTH, PLE_WIDTH) * 2)
        ]
        if not np.array_equal(native_x, oracle_x):
            raise ValueError(
                "PLE replay requires identical embeddings; diagnose the earlier stage first"
            )
        if not native_x.shape[0] == native_ple.shape[0] == oracle_ple.shape[0]:
            raise ValueError("inconsistent token counts")
        return cls(item, native_x, native_ple, oracle_ple, files)


def replay(n, embeddings, weight, scale, output, label):
    import torch

    def bits(tensor):
        return tensor.bfloat16().contiguous().view(torch.uint16).numpy()

    def record(stage, expected, actual):
        reference, candidate = bits(expected), bits(actual)
        np.save(output / f"{label}-{stage}-cpu.npy", reference)
        np.save(output / f"{label}-{stage}-metal.npy", candidate)
        return compare_bits(reference, candidate)

    x = torch.from_numpy(embeddings.copy()).view(torch.bfloat16)
    rows, columns = x.shape[0], weight.shape[0]
    target = n.tensor(torch.empty((rows, columns), dtype=torch.bfloat16))
    n.linear(target, n.tensor(x), n.tensor(weight), rows, HIDDEN_WIDTH, columns)
    metal_linear = n.read(target, (rows, columns)).bfloat16()
    cpu_linear = torch.nn.functional.linear(x, weight)
    report = {"linear_same_input": record("linear", cpu_linear, metal_linear)}
    report["dot_witnesses"] = dot_witnesses(
        x.float().numpy(), weight.float().numpy(), bits(cpu_linear), bits(metal_linear)
    )
    n.clear()

    # Isolate the norm from GEMM error using the CPU projection for BOTH paths.
    scaled = cpu_linear * PLE_SCALE
    source = n.tensor(scaled)
    isolated = n.tensor(torch.empty_like(scaled))
    n.rms_norm(isolated, source, n.tensor(scale), HIDDEN_WIDTH, 1e-6)
    z = scaled.float().reshape(-1, HIDDEN_WIDTH)
    expected = (
        (z * (z.square().mean(-1, keepdim=True) + 1e-6).pow(-0.5) * scale.float())
        .bfloat16()
        .reshape(rows, columns)
    )
    report["norm_same_input"] = record(
        "norm", expected, n.read(isolated, (rows, columns))
    )
    n.clear()

    # Replay the production sequence, including a BF16 boundary after scale.
    source = n.tensor(metal_linear)
    n.scale(source, source, PLE_SCALE)
    metal_scaled = n.read(source, (rows, columns)).bfloat16()
    report["scale_same_input"] = record("scale", metal_linear * PLE_SCALE, metal_scaled)
    n.rms_norm(source, source, n.tensor(scale), HIDDEN_WIDTH, 1e-6)
    actual = n.read(source, (rows, columns)).bfloat16()
    report["ple_replay"] = record("ple", expected, actual)
    n.clear()
    return report, bits(expected), bits(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Pinned Gemma model.safetensors"
    )
    parser.add_argument(
        "--native", type=Path, required=True, help="Existing native short text trace"
    )
    parser.add_argument(
        "--oracle", type=Path, required=True, help="Matching CPU oracle trace"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    if min(args.items) < 0 or len(set(args.items)) != len(args.items):
        parser.error("items must be unique nonnegative indices")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    artifacts = [args.model]
    with safe_open(args.model, framework="pt") as model:
        weight = model.get_tensor(
            "language_model.ple.per_layer_model_projection.weight"
        )
        scale = model.get_tensor("language_model.ple.per_layer_projection_norm.weight")
    if (
        weight.dtype != torch.bfloat16
        or scale.dtype != torch.bfloat16
        or weight.shape != (PLE_WIDTH, HIDDEN_WIDTH)
        or scale.shape != (HIDDEN_WIDTH,)
    ):
        raise ValueError("unsupported PLE tensor layout/dtype")
    report = {
        "kind": "diagnostic",
        "qualification_claim": False,
        "model_revision_expected": MODEL_REVISION,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": sys.argv,
        "torch_version": torch.__version__,
        "owned_storage_limit_bytes": LIMIT,
        "reserve_bytes": 2 * LIMIT,
        "items": [],
    }
    start = time.monotonic()
    sys.path.insert(0, str(ROOT))
    from tests.suite_numeric_api import Numeric

    with Numeric("metal") as n:
        n.configure_metal_budget(reserve_bytes=2 * LIMIT, storage_limit_bytes=LIMIT)
        # Keep one trace live at a time even for an explicit longer item list.
        for item in args.items:
            trace = PLETrace.load(args.native, args.oracle, item)
            artifacts.extend(trace.files)
            result, cpu, metal = replay(
                n, trace.embeddings, weight, scale, args.output, f"item-{item}"
            )
            result.update(
                item=item,
                tokens=trace.embeddings.shape[0],
                cpu_matches_recorded=compare_bits(trace.oracle_ple, cpu),
                metal_matches_recorded=compare_bits(trace.native_ple, metal),
            )
            report["items"].append(result)
        usage = n.memory_usage()
        report["numeric_live_bytes"], report["numeric_peak_bytes"] = usage
        if usage.live_bytes or usage.peak_bytes > LIMIT:
            raise RuntimeError("numeric storage did not respect its lifetime/budget")
    report["seconds"] = time.monotonic() - start
    report["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["replay_verdict"] = (
        "PASS"
        if all(
            v[k]["bit_exact"]
            for v in report["items"]
            for k in ("cpu_matches_recorded", "metal_matches_recorded")
        )
        else "FAIL"
    )
    report["strict_ple_gate"] = (
        "PASS" if all(v["ple_replay"]["bit_exact"] for v in report["items"]) else "FAIL"
    )
    artifacts += [
        Path(__file__),
        ROOT / "tests/suite_numeric_api.py",
        ROOT / "ds4_numeric_metal.m",
        ROOT / "metal/numeric.metal",
        ROOT / "tests/libds4_numeric_metal.so",
    ]
    artifacts += sorted(args.output.glob("*.npy"))
    report["artifacts"] = [file_record(p) for p in artifacts]
    (args.output / "diagnosis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "replay_verdict",
                    "strict_ple_gate",
                    "numeric_peak_bytes",
                    "seconds",
                )
            }
        )
    )
    return 0 if report["replay_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
