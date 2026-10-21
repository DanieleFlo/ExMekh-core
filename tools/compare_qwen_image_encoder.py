#!/usr/bin/env python3
"""Compare complete native conditioning boundaries with the pinned oracle.

Input identity is checked before numerics. T00 thresholds are read verbatim;
max/relative/RMSE are retained for every boundary, including failing ones.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bf16(path):
    return (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def compare(oracle, native):
    metadata = json.loads((oracle / "oracle.json").read_text())
    inputs = np.load(oracle / "inputs.npz")
    identity = {
        "tokens": np.array_equal(
            inputs["input_ids"].ravel(), np.fromfile(native / "tokens.u32", dtype="<u4")
        ),
        "positions": (oracle / "positions.u32").read_bytes()
        == (native / "positions.u32").read_bytes(),
        "image_mask": np.array_equal(
            inputs["input_ids"].ravel() == 151655,
            np.fromfile(native / "image_mask.u8", dtype="u1"),
        ),
    }
    if "image_grid_thw" in inputs:
        identity["grids"] = np.array_equal(
            inputs["image_grid_thw"][:, 1:].ravel(),
            np.fromfile(native / "grids.u32", dtype="<u4"),
        )
        identity["pixels"] = np.array_equal(
            inputs["pixel_values"].ravel(),
            np.fromfile(native / "pixels.f32", dtype="<f4"),
        )
        pixels = np.fromfile(native / "pixels.f32", dtype="<f4")
        identity["pixels_max_abs"] = float(
            np.max(np.abs(inputs["pixel_values"].ravel() - pixels))
        )
    thresholds = json.loads(
        (
            ROOT / "docs/performance/native-suite-t00-20261020/tolerances.json"
        ).read_text()
    )["qwen_latent_boundary"]
    boundaries = []
    for path in sorted(native.glob("*.bf16")):
        if not (oracle / path.name).exists():
            continue
        a, b = bf16(path), bf16(oracle / path.name)
        if a.shape != b.shape:
            boundaries.append(
                {
                    "name": path.stem,
                    "shape": list(a.shape),
                    "oracle_shape": list(b.shape),
                    "verdict": "FAIL",
                }
            )
            continue
        delta = a.astype(np.float64) - b
        rmse = float(np.sqrt(np.mean(delta**2)))
        relative = float(
            np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-30)
        )
        finite = bool(np.isfinite(a).all())
        boundaries.append(
            {
                "name": path.stem,
                "elements": int(a.size),
                "max_abs": float(np.max(np.abs(delta))),
                "rmse": rmse,
                "relative_l2": relative,
                "bit_exact": path.read_bytes() == (oracle / path.name).read_bytes(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "verdict": "PASS"
                if finite
                and rmse <= thresholds["rmse_max"]
                and relative <= thresholds["relative_l2"]
                else "FAIL",
            }
        )
    expected = {"embeddings-0", "conditioning-0"} | {f"text-{i}" for i in range(36)}
    if metadata["references"]:
        expected |= (
            {f"vision-{i}" for i in range(27)}
            | {f"deepstack-{i}" for i in range(3)}
            | {"vision_merged-0"}
        )
    identity["complete_boundaries"] = expected == {b["name"] for b in boundaries}
    passed = all(v for k, v in identity.items() if k != "pixels_max_abs") and all(
        b["verdict"] == "PASS" for b in boundaries
    )
    return {
        "case": metadata["case"],
        "identity": identity,
        "thresholds": thresholds,
        "boundaries": boundaries,
        "verdict": "PASS" if passed else "FAIL",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("oracle", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.oracle, args.native)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "boundaries"}, indent=2))
    for item in result["boundaries"]:
        print(item["name"], item["verdict"], item.get("rmse"), item.get("relative_l2"))
    raise SystemExit(result["verdict"] != "PASS")
