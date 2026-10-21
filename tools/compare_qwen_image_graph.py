#!/usr/bin/env python3
"""Compare complete T09/T10 tensors using the unchanged T00 ceilings."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from suite_numeric_metrics import compare, rgba_metrics


def read(p):
    if p.suffix == ".bf16":
        return (np.fromfile(p, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    return np.fromfile(p, dtype="<f4")


def latent_passed(metrics, limits):
    return (
        metrics["relative_l2"] <= limits["relative_l2"]
        and metrics["rmse"] <= limits["rmse_max"]
    )


def compare_graph(oracle, native):
    tolerances = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/performance/native-suite-t00-20261020/tolerances.json"
        ).read_text()
    )
    result = {"case": oracle.name, "boundaries": [], "passed": True}
    meta = json.loads((oracle / "oracle.json").read_text())
    expected = {"encoded-0.bf16", "pixels.f32"}
    if "steps" in meta:
        if meta["steps"] != 8 or meta["cfg"] != 1:
            raise ValueError("expected the frozen eight-step CFG=1 recipe")
        expected = set()
        for step in range(8):
            expected.update(f"step-{step}-block-{layer}.bf16" for layer in range(32))
            expected.update(
                f"step-{step}-{stage}-0.bf16"
                for stage in ("joint", "modulation", "velocity")
            )
            expected.add(f"step-{step}-latent-{step}.f32")
    fixture = oracle / ("input.ds4g" if "steps" in meta else "input.ds4v")
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != meta["fixture_sha256"]:
        raise ValueError("oracle fixture identity mismatch")
    files = {src.name for src in oracle.iterdir() if src.suffix in (".bf16", ".f32")}
    for name in sorted(files | expected):
        src = oracle / name
        if src.suffix not in [".bf16", ".f32"] or src.name in [
            "decode.f32",
            "sigmas.f32",
        ]:
            continue
        dest = native / src.name
        required = src.name in expected or src.name.startswith("uncached-same-input-")
        if not dest.exists() or not src.exists():
            result["boundaries"].append(
                {
                    "name": src.name,
                    "passed": False,
                    "required": required,
                    "error": "missing oracle/native output",
                }
            )
            if required:
                result["passed"] = False
            continue
        if src.name == "pixels.f32":
            shape = (meta["height"], meta["width"], 4)
            m = rgba_metrics(read(src).reshape(shape), read(dest).reshape(shape))
            t = tolerances["qwen_rgba"]
            passed = (
                m["rmse"] <= t["rmse_max"]
                and m["ssim_rgb"] >= t["ssim_rgb_min"]
                and m["alpha_mae"] <= t["alpha_mae_max"]
            )
        else:
            m = compare(read(src), read(dest))
            t = tolerances["qwen_latent_boundary"]
            passed = latent_passed(m, t)
        # T10 gates normalized encode latents and output pixels. Internal VAE
        # feature maps are retained for diagnosis; T00 defines no feature-map
        # absolute ceiling for their very different activation scales.
        result["boundaries"].append(
            {
                "name": src.name,
                **m,
                "passed": passed,
                "required": required,
                "native_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            }
        )
        if required:
            result["passed"] &= passed
    for step in range(8):
        if "steps" in meta and step:
            full = native / f"uncached-same-input-velocity-{step}.bf16"
            cached = native / f"step-{step}-velocity-0.bf16"
            m = compare(read(full), read(cached))
            t = tolerances["qwen_latent_boundary"]
            passed = latent_passed(m, t)
            result["boundaries"].append(
                {"name": f"cache-same-input-{step}", **m, "passed": passed}
            )
            result["passed"] &= passed
        uncached = native / f"uncached-latent-{step}.f32"
        if "steps" in meta and not uncached.exists():
            result["boundaries"].append(
                {
                    "name": f"cache-equivalence-{step}",
                    "passed": False,
                    "required": False,
                    "error": "missing uncached output",
                }
            )
        if uncached.exists():
            cached = native / f"step-{step}-latent-{step}.f32"
            m = compare(read(uncached), read(cached))
            t = tolerances["qwen_latent_boundary"]
            passed = latent_passed(m, t)
            result["boundaries"].append(
                {
                    "name": f"uncached-trajectory-{step}",
                    **m,
                    "passed": passed,
                    "required": False,
                }
            )
    tiled = native / "pixels-tiled.f32"
    if meta.get("width", 0) >= 512 and not tiled.exists():
        result["boundaries"].append(
            {
                "name": "tiled-equivalence",
                "passed": False,
                "error": "missing tiled output",
            }
        )
        result["passed"] = False
    if tiled.exists():
        shape = (meta["height"], meta["width"], 4)
        ref, actual = (
            read(native / "pixels.f32").reshape(shape),
            read(tiled).reshape(shape),
        )
        m = rgba_metrics(ref, actual)
        # Include the entire tile border strips, not just aggregate RGB.
        edge = np.zeros(shape[:2], bool)
        plan = json.loads((native / "tile-plan.json").read_text())
        if (
            plan["core"] < 2
            or plan["halo"] != 14
            or plan["overlap"] != 1
            or plan["scale"] != 16
        ):
            raise ValueError("invalid decoder tile metadata")
        stride = (plan["core"] - plan["overlap"]) * plan["scale"]
        for axis in (0, 1):
            for start in range(stride, shape[axis], stride):
                slices = [slice(None), slice(None)]
                slices[axis] = slice(max(0, start - 8), min(shape[axis], start + 8))
                edge[tuple(slices)] = True
        m["alpha_edge_mae"] = float(
            np.mean(np.abs(ref[..., 3][edge] - actual[..., 3][edge]))
        )
        t = tolerances["qwen_rgba"]
        passed = (
            m["rmse"] <= t["rmse_max"]
            and m["ssim_rgb"] >= t["ssim_rgb_min"]
            and m["alpha_mae"] <= t["alpha_mae_max"]
            and m["alpha_edge_mae"]
            <= tolerances["image_task_gates"]["alpha_edge_mae_max"]
        )
        result["boundaries"].append(
            {"name": "tiled-equivalence", **m, "passed": passed}
        )
        result["passed"] &= passed
    if "width" in meta and not (native / "result.png").exists():
        result["png_alpha_preserved"] = False
        result["passed"] = False
    if (native / "result.png").exists():
        from PIL import Image

        with Image.open(native / "result.png") as image:
            actual = np.asarray(image)
            pixels = read(native / "pixels.f32").reshape(actual.shape)
            valid = (
                image.mode == "RGBA"
                and image.size == (meta["width"], meta["height"])
                and np.array_equal(actual, np.rint(pixels * 255).astype(np.uint8))
            )
            result["png_alpha_preserved"] = bool(valid)
            result["passed"] &= valid
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_graph(args.oracle, args.native)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "case": args.oracle.name,
            "boundaries": [],
            "passed": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    failed = [
        b for b in result["boundaries"] if not b["passed"] and b.get("required", True)
    ]
    diagnostic_failed = [
        b["name"]
        for b in result["boundaries"]
        if not b["passed"] and not b.get("required", True)
    ]
    print(
        json.dumps(
            {
                "case": result["case"],
                "error": result.get("error"),
                "passed": result["passed"],
                "boundaries": len(result["boundaries"]),
                "failed": failed[:6],
                "diagnostic_failed": diagnostic_failed,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
