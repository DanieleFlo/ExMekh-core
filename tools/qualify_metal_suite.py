#!/usr/bin/env python3
"""Bounded Apple Silicon development subset of the existing suite oracles.

Run text, then media, separately from LLM and image jobs. Frozen full-corpus
qualifiers remain the release gates; this explicit subset never claims them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .qualify_metal_command import run
    from .suite_numeric_metrics import embedding_metrics
    from .suite_visual_cases import write_fixture
else:
    from qualify_metal_command import run
    from suite_numeric_metrics import embedding_metrics
    from suite_visual_cases import write_fixture


def compare_vectors(native: Path, oracle: Path, dimensions: int) -> dict:
    actual = np.fromfile(native, np.float32).reshape(-1, dimensions)
    expected = np.load(oracle)[:, :dimensions].copy()
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    metrics = embedding_metrics(expected, actual)
    ceilings = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )["gemma_forward"]
    passed = (
        metrics["max_abs"] <= ceilings["max_abs"]
        and metrics["rmse"] <= ceilings["rmse_max"]
        and metrics["cosine_min"] >= ceilings["cosine_min"]
        and metrics["unit_norm_error_max"] <= ceilings["unit_norm_error_max"]
    )
    return {
        "metrics": metrics,
        "tolerances": ceilings,
        "status": "PASS" if passed else "FAIL",
    }


def compare_boundaries(native: Path, oracle: Path) -> dict:
    """Apply the existing strict BF16/probability gate to every oracle dump."""
    references = sorted(
        path
        for path in oracle.iterdir()
        if path.suffix == ".bf16" or "-probabilities-" in path.name
    )
    if not references:
        raise ValueError("oracle has no internal boundaries")
    boundaries = []
    for reference in references:
        candidate = native / reference.name
        bf16 = reference.suffix == ".bf16"
        dtype = np.uint16 if bf16 else np.float32
        if any(
            path.stat().st_size % np.dtype(dtype).itemsize
            for path in (reference, candidate)
        ):
            raise ValueError(f"truncated boundary: {reference.name}")
        expected = np.fromfile(reference, dtype)
        actual = np.fromfile(candidate, dtype)
        if not expected.size or expected.shape != actual.shape:
            raise ValueError(f"invalid boundary shape: {reference.name}")
        exact = expected.tobytes() == actual.tobytes()
        if bf16:
            expected = (expected.astype(np.uint32) << 16).view(np.float32)
            actual = (actual.astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"non-finite boundary: {reference.name}")
        delta = actual.astype(np.float64) - expected
        boundaries.append(
            {
                "file": reference.name,
                "elements": int(expected.size),
                "bit_exact": exact,
                "max_abs": float(np.abs(delta).max()),
                "rmse": float(np.sqrt(np.square(delta).mean())),
            }
        )
    return {
        "criterion": "bit_exact",
        "status": "PASS" if all(row["bit_exact"] for row in boundaries) else "FAIL",
        "exact": sum(row["bit_exact"] for row in boundaries),
        "count": len(boundaries),
        "boundaries": boundaries,
    }


def analyze_existing(output: Path, kind: str) -> dict:
    """Recheck saved vectors and mandatory hidden states without model execution."""
    vectors = {
        str(dimensions): compare_vectors(
            output / f"native-{dimensions}/vectors.f32",
            output / "oracle/vectors.npy",
            dimensions,
        )
        for dimensions in (128, 256, 512, 768)
    }
    hidden = compare_boundaries(output / "native-128", output / "oracle")
    report = {
        "scope": "Saved local subset; independent strict internal and final-vector gates",
        "kind": kind,
        "native_trace": str(output / "native-128"),
        "oracle_trace": str(output / "oracle"),
        "output_vectors": vectors,
        "internal_boundaries": hidden,
        "status": "PASS"
        if hidden["status"] == "PASS"
        and all(row["status"] == "PASS" for row in vectors.values())
        else "FAIL",
    }
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("text", "media"), required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--text-asset", type=Path)
    parser.add_argument("--media-asset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Compare existing vectors and strict internal dumps; never open a model",
    )
    args = parser.parse_args()
    if args.analyze_only:
        report = analyze_existing(args.output, args.kind)
        print(report["status"])
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    if not all((args.prepared, args.bundle, args.text_asset)):
        parser.error("--prepared, --bundle and --text-asset are required for inference")
    args.output.mkdir(parents=True, exist_ok=False)
    corpus_root = Path("tests/fixtures/native-suite-v1")
    corpus = json.loads((corpus_root / "corpus.json").read_text())
    if args.kind == "text":
        texts = [
            "Il gatto dorme sul divano.",
            "A cat sleeps on the sofa.",
            "The train arrives at the station.",
        ]
        cases = [{"id": f"text-{i}", "text": text} for i, text in enumerate(texts)]
        base = [
            "tests/probe_native_suite",
            str(args.bundle),
            str(args.prepared / "models"),
            str(args.text_asset),
            "gemma",
        ]
        oracle_script = "tools/run_embedding_gemma2_oracle.py"
    else:
        if args.media_asset is None:
            parser.error("--media-asset is required for media")
        selected = [
            next(item for item in corpus["items"] if item["id"] == name)
            for name in ("image-0", "audio-0", "video-0", "interleaved-0")
        ]
        cases = []
        for item in selected:
            prompt = []
            media = []
            for part in item["content"]:
                if part["type"] == "text":
                    prompt.append(part["text"])
                else:
                    medium = {
                        "type": part["type"],
                        "path": str(corpus_root / part["path"]),
                    }
                    if part["type"] in ("image", "video"):
                        medium["soft"] = 70
                    if part["type"] == "video":
                        medium["indices"] = [0, 1]
                    media.append(medium)
                    prompt.append(f"<|{part['type']}|>")
            cases.append({"id": item["id"], "prompt": " ".join(prompt), "media": media})
        write_fixture(args.output / "cases.bin", cases)
        base = [
            "tests/probe_gemma_visual",
            str(args.bundle),
            str(args.prepared / "models"),
            str(args.text_asset),
            str(args.media_asset),
            str(args.output / "cases.bin"),
        ]
        oracle_script = "tools/run_gemma_visual_oracle.py"
    case_path = args.output / "cases.json"
    case_path.write_text(json.dumps(cases, indent=2) + "\n")
    checks = []
    # First 128 dimensions, then every exposed Matryoshka width. Only three
    # short texts/four brief media items are passed to each native process.
    for dimensions in (128, 256, 512, 768):
        native = args.output / f"native-{dimensions}"
        command = base + [str(native)] + (texts if args.kind == "text" else [])
        prefix = (
            "DS4_GEMMA_DIMENSIONS"
            if args.kind == "text"
            else "DS4_GEMMA_VISUAL_DIMENSIONS"
        )
        checks.append(
            run(
                ["env", f"{prefix}={dimensions}", *command],
                args.output / f"run-{dimensions}",
                120,
                0,
            )
        )
        if checks[-1]["status"] != "PASS":
            break
    if len(checks) == 4 and all(c["status"] == "PASS" for c in checks):
        oracle = args.output / "oracle"
        checks.append(
            run(
                [
                    sys.executable,
                    oracle_script,
                    "--prepared",
                    str(args.prepared),
                    "--cases",
                    str(case_path),
                    "--output",
                    str(oracle),
                    "--device",
                    "cpu",
                ],
                args.output / "run-oracle",
                120,
                0,
            )
        )
        if checks[-1]["status"] == "PASS":
            for dimensions in (128, 256, 512, 768):
                checks.append(
                    compare_vectors(
                        args.output / f"native-{dimensions}/vectors.f32",
                        oracle / "vectors.npy",
                        dimensions,
                    )
                )
    report = {
        "scope": "bounded vector and strict internal subset; full-corpus qualification remains separate",
        "kind": args.kind,
        "cases": cases,
        "checks": checks,
        "status": "PASS"
        if len(checks) == 9 and all(c["status"] == "PASS" for c in checks)
        else "FAIL"
        if any(c["status"] == "FAIL" for c in checks)
        else "SKIP_RESOURCE_LIMIT"
        if any(c["status"] == "SKIP_RESOURCE_LIMIT" for c in checks)
        else "NOT_VERIFIED",
    }
    if args.kind == "text" and (args.output / "native-768/vectors.f32").exists():
        vectors = np.fromfile(
            args.output / "native-768/vectors.f32", np.float32
        ).reshape(-1, 768)
        report["ranking"] = {
            "related": float(vectors[0] @ vectors[1]),
            "unrelated": float(vectors[0] @ vectors[2]),
        }
        if report["ranking"]["related"] <= report["ranking"]["unrelated"]:
            report["status"] = "FAIL"
    report["output_vector_status"] = report["status"]
    if len(checks) == 9 and checks[4]["status"] == "PASS":
        analysis = analyze_existing(args.output, args.kind)
        report["internal_boundaries"] = analysis["internal_boundaries"]
        if analysis["status"] != "PASS":
            report["status"] = "FAIL"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
