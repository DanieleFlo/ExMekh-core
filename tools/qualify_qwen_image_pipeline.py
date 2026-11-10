#!/usr/bin/env python3
"""Qualify the native C image pipeline against serial frozen components.

Default fixtures cover generation, reference order, multi-edit, alpha,
typography and a second aspect ratio. --full-corpus adds all T00 recipes at
their frozen sizes/seeds; none are silently resized to fit the device.
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .qualify_metal_command import run as bounded_run
    from .suite_numeric_metrics import compare, rgba_metrics
else:
    from qualify_metal_command import run as bounded_run
    from suite_numeric_metrics import compare, rgba_metrics


def compare_boundary(reference, candidate, limits):
    metrics = compare(reference, candidate)
    metrics["verdict"] = (
        "PASS"
        if metrics["relative_l2"] <= limits["relative_l2"]
        and metrics["rmse"] <= limits["rmse_max"]
        else "FAIL"
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-corpus", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--only-admitted",
        action="store_true",
        help="Run only cases admitted in --admission-report; keep excluded cases separate",
    )
    p.add_argument("--admission-report", type=Path)
    p.add_argument(
        "--case-ids",
        nargs="+",
        help="Explicit development subset, never labelled full corpus",
    )
    p.add_argument(
        "--native-only",
        action="store_true",
        help="Functional native subset only; leave independent quality gates open",
    )
    p.add_argument(
        "--skip-native", action="store_true", help="Reuse existing native artifacts"
    )
    p.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Compare existing independent artifacts",
    )
    p.add_argument("--oracle-device", choices=("cuda", "cpu", "mps"), default="cuda")
    p.add_argument(
        "--oracle-encoder-device",
        choices=("cuda", "cpu", "mps"),
        help="Explicit text/vision encoder reference device; otherwise use --oracle-device",
    )
    p.add_argument(
        "--oracle-vae-device",
        choices=("cuda", "cpu", "mps"),
        help="Explicit reference device for VAE encode/decode; otherwise use --oracle-device",
    )
    p.add_argument("--oracle-offload", action="store_true")
    p.add_argument(
        "--bounded",
        action="store_true",
        help="One image case, 300-second subprocess deadlines and Metal memory monitoring",
    )
    a = p.parse_args()
    oracle_devices = {
        "encoder": a.oracle_encoder_device or a.oracle_device,
        "encode": a.oracle_vae_device or a.oracle_device,
        "dit": a.oracle_device,
        "decode": a.oracle_vae_device or a.oracle_device,
    }
    a.output.mkdir(parents=True, exist_ok=a.skip_native)
    import numpy as np
    from PIL import Image

    fixtures = a.output / "fixtures"
    fixtures.mkdir(exist_ok=a.skip_native)
    y, x = np.mgrid[:256, :256]
    rgb = np.stack(((x * 3 + y) % 256, (y * 5) % 256, (x + y * 2) % 256), -1).astype(
        np.uint8
    )
    other = np.roll(rgb, 47, axis=1).copy()
    rgba = np.concatenate((rgb, ((x + y) % 256).astype(np.uint8)[..., None]), -1)

    def reference(name, pixels):
        h, w, c = pixels.shape
        path = fixtures / f"{name}.ds4i"
        path.write_bytes(struct.pack("<3I", w, h, c) + pixels.tobytes())
        return str(path)

    r0, r1, ra = (
        reference("reference", rgb),
        reference("second", other),
        reference("rgba", rgba),
    )
    prompt = "A red square beside a blue circle, clean white background."
    specs = [
        {
            "id": name,
            "prompt": prompt,
            "width": 256,
            "height": 256,
            "references": refs,
            "seed": 20261020,
        }
        for name, refs in [
            ("text", []),
            ("reference", [r0]),
            ("multi", [r0, r1]),
            ("reversed", [r1, r0]),
            ("rgba", [ra]),
        ]
    ]
    small = reference("small", rgb[:96, :144].copy())
    specs.append(
        {
            "id": "resized-reference",
            "prompt": prompt,
            "width": 256,
            "height": 256,
            "references": [small],
            "seed": 20261020,
            "reference_resolution": 256,
        }
    )
    specs += [
        {
            "id": "alpha-512",
            "prompt": "A single red circle with a transparent background and clean edges.",
            "width": 512,
            "height": 512,
            "references": [],
            "seed": 20261021,
        },
        {
            "id": "type-wide",
            "prompt": 'A white sign with the exact text "DS4 OPEN".',
            "width": 768,
            "height": 512,
            "references": [],
            "seed": 20261020,
        },
    ]
    if a.full_corpus:
        corpuspath = Path("tests/fixtures/native-suite-v1/corpus.json")
        corpus = json.loads(corpuspath.read_text())
        refs = {}
        for item in corpus["items"]:
            if item["id"].startswith("image-"):
                media = corpuspath.parent / item["artifact"]["path"]
                if (
                    hashlib.sha256(media.read_bytes()).hexdigest()
                    != item["artifact"]["sha256"]
                ):
                    raise ValueError("frozen T00 reference changed")
                # Preserve frozen media geometry. An unsupported reference
                # must fail admission, never become an easier resized corpus.
                image = Image.open(media).convert("RGB")
                refs[item["id"]] = reference(item["id"], np.array(image))
        for recipe in corpus["qwen_recipes"]:
            for seed in recipe["seeds"]:
                specs.append(
                    {
                        "id": f"{recipe['id']}-{seed}",
                        "prompt": recipe["prompt"],
                        "width": recipe["width"],
                        "height": recipe["height"],
                        "references": [refs[i] for i in recipe["reference_ids"]],
                        "seed": seed,
                        "split": recipe["split"],
                        "kind": recipe["kind"],
                        "reference_resolution": 1024,
                    }
                )
    excluded = []
    if a.only_admitted:
        if not a.admission_report:
            raise ValueError("--only-admitted needs --admission-report")
        admitted = {
            c["id"]
            for c in json.loads(a.admission_report.read_text())
            if c["status"] == 0
        }
        excluded = [c["id"] for c in specs if c["id"] not in admitted]
        specs = [c for c in specs if c["id"] in admitted]
    if a.case_ids:
        specs = [c for c in specs if c["id"] in a.case_ids]
        if len(specs) != len(a.case_ids):
            raise ValueError("case id not found")
    if a.bounded and len(specs) != 1:
        raise ValueError(
            "bounded qualification requires exactly one --case-ids selection"
        )
    inputs = []
    for case in specs:
        if a.plan_only:
            # Admission uses shape and counts only. Keep absent oracle noise
            # from blocking a resource check; these fixtures never run inference.
            noise = np.zeros(
                (case["height"] // 16, case["width"] // 16, 64), dtype=np.float32
            )
            case["noise_kind"] = "admission_zero"
        elif "split" in case:
            noise_root = Path("performance-results/native-suite-qwen-noise-20261020")
            frozen = (
                noise_root / f"{case['width']}x{case['height']}-seed-{case['seed']}.npz"
            )
            manifest = json.loads(
                Path(
                    "docs/performance/native-suite-t00-20261020/noise.json"
                ).read_text()
            )
            record = next(r for r in manifest["latents"] if r["path"] == frozen.name)
            if (
                hashlib.sha256(frozen.read_bytes()).hexdigest()
                != record["artifact_sha256"]
            ):
                raise ValueError("frozen T00 noise changed")
            with np.load(frozen) as values:
                noise = values["latents"]
        else:
            noise = np.random.default_rng(case["seed"]).standard_normal(
                (case["height"] // 16, case["width"] // 16, 64), dtype=np.float32
            )
        path = fixtures / f"{case['id']}-noise.f32"
        noise.tofile(path)
        case["noise"] = str(path)
        text = case["prompt"].encode()
        data = (
            struct.pack(
                "<8s4IQ",
                b"DS4PIP2\0",
                case["width"],
                case["height"],
                len(case["references"]),
                len(text),
                case["seed"],
            )
            + struct.pack("<I", case.get("reference_resolution", 0))
            + text
        )
        for ref in case["references"]:
            data += Path(ref).read_bytes()
        data += noise.tobytes()
        fixture = fixtures / f"{case['id']}.ds4p"
        fixture.write_bytes(data)
        inputs.append(fixture)
        case["fixture_sha256"] = hashlib.sha256(data).hexdigest()
    cases = a.output / "cases.json"
    cases.write_text(json.dumps(specs, indent=2) + "\n")
    runs_path = a.output / "runs.json"
    runs = (
        json.loads(runs_path.read_text())
        if a.skip_native and runs_path.exists()
        else []
    )

    def run(name, command, env=None):
        start = time.monotonic()
        if a.bounded:
            environment = [f"{key}={value}" for key, value in (env or {}).items()]
            evidence = bounded_run(
                ["env", *environment, *map(str, command)],
                a.output / f"run-{name}",
                300,
                0,
                "S3 256px image or independent reference component",
            )
            (a.output / f"{name}.log").write_bytes(
                (a.output / f"run-{name}/command.log").read_bytes()
            )
            result = subprocess.CompletedProcess(
                command, 0 if evidence["status"] == "PASS" else 1
            )
        else:
            with (a.output / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    list(map(str, command)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **(env or {})},
                    check=False,
                )
        runs.append(
            {
                "name": name,
                "command": list(map(str, command)),
                "seconds": time.monotonic() - start,
                "returncode": result.returncode,
            }
        )
        runs_path.write_text(json.dumps(runs, indent=2) + "\n")
        print(name, result.returncode, round(runs[-1]["seconds"], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed, see {a.output / name}.log")

    if not a.skip_native:
        run(
            "native",
            [
                "tests/probe_qwen_image_pipeline",
                a.bundle,
                a.prepared / "models",
                a.tokenizer,
                a.output / "native",
                *inputs,
            ],
            {"DS4_SUITE_PLAN_ONLY": "1"}
            if a.plan_only
            else {"DS4_SUITE_CASES_ONLY": "1"}
            if a.native_only or a.bounded
            else None,
        )
    if a.plan_only:
        plans = [
            dict(
                id=c["id"],
                **json.loads((a.output / f"native/case-{i}/plan.json").read_text()),
            )
            for i, c in enumerate(specs)
        ]
        (a.output / "admission.json").write_text(json.dumps(plans, indent=2) + "\n")
        print(
            "admission",
            sum(p["status"] == 0 for p in plans),
            "/",
            len(plans),
            flush=True,
        )
        return
    if a.native_only:
        (a.output / "functional.json").write_text(
            json.dumps(
                {
                    "native_status": "PASS",
                    "quality_status": "NOT_VERIFIED",
                    "scope": "explicit native subset",
                    "case_ids": [c["id"] for c in specs],
                },
                indent=2,
            )
            + "\n"
        )
        return
    if not a.skip_oracle:
        for stage in ("encoder", "encode", "dit", "decode"):
            run(
                stage,
                [
                    sys.executable,
                    "tools/run_qwen_image_pipeline_oracle.py",
                    "--prepared",
                    a.prepared,
                    "--cases",
                    cases,
                    "--output",
                    a.output / "oracle",
                    "--stage",
                    stage,
                    "--device",
                    oracle_devices[stage],
                    *(["--offload"] if a.oracle_offload else []),
                ],
            )
    tol = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )

    def bf16(path):
        return (np.fromfile(path, "<u2").astype(np.uint32) << 16).view(np.float32)

    limits = tol["qwen_latent_boundary"]
    report = {"cases": []}
    passed = True
    for i, case in enumerate(specs):
        native = a.output / f"native/case-{i}"
        oracle = a.output / "oracle" / case["id"]
        boundaries = []
        ok = True
        conditioning = compare_boundary(
            bf16(oracle / "conditioning.bf16"),
            bf16(native / "conditioning.bf16"),
            limits,
        )
        ok &= conditioning["verdict"] == "PASS"
        references = None
        if case["references"]:
            references = compare_boundary(
                bf16(oracle / "references.bf16"),
                np.concatenate(
                    [
                        bf16(native / f"reference-{j}.bf16")
                        for j in range(len(case["references"]))
                    ]
                ),
                limits,
            )
            ok &= references["verdict"] == "PASS"
        for step in range(8):
            m = compare_boundary(
                np.fromfile(oracle / f"latent-{step}.f32", np.float32),
                np.fromfile(native / f"latent-{step}.f32", np.float32),
                limits,
            )
            boundaries.append(m)
            ok &= m["verdict"] == "PASS"
        shape = (case["height"], case["width"], 4)
        m = rgba_metrics(
            np.fromfile(oracle / "rgba.f32", np.float32).reshape(shape),
            np.fromfile(native / "rgba.f32", np.float32).reshape(shape),
        )
        ok &= (
            m["rmse"] <= tol["qwen_rgba"]["rmse_max"]
            and m["ssim_rgb"] >= tol["qwen_rgba"]["ssim_rgb_min"]
            and m["alpha_mae"] <= tol["qwen_rgba"]["alpha_mae_max"]
        )
        record = {
            "id": case["id"],
            "width": case["width"],
            "height": case["height"],
            "seed": case["seed"],
            "references": case["references"],
            "fixture_sha256": case["fixture_sha256"],
            "conditioning": conditioning,
            "references_encoded": references,
            "latents": boundaries,
            "rgba": m,
            "runtime": json.loads((native / "report.json").read_text()),
            "verdict": "PASS" if ok else "FAIL",
        }
        report["cases"].append(record)
        passed &= ok
    report.update(
        verdict="PASS" if passed else "FAIL",
        oracle_devices=oracle_devices,
        full_t00_corpus=a.full_corpus and not excluded and not a.case_ids,
        excluded=excluded,
        human_task_review="NOT_VERIFIED",
        tolerance_sha256=hashlib.sha256(
            Path(
                "docs/performance/native-suite-t00-20261020/tolerances.json"
            ).read_bytes()
        ).hexdigest(),
    )
    (a.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
