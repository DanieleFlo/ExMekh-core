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

from suite_numeric_metrics import compare, rgba_metrics


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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from PIL import Image

    fixtures = a.output / "fixtures"
    fixtures.mkdir()
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
    inputs = []
    for case in specs:
        if "split" in case:
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
    runs = []

    def run(name, command, env=None):
        start = time.monotonic()
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
        (a.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(name, result.returncode, round(runs[-1]["seconds"], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed, see {a.output / name}.log")

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
        {"DS4_SUITE_PLAN_ONLY": "1"} if a.plan_only else None,
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
            ],
        )
    tol = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )
    report = {"cases": []}
    passed = True
    for i, case in enumerate(specs):
        native = a.output / f"native/case-{i}"
        oracle = a.output / "oracle" / case["id"]
        boundaries = []
        ok = True
        for step in range(8):
            m = compare(
                np.fromfile(oracle / f"latent-{step}.f32", np.float32),
                np.fromfile(native / f"latent-{step}.f32", np.float32),
            )
            m["verdict"] = (
                "PASS"
                if m["relative_l2"] <= tol["qwen_latent_boundary"]["relative_l2"]
                and m["rmse"] <= tol["qwen_latent_boundary"]["rmse_max"]
                else "FAIL"
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
            "latents": boundaries,
            "rgba": m,
            "runtime": json.loads((native / "report.json").read_text()),
            "verdict": "PASS" if ok else "FAIL",
        }
        report["cases"].append(record)
        passed &= ok
    report.update(
        verdict="PASS" if passed else "FAIL",
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
