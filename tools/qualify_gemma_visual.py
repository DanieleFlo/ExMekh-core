#!/usr/bin/env python3
"""Sequential native/oracle T14 gates, frozen corpus retrieval and fault replay."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from suite_numeric_metrics import compare, embedding_metrics
from suite_visual_cases import frames, write_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--text-asset", type=Path, required=True)
    parser.add_argument("--processor-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("performance-results/native-suite-gemma-corpus-20261020-r2"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np

    root = Path("tests/fixtures/native-suite-v1")
    corpus = json.loads((root / "corpus.json").read_text())
    cases = []
    for item in corpus["items"]:
        if len(item["content"]) != 1 or item["content"][0]["type"] not in (
            "image",
            "video",
        ):
            continue
        part = item["content"][0]
        media = {"type": part["type"], "path": str(root / part["path"])}
        if media["type"] == "video":
            with np.load(media["path"]) as data:
                media["indices"] = list(range(0, len(data["frames"]), int(data["fps"])))
        cases.append({"id": item["id"], "split": item["split"], "media": [media]})
    image0, image1 = cases[0]["media"][0], cases[1]["media"][0]
    video0 = next(c["media"][0] for c in cases if c["id"] == "video-0")
    cases += [
        {"id": "two-images", "media": [image0, image1]},
        {"id": "reversed-images", "media": [image1, image0]},
        {
            "id": "image-video",
            "prompt": "Colori <|image|> movimento <|video|>",
            "media": [image0, video0],
        },
        {
            "id": "video-reversed",
            "media": [{**video0, "indices": list(reversed(video0["indices"]))}],
        },
        {"id": "budget-70", "task": 1, "media": [{**image0, "soft": 70}]},
    ]
    # Reversed content retains monotonically increasing presentation times.
    values, _ = frames(cases[-2]["media"][0])
    np.savez(args.output / "reversed.npz", frames=values)
    cases[-2]["media"][0] = {"type": "video", "path": str(args.output / "reversed.npz")}
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    write_fixture(args.output / "cases.bin", cases)
    runs = []

    def run(name, command, env=None):
        start = time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                list(map(str, command)),
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
            )
        runs.append(
            {
                "name": name,
                "command": list(map(str, command)),
                "environment": env or {},
                "seconds": time.monotonic() - start,
                "returncode": result.returncode,
            }
        )
        (args.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(name, result.returncode, round(runs[-1]["seconds"], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name}: {args.output / (name + '.log')}")

    base = [
        "tests/probe_gemma_visual",
        args.bundle,
        args.prepared / "models",
        args.text_asset,
        args.processor_asset,
    ]
    run("native", base + [args.output / "cases.bin", args.output / "native"])
    run(
        "oracle",
        [
            sys.executable,
            "tools/run_gemma_visual_oracle.py",
            "--prepared",
            args.prepared,
            "--cases",
            args.output / "cases.json",
            "--output",
            args.output / "oracle",
        ],
    )
    actual = np.fromfile(args.output / "native/vectors.f32", np.float32).reshape(
        -1, 768
    )
    expected = np.load(args.output / "oracle/vectors.npy")
    official = np.load(args.output / "oracle/official-vectors.npy")
    tolerances = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )
    report = {
        "cases": len(cases),
        "tolerances": tolerances,
        "boundaries": [],
        "inputs": [],
        "dimensions": {},
        "batch": {},
        "faults": [],
        "retrieval": {},
    }
    passed = True

    def gate_vector(metrics):
        t = tolerances["gemma_forward"]
        return (
            metrics["max_abs"] <= t["max_abs"]
            and metrics["rmse"] <= t["rmse_max"]
            and metrics["cosine_min"] >= t["cosine_min"]
            and metrics["unit_norm_error_max"] <= t["unit_norm_error_max"]
        )

    for file in sorted((args.output / "native").glob("*.bf16")):

        def load(path):
            return (np.fromfile(path, np.uint16).astype(np.uint32) << 16).view(
                np.float32
            )

        metrics = compare(load(args.output / "oracle" / file.name), load(file))
        # Raw hidden scales have no normalized-vector meaning. Require exact
        # frozen boundaries; never widen the primitive/embedding ceilings.
        ok = metrics["max_abs"] == 0
        report["boundaries"].append(
            {"file": file.name, **metrics, "verdict": "PASS" if ok else "FAIL"}
        )
        passed &= ok
    for pattern in ("*-pixels.f32", "*-positions.i32", "*-tokens.u32"):
        for file in sorted((args.output / "native").glob(pattern)):
            exact = (
                file.read_bytes() == (args.output / "oracle" / file.name).read_bytes()
            )
            report["inputs"].append({"file": file.name, "exact": exact})
            passed &= exact
    frozen = np.load(args.baseline / "vectors.npz")
    for dimension in (128, 256, 512, 768):
        if dimension != 768:
            run(
                f"dim-{dimension}",
                base + [args.output / "cases.bin", args.output / f"dim-{dimension}"],
                {
                    "DS4_GEMMA_VISUAL_DIMENSIONS": str(dimension),
                    "DS4_GEMMA_VISUAL_NO_TRACE": "1",
                },
            )
            a = np.fromfile(
                args.output / f"dim-{dimension}/vectors.f32", np.float32
            ).reshape(-1, dimension)
        else:
            a = actual
        e = expected[:, :dimension].copy()
        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        metrics = embedding_metrics(e, a)
        report["dimensions"][dimension] = metrics
        passed &= gate_vector(metrics)
        e = official[:, :dimension].copy()
        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        metrics = embedding_metrics(e, a)
        report.setdefault("official_forward", {})[dimension] = metrics
        passed &= gate_vector(metrics)
        f = np.stack([frozen[c["id"]][:dimension] for c in cases[:24]])
        f /= np.linalg.norm(f, axis=-1, keepdims=True)
        metrics = embedding_metrics(f, a[:24])
        report.setdefault("frozen_t00", {})[dimension] = metrics
        passed &= gate_vector(metrics)
    subset = [cases[0], cases[16], cases[24]]
    for i, case in enumerate(subset):
        fixture = args.output / f"single-{i}.bin"
        write_fixture(fixture, [case])
        run(
            f"single-{i}",
            base + [fixture, args.output / f"single-{i}"],
            {"DS4_GEMMA_VISUAL_NO_TRACE": "1"},
        )
        v = np.fromfile(args.output / f"single-{i}/vectors.f32", np.float32)
        metrics = compare(actual[(0, 16, 24)[i]], v)
        report["batch"][i] = metrics
        passed &= metrics["bit_exact"]
    # Native text queries and caption documents complete both retrieval links.
    queries = [
        q for q in corpus["retrieval"] if "image" in q["link"] or "video" in q["link"]
    ]
    texts = sorted(
        {
            "task: search result | query: " + q["text"]
            for q in queries
            if q["link"].startswith("text_to_")
        }
        | {
            "title: none | text: " + i["caption"]
            for i in corpus["items"]
            if i["id"].startswith(("image-", "video-"))
        }
    )
    run(
        "text",
        [
            "tests/probe_native_suite",
            args.bundle,
            args.prepared / "models",
            args.text_asset,
            "gemma",
            args.output / "text",
            *texts,
        ],
    )
    text_vectors = dict(
        zip(
            texts,
            np.fromfile(args.output / "text/vectors.f32", np.float32).reshape(-1, 768),
        )
    )
    visual_vectors = {c["id"]: v for c, v in zip(cases, actual)}
    metrics = {}
    for query in queries:
        link, split = query["link"], query["split"]
        modality = (
            link.removeprefix("text_to_")
            if link.startswith("text_to_")
            else link.removesuffix("_to_text")
        )
        items = [
            i
            for i in corpus["items"]
            if i["split"] == split and i["id"].startswith(modality + "-")
        ]
        ids = [i["id"] for i in items]
        if link.startswith("text_to_"):
            q = text_vectors["task: search result | query: " + query["text"]]
            d = np.stack([visual_vectors[i] for i in ids])
            relevant = set(query["relevant_ids"])
        else:
            q = visual_vectors[query["input_id"]]
            d = np.stack(
                [text_vectors["title: none | text: " + i["caption"]] for i in items]
            )
            relevant = {
                i["id"] for i in items if i["caption"] == query["relevant_text"]
            }
        for dim in (128, 256, 512, 768):
            scores = (d[:, :dim] @ q[:dim]) / (
                np.linalg.norm(d[:, :dim], axis=-1) * np.linalg.norm(q[:dim])
            )
            order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
            ranked = [ids[i] for i in order]
            dcg = sum(
                1 / math.log2(i + 2) for i, k in enumerate(ranked[:10]) if k in relevant
            )
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
            metrics.setdefault(f"{split}/{link}/{dim}", []).append(
                (
                    float(ranked[0] in relevant),
                    len(set(ranked[:5]) & relevant) / len(relevant),
                    dcg / ideal,
                )
            )
    frozen_metrics = json.loads((args.baseline / "retrieval.json").read_text())[
        "metrics"
    ]
    for key, values in metrics.items():
        values = np.mean(values, axis=0)
        record = dict(
            zip(("recall_at_1", "recall_at_5", "ndcg_at_10"), map(float, values))
        )
        ok = all(
            frozen_metrics[key][name] - record[name]
            <= tolerances["retrieval"][name + "_drop_max"]
            for name in record
        )
        record["verdict"] = "PASS" if ok else "FAIL"
        report["retrieval"][key] = record
        passed &= ok
    for stage in (
        "vision-patches",
        "vision-layer",
        "vision-projected",
        "embeddings",
        "layer",
        "complete",
    ):
        run(
            f"cancel-{stage}",
            base + [args.output / "single-0.bin", args.output / f"cancel-{stage}"],
            {"DS4_GEMMA_VISUAL_CANCEL": stage},
        )
        report["faults"].append(stage)
    run(
        "faults",
        base + [args.output / "single-0.bin", args.output / "faults"],
        {"DS4_GEMMA_VISUAL_FAULTS": "1", "DS4_GEMMA_VISUAL_NO_TRACE": "1"},
    )
    report["sensitivity"] = {
        "distinct_images": len({v.tobytes() for v in actual[:16]}),
        "distinct_videos": len({v.tobytes() for v in actual[16:24]}),
        "frame_order_changes_vector": not np.array_equal(actual[16], actual[27]),
        "image_order_changes_vector": not np.array_equal(actual[24], actual[25]),
    }
    passed &= (
        report["sensitivity"]["distinct_images"] == 16
        and report["sensitivity"]["distinct_videos"] == 8
        and all(
            report["sensitivity"][k]
            for k in ("frame_order_changes_vector", "image_order_changes_vector")
        )
    )
    report["artifacts"] = [
        {
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in (
            args.text_asset,
            args.processor_asset,
            args.output / "cases.json",
            args.output / "native/vectors.f32",
            args.output / "oracle/vectors.npy",
        )
    ]
    report["verdict"] = "PASS" if passed else "FAIL"
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
