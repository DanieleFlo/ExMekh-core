#!/usr/bin/env python3
"""Run sequential native/oracle Gemma text gates; preserve complete evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from suite_numeric_metrics import compare, embedding_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np

    corpus = json.loads(Path("tests/fixtures/native-suite-v1/corpus.json").read_text())
    cases = [
        {"id": i["id"], "split": i["split"], "text": i["content"][0]["text"]}
        for i in corpus["items"]
        if len(i["content"]) == 1 and i["content"][0]["type"] == "text"
    ]
    cases += [
        {"id": f"extra-{i}", "split": "diagnostic", "text": t}
        for i, t in enumerate(
            [
                "Caffe\u0300, Ελληνικά, العربية, 中文, 日本語 🌈",
                "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
                "",
                "a " * 513,
                "a " * 514,
                "a " * 1025,
            ]
        )
    ]
    (a.output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
    )
    runs = []

    def run(name, command, env=None):
        start = time.monotonic()
        with (a.output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [str(x) for x in command],
                check=False,
                env={**os.environ, **(env or {})},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        runs.append(
            {
                "name": name,
                "command": list(map(str, command)),
                "env": env or {},
                "seconds": time.monotonic() - start,
                "returncode": result.returncode,
            }
        )
        (a.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(name, result.returncode, round(runs[-1]["seconds"], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {a.output / name}.log")

    base = [
        "tests/probe_native_suite",
        a.bundle,
        a.prepared / "models",
        a.asset,
        "gemma",
    ]
    run("native", base + [a.output / "native", *[c["text"] for c in cases]])
    run(
        "oracle",
        [
            sys.executable,
            "tools/run_embedding_gemma2_oracle.py",
            "--prepared",
            a.prepared,
            "--cases",
            a.output / "cases.json",
            "--output",
            a.output / "oracle",
        ],
    )
    native = np.fromfile(a.output / "native/vectors.f32", np.float32).reshape(-1, 768)
    oracle = np.load(a.output / "oracle/vectors.npy")
    tolerances = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )["gemma_forward"]
    report = {
        "cases": len(cases),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tolerances": tolerances,
        "vectors": [],
        "boundaries": [],
        "dimensions": {},
        "batch": {},
        "tasks": {},
    }
    passed = True

    def gate(metrics):
        return (
            metrics["max_abs"] <= tolerances["max_abs"]
            and metrics["rmse"] <= tolerances["rmse_max"]
            and metrics["cosine_min"] >= tolerances["cosine_min"]
            and metrics["unit_norm_error_max"] <= tolerances["unit_norm_error_max"]
        )

    for i, c in enumerate(cases):
        metrics = embedding_metrics(oracle[i : i + 1], native[i : i + 1])
        ok = gate(metrics)
        passed &= ok
        ids = np.fromfile(a.output / f"native/item-{i}-tokens.u32", np.uint32)
        expected = np.fromfile(a.output / f"oracle/item-{i}-tokens.u32", np.uint32)
        exact = np.array_equal(ids, expected)
        passed &= exact
        report["vectors"].append(
            {
                "id": c["id"],
                "split": c["split"],
                "tokens": len(ids),
                "tokens_exact": bool(exact),
                "metrics": metrics,
                "verdict": "PASS" if ok and exact else "FAIL",
            }
        )
        for stage, layer in [
            ("embeddings", 0),
            ("ple", 0),
            *[("layer", j) for j in range(24)],
            ("projected", 24),
        ]:

            def load(which, i=i, stage=stage, layer=layer):
                v = np.fromfile(
                    a.output / f"{which}/item-{i}-{stage}-{layer}.bf16", np.uint16
                )
                return (v.astype(np.uint32) << 16).view(np.float32)

            report["boundaries"].append(
                dict(
                    id=c["id"],
                    stage=stage,
                    layer=layer,
                    **compare(load("oracle"), load("native")),
                )
            )
    # Frozen T00 text boundaries now replay exactly. Enforce that stronger
    # observed criterion without applying unit-vector ceilings to raw hidden
    # scales. Extra window/length stress remains an explicit diagnostic.
    frozen_ids = {c["id"] for c in cases if c["split"] != "diagnostic"}
    hidden = [b for b in report["boundaries"] if b["id"] in frozen_ids]
    hidden_exact = all(b["bit_exact"] for b in hidden)
    report["hidden_t00"] = {
        "cases": len(frozen_ids),
        "boundaries": len(hidden),
        "criterion": "bit_exact",
        "verdict": "PASS" if hidden_exact else "FAIL",
    }
    passed &= hidden_exact
    padded = embedding_metrics(np.load(a.output / "oracle/padded-vectors.npy"), native)
    report["padded_reference"] = padded
    passed &= gate(padded)
    subset = [cases[i]["text"] for i in (0, 1, len(cases) - 6)]
    dimension_vectors = {768: (native, oracle)}
    for dimension in (128, 256, 512):
        run(
            f"dim-{dimension}",
            base + [a.output / f"dim-{dimension}", *[c["text"] for c in cases]],
            {"DS4_GEMMA_DIMENSIONS": str(dimension)},
        )
        actual = np.fromfile(
            a.output / f"dim-{dimension}/vectors.f32", np.float32
        ).reshape(-1, dimension)
        expected = oracle[:, :dimension].copy()
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        dimension_vectors[dimension] = (actual, expected)
        metrics = embedding_metrics(expected, actual)
        passed &= gate(metrics)
        report["dimensions"][dimension] = metrics
    for i, text in enumerate(subset):
        run(f"single-{i}", base + [a.output / f"single-{i}", text])
        actual = np.fromfile(a.output / f"single-{i}/vectors.f32", np.float32)
        original = native[[0, 1, len(cases) - 6][i]]
        metrics = compare(original, actual)
        passed &= metrics["bit_exact"]
        report["batch"][i] = metrics
    prompts = json.loads(
        (
            a.prepared
            / "models/google/embeddinggemma-2/config_sentence_transformers.json"
        ).read_text()
    )["prompts"]
    for task, name in enumerate(
        ("query", "document", "CodeRetrieval", "SentenceSimilarity"), 1
    ):
        text = "Trova informazioni sul codice Python."
        prefix = prompts[name]
        run(
            f"task-{task}",
            base + [a.output / f"task-{task}", text, prefix + text],
            {"DS4_GEMMA_TASK": str(task)},
        )
        values = np.fromfile(a.output / f"task-{task}/vectors.f32", np.float32).reshape(
            2, 768
        )
        metrics = compare(values[0], values[1])
        passed &= metrics["bit_exact"]
        report["tasks"][name] = metrics
    # Ranking stability on each frozen text split, excluding identity matches.
    report["ranking"] = {}
    for dimension, (actual_vectors, expected_vectors) in dimension_vectors.items():
        for split in ("dev", "heldout"):
            idx = [i for i, c in enumerate(cases) if c["split"] == split]
            actual = actual_vectors[idx] @ actual_vectors[idx].T
            expected = expected_vectors[idx] @ expected_vectors[idx].T
            np.fill_diagonal(actual, -np.inf)
            np.fill_diagonal(expected, -np.inf)
            ar = np.argsort(-actual, axis=-1)
            er = np.argsort(-expected, axis=-1)
            top1 = float(np.mean(ar[:, 0] == er[:, 0]))
            overlap = float(
                np.mean([len(set(x[:5]) & set(y[:5])) / 5 for x, y in zip(ar, er)])
            )
            ndcg = []
            for x, y in zip(ar, er):
                relevant = {int(v): 10 - i for i, v in enumerate(y[:10])}
                dcg = sum(
                    (2 ** relevant.get(int(v), 0) - 1) / np.log2(i + 2)
                    for i, v in enumerate(x[:10])
                )
                ideal = sum((2 ** (10 - i) - 1) / np.log2(i + 2) for i in range(10))
                ndcg.append(dcg / ideal)
            ranking = {
                "top1_match": top1,
                "top5_overlap": overlap,
                "ndcg10": float(np.mean(ndcg)),
            }
            retrieval = json.loads(
                Path(
                    "docs/performance/native-suite-t00-20261020/tolerances.json"
                ).read_text()
            )["retrieval"]
            ranking["verdict"] = (
                "PASS"
                if (
                    1 - top1 <= retrieval["recall_at_1_drop_max"]
                    and 1 - overlap <= retrieval["recall_at_5_drop_max"]
                    and 1 - ranking["ndcg10"] <= retrieval["ndcg_at_10_drop_max"]
                )
                else "FAIL"
            )
            passed &= ranking["verdict"] == "PASS"
            report["ranking"][f"{split}-{dimension}"] = ranking
    report["oracle_padding"] = json.loads((a.output / "oracle/oracle.json").read_text())
    report["verdict"] = "PASS" if passed else "FAIL"
    report["artifacts"] = [
        {
            "path": str(x),
            "bytes": x.stat().st_size,
            "sha256": hashlib.sha256(x.read_bytes()).hexdigest(),
        }
        for x in (
            a.asset,
            a.output / "cases.json",
            a.output / "native/vectors.f32",
            a.output / "oracle/vectors.npy",
        )
    ]
    (a.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
