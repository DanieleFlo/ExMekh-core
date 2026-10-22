#!/usr/bin/env python3
"""Sequential T15/T16 native/oracle gates using the frozen four-modal corpus."""

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

from suite_numeric_metrics import compare, embedding_metrics
from suite_visual_cases import frames, write_fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "bundle", "text-asset", "processor-asset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "performance-results/native-suite-gemma-corpus-20261022-fp32-reduction"
        ),
    )
    p.add_argument(
        "--extra-cases",
        type=Path,
        default=Path("tests/fixtures/native-suite-audio-v2/cases.json"),
    )
    a = p.parse_args()
    sys.path.insert(0, str((a.prepared / "deps").resolve()))
    import numpy as np

    a.output.mkdir(parents=True, exist_ok=False)
    root = Path("tests/fixtures/native-suite-v1")
    corpus = json.loads((root / "corpus.json").read_text())
    cases = []
    for item in corpus["items"]:
        parts = []
        media = []
        for part in item["content"]:
            if part["type"] == "text":
                parts.append(part["text"])
                continue
            m = {"type": part["type"], "path": str(root / part["path"])}
            if m["type"] == "video":
                with np.load(m["path"]) as data:
                    m["indices"] = list(range(0, len(data["frames"]), int(data["fps"])))
            media.append(m)
            parts.append(f"<|{m['type']}|>")
        cases.append({"id": item["id"], "prompt": " ".join(parts), "media": media})
    frozen_count = len(cases)
    audio0 = next(c["media"][0] for c in cases if c["id"] == "audio-0")
    audio1 = next(c["media"][0] for c in cases if c["id"] == "audio-1")
    image = next(c["media"][0] for c in cases if c["id"] == "image-0")
    video = next(c["media"][0] for c in cases if c["id"] == "video-0")
    cases += [
        {"id": "two-clips", "media": [audio0, audio1]},
        {"id": "reversed-clips", "media": [audio1, audio0]},
        {"id": "repeat-clip", "media": [audio0, audio0]},
        {
            "id": "all-four",
            "prompt": "Suono <|audio|> colori <|image|> movimento <|video|>",
            "media": [audio0, image, video],
        },
        {
            "id": "all-four-changed-audio",
            "prompt": "Suono <|audio|> colori <|image|> movimento <|video|>",
            "media": [audio1, image, video],
        },
    ]
    for kind, replacement in (("image", "image-1"), ("video", "video-1")):
        media = [audio0, image, video].copy()
        media[1 if kind == "image" else 2] = next(
            c["media"][0] for c in cases if c["id"] == replacement
        )
        cases.append(
            {
                "id": f"all-four-changed-{kind}",
                "prompt": "Suono <|audio|> colori <|image|> movimento <|video|>",
                "media": media,
            }
        )
    cases.append(
        {
            "id": "all-four-changed-text",
            "prompt": "Altri suoni <|audio|> forme <|image|> direzioni <|video|>",
            "media": [audio0, image, video],
        }
    )
    if a.extra_cases:
        cases.extend(json.loads(a.extra_cases.read_text()))
    # A real in-memory-decoded video with an explicitly selected audio track.
    sys.path.insert(0, "performance-results/native-suite-t13-t14-20261021/development")
    import av

    pictures, _ = frames(video)
    path = a.output / "video-audio.mkv"
    waveform = a.output / "video-audio.wav"
    with wave.open(str(waveform), "wb") as f:
        f.setparams((1, 2, 16000, 32000, "NONE", "not compressed"))
        f.writeframes(bytes(64000))
    with av.open(str(path), "w", format="matroska") as container:
        vs = container.add_stream("ffv1", rate=1)
        vs.width = pictures.shape[2]
        vs.height = pictures.shape[1]
        vs.pix_fmt = "bgr0"
        aus = container.add_stream("pcm_s16le", rate=16000)
        aus.layout = "mono"
        for i, picture in enumerate(pictures[:2]):
            frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
            frame.pts = i
            for packet in vs.encode(frame):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)
        for i in range(2):
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 16000), np.int16), format="s16", layout="mono"
            )
            frame.sample_rate = 16000
            frame.pts = i * 16000
            for packet in aus.encode(frame):
                container.mux(packet)
        for packet in aus.encode():
            container.mux(packet)
    np.savez(a.output / "video-audio.npz", frames=pictures[:2], timestamps=[0.0, 1.0])
    cases.append(
        {
            "id": "video-with-audio",
            "media": [
                {
                    "type": "video",
                    "path": str(a.output / "video-audio.npz"),
                    "container": str(path),
                    "audio_path": str(waveform),
                }
            ],
        }
    )
    for samples in (161, 16017, 480000):
        path = a.output / f"silence-{samples}.wav"
        with wave.open(str(path), "wb") as f:
            f.setparams((1, 2, 16000, samples, "NONE", "not compressed"))
            f.writeframes(bytes(samples * 2))
        cases.append(
            {
                "id": f"silence-{samples}",
                "media": [{"type": "audio", "path": str(path)}],
            }
        )
    queries = corpus["retrieval"].copy()
    # Composed retrieval ground truth comes from the existing frozen captions.
    # Its independent reference rankings are measured with the pinned oracle.
    for item in corpus["items"]:
        if item["id"].startswith("interleaved-"):
            queries.extend(
                [
                    {
                        "link": "text_to_interleaved",
                        "split": item["split"],
                        "text": item["caption"],
                        "relevant_ids": [
                            i["id"]
                            for i in corpus["items"]
                            if i["split"] == item["split"]
                            and i["id"].startswith("interleaved-")
                            and i["caption"] == item["caption"]
                        ],
                    },
                    {
                        "link": "interleaved_to_text",
                        "split": item["split"],
                        "input_id": item["id"],
                        "relevant_text": item["caption"],
                    },
                ]
            )
    texts = sorted(
        {
            "task: search result | query: " + q["text"]
            for q in queries
            if q["link"].startswith("text_to_")
        }
        | {
            "title: none | text: " + i["caption"]
            for i in corpus["items"]
            if not i["id"].startswith("text-")
        }
    )
    text_start = len(cases)
    cases += [
        {"id": f"retrieval-{i}", "prompt": t, "media": []} for i, t in enumerate(texts)
    ]
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    write_fixture(a.output / "cases.bin", cases)
    runs = []

    def run(name, cmd, env=None):
        start = time.monotonic()
        with (a.output / f"{name}.log").open("w") as log:
            r = subprocess.run(
                list(map(str, cmd)),
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
                check=False,
            )
        runs.append(
            {
                "name": name,
                "command": list(map(str, cmd)),
                "environment": env or {},
                "seconds": time.monotonic() - start,
                "returncode": r.returncode,
            }
        )
        (a.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(name, r.returncode, round(runs[-1]["seconds"], 3), flush=True)
        if r.returncode:
            raise RuntimeError(name)

    base = [
        "tests/probe_gemma_visual",
        a.bundle,
        a.prepared / "models",
        a.text_asset,
        a.processor_asset,
    ]

    def compress(folder):
        for file in folder.iterdir():
            if (
                file.suffix not in (".bf16", ".u32", ".i32", ".u8", ".f32")
                or file.name == "vectors.f32"
            ):
                continue
            with (
                file.open("rb") as src,
                gzip.open(str(file) + ".gz", "wb", compresslevel=1) as dst,
            ):
                shutil.copyfileobj(src, dst)
            file.unlink()

    env = {"DS4_GEMMA_AUDIO_TRACE_ONLY": "1"}
    run("native", base + [a.output / "cases.bin", a.output / "native"], env)
    compress(a.output / "native")
    run(
        "oracle",
        [
            sys.executable,
            "tools/run_gemma_visual_oracle.py",
            "--prepared",
            a.prepared,
            "--cases",
            a.output / "cases.json",
            "--output",
            a.output / "oracle",
        ],
        env,
    )
    compress(a.output / "oracle")
    actual = np.fromfile(a.output / "native/vectors.f32", np.float32).reshape(-1, 768)
    expected = np.load(a.output / "oracle/vectors.npy")
    official = np.load(a.output / "oracle/official-vectors.npy")
    thresholds = json.loads(
        Path("docs/performance/native-suite-t00-20261020/tolerances.json").read_text()
    )
    report = {
        "cases": len(cases),
        "frozen_cases": frozen_count,
        "thresholds": thresholds,
        "boundaries": [],
        "inputs": [],
        "dimensions": {},
        "retrieval": {},
        "batch": [],
        "faults": [],
    }
    ok = True

    def vector_gate(m):
        t = thresholds["gemma_forward"]
        return (
            m["max_abs"] <= t["max_abs"]
            and m["rmse"] <= t["rmse_max"]
            and m["cosine_min"] >= t["cosine_min"]
            and m["unit_norm_error_max"] <= t["unit_norm_error_max"]
        )

    for file in sorted((a.output / "native").glob("*")):
        if file.suffix != ".gz":
            continue
        other = a.output / "oracle" / file.name
        raw = gzip.decompress(file.read_bytes())
        ref = gzip.decompress(other.read_bytes())
        if ".bf16" in file.name or "-probabilities-" in file.name:

            def load(f, bf16=".bf16" in file.name):
                return (
                    (np.frombuffer(f, np.uint16).astype(np.uint32) << 16).view(
                        np.float32
                    )
                    if bf16
                    else np.frombuffer(f, np.float32)
                )

            m = compare(load(ref), load(raw))
            passed = m["max_abs"] == 0
            report["boundaries"].append(
                {"file": file.name, **m, "verdict": "PASS" if passed else "FAIL"}
            )
            ok &= passed
        elif any(
            x in file.name
            for x in ("-pixels.", "-positions.", "-tokens.", "-mask.", "-features.")
        ):
            if file.name.endswith("features.f32.gz"):
                m = compare(
                    np.frombuffer(ref, np.float32), np.frombuffer(raw, np.float32)
                )
                passed = (
                    m["max_abs"] <= thresholds["primitive_float32"]["max_abs"]
                    and m["relative_l2"]
                    <= thresholds["primitive_float32"]["relative_l2"]
                )
            else:
                m = {"bit_exact": raw == ref}
                passed = m["bit_exact"]
            report["inputs"].append(
                {"file": file.name, **m, "verdict": "PASS" if passed else "FAIL"}
            )
            ok &= passed
    frozen = np.load(a.baseline / "vectors.npz")
    byid = {c["id"]: i for i, c in enumerate(cases)}
    for dim in (128, 256, 512, 768):
        if dim == 768:
            v = actual
        else:
            run(
                f"dim-{dim}",
                base + [a.output / "cases.bin", a.output / f"dim-{dim}"],
                {
                    "DS4_GEMMA_VISUAL_DIMENSIONS": str(dim),
                    "DS4_GEMMA_VISUAL_NO_TRACE": "1",
                },
            )
            v = np.fromfile(a.output / f"dim-{dim}/vectors.f32", np.float32).reshape(
                -1, dim
            )
            compress(a.output / f"dim-{dim}")
        report["dimensions"][dim] = {}
        refs = {
            "components": expected,
            "official": official,
            "frozen": np.stack([frozen[c["id"]] for c in cases[:frozen_count]]),
        }
        for name, e in refs.items():
            e = e[:, :dim].copy()
            e /= np.linalg.norm(e, axis=-1, keepdims=True)
            m = embedding_metrics(e, v[: len(e)])
            passed = vector_gate(m)
            ok &= passed
            report["dimensions"][dim][name] = {
                **m,
                "verdict": "PASS" if passed else "FAIL",
            }

        def retrieval(values, dim=dim):
            metric = {}
            for query in queries:
                link = query["link"]
                modality = (
                    link.removeprefix("text_to_")
                    if link.startswith("text_to_")
                    else link.removesuffix("_to_text")
                )
                items = [
                    i
                    for i in corpus["items"]
                    if i["split"] == query["split"]
                    and i["id"].startswith(modality + "-")
                ]
                ids = [i["id"] for i in items]
                if link.startswith("text_to_"):
                    q = values[
                        text_start
                        + texts.index("task: search result | query: " + query["text"])
                    ]
                    d = np.stack([values[byid[i]] for i in ids])
                    relevant = set(query["relevant_ids"])
                else:
                    q = values[byid[query["input_id"]]]
                    d = np.stack(
                        [
                            values[
                                text_start
                                + texts.index("title: none | text: " + i["caption"])
                            ]
                            for i in items
                        ]
                    )
                    relevant = {
                        i["id"] for i in items if i["caption"] == query["relevant_text"]
                    }
                scores = (d @ q) / (np.linalg.norm(d, axis=-1) * np.linalg.norm(q))
                ranked = [
                    ids[j]
                    for j in sorted(
                        range(len(ids)), key=lambda j: (-float(scores[j]), ids[j])
                    )
                ]
                dcg = sum(
                    1 / math.log2(j + 2)
                    for j, k in enumerate(ranked[:10])
                    if k in relevant
                )
                ideal = sum(1 / math.log2(j + 2) for j in range(min(10, len(relevant))))
                key = f"{query['split']}/{query['link']}/{dim}"
                metric.setdefault(key, []).append(
                    (
                        float(ranked[0] in relevant),
                        len(set(ranked[:5]) & relevant) / len(relevant),
                        dcg / ideal,
                    )
                )
            return {
                key: dict(
                    zip(
                        ("recall_at_1", "recall_at_5", "ndcg_at_10"),
                        map(float, np.mean(values, axis=0)),
                    )
                )
                for key, values in metric.items()
            }

        metric = retrieval(v)
        reference_metric = retrieval(official[:, :dim])
        baseline = json.loads((a.baseline / "retrieval.json").read_text())["metrics"]
        for key, record in metric.items():
            reference = baseline.get(key, reference_metric[key])
            passed = all(
                reference[k] - record[k] <= thresholds["retrieval"][k + "_drop_max"]
                for k in record
            )
            report["retrieval"][key] = {
                **record,
                "reference": reference,
                "verdict": "PASS" if passed else "FAIL",
            }
            ok &= passed
    # Isolated and ragged 1/2/3 batches must return the same per-item vectors.
    selected = [byid["audio-0"], byid["image-0"], byid["all-four"]]
    for size in (1, 2, 3):
        fixture = a.output / f"batch-{size}.bin"
        write_fixture(fixture, [cases[j] for j in selected[:size]])
        run(
            f"batch-{size}",
            base + [fixture, a.output / f"batch-{size}"],
            {"DS4_GEMMA_VISUAL_NO_TRACE": "1"},
        )
        v = np.fromfile(a.output / f"batch-{size}/vectors.f32", np.float32).reshape(
            -1, 768
        )
        passed = np.array_equal(v, actual[selected[:size]])
        report["batch"].append({"size": size, "bit_exact": passed})
        ok &= passed
        compress(a.output / f"batch-{size}")
    write_fixture(a.output / "fault.bin", [cases[byid["all-four"]]])
    for stage in (
        "audio-subsample",
        "audio-ff1",
        "audio-probabilities",
        "audio-layer",
        "audio-projected",
        "vision-patches",
        "embeddings",
        "layer",
        "complete",
    ):
        run(
            f"cancel-{stage}",
            base + [a.output / "fault.bin", a.output / f"cancel-{stage}"],
            {**env, "DS4_GEMMA_VISUAL_CANCEL": stage},
        )
        compress(a.output / f"cancel-{stage}")
        report["faults"].append(stage)
    run(
        "faults",
        base + [a.output / "fault.bin", a.output / "faults"],
        {"DS4_GEMMA_VISUAL_FAULTS": "1", "DS4_GEMMA_VISUAL_NO_TRACE": "1"},
    )
    # Compare co-resident and phase-release execution, including warm replay.
    report["plans"] = {}
    for name, limit in (("co-resident", "16000000000"), ("phase-release", "2000000000")):
        run(
            name,
            base + [a.output / "fault.bin", a.output / name],
            {
                **env,
                "DS4_GEMMA_VISUAL_DEVICE_LIMIT": limit,
                "DS4_GEMMA_VISUAL_BENCH": "1",
            },
        )
        plan = json.loads((a.output / name / "plan.json").read_text())
        trace = [
            json.loads(line)
            for line in (a.output / name / "residency.jsonl").read_text().splitlines()
        ]
        v = np.fromfile(a.output / name / "vectors.f32", np.float32)
        passed = np.array_equal(v, actual[byid["all-four"]]) and plan[
            "release_encoders"
        ] == (name == "phase-release")
        if name == "phase-release":
            passed &= trace[-1]["vision"] == trace[-1]["audio"] == 0
        report["plans"][name] = {
            "plan": plan,
            "trace": trace,
            "timing": json.loads((a.output / name / "timing.json").read_text()),
            "verdict": "PASS" if passed else "FAIL",
        }
        ok &= passed
        compress(a.output / name)
    report["selective"] = {}
    for name, forbidden in (
        ("text-0", ("vision", "audio")),
        ("image-0", ("audio",)),
        ("audio-0", ("vision",)),
        ("video-0", ("audio",)),
    ):
        fixture = a.output / f"selective-{name}.bin"
        write_fixture(fixture, [cases[byid[name]]])
        run(f"selective-{name}", base + [fixture, a.output / f"selective-{name}"], env)
        trace = [
            json.loads(line)
            for line in (a.output / f"selective-{name}/residency.jsonl")
            .read_text()
            .splitlines()
        ]
        passed = all(row[key] == 0 for row in trace for key in forbidden)
        report["selective"][name] = {
            "trace": trace,
            "verdict": "PASS" if passed else "FAIL",
        }
        ok &= passed
        compress(a.output / f"selective-{name}")
    report["sensitivity"] = {
        "audio_order": not np.array_equal(
            actual[byid["two-clips"]], actual[byid["reversed-clips"]]
        ),
        "audio_in_all_four": not np.array_equal(
            actual[byid["all-four"]], actual[byid["all-four-changed-audio"]]
        ),
    }
    for kind in ("image", "video", "text"):
        report["sensitivity"][kind + "_in_all_four"] = not np.array_equal(
            actual[byid["all-four"]], actual[byid["all-four-changed-" + kind]]
        )
    ok &= all(report["sensitivity"].values())
    report["artifacts"] = [
        {
            "path": str(f),
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        for f in (
            a.bundle,
            a.text_asset,
            a.processor_asset,
            a.output / "cases.json",
            a.output / "native/vectors.f32",
            a.output / "oracle/vectors.npy",
            a.baseline / "vectors.npz",
            a.baseline / "retrieval.json",
        )
    ]
    report["verdict"] = "PASS" if ok else "FAIL"
    (a.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
