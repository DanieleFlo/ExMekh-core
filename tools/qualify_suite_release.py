"""Native HTTP lifetime acceptance; Python is a development client only.

Use one running ds4-server-suite, one SDK client, and no external model worker.
The JSONL is append-only and flushed after every cycle, including failures.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "Qwen/Qwen-Image-2.1-Turbo"
EMBED = "google/embeddinggemma-2"


def timing(samples):
    values = sorted(samples)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite timing samples required")

    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = deviation / mean
    return {
        "samples": samples,
        "min": values[0],
        "p10": percentile(0.1),
        "median": statistics.median(values),
        "mean": mean,
        "p90": percentile(0.9),
        "max": values[-1],
        "stddev": deviation,
        "cv": cv,
        "verdict": "NEED_MORE_DATA" if len(values) < 5 or cv > 0.05 else "STABLE",
    }


def memory_sample(pid):
    if not pid:
        return None
    process = Path(f"/proc/{pid}")
    status = {}
    for line in (process / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile", "VmLck", "Threads"):
            status[key] = value.strip()
    status["fds"] = len(list((process / "fd").iterdir()))
    return status


def fixture_video(path):
    import av  # Development-only fixture encoder, absent from the C runtime.
    import numpy as np

    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width, stream.height, stream.pix_fmt = 48, 48, "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 48, 3), i * 50, np.uint8), format="rgb24"
            )
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def run(args):
    from openai import OpenAI
    from PIL import Image

    if args.cycles < 50:
        raise ValueError("T21 requires at least 50 cycles after warm-up")
    args.output.mkdir(parents=True, exist_ok=False)
    vector_hashes = {}
    image_hashes = {}
    samples = {"image": [], "embedding": [], "cycle": []}
    snapshots = []
    report = {
        "version": 1,
        "status": "RUNNING",
        "completed_cycles": 0,
        "warmup_cycles": 1,
        "required_cycles": args.cycles,
        "scope": "HTTP lifetime and deterministic residency; no quality promotion",
    }
    with OpenAI(
        base_url=args.url, api_key="local", timeout=600, max_retries=0
    ) as client:
        catalog = [m.model_dump(exclude_unset=True) for m in client.models.list().data]
        (args.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        chat_model = args.chat_model or next(
            m["id"] for m in catalog if m["id"] not in (IMAGE, EMBED)
        )

        def upload(path, mime):
            return client.files.create(
                file=(path.name, path.read_bytes(), mime), purpose="ds4_media"
            ).id

        image = upload(
            ROOT / "tests/fixtures/native-suite-v1/media/image-0.png", "image/png"
        )
        audio = upload(
            ROOT / "tests/fixtures/native-suite-audio-v2/it.wav", "audio/wav"
        )
        video_path = args.video or fixture_video(args.output / "fixture.mp4")
        video = upload(video_path, "video/mp4")
        fixtures = [
            ROOT / "tests/fixtures/native-suite-v1/media/image-0.png",
            ROOT / "tests/fixtures/native-suite-audio-v2/it.wav",
            video_path,
        ]
        (args.output / "fixtures.json").write_text(
            json.dumps(
                [
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for path in fixtures
                ],
                indent=2,
            )
            + "\n"
        )
        parts = [
            [{"type": "text", "text": "Un quadrato rosso"}],
            [{"type": "image", "file_id": image}],
            [{"type": "video", "file_id": video, "audio": "exclude"}],
            [{"type": "audio", "file_id": audio}],
        ]
        parts.append([part for row in parts for part in row])

        tools = [
            {
                "type": "function",
                "name": name,
                "description": "Perform the explicitly requested native suite operation.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
            for name in ("suite_image", "suite_embedding")
        ]

        def converse(items, cached, choice):
            response = client.responses.create(
                model=chat_model,
                input=items,
                max_output_tokens=128,
                temperature=0,
                reasoning={"effort": "none"},
                tools=tools,
                tool_choice=choice,
            )
            hit = response.usage.input_tokens_details.cached_tokens
            if cached and not hit:
                raise AssertionError("full prefill after a valid live continuation")
            return response, hit

        def call_id(response, name):
            calls = [item for item in response.output if item.type == "function_call"]
            if (
                len(calls) != 1
                or calls[0].name != name
                or json.loads(calls[0].arguments) != {}
            ):
                raise AssertionError(
                    f"unexpected registered tool event: {response.output!r}"
                )
            return calls[0].call_id

        with (args.output / "cycles.jsonl").open("x") as log:
            try:
                for cycle in range(args.cycles + 1):
                    begin = time.perf_counter()
                    code = f"DS4CODE{cycle:03d}"
                    before, _ = converse(
                        [
                            {
                                "role": "user",
                                "content": f"Remember {code}. First call suite_image. After its result call suite_embedding. "
                                "After the embedding result reply only with the remembered code.",
                            }
                        ],
                        False,
                        "required",
                    )
                    image_call = call_id(before, "suite_image")
                    # Repeated and new operations both execute: no receipt replay.
                    seed = 17 if cycle % 2 else 1000 + cycle
                    start = time.perf_counter()
                    result = client.images.generate(
                        model=IMAGE,
                        prompt="A red square on white.",
                        size="64x96",
                        background="transparent",
                        extra_body={"ds4": {"version": 1, "seed": seed}},
                    ).model_dump(exclude_unset=True)
                    image_time = time.perf_counter() - start
                    png = base64.b64decode(result["data"][0]["b64_json"], validate=True)
                    decoded = Image.open(io.BytesIO(png))
                    if decoded.size != (64, 96) or decoded.mode != "RGBA":
                        raise AssertionError(
                            "unexpected image geometry or alpha channel"
                        )
                    digest = hashlib.sha256(png).hexdigest()
                    (args.output / f"image-{cycle:03d}.png").write_bytes(png)
                    if seed in image_hashes and image_hashes[seed] != digest:
                        raise AssertionError(
                            "residency changed deterministic image output"
                        )
                    image_hashes[seed] = digest
                    file_id = result["data"][0]["ds4"]["file_id"]
                    if client.files.content(file_id).read() != png:
                        raise AssertionError("output file differs from committed PNG")
                    next_tool, after_image = converse(
                        [
                            {
                                "type": "function_call_output",
                                "call_id": image_call,
                                "output": "Image complete. Call suite_embedding next.",
                            }
                        ],
                        True,
                        "required",
                    )
                    embedding_call = call_id(next_tool, "suite_embedding")
                    dimensions = (128, 256, 512, 768)[cycle % 4]
                    start = time.perf_counter()
                    vectors = client.post(
                        "/embeddings",
                        cast_to=dict,
                        body={
                            "model": EMBED,
                            "input": [{"content": row} for row in parts],
                            "dimensions": dimensions,
                            "ds4": {"version": 1},
                        },
                    )
                    embed_time = time.perf_counter() - start
                    matrix = [row["embedding"] for row in vectors["data"]]
                    if len(matrix) != 5 or any(
                        len(row) != dimensions for row in matrix
                    ):
                        raise AssertionError("missing modality or vector dimension")
                    for row in matrix:
                        if (
                            any(not math.isfinite(v) for v in row)
                            or abs(math.sqrt(sum(v * v for v in row)) - 1) > 1e-5
                        ):
                            raise AssertionError("embedding norm or finite gate failed")
                    digest_vectors = hashlib.sha256(
                        json.dumps(matrix, separators=(",", ":")).encode()
                    ).hexdigest()
                    if (
                        dimensions in vector_hashes
                        and vector_hashes[dimensions] != digest_vectors
                    ):
                        raise AssertionError(
                            "residency changed deterministic multimodal vectors"
                        )
                    if dimensions not in vector_hashes:
                        (args.output / f"vectors-{dimensions}.json").write_text(
                            json.dumps(matrix) + "\n"
                        )
                    vector_hashes[dimensions] = digest_vectors
                    final, after_embedding = converse(
                        [
                            {
                                "type": "function_call_output",
                                "call_id": embedding_call,
                                "output": "Embedding complete. Return the remembered code.",
                            }
                        ],
                        True,
                        "auto",
                    )
                    if code not in final.output_text:
                        raise AssertionError(
                            f"conversation lost {code}: {final.output_text!r}"
                        )
                    snapshot = vectors["ds4"].get("resources")
                    if not snapshot or not result["ds4"].get("resources"):
                        raise AssertionError(
                            "start the qualification server with DS4_SUITE_RESOURCE_METRICS=1"
                        )
                    if snapshot:
                        if any(any(owner["reserved"]) for owner in snapshot["owners"]):
                            raise AssertionError("operation left a live reservation")
                        snapshots.append(snapshot)
                    elapsed = time.perf_counter() - begin
                    record = {
                        "cycle": cycle,
                        "warmup": cycle == 0,
                        "status": "PASS",
                        "seconds": elapsed,
                        "image_seconds": image_time,
                        "embedding_seconds": embed_time,
                        "image_sha256": digest,
                        "embedding_sha256": digest_vectors,
                        "dimensions": dimensions,
                        "cached_after_image": after_image,
                        "cached_after_embedding": after_embedding,
                        "image": result["ds4"],
                        "embedding": vectors["ds4"],
                        "process": memory_sample(args.pid),
                    }
                    log.write(json.dumps(record, separators=(",", ":")) + "\n")
                    log.flush()
                    if cycle:
                        for key, value in (
                            ("image", image_time),
                            ("embedding", embed_time),
                            ("cycle", elapsed),
                        ):
                            samples[key].append(value)
                        report["completed_cycles"] = cycle
                    print(
                        f"cycle {cycle}/{args.cycles} PASS {elapsed:.3f}s", flush=True
                    )
                if snapshots:
                    native = [
                        next(o for o in s["owners"] if o["owner"] == 500)
                        for s in snapshots[4:]
                    ]
                    if native and any(o != native[0] for o in native):
                        raise AssertionError(
                            "native owner did not return to its bounded warm state"
                        )
                report["status"] = "PASS"
            except Exception as error:
                report["status"] = "FAIL"
                report["error"] = repr(error)
                raise
            finally:
                report["timings"] = {
                    key: timing(values) for key, values in samples.items() if values
                }
                report["owned_memory_observed"] = bool(snapshots)
                (args.output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18095/v1")
    parser.add_argument("--chat-model")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument(
        "--video",
        type=Path,
        help="H264 MP4; otherwise generate a frozen development fixture",
    )
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
