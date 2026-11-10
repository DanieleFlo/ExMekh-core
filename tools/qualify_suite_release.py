"""Native HTTP lifetime acceptance; Python is a development client only.

Use one running ds4-server-suite, one SDK client, and no external model worker.
Validated cycles are durably appended; compatible interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import time
import uuid
from pathlib import Path

if __package__:
    from .qualification_checkpoint import (
        ReleaseCheckpoint,
        atomic_bytes,
        atomic_json,
        digest_file,
    )
    from .qualification_suite import SuiteReleaseHistory, timing
    from .qualify_metal_command import (
        interrupt_on_termination,
    )
    from .qualify_metal_command import (
        sample as metal_memory_sample,
    )
else:
    from qualification_checkpoint import (
        ReleaseCheckpoint,
        atomic_bytes,
        atomic_json,
        digest_file,
    )
    from qualification_suite import SuiteReleaseHistory, timing
    from qualify_metal_command import (
        interrupt_on_termination,
    )
    from qualify_metal_command import (
        sample as metal_memory_sample,
    )

ROOT = Path(__file__).resolve().parents[1]
__all__ = ["run", "timing"]
IMAGE = "Qwen/Qwen-Image-2.1-Turbo"
EMBED = "google/embeddinggemma-2"


def memory_sample(pid):
    if not pid:
        return None
    if platform.system() == "Darwin":
        return metal_memory_sample(pid)
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

    if args.bounded:
        if not 1 <= args.cycles <= 3:
            raise ValueError(
                "bounded development qualification requires one to three cycles"
            )
    elif args.cycles < 50:
        raise ValueError("T21 requires at least 50 cycles after warm-up")
    width, height = map(int, args.image_size.split("x"))
    if width <= 0 or height <= 0:
        raise ValueError("positive image dimensions required")
    resume = bool(getattr(args, "resume", False))
    if resume and not os.environ.get("DS4_QUALIFICATION_PROFILE"):
        raise ValueError(
            "resume requires the qualification controller's binary/model provenance"
        )
    args.output.mkdir(parents=True, exist_ok=resume)
    video_path = args.video or args.output / "fixture.mp4"
    if not args.video and not video_path.exists():
        temporary = args.output / f".fixture-{uuid.uuid4().hex}.mp4"
        try:
            fixture_video(temporary)
            atomic_bytes(video_path, temporary.read_bytes())
        finally:
            temporary.unlink(missing_ok=True)
    fixtures = [
        ROOT / "tests/fixtures/native-suite-v1/media/image-0.png",
        ROOT / "tests/fixtures/native-suite-audio-v2/it.wav",
        video_path,
    ]
    fixture_manifest = [
        {"path": str(p), "sha256": digest_file(p), "bytes": p.stat().st_size}
        for p in fixtures
    ]
    profile = (
        json.loads(Path(os.environ["DS4_QUALIFICATION_PROFILE"]).read_text())
        if os.environ.get("DS4_QUALIFICATION_PROFILE")
        else None
    )
    with OpenAI(
        base_url=args.url, api_key="local", timeout=600, max_retries=0
    ) as client:
        catalog = [m.model_dump(exclude_unset=True) for m in client.models.list().data]
        chat_model = args.chat_model or next(
            m["id"] for m in catalog if m["id"] not in (IMAGE, EMBED)
        )
        identity = {
            "provenance": profile["identity"] if profile else None,
            "configuration": {
                "bounded": args.bounded,
                "cycles": args.cycles,
                "image_size": args.image_size,
                "chat_model": chat_model,
                "fixtures": [
                    {"sha256": row["sha256"], "bytes": row["bytes"]}
                    for row in fixture_manifest
                ],
            },
        }
        segment = profile["server_instance"] if profile else uuid.uuid4().hex
        with ReleaseCheckpoint(args.output, identity, segment, resume) as checkpoint:
            atomic_json(args.output / f"catalog-{segment}.json", catalog)
            atomic_json(args.output / "fixtures.json", fixture_manifest)
            return _run_cycles(args, client, checkpoint, fixtures, chat_model)


def _media_parts(client, fixtures):
    identifiers = [
        client.files.create(
            file=(path.name, path.read_bytes(), mime), purpose="ds4_media"
        ).id
        for path, mime in zip(fixtures, ("image/png", "audio/wav", "video/mp4"))
    ]
    image, audio, video = identifiers
    parts = [
        [{"type": "text", "text": "Un quadrato rosso"}],
        [{"type": "image", "file_id": image}],
        [{"type": "video", "file_id": video, "audio": "exclude"}],
        [{"type": "audio", "file_id": audio}],
    ]
    return [*parts, [part for row in parts for part in row]]


def _run_cycles(args, client, checkpoint, fixtures, chat_model):
    history = SuiteReleaseHistory((), args.image_size)
    report = {
        "version": 2,
        "status": "RUNNING",
        "completed_cycles": checkpoint.completed,
        "warmup_cycles": sum(row["warmup"] for row in checkpoint.records),
        "bounded_development_subset": args.bounded,
        "required_cycles": args.cycles,
        "scope": "HTTP lifetime and deterministic residency; no quality promotion",
        "checkpoint": str(args.output / "checkpoint.json"),
        "resumed": bool(getattr(args, "resume", False)),
    }
    checkpoint.save_summary(report)

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

    try:
        history.restore(checkpoint.records)
        parts = _media_parts(client, fixtures)
        for cycle, warmup in checkpoint.pending_cycles():
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
            seed = history.seed(cycle)
            start = time.perf_counter()
            result = client.images.generate(
                model=IMAGE,
                prompt="A red square on white.",
                size=args.image_size,
                background="transparent",
                extra_body={"ds4": {"version": 1, "seed": seed}},
            ).model_dump(exclude_unset=True)
            image_time = time.perf_counter() - start
            png = base64.b64decode(result["data"][0]["b64_json"], validate=True)
            digest = history.image_digest(cycle, png)
            image_file = checkpoint.save_image(cycle, warmup, png)
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
            dimensions = history.dimensions(cycle)
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
            digest_vectors = history.embedding_digest(cycle, matrix)
            checkpoint.save_vectors(dimensions, matrix)
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
                raise AssertionError(f"conversation lost {code}: {final.output_text!r}")
            snapshot = vectors["ds4"].get("resources")
            if not snapshot or not result["ds4"].get("resources"):
                raise AssertionError(
                    "start the qualification server with DS4_SUITE_RESOURCE_METRICS=1"
                )
            history.validate_resources(snapshot)
            elapsed = time.perf_counter() - begin
            record = {
                "cycle": cycle,
                "warmup": warmup,
                "status": "PASS",
                "seconds": elapsed,
                "image_seconds": image_time,
                "embedding_seconds": embed_time,
                "image_sha256": digest,
                "image_file": image_file,
                "embedding_sha256": digest_vectors,
                "dimensions": dimensions,
                "cached_after_image": after_image,
                "cached_after_embedding": after_embedding,
                "image": result["ds4"],
                "embedding": vectors["ds4"],
                "process": memory_sample(args.pid),
            }
            history.record(checkpoint.append(record))
            checkpoint.save_summary(report)
            print(f"cycle {cycle}/{args.cycles} PASS {elapsed:.3f}s", flush=True)
        history.validate_resting_state()
        report.update(checkpoint.completion())
    except BaseException as error:
        interrupted = not isinstance(error, Exception) or type(error).__name__ in (
            "APIConnectionError",
            "APITimeoutError",
        )
        report["status"] = "INTERRUPTED" if interrupted else "FAIL"
        report["error"] = repr(error)
        if isinstance(error, AssertionError):
            checkpoint.reject_gate(error)
        raise
    finally:
        report["timings"] = history.timings()
        report["owned_memory_observed"] = history.owned_memory_observed
        checkpoint.save_summary(report)
    return report


def main():
    # The controller validates exact URL/output options before launch.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--url", default="http://127.0.0.1:18095/v1")
    parser.add_argument("--chat-model")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument(
        "--bounded",
        action="store_true",
        help="Explicit one-to-three-cycle development subset; no T21 promotion",
    )
    parser.add_argument("--image-size", default="64x96")
    parser.add_argument(
        "--video",
        type=Path,
        help="H264 MP4; otherwise generate a frozen development fixture",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue compatible, validated completed cycles",
    )
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    with interrupt_on_termination():
        report = run(parser.parse_args())
    print(
        f"{report['status']} {report['completed_cycles']}/{report['required_cycles']} cycles",
        flush=True,
    )


if __name__ == "__main__":
    main()
