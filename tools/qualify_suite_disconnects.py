"""Disconnect real HTTP peers at fenced native image boundaries.

Use the qualification-only tests/probe_suite_phase_server binary. The release
binary has no barrier hook. Native exhaustive fault tests cover every DiT
layer; this driver covers both ends of DiT and every named VAE frontier.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image

IMAGE = "Qwen/Qwen-Image-2.1-Turbo"
EMBED = "google/embeddinggemma-2"


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    ready = Path(str(args.barrier) + ".ready")
    cases = [("conditioning", 0)]
    cases += [("block", i) for i in (range(32) if args.all_layers else (0, 31))]
    cases += [("latent", i) for i in (range(8) if args.all_layers else (0, 7))]
    cases += [(stage, i) for stage in ("encoder", "decoder") for i in (0, 1, 2, 3, 5)]
    cases += [("posterior", 0), ("encoded", 0)]
    tools = [
        {
            "type": "function",
            "name": "suite_image",
            "description": "Run the requested image operation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    target = urlparse(args.url)
    args.barrier.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    with OpenAI(
        base_url=args.url, api_key="local", timeout=600, max_retries=0
    ) as client:
        llm = next(
            m.id for m in client.models.list().data if m.id not in (IMAGE, EMBED)
        )
        pixels = io.BytesIO()
        Image.new("RGBA", (256, 256), (220, 40, 30, 128)).save(pixels, format="PNG")
        reference = client.files.create(
            file=("reference.png", pixels.getvalue(), "image/png"), purpose="ds4_media"
        )
        request = {
            "model": IMAGE,
            "prompt": "A red square on white.",
            "size": "64x96",
            "images": [reference.id],
            "ds4": {"version": 1, "seed": 20261020},
        }

        def generate():
            response = client.post("/images/edits", body=request, cast_to=dict)
            (args.output / "last-image-metadata.json").write_text(
                json.dumps(response["ds4"], indent=2) + "\n"
            )
            png = base64.b64decode(response["data"][0]["b64_json"], validate=True)
            return hashlib.sha256(png).hexdigest()

        expected = generate()
        summary = {
            "status": "RUNNING",
            "cases": len(cases),
            "completed": 0,
            "scope": "Real HTTP socket disconnect at named native phases; first/last DiT layer and step unless --all-layers",
            "release_observer_hook": False,
            "initial_png_sha256": expected,
        }
        native_warm = None
        try:
            with (args.output / "cases.jsonl").open("x") as log:
                for stage, layer in cases:
                    before = client.responses.create(
                        model=llm,
                        input=[
                            {
                                "role": "user",
                                "content": "Remember BETA21. Call suite_image. After its result reply only BETA21.",
                            }
                        ],
                        tools=tools,
                        tool_choice="required",
                        max_output_tokens=128,
                        temperature=0,
                        reasoning={"effort": "none"},
                    )
                    calls = [o for o in before.output if o.type == "function_call"]
                    assert len(calls) == 1 and calls[0].name == "suite_image"
                    ready.unlink(missing_ok=True)
                    args.barrier.write_text(f"{stage} {layer}\n")
                    body = json.dumps(request).encode()
                    started = time.monotonic()
                    with socket.create_connection(
                        (target.hostname, target.port or 80), timeout=600
                    ) as peer:
                        peer.sendall(
                            (
                                f"POST /v1/images/edits HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
                            ).encode()
                            + body
                        )
                        deadline = time.monotonic() + 600
                        while (
                            not ready.exists()
                            or ready.read_text().strip() != f"{stage} {layer}"
                        ):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    f"native boundary not reached: {stage}/{layer}"
                                )
                            time.sleep(0.01)
                        assert ready.read_text().strip() == f"{stage} {layer}"
                        peer.shutdown(socket.SHUT_RDWR)
                    # No replay or load/unload call. The next LLM operation can
                    # run only after cancellation drains and restoration ends.
                    after = client.responses.create(
                        model=llm,
                        input=[
                            {
                                "type": "function_call_output",
                                "call_id": calls[0].call_id,
                                "output": "Image request cancelled. Reply only BETA21.",
                            }
                        ],
                        tools=tools,
                        tool_choice="auto",
                        max_output_tokens=24,
                        temperature=0,
                        reasoning={"effort": "none"},
                    )
                    cached = after.usage.input_tokens_details.cached_tokens
                    assert "BETA21" in after.output_text and cached > 0
                    vector = client.embeddings.create(
                        model=EMBED, input="after cancellation", dimensions=128
                    )
                    resources = vector.model_extra["ds4"]["resources"]
                    assert not any(any(o["reserved"]) for o in resources["owners"])
                    native = next(o for o in resources["owners"] if o["owner"] == 500)
                    if native_warm is None:
                        native_warm = native
                    assert native == native_warm
                    files = sorted(
                        p.name for p in args.storage.rglob("*") if p.is_file()
                    )
                    # One input reference and one committed baseline output;
                    # a cancelled image must not publish another file.
                    assert len(files) == 2, files
                    row = {
                        "stage": stage,
                        "layer": layer,
                        "status": "PASS",
                        "cached_tokens": cached,
                        "seconds": time.monotonic() - started,
                        "files": files,
                        "native_owner": native,
                        "reservations_zero": True,
                    }
                    log.write(json.dumps(row, separators=(",", ":")) + "\n")
                    log.flush()
                    summary["completed"] += 1
                    print(f"disconnect {stage}/{layer}: PASS", flush=True)
            args.barrier.unlink(missing_ok=True)
            summary["final_png_sha256"] = generate()
            assert summary["final_png_sha256"] == expected
            summary["status"] = "PASS"
        except Exception as error:
            summary.update(status="FAIL", error=repr(error))
            raise
        finally:
            args.barrier.unlink(missing_ok=True)
            ready.unlink(missing_ok=True)
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18095/v1")
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
