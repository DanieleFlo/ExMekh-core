"""Opt-in real image cancellation after a completed GPU diffusion step."""

import base64
import hashlib
import json
import os
import select
import socket
import time
import unittest
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

URL = os.getenv("DS4_SUITE_HTTP_URL")
CONTROL = os.getenv("DS4_SUITE_PHASE_CONTROL")


@unittest.skipUnless(URL and CONTROL, "requires the native phase qualification server")
class SuitePhaseLive(unittest.TestCase):
    def test_cancel_after_real_latent_restores_llm_and_retry_commits_once(self):
        control = Path(CONTROL)
        signal = control.with_name(control.name + ".ready")
        model = os.environ["DS4_SUITE_LLM_MODEL"]
        observed = {}
        remember = "Remember PHASE27. Reply only PHASE27."
        with OpenAI(
            base_url=URL,
            api_key="local",
            timeout=float(os.getenv("DS4_SUITE_HTTP_TIMEOUT_SECONDS", "300")),
            max_retries=0,
        ) as client:
            first = client.responses.create(
                model=model,
                input=remember,
                reasoning={"effort": "none"},
                max_output_tokens=16,
                temperature=0,
            )
            self.assertIn("PHASE27", first.output_text)
            # The qualification server stores media beneath this private
            # parent. The public API intentionally has no Files list endpoint.
            media = control.parent / "media"
            files_before = {
                str(path) for path in media.rglob("file-*") if path.is_file()
            }
            control.write_text("latent 0\n")
            key = "phase-cancel-" + os.urandom(8).hex()
            body = {
                "model": "Qwen/Qwen-Image-2.1-Turbo",
                "prompt": "A red square on white.",
                "size": "256x256",
                "n": 1,
                "ds4": {"version": 1, "seed": 20261020},
            }
            encoded = json.dumps(body).encode()
            target = urlparse(URL)
            try:
                with socket.create_connection(
                    (target.hostname, target.port or 80)
                ) as peer:
                    peer.sendall(
                        (
                            "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
                            f"Idempotency-Key: {key}\r\nContent-Length: {len(encoded)}\r\n"
                            "Content-Type: application/json\r\n\r\n"
                        ).encode()
                        + encoded
                    )
                    deadline = time.monotonic() + float(
                        os.getenv("DS4_SUITE_PHASE_TIMEOUT_SECONDS", "180")
                    )
                    while not signal.exists():
                        readable, _, _ = select.select([peer], [], [], 0)
                        if readable:
                            reply = peer.recv(8192).decode(errors="replace")
                            self.fail(
                                f"image request ended before latent boundary: {reply}"
                            )
                        self.assertLess(
                            time.monotonic(),
                            deadline,
                            "native latent boundary not reached",
                        )
                        time.sleep(0.1)
                    self.assertEqual(signal.read_text(), "latent 0\n")
            finally:
                control.unlink(missing_ok=True)
            # This LLM request waits for cancellation cleanup and restoration.
            # Pre-commit cancellation has no receipt; a later retry may run.
            final = client.responses.create(
                model=model,
                input=[
                    {"role": "user", "content": remember},
                    *[item.model_dump(exclude_unset=True) for item in first.output],
                    {"role": "user", "content": "Repeat the code you remembered."},
                ],
                reasoning={"effort": "none"},
                max_output_tokens=16,
                temperature=0,
            )
            self.assertIn("PHASE27", final.output_text)
            self.assertGreater(final.usage.input_tokens_details.cached_tokens, 0)
            files_after = {
                str(path) for path in media.rglob("file-*") if path.is_file()
            }
            self.assertEqual(files_before, files_after)
            vectors = client.embeddings.create(
                model="google/embeddinggemma-2",
                input="Recovery after image cancellation",
                dimensions=128,
            )
            self.assertEqual(len(vectors.data[0].embedding), 128)
            resources = vectors.model_extra["ds4"]["resources"]
            self.assertTrue(
                all(not any(owner["reserved"]) for owner in resources["owners"])
            )
            retried = client.post(
                "/images/generations",
                cast_to=dict,
                body=body,
                options={"headers": {"Idempotency-Key": key}},
            )
            duplicate = client.post(
                "/images/generations",
                cast_to=dict,
                body=body,
                options={"headers": {"Idempotency-Key": key}},
            )
            self.assertEqual(retried, duplicate)
            png = base64.b64decode(retried["data"][0]["b64_json"], validate=True)
            file_id = retried["data"][0]["ds4"]["file_id"]
            self.assertEqual(client.files.content(file_id).read(), png)
            published = {str(path) for path in media.rglob("file-*") if path.is_file()}
            self.assertEqual(len(published - files_before), 1)
            output = Path(os.environ["DS4_SUITE_E2E_ARTIFACTS"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "phase-retry.png").write_bytes(png)
            observed.update(
                boundary=signal.read_text(),
                retry_request_id=retried["ds4"]["request_id"],
                retry_png_sha256=hashlib.sha256(png).hexdigest(),
                files_before=sorted(files_before),
                files_after=sorted(files_after),
                first=first.model_dump(mode="json"),
                final=final.model_dump(mode="json"),
                resources=resources,
            )
        output = Path(os.environ["DS4_SUITE_E2E_ARTIFACTS"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "phase-cancel.json").write_text(json.dumps(observed, indent=2) + "\n")
