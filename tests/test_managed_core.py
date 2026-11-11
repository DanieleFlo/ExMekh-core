"""Regression coverage for the headless management HTTP and artifact receipt."""

import hashlib
import http.client
import json
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from managed_assets import Artifact, PinnedStore, Problem  # noqa: E402
from managed_http import ManagedHTTP  # noqa: E402
from managed_server import Core  # noqa: E402
from managed_worker import NativeLaunch, NativeWorker  # noqa: E402


class ServerFixture:
    def __init__(self, server):
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


class ManagedCoreTests(unittest.TestCase):
    def test_regular_launchers_open_managed_http_with_or_without_saved_profile(self):
        launchers = ("start-ds4-server.sh", "agent/start-ds4-server.sh")
        for launcher, configured in product(launchers, (False, True)):
            with (
                self.subTest(launcher=launcher, configured=configured),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                state = root / "state"
                profile = {
                    "model_id": "qwen27",
                    "mode_id": "vram",
                    "context_tokens": 512,
                }
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("DS4_")
                }
                env.update(
                    DS4_STATE_DIR=str(state),
                    DS4_GGUF_DIR=str(root / "gguf"),
                    DS4_BACKEND="cpu",
                    DS4_SERVER_PORT="0",
                )
                if configured:
                    state.mkdir()
                    (state / "profile.json").write_text(
                        json.dumps({"version": 1, "revision": 1, "profile": profile})
                    )
                    env["DS4_MODEL_PROFILE"] = "qwen27"
                with subprocess.Popen(
                    ["bash", launcher],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ) as process:
                    try:
                        self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                        line = process.stdout.readline()
                        endpoint = re.search(r"http://127\.0\.0\.1:\d+/v1", line)
                        self.assertIsNotNone(endpoint, line)
                        with urllib.request.urlopen(
                            endpoint[0] + "/manage/status", timeout=2
                        ) as response:
                            status = json.load(response)
                        self.assertFalse(status["chat_ready"])
                        if configured:
                            self.assertIn(status["state"], {"starting", "error"})
                            self.assertEqual(status["saved_profile"]["context_tokens"], 512)
                        else:
                            self.assertEqual(status["state"], "empty")
                            self.assertIsNone(status["saved_profile"])
                            self.assertEqual(list((root / "gguf").glob("*.gguf*")), [])
                        with self.assertRaises(urllib.error.HTTPError) as unavailable:
                            urllib.request.urlopen(endpoint[0] + "/models", timeout=2)
                        self.assertEqual(unavailable.exception.code, 503)
                    finally:
                        process.terminate()
                        process.communicate(timeout=5)

    def test_http_reports_saved_profile_restore_while_model_work_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 512}
            (state / "profile.json").write_text(
                json.dumps({"version": 1, "revision": 1, "profile": profile})
            )
            entered = threading.Event()
            release = threading.Event()

            def prepare(_core, _job):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release the restore")

            with patch.object(Core, "_run_job", prepare):
                core = Core(state, root / "gguf", "cpu")
                try:
                    server = ManagedHTTP(("127.0.0.1", 0), core, None)
                    core.restore_saved_profile()
                    self.assertTrue(entered.wait(2))
                    with ServerFixture(server) as base:
                        with urllib.request.urlopen(
                            base + "/v1/manage/status", timeout=2
                        ) as response:
                            status = json.load(response)
                        self.assertEqual(status["state"], "starting")
                        self.assertFalse(status["chat_ready"])
                        job = status["activation_job"]
                        self.assertEqual(job["id"], status["job_id"])
                        self.assertEqual(job["kind"], "restore")
                        self.assertEqual(job["model_id"], profile["model_id"])
                        with urllib.request.urlopen(
                            base + "/v1/manage/jobs/" + job["id"], timeout=2
                        ) as response:
                            self.assertEqual(json.load(response)["id"], job["id"])
                        with self.assertRaises(urllib.error.HTTPError) as unavailable:
                            urllib.request.urlopen(base + "/v1/models", timeout=2)
                        self.assertEqual(unavailable.exception.code, 503)
                finally:
                    release.set()
                    core.close()

    def test_saved_profile_can_repair_an_inactive_corrupt_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, models = root / "state", root / "gguf"
            state.mkdir()
            models.mkdir()
            model = models / "Qwen3.8-27B-UD-Q4_K_S.gguf"
            model.write_bytes(b"broken")
            (state / "profile.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "revision": 1,
                        "profile": {
                            "model_id": "qwen27",
                            "mode_id": "vram",
                            "context_tokens": 512,
                        },
                    }
                )
            )
            core = Core(state, models, "cpu")
            try:
                entry = next(
                    item for item in core.catalog()["models"] if item["id"] == "qwen27"
                )
                self.assertTrue(entry["removable"])
                job = core.submit(
                    {"kind": "remove", "model_id": "qwen27"}, "repair-key"
                )
                self.assertNotIn("idempotency", job)
                for _ in range(100):
                    result = core.get_jobs(job["id"])
                    if result["state"] == "completed":
                        break
                    time.sleep(0.01)
                self.assertEqual(result["state"], "completed")
                self.assertEqual(result["allowed_actions"], [])
                self.assertFalse(model.exists())
                self.assertEqual(core.status()["saved_profile"]["model_id"], "qwen27")
            finally:
                core.close()

    def test_other_store_cannot_mutate_an_artifact_during_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"fixture"
            artifact = Artifact(
                "model.gguf", "", len(payload), hashlib.sha256(payload).hexdigest()
            )
            (root / artifact.path).write_bytes(payload)
            verifier, other = PinnedStore(root), PinnedStore(root)

            def report(*_):
                with self.assertRaises(Problem) as busy:
                    other.remove(artifact)
                self.assertEqual(busy.exception.code, "ARTIFACT_BUSY")

            verifier.ensure(artifact, report, lambda: None, import_only=True)
            self.assertTrue(verifier.installed(artifact))

    def test_worker_releases_its_listener_on_invalid_discovery(self):
        script = """
import os
import socket

with socket.socket(fileno=int(os.environ["DS4_MANAGED_LISTENER_FD"])) as listener:
    os.write(int(os.environ["DS4_MANAGED_CONTROL_FD"]), b"opening_model\\nready\\n")
    connection, _ = listener.accept()
    with connection:
        connection.recv(1024)
        connection.sendall(b"HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\n{}")
    listener.accept()
"""
        worker = NativeWorker()
        with self.assertRaises(Problem) as error:
            worker.start(
                NativeLaunch((sys.executable, "-c", script)),
                lambda *_: None,
                lambda: None,
            )
        self.assertEqual(error.exception.code, "WORKER_DISCOVERY_FAILED")
        self.assertFalse(worker.is_running())
        with self.assertRaises(OSError):
            worker.connect()

    def test_empty_core_and_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            core = Core(path / "state", path / "gguf", "cpu")
            try:
                with ServerFixture(
                    ManagedHTTP(("127.0.0.1", 0), core, "secret")
                ) as base:

                    def read(path, **headers):
                        return urllib.request.urlopen(
                            urllib.request.Request(base + path, headers=headers),
                            timeout=5,
                        )

                    with self.assertRaises(urllib.error.HTTPError) as unauthenticated:
                        read("/v1/manage/status")
                    self.assertEqual(unauthenticated.exception.code, 401)
                    with read(
                        "/v1/manage/status", Authorization="Bearer secret"
                    ) as result:
                        self.assertEqual(json.load(result)["state"], "empty")
                    with self.assertRaises(urllib.error.HTTPError) as unavailable:
                        read("/v1/models")
                    self.assertEqual(unavailable.exception.code, 503)
                    self.assertEqual(
                        json.load(unavailable.exception)["error"]["code"],
                        "MODEL_NOT_READY",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as wrong_origin:
                        read(
                            "/v1/manage/status",
                            Authorization="Bearer secret",
                            Origin="https://other.example",
                        )
                    self.assertEqual(wrong_origin.exception.code, 403)
            finally:
                core.close()

    def test_resume_download_and_require_verified_receipt(self):
        payload = b"fixture GGUF bytes"
        requests = []

        class Downloads(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.headers.get("Range"))
                offset = 5
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
                self.send_header("Content-Length", str(len(payload) - offset))
                self.end_headers()
                self.wfile.write(payload[offset:])

        with tempfile.TemporaryDirectory() as directory:
            store = PinnedStore(Path(directory))
            with ServerFixture(ThreadingHTTPServer(("127.0.0.1", 0), Downloads)) as url:
                artifact = Artifact(
                    "model.gguf", url, len(payload), hashlib.sha256(payload).hexdigest()
                )
                partial = store.root / "model.gguf.part"
                partial.write_bytes(payload[:5])
                states = []
                self.assertFalse(store.installed(artifact))
                self.assertEqual(store.remaining(artifact), len(payload) - 5)
                final = store.ensure(
                    artifact, lambda phase, *_: states.append(phase), lambda: None
                )
                self.assertEqual(requests, ["bytes=5-"])
                self.assertIn("downloading", states)
                self.assertIn("verifying", states)
                self.assertTrue(store.installed(artifact))
                self.assertFalse(partial.exists())
                final.write_bytes(b"!" * len(payload))
                self.assertFalse(store.installed(artifact))
                with self.assertRaises(Problem) as mismatch:
                    store.ensure(
                        artifact, lambda *_: None, lambda: None, import_only=True
                    )
                self.assertEqual(mismatch.exception.code, "CHECKSUM_MISMATCH")

    def test_proxy_preserves_multipart_and_propagates_disconnect(self):
        multipart = (
            b'--fixture\r\nContent-Disposition: form-data; name="image";'
            b' filename="sample.png"\r\n\r\n\x00\xff\r\n--fixture--\r\n'
        )
        received = []
        disconnect = threading.Event()

        with (
            tempfile.TemporaryDirectory() as directory,
            socket.socket(socket.AF_UNIX) as native,
        ):
            path = str(Path(directory) / "native.sock")
            native.bind(path)
            native.listen()
            native.settimeout(5)

            class WorkerCore:
                def is_closing(self):
                    return False

                @contextmanager
                def open_inference(self):
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(5)
                        connection.connect(path)
                        yield connection

            def accept():
                connection, _ = native.accept()
                with connection, connection.makefile("rb") as reader:
                    connection.settimeout(5)
                    request_line = reader.readline()
                    headers = http.client.parse_headers(reader)
                    body = reader.read(int(headers["Content-Length"]))
                    received.append(
                        (
                            request_line,
                            headers["Content-Type"],
                            headers["Idempotency-Key"],
                            body,
                        )
                    )
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                        b"event: operation.started\ndata: {}\n\n"
                    )
                    if connection.recv(1) == b"":
                        disconnect.set()

            worker = threading.Thread(target=accept)
            worker.start()
            with ServerFixture(
                ManagedHTTP(("127.0.0.1", 0), WorkerCore(), None)
            ) as base:
                request = urllib.request.Request(
                    base + "/v1/images/edits",
                    data=multipart,
                    headers={
                        "Content-Type": "multipart/form-data; boundary=fixture",
                        "Idempotency-Key": "test-upload",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.readline(), b"event: operation.started\n")
                self.assertTrue(disconnect.wait(5))
            worker.join(5)
            self.assertEqual(
                received,
                [
                    (
                        b"POST /v1/images/edits HTTP/1.1\r\n",
                        "multipart/form-data; boundary=fixture",
                        "test-upload",
                        multipart,
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
