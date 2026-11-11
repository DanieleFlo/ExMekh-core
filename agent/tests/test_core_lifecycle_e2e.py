"""Run the browser against real Core state and HTTP with tiny local artifacts."""

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import httpx
import pytest
from playwright.async_api import Page, async_playwright, expect

from tests.browser_support import BrowserTestServer, get_edge_executable

if os.name != "posix":
    pytest.skip("Core uses Unix sockets and file locks", allow_module_level=True)

with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    import managed_catalog
    from managed_assets import CHUNK, Artifact
    from managed_http import ManagedHTTP
    from managed_server import Core

MODEL = "Qwen3.8-27B UD-Q4_K_S"
ENCODER = "encoder.gguf"
PROFILE = {
    "model_id": "qwen27",
    "mode_id": "vram",
    "context_tokens": 2048,
    "features": ["vision"],
}
NATIVE_WORKER = """
import json
import os
import socket
import time
from pathlib import Path

with (
    socket.socket(fileno=int(os.environ['DS4_MANAGED_CONTROL_FD'])) as control,
    socket.socket(fileno=int(os.environ['DS4_MANAGED_LISTENER_FD'])) as listener,
):
    control.sendall(b'opening_model\\ninitializing_runtime\\n')
    ready = Path(os.environ['TEST_CORE_WORKER_READY'])
    deadline = time.monotonic() + 30
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not ready.exists():
        raise TimeoutError('The test did not release the native worker')
    control.sendall(b'ready\\n')
    body = json.dumps({'data': [{'id': 'test-qwen', 'owned_by': 'ds4.c'}]}).encode()
    while True:
        connection, _ = listener.accept()
        with connection, connection.makefile('rb') as request:
            while request.readline().strip():
                pass
            headers = f'HTTP/1.1 200 OK\\r\\nContent-Length: {len(body)}\\r\\n\\r\\n'
            connection.sendall(headers.encode() + body)
"""


@contextmanager
def serve(server):
    with server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def core_inputs(tmp_path, monkeypatch):
    download = Event()
    model_bytes = b"Q" * (CHUNK * 2)
    files = {"test.gguf": model_bytes, ENCODER: b"tiny vision encoder"}
    ranges = []

    class Downloads(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.removeprefix("/")
            payload = files.get(name)
            if payload is None:
                self.send_error(404)
                return
            offset = 0
            if header := self.headers.get("Range"):
                ranges.append(header)
                offset = int(header.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            if not offset:
                self.wfile.write(payload[:CHUNK])
            if len(payload) > CHUNK and (offset or download.wait(30)):
                self.wfile.write(payload[CHUNK:])

    worker = tmp_path / "native-worker"
    worker.write_text(f"#!{sys.executable}\n{NATIVE_WORKER}")
    worker.chmod(0o700)
    ready = tmp_path / "worker-ready"
    monkeypatch.setenv("DS4_MANAGED_SERVER_BIN", str(worker))
    monkeypatch.setenv("TEST_CORE_WORKER_READY", str(ready))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    gib = 2**30
    host = {
        "backend": "cuda",
        "system": "Linux",
        "ram_bytes": 64 * gib,
        "ram_available_bytes": 48 * gib,
        "unified_memory": False,
        "devices": [
            {
                "name": "Test GPU",
                "total_bytes": 24 * gib,
                "free_bytes": 24 * gib,
                "compute": "8.9",
            }
        ],
        "disk_free_bytes": 100 * gib,
    }
    monkeypatch.setattr(managed_catalog, "hardware", lambda *_: host)
    with serve(ThreadingHTTPServer(("127.0.0.1", 0), Downloads)) as origin:
        artifacts = {
            name: Artifact(
                name, f"{origin}/{name}", len(data), hashlib.sha256(data).hexdigest()
            )
            for name, data in files.items()
        }
        monkeypatch.setitem(
            managed_catalog.MODELS,
            "qwen27",
            {
                "name": MODEL,
                "artifact": artifacts["test.gguf"],
                "vision": artifacts[ENCODER],
            },
        )
        try:
            yield download, ready, ranges
        finally:
            download.set()


@contextmanager
def run_core(tmp_path, release_download, port=0):
    core = Core(tmp_path / "state", tmp_path / "models", "cuda")
    try:
        server = ManagedHTTP(("127.0.0.1", port), core, None)
        core.restore_saved_profile()
        with serve(server) as url:
            yield url + "/v1", server.server_port
    finally:
        release_download.set()
        core.close()


async def download_model(page: Page, release_download: Event) -> None:
    await page.get_by_role("button", name="Download", exact=True).click()
    await expect(page.get_by_text(re.compile(r"50% .*4.* / 8.*?/s"))).to_be_visible()
    await expect(page.get_by_role("progressbar", name="Downloading")).to_be_visible()
    await page.get_by_role("button", name="Pause", exact=True).click()
    await expect(page.get_by_text("Pausing", exact=True)).to_be_visible()
    release_download.set()
    await expect(page.get_by_text("Paused", exact=True)).to_be_visible()
    await page.reload()
    await page.get_by_role("button", name="Resume", exact=True).click()
    await expect(page.get_by_text("Already downloaded", exact=True)).to_be_visible()


async def start_profile(page: Page) -> None:
    await page.get_by_role("button", name="Configure memory and context").click()
    await page.get_by_role("spinbutton", name="Context tokens").fill("2048")
    await page.get_by_role("button", name="Continue", exact=True).click()
    await expect(
        page.get_by_role("checkbox", name="Generate and edit images")
    ).to_be_disabled()
    vision = page.get_by_role("checkbox", name="Analyze images in chat")
    await vision.click()
    await expect(vision).to_be_checked()
    await page.get_by_role("button", name="Review profile").click()
    await expect(page.get_by_text("2,048 tokens", exact=True)).to_be_visible()
    await expect(page.get_by_text("Analyze images in chat", exact=True)).to_be_visible()
    await page.get_by_role("button", name="Download and start").click()


async def test_fresh_install_downloads_starts_and_restores_the_same_profile(
    tmp_path, core_inputs
):
    download, ready, ranges = core_inputs
    saved = tmp_path / "state" / "profile.json"
    assert not saved.exists()
    assert not (tmp_path / "models").exists()

    async with (
        async_playwright() as playwright,
        httpx.AsyncClient(trust_env=False) as http,
    ):
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable()
        )
        try:
            page = await browser.new_page(viewport={"width": 1100, "height": 800})
            with run_core(tmp_path, download) as (base_url, port):
                environment = {
                    "BASE_URL": base_url,
                    "API_KEY": "",
                    "PYTHON_DOTENV_DISABLED": "1",
                }
                with BrowserTestServer(
                    tmp_path / "agent", environment=environment
                ) as agent:
                    await page.goto(agent)
                    await expect(
                        page.get_by_role("heading", name="Choose your chat model")
                    ).to_be_visible()
                    await expect(
                        page.get_by_text("Test GPU", exact=False)
                    ).to_be_visible()
                    await expect(
                        page.get_by_role("radio", name=re.compile("Flash Next"))
                    ).to_be_disabled()
                    assert (await http.get(f"{base_url}/models")).status_code == 503

                    await download_model(page, download)
                    assert ranges == [f"bytes={CHUNK}-"]
                    assert (await http.get(f"{base_url}/models")).status_code == 503
                    assert not saved.exists()
                    await expect(page.locator(".chat-input-textarea")).to_have_count(0)

                    await start_profile(page)
                    await expect(
                        page.get_by_role(
                            "progressbar", name="Initializing the inference session"
                        )
                    ).to_be_visible()
                    await expect(
                        page.get_by_text("Remaining time unavailable", exact=False)
                    ).to_be_visible()
                    await page.reload()
                    await expect(
                        page.get_by_role("heading", name="Core startup")
                    ).to_be_visible()
                    assert not saved.exists()
                    assert (await http.get(f"{base_url}/models")).status_code == 503
                    ready.touch()
                    await expect(
                        page.locator(".chat-input-textarea textarea")
                    ).to_be_visible()
                    status = (await http.get(f"{base_url}/manage/status")).json()
                    assert json.loads(saved.read_text())["profile"] == PROFILE
                    assert status["active_profile"]["available_features"] == ["vision"]
                    assert (await http.get(f"{base_url}/models")).status_code == 200

            ready.unlink()
            with BrowserTestServer(
                tmp_path / "agent", environment=environment
            ) as agent:
                await page.goto(agent)
                await expect(
                    page.get_by_role("heading", name="Connect to Core")
                ).to_be_visible()
                with run_core(tmp_path, download, port) as (base_url, _):
                    await expect(
                        page.get_by_role(
                            "progressbar", name="Initializing the inference session"
                        )
                    ).to_be_visible()
                    await expect(
                        page.get_by_text("Core connected", exact=True)
                    ).to_be_visible()
                    restore = (await http.get(f"{base_url}/manage/status")).json()
                    assert restore["boot_id"] != status["boot_id"]
                    assert restore["saved_profile"] == PROFILE
                    assert restore["activation_job"]["kind"] == "restore"
                    assert (await http.get(f"{base_url}/models")).status_code == 503
                    ready.touch()
                    await expect(
                        page.locator(".chat-input-textarea textarea")
                    ).to_be_visible()
                    assert (await http.get(f"{base_url}/models")).status_code == 200
        finally:
            await browser.close()
