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
    import managed_hardware
    from managed_assets import CHUNK, Artifact
    from managed_http import ManagedHTTP
    from managed_server import Core

MODEL = "Qwen3.8-27B UD-Q4_K_S"
FLASH_FILE = "flash.gguf"
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
import sys
import time
from pathlib import Path

model = 'test-flash' if Path(sys.argv[sys.argv.index('-m') + 1]).name == 'flash.gguf' else 'test-qwen'
if '--cuda' in sys.argv and model == 'test-flash':
    assert '--mtp' in sys.argv and '--ssd-streaming' not in sys.argv
    assert sys.argv[sys.argv.index('--prefill-chunk') + 1] == '2048'
    assert os.environ['DS4_CUDA_QWEN4_TIERED_CACHE'] == '1'
elif '--cuda' in sys.argv:
    assert '--prefill-chunk' not in sys.argv
else:
    assert '--metal' in sys.argv and '--ssd-streaming' in sys.argv
    assert sys.argv[sys.argv.index('--prefill-chunk') + 1] == '64'

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
    body = json.dumps({'data': [{'id': model, 'owned_by': 'ds4.c'}]}).encode()
    while True:
        connection, _ = listener.accept()
        with connection, connection.makefile('rb') as request:
            line = request.readline()
            length = 0
            while header := request.readline().strip():
                if header.lower().startswith(b'content-length:'):
                    length = int(header.partition(b':')[2])
            if line.startswith(b'POST /v1/responses '):
                request.read(length)
                connection.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Type: text/event-stream\\r\\nConnection: close\\r\\n\\r\\n')
                assert request.read(1) == b''
                ready.with_name('request-cancelled').touch()
                release = ready.with_name('request-released')
                deadline = time.monotonic() + 15
                while not release.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                continue
            headers = (
                f'HTTP/1.1 200 OK\\r\\nContent-Length: {len(body)}\\r\\n'
                'Connection: close\\r\\n\\r\\n'
            )
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
    model_bytes = b"Q" * (CHUNK * 2)
    files = {
        "test.gguf": model_bytes,
        ENCODER: b"V" * (CHUNK * 2),
        FLASH_FILE: b"flash",
    }
    downloads = {name: Event() for name in files}
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
            if offset or downloads[name].wait(30):
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
        "execution_device_uuid": "GPU-test",
        "devices": [
            {
                "name": "Test GPU",
                "uuid": "GPU-test",
                "total_bytes": 24 * gib,
                "free_bytes": 24 * gib,
                "compute": "8.9",
            }
        ],
        "disk_free_bytes": 100 * gib,
    }
    monkeypatch.setattr(managed_hardware, "hardware", lambda *_: host)
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
        monkeypatch.setitem(
            managed_catalog.MODELS,
            "flash",
            {**managed_catalog.MODELS["flash"], "artifact": artifacts[FLASH_FILE]},
        )
        try:
            yield downloads, ready, ranges
        finally:
            for download in downloads.values():
                download.set()


@contextmanager
def run_core(tmp_path, downloads, port=0, *, backend="cuda"):
    core = Core(tmp_path / "state", tmp_path / "models", backend)
    try:
        server = ManagedHTTP(("127.0.0.1", port), core, None)
        core.restore_saved_profile()
        with serve(server) as url:
            yield url + "/v1", server.server_port
    finally:
        for download in downloads.values():
            download.set()
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


@pytest.mark.parametrize("separate_download", [True, False], ids=["resume", "queue"])
async def test_fresh_install_downloads_starts_and_restores_the_same_profile(
    tmp_path, core_inputs, separate_download
):
    downloads, ready, ranges = core_inputs
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
            page = await browser.new_page(
                viewport={"width": 1100 if separate_download else 390, "height": 800}
            )
            if not separate_download:
                await page.add_init_script(
                    "localStorage.setItem('exmekh-theme', 'light')"
                )
            with run_core(tmp_path, downloads) as (base_url, port):
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

                    if separate_download:
                        await download_model(page, downloads["test.gguf"])
                        assert ranges == [f"bytes={CHUNK}-"]
                        assert (await http.get(f"{base_url}/models")).status_code == 503
                        assert not saved.exists()
                        await expect(
                            page.locator(".chat-input-textarea")
                        ).to_have_count(0)

                    await start_profile(page)
                    await expect(
                        page.get_by_role("heading", name="Prepare model files")
                    ).to_be_visible()
                    chat_file = page.get_by_role("listitem").filter(
                        has_text="test.gguf"
                    )
                    vision_file = page.get_by_role("listitem").filter(has_text=ENCODER)
                    if not separate_download:
                        await expect(
                            chat_file.get_by_role("progressbar", name="Downloading")
                        ).to_be_visible()
                        await expect(
                            chat_file.get_by_text("50%", exact=False)
                        ).to_be_visible()
                        await expect(
                            vision_file.get_by_text("Queued", exact=True)
                        ).to_be_visible()
                        await page.reload()
                        await expect(
                            vision_file.get_by_text("Queued", exact=True)
                        ).to_be_visible()
                        downloads["test.gguf"].set()
                    await expect(
                        chat_file.get_by_text("Verified", exact=True)
                    ).to_be_visible()
                    await expect(
                        vision_file.get_by_role("progressbar", name="Downloading")
                    ).to_be_visible()
                    await expect(
                        page.get_by_text("1 of 2 files verified", exact=True)
                    ).to_be_visible()
                    assert await page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth"
                    )
                    await page.screenshot(
                        path=tmp_path / "download-queue.png", full_page=True
                    )
                    downloads[ENCODER].set()
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
                    await expect(vision_file).to_have_count(0)
                    await expect(
                        page.get_by_role("navigation", name="Setup steps")
                    ).to_have_count(0)
                    await page.screenshot(
                        path=tmp_path / "model-startup.png", full_page=True
                    )
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
                    history = (await http.get(f"{base_url}/manage/jobs")).json()
                    activation = next(
                        item for item in history if item["kind"] == "apply"
                    )
                    assert [
                        (item["name"], item["state"]) for item in activation["files"]
                    ] == [("test.gguf", "verified"), (ENCODER, "verified")]

            ready.unlink()
            with BrowserTestServer(
                tmp_path / "agent", environment=environment
            ) as agent:
                await page.goto(agent)
                await expect(
                    page.get_by_role("heading", name="Connect to Core")
                ).to_be_visible()
                with run_core(tmp_path, downloads, port) as (base_url, _):
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


@pytest.mark.parametrize(
    ("backend", "first_mode", "next_mode"),
    [("cuda", "vram", "tiered"), ("metal", "ssd", "ssd")],
)
async def test_busy_core_stops_the_request_before_switching_models(
    tmp_path, core_inputs, monkeypatch, backend, first_mode, next_mode
):
    downloads, ready, _ = core_inputs
    ready.touch()
    gib = 2**30
    host = {
        "backend": backend,
        "ram_bytes": 24 * gib,
        "ram_available_bytes": 18 * gib,
        "ram_stream_available_bytes": 14 * gib,
        "disk_free_bytes": 100 * gib,
    }
    if backend == "metal":
        host.update(
            system="Darwin",
            unified_memory=True,
            metal={"working_set_bytes": 18 * gib, "max_buffer_bytes": 2 * gib},
            devices=[],
        )
    else:
        host.update(
            system="Linux",
            unified_memory=False,
            execution_device_uuid="GPU-test",
            devices=[
                {
                    "name": "Test RTX GPU",
                    "uuid": "GPU-test",
                    "compute": "8.6",
                    "total_bytes": 24 * gib,
                    "free_bytes": 22 * gib,
                }
            ],
        )
    monkeypatch.setattr(managed_hardware, "hardware", lambda *_: host)
    state = tmp_path / "state"
    state.mkdir()
    profile = {**PROFILE, "mode_id": first_mode, "features": []}
    (state / "profile.json").write_text(
        json.dumps({"version": 1, "revision": 1, "profile": profile})
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "test.gguf").write_bytes(b"Q" * (CHUNK * 2))
    (models / FLASH_FILE).write_bytes(b"flash")

    with run_core(tmp_path, downloads, backend=backend) as (base_url, _):
        with BrowserTestServer(
            tmp_path / "agent", environment={"BASE_URL": base_url}
        ) as agent:
            async with (
                async_playwright() as playwright,
                httpx.AsyncClient(trust_env=False) as http,
            ):
                browser = await playwright.chromium.launch()
                try:
                    page = await browser.new_page()
                    await page.goto(agent)
                    await expect(
                        page.locator(".chat-input-textarea textarea")
                    ).to_be_visible()
                    async with http.stream(
                        "POST", f"{base_url}/responses", json={"input": "hello"}
                    ) as response:
                        assert response.status_code == 200
                        await page.locator(
                            '.lm-header button:has(i:text-is("tune"))'
                        ).click()
                        await expect(
                            page.get_by_role("button", name="Stop active requests")
                        ).to_be_visible()
                        await page.get_by_role(
                            "link", name="Reconfigure startup"
                        ).click()
                        await page.get_by_role(
                            "radio", name=re.compile("Flash Next")
                        ).check()
                        review = page.get_by_role("link", name="Review", exact=True)
                        await expect(review).to_have_attribute(
                            "href", re.compile("model=flash")
                        )
                        await review.click()
                        apply = page.get_by_role("button", name="Apply profile")
                        await expect(apply).to_be_disabled()

                        next_profile = {
                            **profile,
                            "model_id": "flash",
                            "mode_id": next_mode,
                        }
                        plan = await http.post(
                            f"{base_url}/manage/plans", json=next_profile
                        )
                        plan.raise_for_status()
                        request = {"kind": "apply", "plan_id": plan.json()["id"]}
                        busy = await http.post(f"{base_url}/manage/jobs", json=request)
                        assert (
                            busy.status_code == 409
                            and busy.json()["error"]["code"] == "INFERENCE_BUSY"
                        )

                        await page.get_by_role(
                            "button", name="Stop active requests"
                        ).click()
                        await expect(
                            page.get_by_text("Core is stopping requests", exact=False)
                        ).to_be_visible()
                        await expect(apply).to_be_disabled()
                        stopping = (await http.get(f"{base_url}/manage/status")).json()
                        assert (
                            stopping["active_requests"]
                            == stopping["stopping_requests"]
                            == 1
                        )
                        assert stopping["active_profile"]["model_id"] == "qwen27"
                        ready.with_name("request-released").touch()
                        await response.aread()
                        assert ready.with_name("request-cancelled").exists()
                        await expect(apply).to_be_enabled()
                        await apply.click()
                    await expect(
                        page.locator(".chat-input-textarea textarea")
                    ).to_be_visible()
                    final = (await http.get(f"{base_url}/manage/status")).json()
                    assert final["chat_ready"]
                    assert final["boot_id"] == stopping["boot_id"]
                    assert final["saved_profile"]["mode_id"] == next_mode
                    assert (
                        final["active_profile"]["model_id"]
                        == final["saved_profile"]["model_id"]
                        == "flash"
                    )
                finally:
                    await browser.close()
