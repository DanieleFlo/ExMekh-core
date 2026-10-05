"""Shared browser actions and isolated server lifecycles for Playwright tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator, Mapping
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import httpx
from playwright.async_api import Page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / ".test-tmp"
EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
SERVER_RUNNER = """
import os
import runpy
from pathlib import Path

from exmekh_agent.config import config

config.data_dir = Path(os.environ["EXMEKH_AGENT_TEST_DATA_DIR"])
if os.environ.get("EXMEKH_AGENT_TEST_SCRIPT"):
    runpy.run_path(os.environ["EXMEKH_AGENT_TEST_SCRIPT"], run_name="__main__")
else:
    runpy.run_module(os.environ["EXMEKH_AGENT_TEST_MODULE"], run_name="__main__")
"""


def get_edge_executable() -> str | None:
    """Use installed Edge when available, otherwise Playwright's Chromium."""
    return next(
        (str(path) for path in EDGE_PATHS if path.exists()), shutil.which("msedge")
    )


class BrowserTestServer:
    """Own startup, diagnostics, and process teardown for one isolated server."""

    def __init__(
        self,
        data_dir: Path,
        *,
        module: str = "exmekh_agent.main",
        script: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._module = module
        self._script = script
        self._environment = environment or {}
        self._process: subprocess.Popen | None = None
        self.url = ""

    def __enter__(self) -> str:
        with socket.socket() as available_port:
            available_port.bind(("127.0.0.1", 0))
            port = available_port.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        ARTIFACT_ROOT.mkdir(exist_ok=True)
        # Keep logs outside storage: Windows workers can retain inherited handles.
        log_path = ARTIFACT_ROOT / f"server-{port}-{uuid4().hex}.log"
        environment = dict(os.environ)
        environment.update(self._environment)
        environment.pop("PYTEST_CURRENT_TEST", None)
        environment.pop("EXMEKH_AGENT_TEST_SCRIPT", None)
        environment.update(
            EXMEKH_AGENT_TEST_DATA_DIR=str(self._data_dir),
            EXMEKH_AGENT_TEST_MODULE=self._module,
            EXMEKH_AGENT_PORT=str(port),
            EXMEKH_AGENT_HOST="127.0.0.1",
            PYTHONPATH=os.pathsep.join(
                [
                    str(PROJECT_ROOT / "src"),
                    str(PROJECT_ROOT),
                    environment.get("PYTHONPATH", ""),
                ]
            ),
        )
        if self._script is not None:
            environment["EXMEKH_AGENT_TEST_SCRIPT"] = str(self._script)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-c", SERVER_RUNNER],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if not self._wait_until_ready():
                self.close()
                output = log_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"Server startup failed on {self.url}. Output: {output}"
                )
        except BaseException:
            self.close()
            raise
        return self.url

    def _wait_until_ready(self) -> bool:
        assert self._process is not None
        deadline = time.monotonic() + 30
        with httpx.Client(timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline and self._process.poll() is None:
                try:
                    if client.get(self.url).status_code == 200:
                        return True
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
        return False

    def close(self) -> None:
        """Wait for process termination before callers remove its data directory."""
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def start_mock_server(
    stream_delay: float = 0,
    entry_module: str = "tests.run_mock_agent",
    *,
    steering_first_response: str = "tool",
) -> Generator[str]:
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    environment = {
        "EXMEKH_AGENT_MOCK_CLIENT": "1",
        "EXMEKH_AGENT_MOCK_STREAM_DELAY": str(stream_delay),
        "EXMEKH_AGENT_STEERING_FIRST_RESPONSE": steering_first_response,
    }
    with tempfile.TemporaryDirectory(
        prefix="server-data-", dir=ARTIFACT_ROOT
    ) as storage:
        with BrowserTestServer(
            Path(storage), module=entry_module, environment=environment
        ) as url:
            yield url


async def send_command(page: Page, text: str) -> None:
    """Wait for the server to publish the input, including queued messages."""
    previous_inputs = await page.locator(".msg-user, .exmekh-queue-item").count()
    textarea = page.locator("textarea").first
    await textarea.wait_for(state="visible", timeout=8000)
    await textarea.click()
    await textarea.fill(text)
    await asyncio.sleep(0.2)
    send_button = page.locator("button:has(i:has-text('arrow_upward'))").first
    if await send_button.is_visible():
        await send_button.click()
    else:
        await textarea.press("Enter")
    await page.wait_for_function(
        "previous => document.querySelectorAll('.msg-user, .exmekh-queue-item').length > previous",
        arg=previous_inputs,
        timeout=15000,
    )


async def wait_turn_completion(page: Page, timeout: float = 15000) -> None:
    send_button = page.locator("button:has(i:has-text('arrow_upward'))").first
    await send_button.wait_for(state="visible", timeout=timeout)
    await page.locator(".chat-message-streaming").wait_for(
        state="hidden", timeout=timeout
    )


async def wait_for_mock_requests(
    page: Page, stats_url: str, count: int, timeout: float = 10
) -> None:
    """Wait for the provider to receive requests before interacting with its turn."""

    async def wait_for_provider() -> None:
        while True:
            response = await page.request.get(stats_url)
            stats = await response.json()
            if len(stats["requests"]) >= count:
                return
            await asyncio.sleep(0.1)

    await asyncio.wait_for(wait_for_provider(), timeout)
