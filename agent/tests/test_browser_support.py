from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests import browser_support
from tests.browser_support import BrowserTestServer

SERVER_SCRIPT = """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exmekh_agent.config import config

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(config.data_dir).encode())

# Exceeds an unread pipe's capacity; startup must still complete.
print("server output " * 10000, flush=True)
ThreadingHTTPServer(("127.0.0.1", config.port), Handler).serve_forever()
"""


def test_servers_use_independent_ports_and_close_on_caller_error(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT, encoding="utf-8")
    first_data = tmp_path / "first"
    second_data = tmp_path / "second"
    first = BrowserTestServer(first_data, script=script)
    second = BrowserTestServer(second_data, script=script)
    with pytest.raises(RuntimeError, match="caller failed"):
        with first as first_url, second as second_url:
            assert first_url != second_url
            with httpx.Client(trust_env=False) as client:
                assert client.get(first_url).text == str(first_data)
                assert client.get(second_url).text == str(second_data)
            raise RuntimeError("caller failed")
    first.close()
    second.close()
    with httpx.Client(timeout=1, trust_env=False) as client:
        for url in (first.url, second.url):
            with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
                client.get(url)


def test_failed_startup_reports_child_process_output(tmp_path: Path) -> None:
    script = tmp_path / "failed.py"
    script.write_text(
        'raise RuntimeError("server initialization failed")', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="server initialization failed"):
        with BrowserTestServer(tmp_path, script=script):
            pytest.fail("failed server must not enter its context")


def test_browser_uses_bundled_chromium_when_edge_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser_support, "EDGE_PATHS", ())
    monkeypatch.setattr(browser_support.shutil, "which", lambda _: None)
    assert browser_support.get_edge_executable() is None


def test_browser_uses_edge_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_support, "EDGE_PATHS", ())
    monkeypatch.setattr(browser_support.shutil, "which", lambda _: "/usr/bin/msedge")
    assert browser_support.get_edge_executable() == "/usr/bin/msedge"


def test_browser_prefers_installed_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edge = tmp_path / "msedge.exe"
    edge.touch()
    monkeypatch.setattr(browser_support, "EDGE_PATHS", (edge,))
    assert browser_support.get_edge_executable() == str(edge)
