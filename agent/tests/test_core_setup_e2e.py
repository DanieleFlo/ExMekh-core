"""Browser coverage of the managed Core setup and restart gate."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import BrowserTestServer

FEATURES = [
    {"id": key, "available": key == "vision", "reason": "Unavailable"}
    for key in ("vision", "images", "embeddings")
]
PROFILE = {
    "model_id": "test",
    "mode_id": "vram",
    "context_tokens": 4096,
    "features": [],
}
CATALOG = {
    "hardware": {
        "backend": "cuda",
        "ram_bytes": 64 * 2**30,
        "unified_memory": False,
        "devices": [],
        "disk_free_bytes": 80 * 2**30,
    },
    "models": [
        {
            "id": "test",
            "name": "Test model",
            "filename": "test.gguf",
            "size_bytes": 1000,
            "installed": False,
            "importable": False,
            "features": FEATURES,
            "modes": [
                {
                    "mode_id": "vram",
                    "label": "Fully in VRAM",
                    "available": True,
                    "recommended_context_tokens": 4096,
                    "estimated_max_context_tokens": 8192,
                }
            ],
        }
    ],
}
ACTIVITY = {
    "id": "load-1",
    "model_id": "test",
    "state": "running",
    "phase": "opening_model",
    "kind": "apply",
    "elapsed_seconds": 3,
    "progress": {"percent": 33, "unit": "milestones"},
    "profile": PROFILE,
}


@pytest.fixture
def core_http():
    state = {
        "version": 1,
        "boot_id": "boot-1",
        "revision": 0,
        "state": "empty",
        "chat_ready": False,
    }

    class CoreFixture(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            data = {
                "/v1/manage/status": state,
                "/v1/manage/catalog": CATALOG,
                "/v1/manage/jobs": [],
                "/v1/manage/jobs/load-1": ACTIVITY,
                "/v1/manage/hardware": CATALOG["hardware"],
            }
            self.reply(data[self.path]) if self.path in data else self.reply({}, 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/manage/plans":
                assert payload == PROFILE
                self.reply({"id": "plan-1", "missing_bytes": 1000})
            elif self.path == "/v1/manage/jobs" and payload.get("kind") == "apply":
                assert payload["plan_id"] == "plan-1"
                state.update(state="starting", job_id="load-1")
                self.reply(ACTIVITY, 202)
            else:
                self.reply({}, 404)

    with ThreadingHTTPServer(("127.0.0.1", 0), CoreFixture) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", state
        finally:
            server.shutdown()
            thread.join()


async def test_startup_gate_remounts_and_settings_are_separate(tmp_path, core_http):
    base_url, state = core_http
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent_url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                browser_page = await browser.new_page(
                    viewport={"width": 390, "height": 844}
                )
                await browser_page.goto(agent_url)
                await expect(browser_page.get_by_text("Test model")).to_be_visible()
                await browser_page.get_by_role(
                    "button", name="Configure", exact=True
                ).click()
                await expect(
                    browser_page.get_by_text("Estimated maximum · 8,192 tokens")
                ).to_be_visible()
                await browser_page.get_by_role(
                    "button", name="Continue", exact=True
                ).click()
                await expect(
                    browser_page.get_by_text("Analyze images in chat")
                ).to_be_visible()
                await browser_page.get_by_role(
                    "button", name="Continue with chat only"
                ).click()
                await browser_page.get_by_role(
                    "button", name="Download and start"
                ).click()
                await expect(
                    browser_page.get_by_text("Opening model weights")
                ).to_be_visible()
                await browser_page.reload()
                await expect(
                    browser_page.get_by_text("Opening model weights")
                ).to_be_visible()
                await expect(
                    browser_page.get_by_role("button", name="Open chat")
                ).to_have_count(0)
                state.update(
                    state="ready",
                    chat_ready=True,
                    job_id=None,
                    saved_profile=PROFILE,
                    active_profile={
                        **PROFILE,
                        "available_features": [],
                        "api_model": "test",
                        "owned_by": "ds4.c",
                        "runtime_id": "worker-1",
                    },
                )
                await browser_page.get_by_role("button", name="Open chat").click()
                await expect(browser_page.locator(".lm-header")).to_be_visible()
                draft = browser_page.locator(".chat-input-textarea textarea")
                await draft.fill("Keep this draft")
                await browser_page.locator(
                    '.lm-header button:has(i:text-is("tune"))'
                ).click()
                await expect(
                    browser_page.get_by_role("link", name="Reconfigure startup")
                ).to_be_visible()
                await browser_page.get_by_role(
                    "link", name="Conversations", exact=True
                ).click()
                await expect(
                    browser_page.get_by_text("Defaults for conversations in Agent.")
                ).to_be_visible()
                assert await browser_page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                await browser_page.get_by_role("link", name="Back to chat").click()
                await expect(draft).to_have_value("Keep this draft")
                active = state["active_profile"]
                state.update(
                    boot_id="boot-2",
                    state="starting",
                    chat_ready=False,
                    active_profile=None,
                    job_id="load-1",
                )
                await expect(
                    browser_page.get_by_text("Opening model weights")
                ).to_be_visible()
                await expect(draft).to_have_count(0)
                state.update(
                    state="ready",
                    chat_ready=True,
                    job_id=None,
                    active_profile={**active, "runtime_id": "worker-2"},
                )
                await expect(draft).to_have_value("Keep this draft")
            finally:
                await browser.close()
