"""Browser coverage of the managed Core setup and restart gate."""

import json
import re
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from playwright.async_api import async_playwright, expect

from tests.audit_theme_contrast import PROBE
from tests.browser_support import BrowserTestServer, get_edge_executable

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
ACTIVE_PROFILE = {
    **PROFILE,
    "available_features": [],
    "api_model": "test",
    "owned_by": "ds4.c",
    "runtime_id": "worker-1",
}


@pytest.fixture
def core_http():
    catalog = deepcopy(CATALOG)
    jobs = {}
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
            if not state.get("reachable", True):
                self.reply({"error": {"message": "Server unavailable"}}, 503)
                return
            if state.get("manual"):
                if self.path == "/v1/models":
                    self.reply({"data": [{"id": "test", "context_length": 4096}]})
                else:
                    self.reply({}, 404)
                return
            data = {
                "/v1/manage/status": state,
                "/v1/manage/catalog": catalog,
                "/v1/manage/jobs": list(reversed(jobs.values())),
                "/v1/manage/hardware": catalog["hardware"],
            }
            data.update(
                {f"/v1/manage/jobs/{job_id}": job for job_id, job in jobs.items()}
            )
            self.reply(data[self.path]) if self.path in data else self.reply({}, 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            plan_id = f"plan-{state['revision']}"
            if self.path == "/v1/manage/plans":
                assert payload == PROFILE
                missing = 0 if catalog["models"][0]["installed"] else 1000
                self.reply({"id": plan_id, "missing_bytes": missing})
            elif self.path == "/v1/manage/jobs" and payload.get("kind") == "apply":
                if payload["plan_id"] != plan_id:
                    self.reply(
                        {
                            "error": {
                                "code": "PLAN_STALE",
                                "message": "Review the profile again before applying it.",
                            }
                        },
                        409,
                    )
                    return
                jobs["load-1"] = ACTIVITY
                state.update(state="starting", job_id="load-1")
                self.reply(ACTIVITY, 202)
            elif self.path == "/v1/manage/jobs" and payload == {"kind": "restore"}:
                job = {**ACTIVITY, "kind": "restore"}
                jobs[job["id"]] = job
                state.update(state="starting", job_id=job["id"], error=None)
                self.reply(job, 202)
            else:
                self.reply({}, 404)

    with ThreadingHTTPServer(("127.0.0.1", 0), CoreFixture) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", state, jobs, catalog
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("installed", [False, True], ids=["empty", "downloaded"])
async def test_startup_gate_remounts_and_settings_are_separate(
    tmp_path, core_http, installed
):
    base_url, state, jobs, catalog = core_http
    catalog["models"][0]["installed"] = installed
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent_url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                browser_page = await browser.new_page(
                    viewport={"width": 390, "height": 844}
                )
                await browser_page.goto(agent_url)
                await expect(browser_page.get_by_text("Test model")).to_be_visible()
                await expect(
                    browser_page.get_by_role("radio", name="Test model", exact=False)
                ).to_be_checked()
                await expect(
                    browser_page.locator('nav [aria-current="step"]')
                ).to_have_text("Chat model")
                assert await browser_page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                await browser_page.get_by_role(
                    "button", name="Configure memory and context", exact=True
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
                    "button", name="Start model" if installed else "Download and start"
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
                jobs["load-1"] = {
                    **ACTIVITY,
                    "state": "completed",
                    "phase": "completed",
                }
                state.update(
                    state="ready",
                    chat_ready=True,
                    job_id=None,
                    saved_profile=PROFILE,
                    active_profile=ACTIVE_PROFILE,
                )
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
                    saved_profile=PROFILE,
                    job_id="restore-2",
                    activation_job={
                        **ACTIVITY,
                        "id": "restore-2",
                        "kind": "restore",
                        "phase": "initializing_runtime",
                        "elapsed_seconds": 7,
                        "progress": {
                            "percent": 66.7,
                            "unit": "milestones",
                            "eta_seconds": 4,
                        },
                    },
                )
                await expect(
                    browser_page.get_by_text("Initializing the inference session")
                ).to_be_visible()
                await expect(
                    browser_page.get_by_text("7 s elapsed", exact=False)
                ).to_be_visible()
                await expect(
                    browser_page.get_by_text("66.7% of milestones", exact=False)
                ).to_be_visible()
                await expect(
                    browser_page.get_by_text(
                        "About 4 s remaining, based on a previous load", exact=False
                    )
                ).to_be_visible()
                await expect(
                    browser_page.get_by_role("button", name="Open chat")
                ).to_have_count(0)
                await expect(draft).to_have_count(0)
                state.update(
                    state="ready",
                    chat_ready=True,
                    job_id=None,
                    activation_job=None,
                    active_profile={**active, "runtime_id": "worker-2"},
                )
                await expect(draft).to_have_value("Keep this draft")
            finally:
                await browser.close()


async def test_failed_activation_stays_visible_when_previous_model_is_restored(
    tmp_path, core_http
):
    base_url, state, jobs, _ = core_http
    error = {
        "code": "WORKER_START_FAILED",
        "message": "The new profile could not load.",
    }
    jobs["load-1"] = {**ACTIVITY, "state": "failed", "error": error}
    state.update(
        state="ready",
        chat_ready=True,
        active_profile=ACTIVE_PROFILE,
        saved_profile=PROFILE,
        error=error,
    )
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent_url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(f"{agent_url}/startup?job=load-1")
                await expect(page.get_by_text(error["message"])).to_be_visible()
                await expect(page.locator(".lm-header")).to_have_count(0)
                await page.get_by_role("button", name="Open chat").click()
                await expect(page.locator(".lm-header")).to_be_visible()
            finally:
                await browser.close()


async def test_agent_opened_before_core_follows_the_restore(tmp_path, core_http):
    base_url, state, jobs, _ = core_http
    state["reachable"] = False
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent_url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(agent_url)
                await expect(
                    page.get_by_role("heading", name="Connect to Core")
                ).to_be_visible()
                jobs["load-1"] = {**ACTIVITY, "kind": "restore"}
                state.update(
                    reachable=True,
                    state="starting",
                    saved_profile=PROFILE,
                    job_id="load-1",
                )
                await expect(page.get_by_text("Opening model weights")).to_be_visible()
                await expect(page.get_by_text("Core connected")).to_be_visible()
                await expect(page.locator(".lm-header")).to_have_count(0)
                jobs["load-1"] = {
                    **jobs["load-1"],
                    "state": "completed",
                    "phase": "completed",
                }
                state.update(
                    state="ready",
                    chat_ready=True,
                    active_profile=ACTIVE_PROFILE,
                    job_id=None,
                )
                await expect(page.locator(".lm-header")).to_be_visible()
            finally:
                await browser.close()


@pytest.mark.parametrize(
    ("width", "theme"), [(1100, "dark"), (390, "light")], ids=["desktop", "mobile"]
)
async def test_settings_navigation_and_setup_layout(tmp_path, core_http, width, theme):
    base_url, state, _, catalog = core_http
    model = catalog["models"][0]
    model["installed"] = True
    catalog["models"].append(
        {
            **model,
            "id": "stream",
            "name": "Streaming model",
            "installed": False,
            "modes": [
                {
                    **model["modes"][0],
                    "mode_id": "ssd",
                    "label": "SSD streaming",
                    "recommended_context_tokens": 2048,
                    "estimated_max_context_tokens": 4096,
                },
                *model["modes"],
            ],
        }
    )
    state.update(
        state="ready",
        chat_ready=True,
        saved_profile=PROFILE,
        active_profile={**ACTIVE_PROFILE, "available_features": ["images"]},
    )
    environment = {"BASE_URL": base_url, "API_KEY": "", "PYTHON_DOTENV_DISABLED": "1"}
    with BrowserTestServer(tmp_path / "agent", environment=environment) as agent:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            try:
                page = await browser.new_page(
                    viewport={"width": width, "height": 844},
                    has_touch=width == 390,
                )
                await page.add_init_script(
                    f"localStorage.setItem('exmekh-theme', '{theme}')"
                )
                await page.goto(agent)
                await page.locator('.lm-header button:has(i:text-is("tune"))').click()
                settings = page.get_by_role(
                    "navigation", name="System settings sections"
                )
                await expect(settings.locator('[aria-current="page"]')).to_have_text(
                    "Server"
                )
                await settings.get_by_role("link", name="Models", exact=True).click()
                await expect(
                    page.get_by_role("tab", name="Installed")
                ).to_have_attribute("aria-selected", "true")
                await expect(
                    page.get_by_text("Already downloaded", exact=True)
                ).to_be_visible()
                await page.get_by_role("tab", name="Catalog").click()
                await expect(
                    page.get_by_text("Streaming model", exact=True)
                ).to_be_visible()
                await settings.get_by_role("link", name="Features", exact=True).click()
                await expect(
                    page.get_by_text("Available for this profile", exact=True)
                ).to_be_visible()
                await expect(
                    page.get_by_text("Unavailable for the active profile", exact=True)
                ).to_have_count(2)
                await settings.get_by_role("link", name="Activity", exact=True).click()
                await expect(page.get_by_text("No recent activity.")).to_be_visible()
                await settings.get_by_role("link", name="Server", exact=True).click()
                await page.get_by_role("link", name="Reconfigure startup").click()

                steps = page.get_by_role("navigation", name="Setup steps")
                heading = page.get_by_role("heading", name="Choose your chat model")
                await expect(heading).to_be_visible()
                await expect(steps.locator('[aria-current="step"]')).to_have_text(
                    "Chat model"
                )
                await expect(steps.get_by_text("Review", exact=True)).to_have_attribute(
                    "aria-disabled", "true"
                )
                assert await page.evaluate("window.exmekhTheme.mode") == theme
                assert await page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                await page.wait_for_function(
                    """mobile => {
                        const nav = document.querySelector('.core-navigation');
                        const heading = document.querySelector('.core-heading');
                        if (!nav || !heading) return false;
                        const sidebar = nav.getBoundingClientRect();
                        const content = heading.getBoundingClientRect();
                        return mobile ? sidebar.bottom <= content.top : sidebar.right <= content.left;
                    }""",
                    arg=width == 390,
                )

                installed = page.get_by_role("radio", name=re.compile("^Test model"))
                await expect(installed).to_be_checked()
                await installed.press("ArrowDown")
                await expect(
                    page.get_by_role("radio", name=re.compile("^Streaming model"))
                ).to_be_checked()
                next_step = page.get_by_role(
                    "button", name="Configure memory and context"
                )
                await expect(next_step).to_be_enabled()
                button_box = await next_step.bounding_box()
                assert button_box and button_box["height"] >= 44
                contrast = await page.locator(".core-console").evaluate(PROBE)
                assert contrast["checked"] > 0
                assert not [
                    item
                    for item in contrast["findings"]
                    if item["status"] == "NOT PASS"
                ], contrast["findings"]
                await page.screenshot(
                    path=tmp_path / "model-selection.png", full_page=True
                )
                await next_step.click()
                await expect(steps.locator('[aria-current="step"]')).to_have_text(
                    "Memory and context"
                )
                await expect(
                    page.get_by_text("Estimated maximum · 4,096 tokens", exact=False)
                ).to_be_visible()
                context = page.get_by_role("spinbutton", name="Context tokens")
                await expect(context).to_have_value("2048")
                await context.fill("4097")
                await page.get_by_role("button", name="Continue", exact=True).click()
                await expect(
                    page.get_by_text("Enter a whole number from 1 to 4,096 tokens.")
                ).to_be_visible()
                await page.get_by_role("button", name="Use recommended").click()
                await expect(context).to_have_value("2048")
                memory_mode = page.get_by_role("combobox", name="Memory mode")
                await memory_mode.click()
                await page.get_by_role("option", name="Fully in VRAM").click()
                await expect(context).to_have_value("4096")
                await memory_mode.click()
                await page.get_by_role("option", name="SSD streaming").click()
                await expect(context).to_have_value("2048")
                await page.get_by_role("button", name="Continue", exact=True).click()
                await expect(steps.locator('[aria-current="step"]')).to_have_text(
                    "Optional features"
                )
                await steps.get_by_role("link", name="Memory and context").click()
                await expect(context).to_have_value("2048")
            finally:
                await browser.close()


async def test_failed_restore_offers_retry_and_keeps_chat_blocked(tmp_path, core_http):
    base_url, state, jobs, _ = core_http
    state.update(
        state="error",
        saved_profile=PROFILE,
        error={
            "code": "ARTIFACT_MISSING",
            "message": "Place the expected file on Core and retry.",
        },
    )
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            try:
                page = await browser.new_page()
                await page.goto(agent)
                await expect(
                    page.get_by_text("Place the expected file on Core and retry.")
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Open chat")
                ).to_have_count(0)
                await page.get_by_role("button", name="Retry saved profile").click()
                await expect(
                    page.get_by_role("progressbar", name="Opening model weights")
                ).to_be_visible()
                await expect(page.locator(".chat-input-textarea")).to_have_count(0)
                jobs["load-1"] = {
                    **jobs["load-1"],
                    "state": "completed",
                    "phase": "completed",
                }
                state.update(
                    state="ready",
                    chat_ready=True,
                    active_profile=ACTIVE_PROFILE,
                    job_id=None,
                )
                await expect(
                    page.locator(".chat-input-textarea textarea")
                ).to_be_visible()
            finally:
                await browser.close()


async def test_expired_plan_can_be_reviewed_and_applied_again(tmp_path, core_http):
    base_url, state, _, catalog = core_http
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            try:
                page = await browser.new_page()
                await page.goto(
                    f"{agent}/setup/review?model=test&mode=vram&tokens=4096"
                )
                apply = page.get_by_role("button", name="Download and start")
                await expect(apply).to_be_visible()
                catalog["models"][0]["installed"] = True
                state["revision"] += 1
                await apply.click()
                await expect(apply).to_be_hidden()
                await page.get_by_role("button", name="Review updated profile").click()
                await expect(
                    page.get_by_text("4,096 tokens", exact=True)
                ).to_be_visible()
                await page.get_by_role("button", name="Start model").click()
                await expect(
                    page.get_by_role("progressbar", name="Opening model weights")
                ).to_be_visible()
            finally:
                await browser.close()


async def test_model_row_keeps_the_controls_for_a_paused_download(tmp_path, core_http):
    base_url, _, jobs, _ = core_http
    jobs["download"] = {
        **ACTIVITY,
        "id": "download",
        "kind": "download",
        "state": "paused",
        "phase": "downloading",
        "allowed_actions": ["resume"],
    }
    jobs["import"] = {
        **ACTIVITY,
        "id": "import",
        "kind": "import",
        "state": "failed",
        "phase": "verifying",
    }
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            try:
                page = await browser.new_page()
                await page.goto(agent)
                await expect(
                    page.get_by_role("button", name="Resume", exact=True)
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Download", exact=True)
                ).to_have_count(0)
            finally:
                await browser.close()


@pytest.mark.parametrize("manual", [False, True], ids=["empty-core", "manual-server"])
async def test_startup_reconnects_when_the_server_mode_changes(
    tmp_path, core_http, manual
):
    base_url, state, _, _ = core_http
    state.update(state="ready", chat_ready=True, active_profile=ACTIVE_PROFILE)
    with BrowserTestServer(tmp_path, environment={"BASE_URL": base_url}) as agent:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            try:
                page = await browser.new_page()
                await page.goto(agent)
                chat = page.locator(".chat-input-textarea textarea")
                await expect(chat).to_be_visible()
                state.update(reachable=False)
                await expect(
                    page.get_by_text("Reconnecting to Core", exact=True)
                ).to_be_visible()
                await expect(chat).to_have_count(0)
                state.update(
                    reachable=True,
                    boot_id="boot-2",
                    state="empty",
                    chat_ready=False,
                    active_profile=None,
                    manual=manual,
                )
                if manual:
                    await expect(chat).to_be_visible()
                else:
                    await expect(
                        page.get_by_role("heading", name="Choose your chat model")
                    ).to_be_visible()
                    await expect(chat).to_have_count(0)
            finally:
                await browser.close()
