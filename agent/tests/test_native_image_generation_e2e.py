"""Browser -> real LLM -> registered tool -> native image service -> vision/UI."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.serial,
    pytest.mark.skipif(
        not os.getenv("DS4_SUITE_HTTP_URL"),
        reason="set DS4_SUITE_HTTP_URL to a native server with vision",
    ),
]


async def test_real_image_generation_delivers_pixels_to_model_and_persists_in_ui(
    tmp_path,
):
    artifacts = Path(os.getenv("DS4_SUITE_E2E_ARTIFACTS", str(tmp_path / "evidence")))
    artifacts.mkdir(parents=True, exist_ok=True)
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_native_image_generation_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            page = await browser.new_page(viewport={"width": 1360, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                await page.goto(url, wait_until="networkidle")
                await send_command(
                    page,
                    "Remember code IMAGE-E2E-17. Generate exactly two images by calling generate-image exactly once "
                    "with prompt 'A red square on white', n 2, width 256, height 256, seed 171, "
                    "background opaque, include_image_in_result true, preview_scale 0.5. "
                    "Set execution_mode wait and wait_seconds 300 for the generation. "
                    "After success finish with IMAGE-E2E-17.",
                )
                deadline = time.monotonic() + 120
                while True:
                    state = await (
                        await page.request.get(
                            f"{url}/test/native-image-stats", timeout=2000
                        )
                    ).json()
                    if state["image_requests"]:
                        break
                    assert time.monotonic() < deadline, state
                    await asyncio.sleep(0.2)
                assert len(state["image_responses"]) < len(state["image_requests"]), (
                    state
                )
                heartbeats = []
                for _ in range(3):
                    start = time.monotonic()
                    response = await page.request.get(
                        f"{url}/test/native-image-stats", timeout=2000
                    )
                    assert response.ok
                    heartbeats.append(time.monotonic() - start)
                await page.locator("textarea").fill("Draft while generating")
                await expect(page.locator("textarea")).to_have_value(
                    "Draft while generating"
                )
                await page.locator("textarea").fill("")
                await wait_turn_completion(page, timeout=600000)
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "IMAGE-E2E-17"
                )
                previews = page.locator(".msg-assistant .exmekh-image-thumbnail")
                await expect(previews).to_have_count(2)
                await page.screenshot(
                    path=str(artifacts / "generated-images.png"), full_page=True
                )
                await previews.first.click()
                await expect(page.locator(".exmekh-image-preview")).to_be_visible()
                await page.screenshot(
                    path=str(artifacts / "image-preview.png"), full_page=True
                )
                await page.get_by_role("button", name="Close image preview").click()
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                state = await (
                    await page.request.get(f"{url}/test/native-image-stats")
                ).json()
                assert len(state["image_requests"]) == 1, state
                assert state["image_requests"][0]["body"]["size"] == "256x256"
                assert state["image_requests"][0]["body"]["n"] == 2
                assert state["image_requests"][0]["idempotency_key"]
                assert state["image_responses"][0]["status"] == 200
                outputs = state["image_responses"][0]["images"]
                assert [row["seed"] for row in outputs] == [171, 172]
                assert {row["png_sha256"] for row in outputs} == {
                    row["sha256"] for row in state["files"]
                }
                assert all(row["size"] == [256, 256] for row in state["files"])
                assert any(
                    call["input_image_sizes"] == [[128, 128], [128, 128]]
                    and call["tool_attachment_counts"] == [2]
                    for call in state["provider_calls"]
                ), state
                await send_command(
                    page,
                    "Without generating another image, repeat the code you remembered earlier.",
                )
                await wait_turn_completion(page, timeout=600000)
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "IMAGE-E2E-17"
                )
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                state = await (
                    await page.request.get(f"{url}/test/native-image-stats")
                ).json()
                assert len(state["image_requests"]) == 1, state
                assert state["provider_calls"][-1]["input_image_sizes"] == [
                    [128, 128],
                    [128, 128],
                ], state
                assert not errors, errors
                state["ui_heartbeat_seconds"] = heartbeats
                (artifacts / "browser.json").write_text(json.dumps(state, indent=2))
            finally:
                await page.screenshot(
                    path=str(artifacts / "last-page.png"), full_page=True
                )
                current = await (
                    await page.request.get(f"{url}/test/native-image-stats")
                ).json()
                (artifacts / "last-state.json").write_text(
                    json.dumps(current, indent=2)
                )
                await browser.close()
