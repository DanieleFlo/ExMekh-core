from __future__ import annotations

import re
from collections.abc import Generator

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def model_server() -> Generator[str]:
    yield from start_mock_server(0.0, "tests.run_model_discovery_agent")


async def test_loaded_qwen_model_context_and_requests_follow_server_restarts(
    model_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(model_server, wait_until="networkidle")
            label = page.locator(".input-bar-model-name")
            tokens = page.locator(".input-bar-context-tokens")
            await expect(label).to_have_text("qwen3.8-flash-next", timeout=15000)
            await expect(tokens).to_have_text(re.compile(r"/16\.4k$"))
            # Keep the same page and chat open. The regular health timer must
            # refresh both model identity and actual allocation, without reload.
            cases = [
                ("qwen3.8-flash-next", 8192, "8.2k"),
                ("Qwen3.8-27B-UD-Q4_K_S", 28000, "28.0k"),
                ("qwen3.8-flash-next", 16384, "16.4k"),
            ]
            for model, length, compact in cases:
                response = await page.request.post(
                    model_server + "/test/loaded-model",
                    data={"model": model, "context_length": length},
                )
                assert response.ok
                await expect(label).to_have_text(model, timeout=20000)
                await expect(tokens).to_have_text(
                    re.compile(r"/" + re.escape(compact) + r"$"), timeout=20000
                )
                await page.locator(".input-bar-context-info").hover()
                await expect(page.locator(".q-tooltip").last).to_contain_text(
                    f"{length:,}"
                )
                await page.mouse.move(0, 0)
                if length != 8192:
                    await send_command(page, "Check the active model")
                    await wait_turn_completion(page)
                    calls_response = await page.request.get(
                        model_server + "/test/model-calls"
                    )
                    calls = await calls_response.json()
                    assert calls[-1] == {"model": model, "context_length": length}
        finally:
            await browser.close()
