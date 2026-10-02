"""Browser regression for steering attachments surviving skill compaction and reload."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)
from tests.image_samples import pdf_bytes, png_bytes


@pytest.fixture(scope="module")
def recovery_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    with BrowserTestServer(
        tmp_path_factory.mktemp("steering-recovery"),
        module="tests.run_steering_fe",
        environment={
            "FE_STEERING_FIRST_RESPONSE": "nested_skill",
            "FE_STEERING_RECOVER_OUTPUT": "1",
        },
    ) as url:
        yield url


async def test_nested_steering_output_and_pdf_images_are_read_after_browser_reload(
    recovery_server: str,
) -> None:
    server = recovery_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(),
            headless=True,
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await page.locator(".input-bar-approval-select").first.click()
            await page.get_by_role("option", name="Approve All", exact=True).click()
            await send_command(page, "Start nested steering test")
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await page.request.post(f"{server}/test/release-inference")

            async def wait_for_skill() -> None:
                while True:
                    stats = await (
                        await page.request.get(f"{server}/test/steering-stats")
                    ).json()
                    if len(stats["requests"]) == 2:
                        return
                    await asyncio.sleep(0.1)

            await asyncio.wait_for(wait_for_skill(), 5)
            await page.locator(".qs-file-picker").set_input_files(
                [
                    {"name": "red.png", "mimeType": "image/png", "buffer": png_bytes()},
                    {
                        "name": "report.pdf",
                        "mimeType": "application/pdf",
                        "buffer": pdf_bytes(),
                    },
                ]
            )
            await expect(page.locator(".qs-attachment-chip")).to_have_count(2)
            await send_command(page, "Read the attached correction")
            row = page.locator(".qs-queue-item").filter(
                has_text="Read the attached correction"
            )
            await row.get_by_role("button", name="Steering").click()
            await expect(page.locator(".msg-user-steering")).to_contain_text(
                "Read the attached correction"
            )
            await page.request.post(f"{server}/test/release-reply")
            await wait_turn_completion(page)
            stats = await (
                await page.request.get(f"{server}/test/steering-stats")
            ).json()
            queued = stats["session"]["messages"][1]["steering_messages"][0]
            assert len(queued["image_urls"]) == 3
            for _ in range(2):
                await page.reload(wait_until="networkidle")
                await send_command(page, "Inspect saved guidance")
                await wait_turn_completion(page)
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "Recovered steering with 3 images"
                )
                stats = await (
                    await page.request.get(f"{server}/test/steering-stats")
                ).json()
                request = stats["requests"][-1]
                tool = next(
                    item for item in reversed(request) if item["role"] == "tool"
                )
                assert "<type>UserMessage</type>" in tool["content"]
                assert queued["text"] in tool["content"]
                assert request[-1]["role"] == "user"
                urls = [
                    part["image_url"]["url"]
                    for part in request[-1]["content"]
                    if part["type"] == "image_url"
                ]
                assert urls == queued["image_urls"]
        finally:
            await browser.close()
