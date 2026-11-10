"""Progress remains visible during a tool and original pixels survive later errors."""

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)


@pytest.mark.parametrize("post_image_error", [False, True])
async def test_progress_refresh_batch_and_final_pixels(tmp_path, post_image_error):
    with BrowserTestServer(
        tmp_path / "state",
        module="tests.run_image_progress_agent",
        environment={
            "EXMEKH_AGENT_TEST_HOLD_IMAGES": "1",
            "EXMEKH_AGENT_TEST_POST_IMAGE_ERROR": "1" if post_image_error else "",
        },
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page(viewport={"width": 1360, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Generate two images")
                labels = page.locator(".image-generation-progress-label")
                await expect(labels.first).to_contain_text("42%")
                await expect(labels.last).to_contain_text("42%")
                await expect(
                    page.locator(".generated-image-slot [role='progressbar']")
                ).to_have_count(0)
                await expect(
                    page.locator(".generated-image-slot").first
                ).not_to_contain_text("Generating image")
                # Allow initial tool transcript/context publication to settle;
                # then trigger a controlled progress-only update.
                await page.wait_for_timeout(1200)
                await page.locator(".image-generation-progress").first.evaluate(
                    "node => node.dataset.progressMarker = 'same'"
                )
                await page.request.post(url + "/test/advance-image-progress")
                await expect(labels.first).to_contain_text("64%")
                await expect(labels.last).to_contain_text("64%")
                await page.wait_for_timeout(400)
                assert await page.locator("[data-progress-marker='same']").count() == 1
                await page.reload(wait_until="networkidle")
                await expect(labels.first).to_contain_text("64%")
                await expect(labels.last).to_contain_text("64%")
                await page.screenshot(
                    path=str(tmp_path / "progress.png"), full_page=True
                )
                await page.request.post(url + "/test/release-images")
                await wait_turn_completion(page)
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                await expect(page.locator(".image-generation-progress")).to_have_count(
                    0
                )
                if post_image_error:
                    await expect(page.locator(".msg-assistant").last).to_contain_text(
                        "Controlled vision rejection"
                    )
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                assert not errors
            finally:
                await browser.close()
