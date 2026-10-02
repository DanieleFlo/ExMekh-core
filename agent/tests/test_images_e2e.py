from __future__ import annotations

from collections.abc import Generator

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    start_mock_server,
    wait_turn_completion,
)
from tests.image_samples import pdf_bytes, png_bytes


@pytest.fixture(scope="module")
def image_server() -> Generator[str]:
    yield from start_mock_server(0, "tests.run_images_agent")


async def test_images_pdf_preview_persistence_and_provider_protocol(image_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(image_server, wait_until="networkidle")
            files = [
                {"name": "red.png", "mimeType": "image/png", "buffer": png_bytes()},
                {
                    "name": "blue.png",
                    "mimeType": "image/png",
                    "buffer": png_bytes("blue"),
                },
                {
                    "name": "report.pdf",
                    "mimeType": "application/pdf",
                    "buffer": pdf_bytes(),
                },
            ]
            await page.locator(".exmekh-file-picker").set_input_files(files)
            await expect(page.locator(".exmekh-attachment-chip")).to_have_count(3)
            await expect(
                page.locator(".exmekh-attachments-bar .exmekh-image-thumbnail")
            ).to_have_count(4)
            await expect(page.locator(".exmekh-message-attachment-chip")).to_have_count(
                0
            )
            await page.locator(".exmekh-image-thumbnail").first.click()
            await expect(page.locator(".exmekh-image-preview")).to_be_visible()
            await expect(page.locator(".exmekh-preview-image img")).to_be_visible()
            assert (
                await page.locator(".exmekh-image-dialog .q-dialog__backdrop").evaluate(
                    "(el) => getComputedStyle(el).backdropFilter"
                )
                == "blur(8px)"
            )
            await page.get_by_role("button", name="Close image preview").click()
            await expect(page.locator(".exmekh-image-preview")).to_have_count(0)
            await page.locator("textarea").fill("Compare the attachments")
            await page.locator("textarea").press("Enter")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant").last).to_contain_text(
                "Received 4 images"
            )
            await expect(
                page.locator(".exmekh-message-images .exmekh-image-thumbnail")
            ).to_have_count(4)
            await expect(page.locator(".exmekh-message-attachment-chip")).to_have_count(
                0
            )
            stats = await (
                await page.request.get(image_server + "/test/image-stats")
            ).json()
            user = next(
                item
                for item in reversed(stats["requests"][-1])
                if item["role"] == "user"
            )
            assert [part["type"] for part in user["content"]] == ["text"] + [
                "image_url"
            ] * 4
            assert len(stats["session"]["messages"][0]["image_urls"]) == 4
            await page.reload(wait_until="networkidle")
            await expect(
                page.locator(".exmekh-message-images .exmekh-image-thumbnail")
            ).to_have_count(4)
            await expect(page.locator(".exmekh-message-attachment-chip")).to_have_count(
                0
            )
            await page.locator(
                ".exmekh-message-images .exmekh-image-thumbnail"
            ).last.click()
            await expect(page.locator(".exmekh-image-preview")).to_be_visible()
            await page.keyboard.press("Escape")
            await expect(page.locator(".exmekh-image-preview")).to_have_count(0)
            # An image copied to the clipboard follows the same multipart path.
            await page.locator("textarea").evaluate(
                """(el) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([new Uint8Array(%s)], 'clipboard.png', {type: 'image/png'}));
                el.dispatchEvent(new ClipboardEvent('paste', {clipboardData: transfer, bubbles: true}));
            }"""
                % list(png_bytes("green"))
            )
            await expect(
                page.locator(".exmekh-attachments-bar .exmekh-image-thumbnail")
            ).to_have_count(1)
            await page.locator("textarea").press("Enter")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant").last).to_contain_text(
                "Received 1 images"
            )
            stats = await (
                await page.request.get(image_server + "/test/image-stats")
            ).json()
            users = [item for item in stats["requests"][-1] if item["role"] == "user"]
            assert len(users[0]["content"]) == 5
            assert len(users[-1]["content"]) == 2
        finally:
            await browser.close()
