from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    start_mock_server,
    wait_turn_completion,
)


async def test_assistant_tool_image_preview_is_visible_and_persistent():
    server = start_mock_server(0, "tests.run_native_media_preview_agent")
    url = next(server)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await page.locator("textarea").fill("Generate a red image")
                await page.locator("textarea").press("Enter")
                await wait_turn_completion(page)
                preview = page.locator(".msg-assistant .exmekh-image-thumbnail")
                await expect(preview).to_have_count(1)
                await preview.click()
                await expect(page.locator(".exmekh-image-preview")).to_be_visible()
                await page.get_by_role("button", name="Close image preview").click()
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(1)
            finally:
                await browser.close()
    finally:
        server.close()
