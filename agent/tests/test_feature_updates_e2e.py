from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)


@pytest.mark.serial(reason="The image clipboard is shared between browsers")
@pytest.mark.parametrize("include", [False, True])
async def test_generated_batch_loading_order_gallery_and_model_delivery(
    tmp_path, include
):
    with BrowserTestServer(
        tmp_path / "state",
        module="tests.run_native_media_preview_agent",
        environment={
            "INCLUDE_GENERATED_IMAGES_IN_CONTEXT": "true",
            "EXMEKH_AGENT_TEST_HOLD_IMAGES": "1",
        },
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                context = await browser.new_context(
                    permissions=["clipboard-read", "clipboard-write"],
                    viewport={"width": 1360, "height": 900},
                )
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(url, wait_until="networkidle")
                if not include:
                    await page.locator(
                        '.lm-header button:has(i:text-is("tune"))'
                    ).click()
                    toggle = page.get_by_role(
                        "switch", name="Send generated images to the agent"
                    )
                    await expect(toggle).to_be_checked()
                    await toggle.click()
                    await expect(toggle).not_to_be_checked()
                    await page.get_by_role(
                        "button", name="Save global settings", exact=True
                    ).click()
                    await expect(
                        page.get_by_text("Global settings saved", exact=True)
                    ).to_be_visible()
                    saved = await (
                        await page.request.get(url + "/test/media-state")
                    ).json()
                    assert saved["include_generated_images"] is False
                    assert saved["saved_image_preference"]
                await send_command(
                    page,
                    "Generate two images"
                    if include
                    else "Generate two images in background",
                )
                await expect(page.locator(".generated-image-loading")).to_have_count(2)
                await page.wait_for_function("""() => {
                    const boxes = [...document.querySelectorAll('.generated-image-slot')]
                        .filter(node => node.querySelector('.generated-image-loading'))
                        .map(node => node.getBoundingClientRect())
                        .filter(box => box.width > 0 && box.height > 0);
                    return boxes.length === 2 && boxes.every(box => Math.abs(box.width / box.height - 2) < 0.02);
                }""")
                await page.request.post(url + "/test/release-images")
                await wait_turn_completion(page)
                previews = page.locator(".msg-assistant .exmekh-image-thumbnail")
                await expect(previews).to_have_count(2)
                await expect(page.locator(".generated-image-loading")).to_have_count(0)
                order = await page.locator(".msg-assistant").evaluate(
                    "el => [...el.querySelectorAll('.tool-event-card,.generated-images')].map(node => node.classList.contains('generated-images') ? 'images' : node.textContent)"
                )
                assert "generate-image" in order[0]
                assert order[1] == "images"
                assert "inspect-file" in order[2]
                await previews.first.click()
                await expect(
                    page.get_by_role("button", name="Next image", exact=True)
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Previous image", exact=True)
                ).not_to_be_visible()
                first_source = await page.locator(
                    ".exmekh-preview-image img"
                ).first.get_attribute("src")
                await page.get_by_role("button", name="Next image", exact=True).click()
                await expect(
                    page.get_by_role("button", name="Previous image", exact=True)
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Next image", exact=True)
                ).not_to_be_visible()
                assert (
                    await page.locator(".exmekh-preview-image img").first.get_attribute(
                        "src"
                    )
                    != first_source
                )
                async with page.expect_download() as download:
                    await page.get_by_role(
                        "button", name="Download image", exact=True
                    ).click()
                assert (await download.value).suggested_filename == "image-2.png"
                await page.get_by_role("button", name="Copy image", exact=True).click()
                await page.wait_for_function(
                    "async () => (await navigator.clipboard.read()).some(item => item.types.includes('image/png'))"
                )
                evidence = Path(".test-tmp") / f"update-gallery-{include}.png"
                await page.screenshot(path=str(evidence))
                await page.get_by_role(
                    "button", name="Previous image", exact=True
                ).click()
                await expect(
                    page.locator(".exmekh-preview-image img").first
                ).to_have_attribute("src", first_source)
                await page.get_by_role(
                    "button", name="Close image preview", exact=True
                ).click()
                if not include:
                    await expect(page.locator(".generated-image-attach")).to_have_count(
                        2
                    )
                    await page.request.post(url + "/test/legacy-image-policy")
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                state = await (await page.request.get(url + "/test/media-state")).json()
                assert state["generated_calls"] == [include]
                assert state["provider_calls"][-1][0] is include
                assert (
                    len(
                        state["session"]["messages"][-1]["tool_events"][0]["image_urls"]
                    )
                    == 2
                )
                await expect(page.locator(".generated-image-attach")).to_have_count(
                    0 if include else 2
                )
                if not include:
                    await page.locator(".generated-image-attach").first.click()
                    await expect(page.locator(".exmekh-attachment-chip")).to_have_count(
                        1
                    )
                    await send_command(page, "Describe attached image")
                    await wait_turn_completion(page)
                    await expect(page.locator(".msg-assistant").last).to_contain_text(
                        "Attachment received"
                    )
                    state = await (
                        await page.request.get(url + "/test/media-state")
                    ).json()
                    assert state["generated_calls"] == [False]
                    assert state["user_image_counts"][-1] == 1
                    assert len(state["session"]["messages"][-2]["image_urls"]) == 1
                assert not errors
            finally:
                await browser.close()


@pytest.mark.parametrize("approved", [False, True])
async def test_approval_shield_changes_during_the_real_tool_lifecycle(
    tmp_path, approved
):
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_approval_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Run the command")
                await expect(page.locator(".tool-approval-pending")).to_have_count(1)
                await page.get_by_role(
                    "button", name="Approve" if approved else "Decline", exact=True
                ).click()
                await wait_turn_completion(page)
                state = "approved" if approved else "denied"
                await expect(page.locator(f".tool-approval-{state}")).to_have_count(1)
                await page.reload(wait_until="networkidle")
                await expect(page.locator(f".tool-approval-{state}")).to_have_count(1)
            finally:
                await browser.close()


async def test_failed_image_finishes_and_retains_the_error(tmp_path):
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_native_media_preview_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Generate two images with failure")
                await expect(page.locator(".generated-image-loading")).to_have_count(2)
                await wait_turn_completion(page)
                await expect(
                    page.locator(".generated-image-unavailable")
                ).to_have_count(2)
                await expect(page.locator(".generated-image-loading")).to_have_count(0)
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".generated-image-unavailable")
                ).to_have_count(2)
            finally:
                await browser.close()


@pytest.mark.parametrize("include", [False, True])
async def test_multiple_image_tool_calls_finish_and_persist(tmp_path, include):
    with BrowserTestServer(
        tmp_path / "state",
        module="tests.run_native_media_preview_agent",
        environment={"INCLUDE_GENERATED_IMAGES_IN_CONTEXT": str(include).lower()},
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Generate multiple images")
                await wait_turn_completion(page)
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
                state = await (await page.request.get(url + "/test/media-state")).json()
                assert state["generated_calls"] == [include, include]
                tools = state["session"]["messages"][-1]["tool_events"]
                images = [
                    tool for tool in tools if tool["tool_name"] == "generate-image"
                ]
                assert len(images) == 2
                assert all(
                    tool["status"] == "completed" and tool["image_state"] == "completed"
                    for tool in images
                )
                assert state["provider_calls"][-1][:2] == [include, include]
                await page.reload(wait_until="networkidle")
                await expect(
                    page.locator(".msg-assistant .exmekh-image-thumbnail")
                ).to_have_count(2)
            finally:
                await browser.close()
