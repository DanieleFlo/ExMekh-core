import json
import re
from pathlib import Path

from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)


async def test_two_chat_requests_stream_and_steer_without_reloading(tmp_path):
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_concurrent_requests_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await page.locator(".input-bar-approval-select").click()
                await page.get_by_role("option", name="Approve All", exact=True).click()
                await send_command(page, "First request")
                await expect(page.locator(".chat-message-streaming")).to_be_visible()
                await expect(page).to_have_url(re.compile(r"/first-request--"))
                first_url = page.url
                await page.get_by_role(
                    "button", name="New project chat", exact=True
                ).first.click()
                await expect(page).not_to_have_url(first_url)
                await expect(page.locator(".msg-user:visible")).to_have_count(0)
                await send_command(page, "Second request")
                await expect(page).to_have_url(re.compile(r"/second-request--"))
                second_url = page.url
                await expect(page.locator(".chat-message-streaming")).to_be_visible()
                # Browser history changes the selected chat through the application's router.
                await page.go_back()
                await expect(page).to_have_url(first_url)
                await expect(page.locator(".chat-message-streaming")).to_be_visible()
                await send_command(page, "First correction")
                row = page.locator(".exmekh-queue-item").filter(
                    has_text="First correction"
                )
                await expect(row.get_by_role("button", name="Steering")).to_be_enabled()
                await row.get_by_role("button", name="Steering").click()
                await expect(page.locator(".msg-user-steering")).to_contain_text(
                    "First correction"
                )
                await page.request.post(url + "/test/release/0/inference")
                await page.request.post(url + "/test/release/0/reply")
                await wait_turn_completion(page)
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "First correction"
                )
                await page.go_forward()
                await expect(page).to_have_url(second_url)
                await expect(page.locator(".chat-message-streaming")).to_be_visible()
                await page.request.post(url + "/test/release/1/inference")
                await wait_turn_completion(page)
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "Initial answer"
                )
                state = await (
                    await page.request.get(url + "/test/concurrent-state")
                ).json()
                assert state["started"] == [True, True]
                assert not state["session"]["messages"][-1]["steering_messages"]
                assert not any(
                    "First correction" in str(item.get("content", ""))
                    for item in state["requests"][1][0]
                )
            finally:
                await page.screenshot(path=".test-tmp/concurrent-requests.png")
                current = await (
                    await page.request.get(url + "/test/concurrent-state")
                ).json()
                Path(".test-tmp/concurrent-requests.json").write_text(
                    json.dumps(current, indent=2)
                )
                await browser.close()
