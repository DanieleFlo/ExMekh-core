import json
import re
from pathlib import Path

from playwright.async_api import async_playwright, expect

from tests.audit_theme_contrast import PROBE
from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    wait_turn_completion,
)


async def test_quick_and_project_chats_switch_while_project_waits_for_provider(
    tmp_path,
):
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_overlapping_chats_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            page = await browser.new_page()

            async def release(stage: str) -> None:
                response = await page.request.post(f"{url}/test/release/{stage}")
                assert response.ok

            try:
                await page.goto(url, wait_until="networkidle")
                theme = page.get_by_role("switch", name="Light mode")
                await theme.click()
                await expect(theme).to_have_attribute("aria-checked", "true")
                document_id = await page.evaluate("window.documentId")
                counter = page.locator(".input-bar-context-tokens")
                positive_count = re.compile(r"^(?!0/)\d")
                stream = page.locator(".chat-message-streaming:visible")
                user_messages = page.locator(".msg-user:visible p")
                quick = page.locator(".sidebar-item").filter(has_text="Quick question")
                project = page.locator(".sidebar-item").filter(
                    has_text="Project question"
                )

                await send_command(page, "Quick question")
                await expect(page).to_have_url(
                    re.compile(r"/quick-chats/quick-question--")
                )
                quick_url = page.url
                await expect(stream).to_contain_text("Quick reply started.")
                await expect(counter).to_have_text(positive_count)
                quick_count = await counter.inner_text()

                await page.get_by_text("Overlap project", exact=True).click()
                await project.click()
                await expect(page).to_have_url(re.compile(r"/project-question--"))
                project_url = page.url
                await send_command(page, "Project question")
                await expect(stream).to_be_visible()
                await expect(stream).not_to_contain_text("Quick reply")
                await page.wait_for_function(
                    "async () => (await (await fetch('/test/overlap')).json()).project_counting",
                    timeout=10000,
                )
                state = await (await page.request.get(url + "/test/overlap")).json()
                assert not state["project_started"]

                # The provider and the project counter are still waiting. Both chats
                # must remain selectable through the sidebar in this same document.
                loader = page.get_by_role("status").filter(
                    has_text="Loading conversation"
                )
                assert (await page.request.post(url + "/test/navigation/false")).ok
                await quick.click()
                await expect(page.locator(".chat-navigation-overlay")).to_have_css(
                    "opacity", "1"
                )
                assert await loader.evaluate(PROBE) == {"checked": 1, "findings": []}
                await expect(page).to_have_url(project_url)
                await expect(quick).to_be_enabled()
                assert (await page.request.post(url + "/test/navigation/true")).ok
                await expect(page).to_have_url(quick_url)
                await expect(loader).not_to_be_visible()
                await expect(stream).to_contain_text("Quick reply started.")
                await project.click()
                await expect(page).to_have_url(project_url)
                await expect(user_messages).to_have_text("Project question")
                await expect(counter).to_have_text(re.compile(r"^—/"))

                await release("project-count")
                await expect(counter).to_have_text(positive_count)
                project_count = await counter.inner_text()
                assert project_count != quick_count
                await release("quick-progress")
                await page.wait_for_function(
                    "async () => (await (await fetch('/test/overlap')).json()).quick_progressed",
                    timeout=10000,
                )
                await expect(counter).to_have_text(project_count)
                await quick.click()
                await expect(stream).to_contain_text("Quick reply continued.")
                await page.get_by_role("tab", name="Raw History").click()
                await expect(
                    page.locator(".live-assistant-card:visible")
                ).to_contain_text("Quick reply continued.")
                await page.get_by_role("tab", name="Chat", exact=True).click()
                await expect(counter).to_have_text(positive_count)
                await expect(counter).not_to_have_text(quick_count)
                await project.click()
                await expect(user_messages).to_have_text("Project question")
                await expect(counter).to_have_text(project_count)
                assert not (
                    await (await page.request.get(url + "/test/overlap")).json()
                )["project_started"]

                await release("quick-finish")
                await expect(stream.locator(".tool-event-card")).to_contain_text(
                    "mock-tool"
                )
                await quick.click()
                await expect(page).to_have_url(quick_url)
                await expect(counter).to_contain_text("7.0k/")
                await project.click()
                await expect(page).to_have_url(project_url)
                await expect(stream.locator(".tool-event-card")).to_contain_text(
                    "mock-tool"
                )
                await release("project-finish")
                await wait_turn_completion(page)
                await expect(
                    page.locator(".msg-assistant:visible").last
                ).to_contain_text("Project tool verified")
                await expect(counter).to_contain_text("2.4k/")
                assert await page.evaluate("window.documentId") == document_id
            finally:
                await page.request.post(url + "/test/navigation/true")
                for stage in (
                    "project-count",
                    "quick-progress",
                    "quick-finish",
                    "project-finish",
                ):
                    await release(stage)
                await browser.close()


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


async def test_mobile_chat_navigation_reveals_the_loader_behind_the_sidebar(tmp_path):
    with BrowserTestServer(
        tmp_path / "state", module="tests.run_overlapping_chats_agent"
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            page = await browser.new_page(viewport={"width": 390, "height": 844})
            try:
                await page.goto(url, wait_until="networkidle")
                await page.get_by_role("button", name="Open projects menu").click()
                await page.get_by_text("Overlap project", exact=True).click()
                assert (await page.request.post(url + "/test/navigation/false")).ok
                await (
                    page.locator(".sidebar-item")
                    .filter(has_text="Project question")
                    .click()
                )
                loader = page.get_by_role("status").filter(
                    has_text="Loading conversation"
                )
                await expect(page.locator(".chat-navigation-overlay")).to_have_css(
                    "opacity", "1"
                )
                await page.wait_for_function(
                    "document.querySelector('.lm-sidebar').getBoundingClientRect().right <= 0",
                    timeout=5000,
                )
                assert (await page.request.post(url + "/test/navigation/true")).ok
                await expect(page).to_have_url(re.compile(r"/project-question--"))
                await expect(loader).not_to_be_visible()
            finally:
                await page.request.post(url + "/test/navigation/true")
                await browser.close()
