from __future__ import annotations

import re
import textwrap

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def navigation_server():
    yield from start_mock_server(0, "tests.run_projects_agent")


async def test_chat_urls_restore_refresh_bookmarks_and_browser_history(
    navigation_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            context = await browser.new_context()
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(navigation_server, wait_until="networkidle")
            await send_command(page, "First bookmark")
            await wait_turn_completion(page)
            await expect(page).to_have_url(
                re.compile(r"/chat/quick-chats/(?:chat-)?first-bookmark--")
            )
            first_url = page.url
            await page.get_by_role("button", name="New Quick Chat", exact=True).click()
            await expect(page.locator(".msg-user")).to_have_count(0)
            await send_command(page, "Second bookmark")
            await wait_turn_completion(page)
            await expect(page).to_have_url(
                re.compile(r"/chat/quick-chats/(?:chat-)?second-bookmark--")
            )
            second_url = page.url
            assert first_url != second_url
            second_page = await page.context.new_page()
            await second_page.goto(first_url, wait_until="networkidle")
            await send_command(second_page, "First tab follow-up")
            await wait_turn_completion(second_page)
            await expect(page.locator(".msg-user")).to_have_count(1)
            await expect(page.locator(".msg-user")).to_contain_text("Second bookmark")
            await send_command(page, "Second tab follow-up")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-user")).to_have_count(2)
            await expect(second_page.locator(".msg-user")).to_have_count(2)
            await expect(page.locator(".msg-user").last).to_contain_text(
                "Second tab follow-up"
            )
            await expect(second_page.locator(".msg-user").last).to_contain_text(
                "First tab follow-up"
            )
            await second_page.close()
            await page.go_back()
            await expect(page.locator(".msg-user").first).to_have_text(
                re.compile("First bookmark")
            )
            await expect(page.locator(".msg-user")).to_have_count(2)
            await page.go_forward()
            await expect(page.locator(".msg-user").first).to_have_text(
                re.compile("Second bookmark")
            )
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".msg-user").first).to_contain_text(
                "Second bookmark"
            )
            await page.goto(first_url, wait_until="networkidle")
            await expect(page.locator(".msg-user").first).to_contain_text(
                "First bookmark"
            )
            assert (
                await page.request.get(
                    navigation_server + "/chat/quick-chats/chat--missing"
                )
            ).status == 404
            assert not errors
        finally:
            await browser.close()


async def test_pending_approval_switch_applies_without_another_message():
    servers = start_mock_server(0, "tests.run_approval_agent")
    url = next(servers)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable(), headless=True
            )
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                await send_command(page, "Run the command")
                approve = page.get_by_role("button", name="Approve", exact=True)
                await expect(approve).to_be_visible()
                await page.locator(".input-bar-approval-select").click()
                await page.get_by_role("option", name="Approve All", exact=True).click()
                await expect(approve).not_to_be_visible()
                await wait_turn_completion(page)
                tool = (
                    page.locator(".tool-event-card").filter(has_text="Tool: bash").first
                )
                await tool.click()
                panel = page.locator(".tool-output-panel").first
                await expect(panel).to_contain_text("approval_probe")
                await expect(panel).not_to_contain_text("declined")
            finally:
                await browser.close()
    finally:
        servers.close()


@pytest.fixture(scope="module")
def layout_server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("corrections-3-ui")
    script = folder / "server.py"
    script.write_text(
        textwrap.dedent("""
        from nicegui import ui
        from exmekh_agent.config import config
        from exmekh_agent.exmekh_bridge.capabilities.capability_repository import CapabilityRepository
        from exmekh_agent.exmekh_bridge.models import ChatMessage, SteeringInsertion, ToolEvent
        from exmekh_agent.ui.agents.agent_designer import AgentDesigner
        from exmekh_agent.ui.chat.chat_view import ChatViewComponent
        from exmekh_agent.ui.shared.markdown_parser import parse_markdown_to_html
        from exmekh_agent.ui.styles import load_css

        @ui.page("/")
        def index():
            load_css()
            with ui.column().classes("w-full h-screen gap-0"):
                designer = AgentDesigner()
                designer.build()
                async def open_board():
                    await designer.open(CapabilityRepository().default_design(), lambda _: None)
                ui.button("Show board", on_click=open_board)
                ui.html(parse_markdown_to_html("First paragraph\\n\\nSecond paragraph\\n\\n```python\\nx = 1\\n    y = 2\\n```"))\
                    .classes("thinking-box whitespace-pre-wrap")
                view = ChatViewComponent(lambda _: None)
                view.build()
                view.start_assistant_stream()
                message = ChatMessage(role="assistant", steering_insertions=[
                    SteeringInsertion(message_id="guidance", after_tool_call_id=None,
                                      tool_name=None, skill_path="/", timestamp=15)])
                view.show_steering_insertions(message)
                def later_tool():
                    view.add_tool_event(ToolEvent(tool_name="later", tool_call_id="later", timestamp=20))
                ui.button("Later tool", on_click=later_tool)
            ui.dark_mode().enable()
        ui.run(host="127.0.0.1", port=config.port, reload=False, show=False)
    """),
        encoding="utf-8",
    )
    with BrowserTestServer(folder / "data", script=script) as url:
        yield url


async def test_live_steering_thinking_spacing_and_open_tool_indicator(layout_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(layout_server, wait_until="networkidle")
            thinking = page.locator(".thinking-box").first
            assert await thinking.evaluate("""el => {
                const style = getComputedStyle(el);
                const paragraph = getComputedStyle(el.querySelector('p'));
                return style.whiteSpace === 'normal' && parseFloat(paragraph.marginBottom) <= 3.2;
            }""")
            assert (
                await thinking.locator("pre").evaluate(
                    "el => getComputedStyle(el).whiteSpace"
                )
                == "pre-wrap"
            )
            await expect(page.locator(".steering-tool-marker")).to_have_count(1)
            await page.get_by_role("button", name="Later tool", exact=True).click()
            tool = page.locator(".tool-event-card").filter(has_text="Tool: later")
            await expect(tool).to_be_visible()
            assert (
                await page.evaluate("""() => [...document.querySelectorAll('.steering-tool-marker, .tool-event-card')]
                .map(el => el.classList.contains('steering-tool-marker') ? 'steering' : 'tool')""")
                == ["steering", "tool"]
            )
            await tool.hover()
            closed_color = await tool.evaluate(
                "el => getComputedStyle(el).borderLeftColor"
            )
            await tool.click()
            await expect(tool).to_have_attribute("aria-expanded", "true")
            open_color = await tool.evaluate(
                "el => getComputedStyle(el).borderLeftColor"
            )
            assert open_color != closed_color
            await tool.click()
            await expect(tool).to_have_attribute("aria-expanded", "false")
            assert (
                await tool.evaluate("el => getComputedStyle(el).borderLeftColor")
                == closed_color
            )
            await page.get_by_role("button", name="Show board", exact=True).click()
            cards = page.locator(".agent-board-node")
            await expect(cards.first).to_be_visible()
            assert await cards.evaluate_all("""nodes => nodes.every((node, i) => {
                const a = node.getBoundingClientRect();
                return nodes.slice(i + 1).every(other => {
                    const b = other.getBoundingClientRect();
                    return a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top;
                });
            })""")
        finally:
            await browser.close()
