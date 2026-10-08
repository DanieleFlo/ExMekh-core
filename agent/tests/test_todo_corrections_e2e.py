from __future__ import annotations

import textwrap

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.fixture(scope="module")
def corrections_server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("todo-corrections")
    capabilities = folder / "capabilities"
    (capabilities / "deep-skill").mkdir(parents=True)
    skill = capabilities / "deep-skill" / "changing.md"
    skill.write_text(
        "---\ndescription: Before\nsub-tools: []\n---\nOriginal instructions",
        encoding="utf-8",
    )
    script = folder / "server.py"
    script.write_text(
        textwrap.dedent("""
        from pathlib import Path

        from nicegui import ui

        from exmekh_agent.config import config
        from exmekh_agent.exmekh_bridge.models import ChatMessage
        from exmekh_agent.exmekh_bridge.capabilities.capability_repository import CapabilityRepository
        from exmekh_agent.state.session_manager import session_manager
        from exmekh_agent.ui.agents.agent_designer import AgentDesigner
        from exmekh_agent.ui.chat.chat_view import ChatViewComponent
        from exmekh_agent.ui.styles import load_css

        @ui.page("/")
        def index():
            load_css()
            session = session_manager.create_session()
            session.messages = [ChatMessage(role=role, content="x")
                                for _ in range(35) for role in ("user", "assistant")]
            with ui.column().classes("w-full h-screen gap-0"):
                designer = AgentDesigner()
                designer.build()
                async def customize():
                    repository = CapabilityRepository()
                    paths = [CAPABILITIES]
                    design = repository.default_design("creative", paths)
                    catalog = repository.catalog(paths)
                    skill = next(item for item in catalog if item.name == "changing")
                    design = repository.add_to_design(design, skill, design.root_id, catalog)
                    await designer.open(design, lambda _: None, paths)
                ui.button("Customize", on_click=customize)
                view = ChatViewComponent(lambda _: None, on_edit_message=lambda *_: None)
                view.build()
            ui.dark_mode().enable()

        ui.run(host="127.0.0.1", port=config.port, reload=False, show=False)
        """).replace("CAPABILITIES", repr(str(capabilities))),
        encoding="utf-8",
    )
    with BrowserTestServer(folder / "data", script=script) as url:
        yield url, skill


async def test_short_messages_layout_and_dense_point_navigation(corrections_server):
    url, _ = corrections_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1800, "height": 900})
            await page.goto(url, wait_until="networkidle")
            chat = page.locator(".chat-scroll-container").first
            dots = page.locator(".chat-poi-dot")
            await expect(dots).to_have_count(36)
            for width in [1800, 768, 390, 320]:
                await page.set_viewport_size({"width": width, "height": 900})
                for index in [0, 1, 17, 34, 35, 0]:
                    dot = dots.nth(index)
                    await dot.click()
                    await expect(dot).to_have_class(
                        "chat-poi-dot "
                        + ("poi-assistant-final" if index == 35 else "poi-user")
                        + " active"
                    )
                    assert await chat.evaluate(
                        """(el, index) => {
                        const row = index === 35 ? el.querySelectorAll('.chat-message-assistant')[34]
                                                : el.querySelectorAll('.chat-message-user')[index];
                        const bounds = row.getBoundingClientRect();
                        const viewport = el.getBoundingClientRect();
                        return bounds.top >= viewport.top - 1 && bounds.bottom <= viewport.bottom + 1;
                    }""",
                        index,
                    )
                first = page.locator(".msg-user").first
                await first.hover()
                assert await first.evaluate("""el => {
                    const bubble = el.getBoundingClientRect();
                    return [...el.querySelectorAll('button')].every(button => {
                        const bounds = button.getBoundingClientRect();
                        return bounds.left >= bubble.left && bounds.right <= bubble.right
                            && bounds.top >= bubble.top && bounds.bottom <= bubble.bottom;
                    });
                }""")
                assert await chat.evaluate("""el => {
                    const bounds = el.getBoundingClientRect();
                    const track = document.getElementById('exmekh-chat-poi-track').getBoundingClientRect();
                    return Math.abs(bounds.right - el.parentElement.getBoundingClientRect().right) < 2
                        && [...el.querySelectorAll('.chat-message-row')].every(row => {
                            const message = row.getBoundingClientRect();
                            return message.width <= 896 && message.right < track.left;
                        });
                }""")
        finally:
            await browser.close()


async def test_open_designer_refreshes_changed_capabilities(corrections_server):
    url, source = corrections_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            await page.goto(url, wait_until="networkidle")
            await page.get_by_role("button", name="Customize", exact=True).click()
            canvas = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            node = canvas.locator(".agent-board-node").filter(has_text="changing")
            await expect(node).to_be_visible()
            node_id = await node.get_attribute("data-node")
            source.write_text(
                "---\ndescription: After\nsub-tools: [calculate]\n---\nUpdated instructions",
                encoding="utf-8",
            )
            await page.wait_for_function(
                """nodeId => {
                const board = document.querySelector('.agent-designer-dialog .agent-board');
                return board && [...board.querySelectorAll('[data-parent]')].some(el => el.dataset.parent === nodeId);
            }""",
                arg=node_id,
                timeout=10000,
            )
            await expect(node).to_have_attribute("data-node", node_id)
            await node.click()
            await canvas.get_by_role("button", name="Edit a copy", exact=True).click()
            editor = (
                page.get_by_role("dialog").filter(has_text="Edit capability copy").last
            )
            await expect(editor.get_by_label("Instructions", exact=True)).to_have_value(
                "Updated instructions"
            )
            await expect(editor.get_by_label("Description", exact=True)).to_have_value(
                "After"
            )
        finally:
            await browser.close()
