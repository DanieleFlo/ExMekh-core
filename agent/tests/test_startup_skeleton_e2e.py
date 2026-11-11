from __future__ import annotations

import asyncio

import pytest
from playwright.async_api import Route, async_playwright, expect

from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.mark.parametrize("mobile", [False, True], ids=["desktop-dark", "mobile-light"])
async def test_startup_skeleton_is_visible_before_nicegui_mounts(tmp_path, mobile):
    sessions = SessionManager(tmp_path / "chats")
    session = sessions.get_current_session()
    sessions.add_message(session, ChatMessage(role="user", content="Test question"))
    sessions.add_message(
        session,
        ChatMessage(
            role="assistant",
            content="Test response",
            raw_history=[{"role": "assistant", "content": "Test response"}],
        ),
    )
    path = ChatRoutes(sessions).path_for(session) if mobile else "/"
    with BrowserTestServer(tmp_path, module="tests.run_projects_agent") as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=get_edge_executable()
            )
            page = await browser.new_page(
                viewport={"width": 390, "height": 844}
                if mobile
                else {"width": 1440, "height": 960},
                reduced_motion="reduce" if mobile else "no-preference",
            )
            if mobile:
                await page.add_init_script(
                    "localStorage.setItem('exmekh-theme', 'light')"
                )

            release_script = asyncio.Event()

            async def delay_nicegui(route: Route) -> None:
                await release_script.wait()
                await route.continue_()

            await page.route("**/_nicegui/**/static/nicegui.js", delay_nicegui)
            try:
                await page.goto(url + path, wait_until="commit")
                startup = page.locator("#exmekh-startup")
                await expect(startup).to_be_visible()
                await expect(startup).to_have_css(
                    "background-color",
                    "rgb(248, 250, 252)" if mobile else "rgb(9, 9, 11)",
                )
                await expect(startup.locator(".startup-brand")).to_have_css(
                    "animation-name", "none" if mobile else "startup-shimmer"
                )
                status = startup.get_by_role("status", name="Loading conversation…")
                await expect(status).to_have_text("Loading conversation…")
                await expect(status.locator(".chat-loading-spinner")).to_have_css(
                    "animation-name", "none" if mobile else "chat-loading-spin"
                )
                startup_chat = await startup.locator(
                    ".startup-conversation"
                ).bounding_box()
                startup_status = await status.bounding_box()
                assert startup_chat is not None and startup_status is not None
                sidebar = startup.locator(".startup-sidebar")
                await expect(sidebar).to_have_css(
                    "display", "none" if mobile else "grid"
                )
                composer = page.locator(".chat-input-textarea textarea")
                await expect(composer).not_to_be_visible()

                release_script.set()
                await expect(startup).to_have_count(0)
                await expect(composer).to_be_visible()
                chat = await page.locator(".chat-page-layout").bounding_box()
                assert chat is not None
                for dimension in ("x", "y", "width"):
                    assert startup_chat[dimension] == pytest.approx(
                        chat[dimension], abs=1
                    )
                assert startup_status["y"] + startup_status[
                    "height"
                ] / 2 == pytest.approx(chat["y"] + chat["height"] / 2, abs=1)
                await expect(page.locator(".msg-assistant")).to_contain_text(
                    "Test response"
                )
                await expect(
                    page.locator(".raw-json-block, .graph-node")
                ).to_have_count(0)
                await page.get_by_role("tab", name="Raw History").click()
                await expect(
                    page.locator(".raw-json-block:visible").last
                ).to_contain_text("Test response")
                await page.get_by_role("tab", name="Graph").click()
                await expect(page.locator(".graph-node").first).to_be_visible()
                await page.get_by_role("tab", name="Chat", exact=True).click()
                await composer.fill("Ready to write")
                await expect(composer).to_have_value("Ready to write")
                await expect(
                    page.get_by_role("switch", name="Light mode")
                ).to_have_attribute("aria-checked", str(mobile).lower())
            finally:
                release_script.set()
                await browser.close()
