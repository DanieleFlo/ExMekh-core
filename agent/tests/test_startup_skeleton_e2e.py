from __future__ import annotations

import asyncio

import pytest
from playwright.async_api import Route, async_playwright, expect

from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.mark.parametrize("mobile", [False, True], ids=["desktop-dark", "mobile-light"])
async def test_startup_skeleton_is_visible_before_nicegui_mounts(tmp_path, mobile):
    sessions = SessionManager(tmp_path / "chats")
    path = (
        ChatRoutes(sessions).path_for(sessions.get_current_session()) if mobile else "/"
    )
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
                startup = page.get_by_role("status", name="Loading conversation…")
                await expect(startup).to_be_visible()
                await expect(startup).to_have_css(
                    "background-color",
                    "rgb(248, 250, 252)" if mobile else "rgb(9, 9, 11)",
                )
                await expect(startup.locator(".startup-brand")).to_have_css(
                    "animation-name", "none" if mobile else "startup-shimmer"
                )
                sidebar = startup.locator(".startup-sidebar")
                await expect(sidebar).to_have_css(
                    "display", "none" if mobile else "grid"
                )
                composer = page.locator(".chat-input-textarea textarea")
                await expect(composer).not_to_be_visible()

                release_script.set()
                await expect(page.locator("#exmekh-startup")).to_have_count(0)
                await expect(composer).to_be_visible()
                await composer.fill("Ready to write")
                await expect(composer).to_have_value("Ready to write")
                await expect(
                    page.get_by_role("switch", name="Light mode")
                ).to_have_attribute("aria-checked", str(mobile).lower())
            finally:
                release_script.set()
                await browser.close()
