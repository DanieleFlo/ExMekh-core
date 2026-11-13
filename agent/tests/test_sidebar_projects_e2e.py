from playwright.async_api import async_playwright, expect

from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from tests.browser_support import BrowserTestServer


async def test_projects_show_more_chats_independently_and_have_jump_dots(tmp_path):
    data = tmp_path / "data"
    sessions = SessionManager(data / "chats")
    project_chats = {}
    for name in ("Core", "Agent"):
        folder = tmp_path / name
        folder.mkdir()
        project = sessions.create_project(folder)
        for index in range(1, 13):
            chat = sessions.create_session(project_id=project.id)
            chat.title = f"{name} conversation {index}"
            sessions.save_session(chat)
            if index == 1:
                project_chats[name] = ChatRoutes(sessions).path_for(chat)

    with BrowserTestServer(
        data, environment={"EXMEKH_AGENT_TEST_BYPASS_CORE_HTTP": "1"}
    ) as url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(
                    viewport={"width": 1250, "height": 600}, reduced_motion="reduce"
                )
                await page.goto(url + project_chats["Agent"])
                rail = page.get_by_role("navigation", name="Project navigation")
                agent_dot = rail.get_by_role("button", name="Jump to Agent")
                core_dot = rail.get_by_role("button", name="Jump to Core")
                await expect(rail.get_by_role("button")).to_have_count(2)
                await expect(agent_dot).to_have_attribute("aria-current", "true")
                agent = page.locator(".sidebar-project").filter(
                    has_text="Agent conversation"
                )
                core = page.locator(".sidebar-project").filter(
                    has_text="Core conversation"
                )
                await expect(agent.locator(".sidebar-item")).to_have_count(5)
                await expect(
                    agent.get_by_text("Agent conversation 1", exact=True)
                ).to_be_visible()
                await agent.get_by_role("button", name="Show more chats (7)").click()
                await expect(agent.locator(".sidebar-item")).to_have_count(10)

                await core_dot.press("Enter")
                await expect(core_dot).to_have_attribute("aria-current", "true")
                await expect(agent_dot).to_have_attribute("aria-current", "false")
                assert (
                    await core_dot.evaluate(
                        "el => getComputedStyle(el, '::before').width"
                    )
                    == "7px"
                )
                assert (
                    await agent_dot.evaluate(
                        "el => getComputedStyle(el, '::before').width"
                    )
                    == "4px"
                )
                assert await core_dot.evaluate(
                    "el => getComputedStyle(el, '::before').backgroundColor"
                ) == await agent_dot.evaluate(
                    "el => getComputedStyle(el, '::before').backgroundColor"
                )
                await expect(
                    core.get_by_role("button", name="Show more chats (7)")
                ).to_be_visible()
                await expect(core.locator(".sidebar-item")).to_have_count(5)
                project_list = page.locator(
                    ".sidebar-projects-content .sidebar-list-scroll"
                )
                await page.wait_for_function(
                    """() => {
                    const list = document.querySelector('.sidebar-projects-content .sidebar-list-scroll');
                    const core = document.getElementById(document.querySelector('.sidebar-project-dot[aria-label="Jump to Core"]').getAttribute('aria-controls'));
                    const header = core.querySelector('.q-item').getBoundingClientRect();
                    const area = list.getBoundingClientRect();
                    return list.scrollTop > 0 && header.top >= area.top - 1 && header.bottom <= area.bottom;
                }""",
                    timeout=5000,
                )
                await project_list.hover()
                await page.mouse.wheel(0, -1000)
                await page.wait_for_function(
                    """() =>
                    document.querySelector('.sidebar-projects-content .sidebar-list-scroll').scrollTop === 0
                """,
                    timeout=5000,
                )
                await expect(agent_dot).to_have_attribute("aria-current", "true")
                await expect(core_dot).to_have_attribute("aria-current", "false")
                await page.mouse.wheel(0, 1000)
                await expect(core_dot).to_have_attribute("aria-current", "true")
                await expect(agent_dot).to_have_attribute("aria-current", "false")
                await expect(page).to_have_url(url + project_chats["Agent"])
                await agent_dot.click()
                await expect(agent_dot).to_have_attribute("aria-current", "true")
                await expect(core_dot).to_have_attribute("aria-current", "false")
                await agent.get_by_role("button", name="Show fewer").click()
                await expect(agent.locator(".sidebar-item")).to_have_count(5)
                await expect(
                    agent.get_by_text("Agent conversation 1", exact=True)
                ).to_be_visible()
                await expect(core.locator(".sidebar-item")).to_have_count(5)

                await page.set_viewport_size({"width": 390, "height": 660})
                menu = page.get_by_role("button", name="Open projects menu")
                if not await rail.is_visible():
                    await menu.click()
                await expect(rail).to_be_visible()
                await core_dot.click()
                await page.wait_for_function(
                    """() => document.querySelector('.sidebar-projects-content .sidebar-list-scroll').scrollTop > 0""",
                    timeout=5000,
                )
            finally:
                await browser.close()
