from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from playwright.async_api import Page, async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def projects_server():
    yield from start_mock_server(0, "tests.run_projects_fe")


async def read_state(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        return (await client.get(f"{url}/test/state")).json()


async def select_approval(page: Page, name: str, project_default: bool = False) -> None:
    selector = (
        page.get_by_label("Default bash approval")
        if project_default
        else page.locator(".input-bar-approval-select")
    )
    await selector.click()
    await page.get_by_role("option", name=name, exact=True).click()


async def select_project_action(row, action_name: str) -> None:
    await row.get_by_role("button", name="Project actions", exact=True).click()
    await row.page.get_by_text(action_name, exact=True).click()


async def test_project_creation_settings_quickchats_and_deletion(
    projects_server: str, tmp_path: Path
) -> None:
    folder = tmp_path / "Project Alpha"
    folder.mkdir()
    screenshots = Path(__file__).resolve().parents[1] / "screenshots"
    screenshots.mkdir(exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 920})
            await page.goto(projects_server, wait_until="networkidle")
            await expect(page.get_by_text("PROJECTS", exact=True)).to_be_visible()
            await expect(page.get_by_text("QUICK CHATS", exact=True)).to_be_visible()
            await expect(page.locator(".q-btn-toggle")).to_have_count(0)
            await expect(page.locator(".input-bar-approval-select")).not_to_be_visible()
            await page.get_by_role("button", name="New project", exact=True).click()
            await page.get_by_label("Project folder", exact=True).fill(str(folder))
            await page.get_by_role(
                "button", name="Project settings", exact=True
            ).last.click()
            await page.get_by_label("System HDS Prompt Name", exact=True).fill(
                "custom-project"
            )
            await select_approval(page, "Approve All", project_default=True)
            await page.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await page.get_by_role("button", name="Create project", exact=True).click()
            project_row = page.locator(".sidebar-project").filter(
                has_text="Project Alpha"
            )
            await expect(project_row).to_be_visible()
            await expect(page.locator(".input-bar-approval-select")).to_be_visible()
            await expect(page.locator(".input-bar-approval-select")).to_contain_text(
                "Approve All"
            )
            data = await read_state(projects_server)
            project_id = data["projects"][0]["id"]
            first_id = data["current_id"]
            assert data["projects"][0]["work_root"] == str(folder)
            current = next(item for item in data["sessions"] if item["id"] == first_id)
            assert (
                current["project_id"] == project_id
                and current["agent_mode"] == "deep_agent"
            )
            await send_command(page, "First project message")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Response in deep_agent"
            )
            await select_approval(page, "Manual")
            await select_project_action(project_row, "Project settings")
            assert (
                await page.get_by_label("Default bash approval").input_value()
                == "Approve All"
            )
            await page.get_by_label("System HDS Prompt Name", exact=True).fill(
                "changed-project"
            )
            await select_approval(page, "Approve via LLM", project_default=True)
            assert await page.get_by_label("Project folder", exact=True).count() == 0
            await page.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(page.locator(".input-bar-approval-select")).to_contain_text(
                "Manual"
            )
            await project_row.get_by_role(
                "button", name="New project chat", exact=True
            ).click()
            await expect(page.locator(".input-bar-approval-select")).to_contain_text(
                "Approve via LLM"
            )
            await send_command(page, "Second project message")
            await wait_turn_completion(page)
            data = await read_state(projects_server)
            assert data["calls"][-1]["name_hds"] == "changed-project"
            assert data["calls"][-1]["tool_approval_mode"] == "Approve via LLM"
            assert data["calls"][-1]["work_root"] == str(folder)
            first = next(item for item in data["sessions"] if item["id"] == first_id)
            assert first["settings"]["tool_approval_mode"] == "Manual"
            assert first["settings"]["name_hds"] == "changed-project"
            await page.reload(wait_until="networkidle")
            await expect(project_row).to_be_visible()
            await page.screenshot(path=str(screenshots / "projects-expanded.png"))
            await page.get_by_role("button", name="New Quick Chat", exact=True).click()
            await expect(page.locator(".input-bar-approval-select")).not_to_be_visible()
            await send_command(page, "Quick message")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Response in quick_chat"
            )
            data = await read_state(projects_server)
            quick_id = data["current_id"]
            assert data["calls"][-1]["agent_mode"] == "quick_chat"
            await select_project_action(project_row, "Delete project")
            await expect(
                page.get_by_text("Delete this project?", exact=True)
            ).to_be_visible()
            await page.get_by_role("button", name="Cancel", exact=True).click()
            assert (await read_state(projects_server))["projects"]
            await select_project_action(project_row, "Delete project")
            await page.get_by_role("button", name="Yes, delete", exact=True).click()
            await expect(project_row).to_have_count(0)
            data = await read_state(projects_server)
            assert not data["projects"]
            assert all(item["project_id"] is None for item in data["sessions"])
            assert any(item["id"] == quick_id for item in data["sessions"])
            assert folder.is_dir()
            await page.reload(wait_until="networkidle")
            await expect(project_row).to_have_count(0)
            await page.screenshot(path=str(screenshots / "projects-sidebar.png"))
        finally:
            await browser.close()


async def test_new_quick_chat_shows_confirmation_after_sidebar_refresh(
    projects_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 920})
            await page.goto(projects_server, wait_until="networkidle")
            new_chat = page.get_by_role("button", name="New Quick Chat", exact=True)
            input_box = page.locator(".exmekh-chat-input-box")
            confirmation = page.get_by_text("Conversation ready", exact=True)

            await new_chat.click()
            await expect(confirmation).to_be_visible()
            first_id = await input_box.get_attribute("data-session-id")
            assert first_id is not None
            await expect(confirmation).to_have_count(0, timeout=10000)

            prompt = "Preserve this quick chat"
            await send_command(page, prompt)
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Response in quick_chat"
            )
            await new_chat.click()
            await expect(confirmation).to_be_visible()
            await expect(input_box).not_to_have_attribute("data-session-id", first_id)
            await expect(page.locator(".msg-user")).to_have_count(0)
            ready_id = await input_box.get_attribute("data-session-id")
            assert ready_id is not None
            quick_chats = page.locator(".sidebar-section-quickchats .sidebar-item")
            await expect(quick_chats.filter(has_text=prompt)).to_have_count(1)
            chat_count = await quick_chats.count()
            await expect(confirmation).to_have_count(0, timeout=10000)

            await new_chat.click()
            await expect(confirmation).to_be_visible()
            await expect(input_box).to_have_attribute("data-session-id", ready_id)
            await expect(quick_chats).to_have_count(chat_count)
            await expect(page.locator(".sidebar-item-selected")).to_have_count(1)
        finally:
            await browser.close()
