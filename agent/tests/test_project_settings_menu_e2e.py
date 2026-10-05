"""Exercise settings hierarchy, icon profiles and additive folder references."""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def settings_server():
    yield from start_mock_server(0, "tests.run_projects_agent")


@pytest.mark.parametrize("width", [1280, 390], ids=["desktop", "phone"])
async def test_project_settings_hierarchy_paths_profiles_and_reload(
    settings_server, tmp_path, width
):
    project_folder = tmp_path / f"Icon project {width}"
    project_folder.mkdir()
    first_folder = tmp_path / "First folder" / "capabilities"
    second_folder = tmp_path / "Second folder" / "capabilities"
    first_folder.mkdir(parents=True)
    (second_folder / "deep-skill").mkdir(parents=True)
    (second_folder / "deep-skill" / "extra-skill.md").write_text(
        "Additional catalog capability.", encoding="utf-8"
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(settings_server, wait_until="networkidle")
            menu = page.get_by_role("button", name="Open projects menu", exact=True)
            if not await page.get_by_role(
                "button", name="New project", exact=True
            ).is_visible():
                await menu.click()
            await page.get_by_role("button", name="New project", exact=True).click()
            creation = page.get_by_role("dialog").filter(has_text="Create project").last
            await creation.get_by_label("Project folder", exact=True).fill(
                str(project_folder)
            )
            await creation.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            settings = (
                page.get_by_role("dialog").filter(has_text="Save project settings").last
            )
            group = settings.get_by_role("group", name="Agent type", exact=True)
            await expect(
                settings.get_by_role("button", name="Customize agent", exact=True)
            ).to_have_count(0)
            await expect(
                settings.get_by_text("Coding agent: preserve history", exact=True)
            ).to_have_count(0)
            approval = settings.get_by_label("Default bash approval", exact=True)
            folders = settings.locator(".capability-paths-section")
            await expect(
                settings.get_by_label("System HDS Prompt Name", exact=True)
            ).to_have_count(0)
            await expect(group.get_by_role("button")).to_have_count(4)
            icons = []
            for profile in ["Generic", "Programmer", "Creative", "Custom"]:
                option = group.get_by_role("button", name=profile, exact=True)
                await expect(option.locator(".q-btn__content > .block")).to_have_count(
                    0
                )
                icons.append(await option.locator(".q-icon").inner_text())
            assert len(set(icons)) == 4
            await page.screenshot(
                path=f"screenshots/project-settings-priority-{width}.png",
                animations="disabled",
            )
            positions = [
                await element.bounding_box()
                for element in [
                    group,
                    approval.locator("xpath=ancestor::label[1]"),
                    folders,
                ]
            ]
            assert positions[0]["y"] < positions[1]["y"] < positions[2]["y"]
            await expect(
                settings.get_by_label("Default capabilities", exact=True)
            ).not_to_be_visible()
            assert await settings.locator(".q-card").evaluate(
                "element => element.scrollWidth <= element.clientWidth"
            )

            await group.get_by_role("button", name="Programmer", exact=True).click()
            await expect(
                group.get_by_role("button", name="Programmer", exact=True)
            ).to_have_attribute("aria-pressed", "true")
            await approval.click()
            await page.get_by_role("option", name="Approve All", exact=True).click()
            await folders.locator(".q-item").first.click()
            default = settings.get_by_label("Default capabilities", exact=True)
            state = await (
                await page.request.get(settings_server + "/test/state")
            ).json()
            default_path = str(Path(state["default_capabilities_path"]).resolve())
            await expect(default).to_have_value(default_path)
            await expect(default).to_have_attribute("readonly", "")
            additional = settings.get_by_label(
                "Additional capabilities folder", exact=True
            )
            await expect(
                settings.get_by_role(
                    "button", name="Remove capabilities folder", exact=True
                )
            ).to_have_count(0)
            add = settings.get_by_role(
                "button", name="Add capabilities folder", exact=True
            )
            for selected, count in [
                (str(first_folder), 1),
                (str(first_folder), 1),
                (default_path, 1),
                (None, 1),
                (str(second_folder), 2),
            ]:
                await page.request.post(
                    settings_server + "/test/folder-selection", data={"path": selected}
                )
                picker_state = await (
                    await page.request.get(settings_server + "/test/state")
                ).json()
                previous_picks = len(picker_state["picker_calls"])
                await add.click()
                await page.wait_for_function(
                    "async count => (await (await fetch('/test/state')).json()).picker_calls.length > count",
                    arg=previous_picks,
                )
                await expect(add).to_be_enabled()
                await expect(additional).to_have_count(count)
            await expect(default).to_have_value(default_path)
            await page.screenshot(
                path=f"screenshots/project-settings-folders-{width}.png",
                animations="disabled",
            )
            await settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(settings).not_to_be_visible()
            await creation.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            await expect(
                group.get_by_role("button", name="Programmer", exact=True)
            ).to_have_attribute("aria-pressed", "true")
            await expect(additional).to_have_count(2)
            await expect(additional.nth(0)).to_have_value(str(first_folder))
            await expect(additional.nth(1)).to_have_value(str(second_folder))
            await settings.get_by_role(
                "button", name="Remove capabilities folder", exact=True
            ).first.click()
            await expect(additional).to_have_count(1)
            await expect(additional).to_have_value(str(second_folder))
            creative = group.get_by_role("button", name="Creative", exact=True)
            await creative.focus()
            await creative.press("Space")
            await expect(creative).to_have_attribute("aria-pressed", "true")
            await settings.get_by_role("button", name="Custom", exact=True).click()
            designer = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            await expect(
                designer.locator(".agent-palette-item").filter(has_text="extra-skill")
            ).to_be_visible()
            await designer.get_by_role("button", name="Cancel", exact=True).click()
            await expect(creative).to_have_attribute("aria-pressed", "true")
            await settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(settings).not_to_be_visible()
            await creation.get_by_role(
                "button", name="Create project", exact=True
            ).click()
            await expect(creation).not_to_be_visible()
            close_menu = page.get_by_role(
                "button", name="Close projects menu", exact=True
            )
            if await close_menu.is_visible():
                await close_menu.click()
            await send_command(page, "Verify additional folders")
            await wait_turn_completion(page)
            state = await (
                await page.request.get(settings_server + "/test/state")
            ).json()
            project = next(
                item
                for item in state["projects"]
                if item["work_root"] == str(project_folder)
            )
            assert project["settings"]["agent_profile"] == "creative"
            assert project["settings"]["capabilities_path"] is None
            assert project["settings"]["additional_capabilities_paths"] == [
                str(second_folder)
            ]
            assert state["calls"][-1]["capabilities_paths"] == [
                default_path,
                str(second_folder),
            ]
            assert state["calls"][-1]["agent_profile"] == "creative"
            await page.reload(wait_until="networkidle")
            row = (
                page.locator(".sidebar-project")
                .filter(has_text=f"Icon project {width}")
                .last
            )
            if not await row.is_visible():
                await menu.click()
            await row.get_by_role("button", name="Project actions", exact=True).click()
            await page.get_by_text("Project settings", exact=True).last.click()
            await expect(creative).to_have_attribute("aria-pressed", "true")
            await folders.locator(".q-item").first.click()
            await expect(default).to_have_value(default_path)
            await expect(additional).to_have_count(1)
            await expect(additional).to_have_value(str(second_folder))
        finally:
            await browser.close()
