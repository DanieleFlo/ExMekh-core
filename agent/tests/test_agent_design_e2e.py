from __future__ import annotations

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
def design_server():
    yield from start_mock_server(0, "tests.run_message_edit_agent")


async def choose_agent(dialog, page, name: str):
    await dialog.get_by_label("Agent type", exact=True).click()
    await page.get_by_role("option", name=name, exact=True).click()


async def test_project_profile_chat_override_and_custom_canvas_mock_request(
    design_server, tmp_path: Path
):
    project_folder = tmp_path / "Design project"
    project_folder.mkdir()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            await page.goto(design_server, wait_until="networkidle")
            await page.add_style_tag(
                content="*, *::before, *::after { transition-duration: 0s !important; animation-duration: 0s !important; }"
            )
            await page.get_by_role("button", name="New project", exact=True).click()
            dialog = page.get_by_role("dialog").filter(has_text="New project").last
            await dialog.get_by_label("Project folder", exact=True).fill(
                str(project_folder)
            )
            await expect(dialog.get_by_label("Agent type", exact=True)).to_have_count(0)
            await expect(
                dialog.get_by_role("button", name="Customize agent", exact=True)
            ).to_have_count(0)
            await dialog.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            project_settings = (
                page.get_by_role("dialog").filter(has_text="Save project settings").last
            )
            await expect(
                project_settings.get_by_label("Agent type", exact=True)
            ).to_have_value("Generic")
            await expect(
                project_settings.get_by_role(
                    "button", name="Customize agent", exact=True
                )
            ).to_be_visible()
            await choose_agent(project_settings, page, "Programmer")
            await project_settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(project_settings).not_to_be_visible()
            # Reopening settings retains the draft without a second selector overwriting it.
            await dialog.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            await expect(
                project_settings.get_by_label("Agent type", exact=True)
            ).to_have_value("Programmer")
            await project_settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(project_settings).not_to_be_visible()
            await expect(dialog.get_by_label("Agent type", exact=True)).to_have_count(0)
            await dialog.get_by_role(
                "button", name="Create project", exact=True
            ).click()
            await expect(dialog).not_to_be_visible()
            await send_command(page, "Verify programmer")
            await wait_turn_completion(page)
            data = await (await page.request.get(design_server + "/test/state")).json()
            assert data["session"]["settings"]["agent_profile"] == "programmer"
            assert (
                "programming assistant" in data["calls"][-1]["messages"][0]["content"]
            )
            await page.get_by_role(
                "button", name="Chat agent settings", exact=True
            ).click()
            chat_dialog = (
                page.get_by_role("dialog").filter(has_text="Save chat agent").last
            )
            await choose_agent(chat_dialog, page, "Creative")
            await chat_dialog.get_by_role(
                "button", name="Save chat agent", exact=True
            ).click()
            await expect(chat_dialog).not_to_be_visible()
            await page.reload(wait_until="networkidle")
            data = await (await page.request.get(design_server + "/test/state")).json()
            assert data["session"]["settings"]["agent_profile"] == "creative"
            await (
                page.locator(".sidebar-project")
                .filter(has_text="Design project")
                .get_by_role("button", name="New project chat", exact=True)
                .click()
            )
            await page.get_by_role(
                "button", name="Chat agent settings", exact=True
            ).click()
            chat_dialog = (
                page.get_by_role("dialog").filter(has_text="Save chat agent").last
            )
            await expect(
                chat_dialog.get_by_label("Agent type", exact=True)
            ).to_have_value("Programmer")
            await choose_agent(chat_dialog, page, "Custom")
            cancelled_canvas = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            await expect(cancelled_canvas).to_be_visible()
            await cancelled_canvas.get_by_role(
                "button", name="Cancel", exact=True
            ).click()
            await expect(cancelled_canvas).not_to_be_visible()
            await expect(
                chat_dialog.get_by_label("Agent type", exact=True)
            ).to_have_value("Programmer")
            await chat_dialog.get_by_role(
                "button", name="Customize agent", exact=True
            ).click()
            canvas = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            await expect(canvas).to_be_visible()
            root = canvas.locator(".agent-board-node").filter(has_text="agent-root")
            await expect(root).to_be_visible()
            await page.screenshot(animations="disabled")
            box = await root.bounding_box()
            await page.mouse.move(box["x"] + 30, box["y"] + 20)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 110, box["y"] + 70, steps=8)
            await page.mouse.up()
            await expect(root).to_have_attribute(
                "style", "left:130.0px;top:220.0px;--node-color:#c084fc"
            )
            await canvas.get_by_role("button", name="Edit a copy", exact=True).click()
            copy_dialog = (
                page.get_by_role("dialog").filter(has_text="Edit capability copy").last
            )
            await copy_dialog.get_by_label("New capability name", exact=True).fill(
                "my-custom-root"
            )
            await copy_dialog.get_by_label("Instructions", exact=True).fill(
                "You are my custom assistant. Always include the marker DESIGN_TEST."
            )
            await copy_dialog.get_by_role(
                "button", name="Save copy", exact=True
            ).click()
            await expect(copy_dialog).not_to_be_visible()
            await expect(
                canvas.locator(".agent-palette-item").filter(has_text="my-custom-root")
            ).to_be_visible()
            await (
                canvas.locator(".agent-palette-item")
                .filter(has_text="date-time")
                .click()
            )
            await expect(
                canvas.locator(".agent-board-node").filter(has_text="date-time")
            ).to_be_visible()
            await canvas.get_by_role("button", name="New HDS", exact=True).click()
            new_hds = (
                page.get_by_role("dialog").filter(has_text="New sub-agent HDS").last
            )
            await new_hds.get_by_label("HDS name", exact=True).fill("reviewer")
            await new_hds.get_by_label("Instructions", exact=True).fill(
                "You are a reviewer. Reply with DELEGATED_TEST and finish the delegated task."
            )
            await new_hds.get_by_role(
                "button", name="Add sub-agent", exact=True
            ).click()
            await expect(new_hds).not_to_be_visible()
            await expect(
                canvas.locator(".agent-board-node").filter(has_text="reviewer")
            ).to_be_visible()
            await page.screenshot(
                path="screenshots/agent-designer.png",
                full_page=True,
                animations="disabled",
            )
            await canvas.get_by_role("button", name="Save agent", exact=True).click()
            await expect(canvas).not_to_be_visible()
            await chat_dialog.get_by_role(
                "button", name="Save chat agent", exact=True
            ).click()
            await send_command(page, "Verify design")
            await wait_turn_completion(page)
            data = await (await page.request.get(design_server + "/test/state")).json()
            assert "DESIGN_TEST" in data["calls"][-1]["messages"][0]["content"]
            design = data["session"]["settings"]["agent_design"]
            saved_root = next(
                node for node in design["nodes"] if node["id"] == design["root_id"]
            )
            assert saved_root["name"] == "my-custom-root" and saved_root["x"] == 130
            await page.reload(wait_until="networkidle")
            data = await (await page.request.get(design_server + "/test/state")).json()
            assert data["session"]["settings"]["agent_design"] == design
            await send_command(page, "Delegate custom agent")
            await wait_turn_completion(page)
            data = await (await page.request.get(design_server + "/test/state")).json()
            assert any(
                "DELEGATED_TEST" in request["messages"][0]["content"]
                for request in data["calls"]
            )
            assert any(
                child["path"] == "my-custom-root/reviewer"
                and child["status"] == "completed"
                for child in data["session"]["sub_agent_conversations"]
            )
            assert (
                data["session"]["messages"][-1]["content"]
                == "Reply: Delegate custom agent"
            )
        finally:
            await browser.close()


async def test_skill_editor_separates_yaml_and_shows_nested_connections(
    design_server, tmp_path: Path
):
    project_folder = tmp_path / "Skill project"
    project_folder.mkdir()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 1100})
            await page.goto(design_server, wait_until="networkidle")
            await page.get_by_role("button", name="New project", exact=True).click()
            project_dialog = (
                page.get_by_role("dialog").filter(has_text="New project").last
            )
            await project_dialog.get_by_label("Project folder", exact=True).fill(
                str(project_folder)
            )
            await project_dialog.get_by_role(
                "button", name="Create project", exact=True
            ).click()
            await expect(project_dialog).not_to_be_visible()
            await page.get_by_role(
                "button", name="Chat agent settings", exact=True
            ).click()
            settings = (
                page.get_by_role("dialog").filter(has_text="Save chat agent").last
            )
            await settings.get_by_role(
                "button", name="Customize agent", exact=True
            ).click()
            canvas = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            await expect(canvas).to_be_visible()
            await (
                canvas.locator(".agent-palette-item")
                .filter(has_text="mock-skill")
                .click()
            )
            mock = canvas.locator(".agent-board-node").filter(has_text="mock-skill")
            nested = canvas.locator(".agent-board-node").filter(has_text="nested-skill")
            await expect(mock).to_be_attached()
            await expect(nested).to_be_attached()
            mock_id = await mock.get_attribute("data-node")
            nested_id = await nested.get_attribute("data-node")
            await expect(
                canvas.locator(
                    f'.agent-board-links path[data-parent="{mock_id}"][data-child="{nested_id}"]'
                )
            ).to_have_attribute("stroke-dasharray", "6 4")
            await expect(
                canvas.locator(f'.agent-board-links path[data-parent="{nested_id}"]')
            ).to_have_count(2)
            await canvas.get_by_role(
                "button", name="mock-skill → shared-tool", exact=True
            ).click()
            await canvas.get_by_label("Connect to", exact=True).click()
            await page.get_by_role(
                "option", name="TOOL · date-time", exact=True
            ).click()
            await canvas.get_by_role("button", name="Connect", exact=True).click()
            await canvas.get_by_role("button", name="Edit a copy", exact=True).click()
            editor = (
                page.get_by_role("dialog").filter(has_text="Edit capability copy").last
            )
            instructions = editor.get_by_label("Instructions", exact=True)
            body = await instructions.input_value()
            assert (
                not body.startswith("---")
                and "sub-tools:" not in body
                and "description:" not in body
            )
            await expect(editor.get_by_label("Description", exact=True)).to_have_value(
                "Produces a demonstration response without performing real operations."
            )
            await expect(
                editor.get_by_text("Sub-skills: nested-skill", exact=True)
            ).to_be_visible()
            await expect(
                editor.get_by_text("Sub-tools: mock-sub-tool, date-time", exact=True)
            ).to_be_visible()
            await editor.get_by_label("New capability name", exact=True).fill(
                "edited-skill-browser"
            )
            await editor.get_by_label("Description", exact=True).fill(
                "Description edited independently"
            )
            await instructions.fill("Only the instructions are editable here.")
            await page.screenshot(
                path="screenshots/skill-editor-description.png",
                full_page=True,
                animations="disabled",
            )
            await editor.get_by_role("button", name="Save copy", exact=True).click()
            await expect(editor).not_to_be_visible()
            edited = canvas.locator(".agent-board-node").filter(
                has_text="edited-skill-browser"
            )
            await expect(edited).to_be_attached()
            assert await edited.get_attribute("data-node") == mock_id
            await canvas.locator(".agent-board-viewport").evaluate(
                "element => { element.scrollLeft = 0; element.scrollTop = element.scrollHeight; }"
            )
            shared = canvas.locator(".agent-board-node").filter(has_text="shared-tool")
            await shared.scroll_into_view_if_needed()
            box = await shared.bounding_box()
            await page.mouse.move(box["x"] + 30, box["y"] + 20)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 60, box["y"] + 30, steps=8)
            await page.mouse.up()
            await expect(shared).to_have_attribute(
                "style", "left:380.0px;top:970.0px;--node-color:#34d399"
            )
            viewport = canvas.locator(".agent-board-viewport")
            scroll_top = await viewport.evaluate("element => element.scrollTop")
            assert scroll_top > 0
            await edited.click()
            await expect(
                canvas.locator(".agent-node-details").get_by_label(
                    "Connect to", exact=True
                )
            ).to_be_visible()
            await expect(viewport).to_have_js_property("scrollTop", scroll_top)
            await page.screenshot(
                path="screenshots/skill-dependency-connections.png",
                full_page=True,
                animations="disabled",
            )
            await canvas.get_by_role("button", name="Save agent", exact=True).click()
            await expect(canvas).not_to_be_visible()
            await settings.get_by_role(
                "button", name="Save chat agent", exact=True
            ).click()
            await expect(settings).not_to_be_visible()
            await page.reload(wait_until="networkidle")
            data = await (await page.request.get(design_server + "/test/state")).json()
            design = data["session"]["settings"]["agent_design"]
            skill = next(
                item
                for item in design["nodes"]
                if item["name"] == "edited-skill-browser"
            )
            assert "Description edited independently" in skill["content"]
            assert "sub-tools:\n- mock-sub-tool\n- date-time" in skill["content"]
            assert design["skill_connections_explicit"]
            await page.get_by_role(
                "button", name="Chat agent settings", exact=True
            ).click()
            settings = (
                page.get_by_role("dialog").filter(has_text="Save chat agent").last
            )
            await settings.get_by_role(
                "button", name="Customize agent", exact=True
            ).click()
            canvas = (
                page.locator(".agent-designer-dialog")
                .filter(has=page.locator(".agent-board-node"))
                .last
            )
            await expect(canvas).to_be_visible()
            shared_id = next(
                item["id"] for item in design["nodes"] if item["name"] == "shared-tool"
            )
            await expect(
                canvas.locator(
                    f'.agent-board-links path[data-parent="{mock_id}"][data-child="{shared_id}"]'
                )
            ).to_have_count(0)
            await expect(
                canvas.locator(
                    f'.agent-board-links path[data-parent="{mock_id}"][data-child="{nested_id}"]'
                )
            ).to_have_count(1)
        finally:
            await browser.close()
