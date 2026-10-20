from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, expect

from tests.board_support import board_snapshot, open_board
from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)
from tests.test_message_editing_e2e import state


@pytest.fixture(scope="module")
def corrections_server():
    yield from start_mock_server(0, "tests.run_board_agent")


async def test_marquee_toggle_group_move_collective_commands_and_arrange(
    corrections_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1800, "height": 1100})
            board = await open_board(page, corrections_server)
            canvas = board.locator(".agent-board")
            bounds = await canvas.bounding_box()
            await page.mouse.move(bounds["x"] + 330, bounds["y"] + 80)
            await page.mouse.down(button="right")
            await page.mouse.move(bounds["x"] + 550, bounds["y"] + 450, steps=8)
            await expect(board.locator(".agent-board-selection")).to_be_visible()
            await expect(board.locator(".agent-board-node.selected")).to_have_count(2)
            await page.mouse.up(button="right")
            await expect(
                board.get_by_text("2 components selected", exact=True)
            ).to_be_visible()
            await expect(board.locator(".agent-board-selection")).to_have_count(0)
            root = board.locator('[data-node="root"]')
            root_style = await root.get_attribute("style")
            await root.click(modifiers=["Control"])
            await expect(
                board.get_by_text("3 components selected", exact=True)
            ).to_be_visible()
            await root.click(modifiers=["Meta"])
            await expect(
                board.get_by_text("2 components selected", exact=True)
            ).to_be_visible()
            before = await board_snapshot(board)
            thinking = board.locator('[data-node="thinking"]')
            box = await thinking.bounding_box()
            await page.mouse.move(box["x"] + 20, box["y"] + 20)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 85, box["y"] + 55, steps=6)
            await page.mouse.up()
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_enabled()
            moved = await board_snapshot(board)
            for old, new in zip(before["nodes"], moved["nodes"], strict=True):
                delta = (65, 35) if old["id"] in {"thinking", "calculate"} else (0, 0)
                assert (new["x"] - old["x"], new["y"] - old["y"]) == delta
            await board.get_by_role("button", name="Undo", exact=True).click()
            await expect(thinking).to_have_css("left", "350px")
            await expect(thinking).to_have_attribute("aria-pressed", "true")
            assert (await board_snapshot(board))["nodes"] == before["nodes"]
            await board.get_by_role(
                "button", name="Disconnect selected", exact=True
            ).click()
            await expect(board.locator('path[data-parent="thinking"]')).to_have_count(0)
            await board.get_by_role("button", name="Undo", exact=True).click()
            await expect(board.locator('path[data-parent="thinking"]')).to_have_count(1)
            await board.get_by_role(
                "button", name="Remove selected (keep root)", exact=True
            ).click()
            await expect(thinking).to_have_count(0)
            await board.get_by_role("button", name="Undo", exact=True).click()
            await expect(board.locator(".agent-board-node.selected")).to_have_count(2)
            await board.get_by_role(
                "button", name="Arrange automatically", exact=True
            ).click()
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_enabled()
            await expect(root).not_to_have_attribute("style", root_style)
            arranged = await board_snapshot(board)
            for index, node in enumerate(arranged["nodes"]):
                for other in arranged["nodes"][index + 1 :]:
                    assert (
                        abs(node["x"] - other["x"]) >= 185
                        or abs(node["y"] - other["y"]) >= 64
                    )
            await board.get_by_role("button", name="Undo", exact=True).click()
            await expect(thinking).to_have_css("left", "350px")
            assert (await board_snapshot(board))["nodes"] == before["nodes"]
        finally:
            await browser.close()


async def test_missing_tool_dependency_warning_can_be_resolved(corrections_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 1000})
            board = await open_board(page, corrections_server)
            await board.locator('[data-node="thinking"]').click()
            await board.locator('.agent-palette-item[title="read-file"]').click()
            warning = board.locator(".agent-node-details").get_by_text(
                "deep-thinking: read-file requires inspect-file in the same scope. Connect the missing tools.",
                exact=True,
            )
            await expect(warning).to_be_visible()
            await board.locator('[data-node="thinking"]').click()
            await board.locator('.agent-palette-item[title="inspect-file"]').click()
            await expect(warning).to_have_count(0)
        finally:
            await browser.close()


async def test_split_earlier_and_latest_reply_reload_and_continue_independently(
    corrections_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(corrections_server, wait_until="networkidle")
            await page.get_by_role("button", name="New Quick Chat", exact=True).click()
            await send_command(page, "First branch input")
            await wait_turn_completion(page)
            await send_command(page, "Obsolete second input")
            await wait_turn_completion(page)
            source = (await state(corrections_server))["session"]
            source_url = page.url
            buttons = page.get_by_role(
                "button", name="Split conversation here", exact=True
            )
            await expect(buttons).to_have_count(2)
            await buttons.first.click()
            await expect(page.locator(".msg-assistant")).to_have_count(1)
            branch_url = page.url
            branch = (await state(corrections_server))["session"]
            assert branch["id"] != source["id"]
            assert branch["title"] == source["title"] + " (1)"
            assert branch["messages"] == source["messages"][:2]
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".msg-assistant")).to_have_count(1)
            await send_command(page, "Alternate continuation")
            await wait_turn_completion(page)
            current = await state(corrections_server)
            assert "Obsolete second input" not in current["calls"][-1]["history"]
            assert "Reply: First branch input" in current["calls"][-1]["history"]
            await page.goto(source_url, wait_until="networkidle")
            await expect(page.locator(".msg-assistant")).to_have_count(2)
            assert (await state(corrections_server))["session"]["messages"] == source[
                "messages"
            ]
            await page.get_by_role(
                "button", name="Split conversation here", exact=True
            ).last.click()
            await expect(page).not_to_have_url(source_url)
            cloned = (await state(corrections_server))["session"]
            assert cloned["title"] == source["title"] + " (2)"
            assert cloned["messages"] == source["messages"]
            await page.goto(branch_url, wait_until="networkidle")
            await expect(page.locator(".msg-assistant")).to_have_count(2)
        finally:
            await browser.close()


async def test_schema_switch_is_saved_for_new_and_existing_projects(
    corrections_server, tmp_path
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            await page.goto(corrections_server, wait_until="networkidle")
            await page.get_by_role("button", name="New project", exact=True).click()
            creation = page.get_by_role("dialog").filter(has_text="Create project").last
            await creation.get_by_label("Project folder", exact=True).fill(
                str(tmp_path)
            )
            await creation.get_by_role(
                "button", name="Project settings", exact=True
            ).click()
            settings = (
                page.get_by_role("dialog").filter(has_text="Save project settings").last
            )
            switch = settings.get_by_role(
                "switch", name="Parameter descriptions in tool schemas", exact=True
            )
            await expect(switch).not_to_be_checked()
            await switch.click()
            await settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(settings).not_to_be_visible()
            await creation.get_by_role(
                "button", name="Create project", exact=True
            ).click()
            await expect(creation).not_to_be_visible()
            await page.reload(wait_until="networkidle")
            row = page.locator(".sidebar-project").filter(has_text=tmp_path.name).last
            await row.get_by_role("button", name="Project actions", exact=True).click()
            await page.get_by_text("Project settings", exact=True).last.click()
            await expect(switch).to_be_checked()
            await switch.click()
            await settings.get_by_role(
                "button", name="Save project settings", exact=True
            ).click()
            await expect(settings).not_to_be_visible()
            await page.reload(wait_until="networkidle")
            await row.get_by_role("button", name="Project actions", exact=True).click()
            await page.get_by_text("Project settings", exact=True).last.click()
            await expect(switch).not_to_be_checked()
        finally:
            await browser.close()
