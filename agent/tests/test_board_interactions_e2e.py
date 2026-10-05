from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import get_edge_executable, start_mock_server


@pytest.fixture(scope="module")
def regression_server():
    yield from start_mock_server(0, "tests.run_board_agent")


async def open_board(page, server):
    await page.goto(server, wait_until="networkidle")
    await page.get_by_role("button", name="Chat agent settings", exact=True).click()
    settings = page.get_by_role("dialog").filter(has_text="Save chat agent").last
    await settings.get_by_role("button", name="Customize agent", exact=True).click()
    board = (
        page.locator(".agent-designer-dialog")
        .filter(has=page.locator(".agent-board-node"))
        .last
    )
    await expect(board.locator(".agent-board-viewport")).to_be_visible()
    await board.locator('[data-node="root"]').click(trial=True)
    return board


async def candidates(page, board):
    await board.get_by_label("Connect to", exact=True).click()
    await expect(page.get_by_role("option").first).to_be_visible()
    labels = await page.get_by_role("option").all_text_contents()
    await page.keyboard.press("Escape")
    return {label.strip() for label in labels}


async def connect(page, board, label):
    await board.get_by_label("Connect to", exact=True).click()
    await page.get_by_role("option", name=label, exact=True).click()
    await board.get_by_role("button", name="Connect", exact=True).click()


async def test_archived_read_outputs_and_xml_headers_are_visible(regression_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(regression_server, wait_until="networkidle")
            for mode in ("none", "brief", "detailed"):
                row = page.locator(".tool-event-card").filter(has_text=mode)
                await row.click()
                await expect(row).to_have_attribute("aria-expanded", "true")
                panel = row.locator("xpath=following-sibling::*[1]")
                await expect(panel).to_contain_text(mode.upper() + " ARCHIVED RESULT")
            await page.get_by_role("tab", name="Raw History", exact=True).click()
            raw = page.locator(
                ".raw-history-container:visible .json-readable-string"
            ).filter(has_text="BRIEF ARCHIVED RESULT")
            await expect(raw).to_contain_text("<header>\n  <name>read-web-page</name>")
            await expect(raw).to_contain_text("<summary-mode>brief</summary-mode>")
            await expect(
                page.locator(".raw-history-container:visible .output-header")
            ).to_have_count(0)
            await page.reload(wait_until="networkidle")
            await page.get_by_role("tab", name="Chat", exact=True).click()
            row = page.locator(".tool-event-card").filter(has_text="detailed")
            await row.click()
            await expect(row.locator("xpath=following-sibling::*[1]")).to_contain_text(
                "DETAILED ARCHIVED RESULT"
            )
        finally:
            await browser.close()


@pytest.mark.parametrize("node_id", ["root", "thinking"])
async def test_small_smooth_pan_live_connections_and_drag_undo(
    regression_server, node_id
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            board = await open_board(page, regression_server)
            viewport = board.locator(".agent-board-viewport")
            if node_id == "root":
                await board.get_by_role("button", name="Pan right", exact=True).click()
                await page.wait_for_function(
                    "Math.abs(document.querySelector('.agent-board-viewport').scrollLeft - 48) <= 1"
                )
                await viewport.focus()
                await page.keyboard.press("ArrowLeft")
                await page.wait_for_function(
                    "document.querySelector('.agent-board-viewport').scrollLeft === 0"
                )
                await page.keyboard.press("ArrowDown")
                await page.wait_for_function(
                    "Math.abs(document.querySelector('.agent-board-viewport').scrollTop - 48) <= 1"
                )
                await page.keyboard.press("ArrowUp")
                await page.wait_for_function(
                    "document.querySelector('.agent-board-viewport').scrollTop === 0"
                )
            node = board.locator(f'[data-node="{node_id}"]')
            links = board.locator(
                f'path[data-parent="{node_id}"], path[data-child="{node_id}"]'
            )
            await node.scroll_into_view_if_needed()
            original_style = await node.get_attribute("style")
            original_position = await node.evaluate(
                "node => [node.style.left, node.style.top]"
            )
            original_curves = await links.evaluate_all(
                "paths => paths.map(path => path.getAttribute('d').trim().split(/\\s+/).join(' '))"
            )
            box = await node.bounding_box()
            assert (
                await page.evaluate(
                    "point => document.elementFromPoint(point.x, point.y).closest('.agent-board-node')?.dataset.node",
                    {"x": box["x"] + 30, "y": box["y"] + 20},
                )
                == node_id
            )
            await page.mouse.move(box["x"] + 30, box["y"] + 20)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 100, box["y"] + 60, steps=5)
            current_curves = await links.evaluate_all(
                "paths => paths.map(path => path.getAttribute('d').trim().split(/\\s+/).join(' '))"
            )
            assert len(current_curves) >= 2
            assert all(
                before != after
                for before, after in zip(original_curves, current_curves, strict=True)
            )
            # Geometry updates before pointerup and before the server commits the move.
            assert await node.get_attribute("style") != original_style
            await page.mouse.up()
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_enabled()
            await page.keyboard.press("Control+z")
            await expect(node).to_have_attribute("style", original_style)
            restored_curves = await links.evaluate_all(
                "paths => paths.map(path => path.getAttribute('d').trim().split(/\\s+/).join(' '))"
            )
            assert restored_curves == original_curves
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_disabled()
            # Cancelling an in-progress drag restores both node and lines without an undo entry.
            box = await node.bounding_box()
            assert (
                await page.evaluate(
                    "point => document.elementFromPoint(point.x, point.y).closest('.agent-board-node')?.dataset.node",
                    {"x": box["x"] + 30, "y": box["y"] + 20},
                )
                == node_id
            ), {
                "node": box,
                "viewport": await viewport.bounding_box(),
                "hit": await page.evaluate(
                    "point => document.elementFromPoint(point.x, point.y)?.outerHTML",
                    {"x": box["x"] + 30, "y": box["y"] + 20},
                ),
            }
            await page.mouse.move(box["x"] + 30, box["y"] + 20)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 80, box["y"] + 40)
            await page.keyboard.press("Control+z")
            await page.mouse.up()
            assert (
                await node.evaluate("node => [node.style.left, node.style.top]")
                == original_position
            )
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_disabled()
        finally:
            await browser.close()


async def test_connect_options_change_with_selection_connections_and_undo(
    regression_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 1000})
            board = await open_board(page, regression_server)
            options = await candidates(page, board)
            assert (
                "HDS · worker" not in options and "SKILL · deep-thinking" not in options
            )
            assert {
                "HDS · free-worker",
                "SKILL · nested",
                "TOOL · calculate",
                "TOOL · date-time",
            } == options
            await connect(page, board, "TOOL · date-time")
            await expect(
                board.locator('path[data-parent="root"][data-child="date-time"]')
            ).to_have_count(1)
            assert "TOOL · date-time" not in await candidates(page, board)
            await page.keyboard.press("Control+z")
            await expect(
                board.locator('path[data-parent="root"][data-child="date-time"]')
            ).to_have_count(0)
            assert "TOOL · date-time" in await candidates(page, board)
            await board.locator('[data-node="thinking"]').click()
            options = await candidates(page, board)
            assert not any(option.startswith("HDS") for option in options)
            assert "TOOL · calculate" not in options
            assert {"SKILL · nested", "TOOL · date-time"} == options
            await connect(page, board, "SKILL · nested")
            await expect(
                board.locator('path[data-parent="thinking"][data-child="nested"]')
            ).to_have_count(1)
            await board.locator('[data-node="nested"]').click()
            assert await candidates(page, board) == {
                "TOOL · calculate",
                "TOOL · date-time",
            }
            await page.keyboard.press("Control+z")
            await expect(
                board.locator('path[data-parent="thinking"][data-child="nested"]')
            ).to_have_count(0)
            await board.locator('[data-node="calculate"]').click()
            await expect(board.get_by_label("Connect to", exact=True)).to_have_count(0)
            await board.get_by_role(
                "button", name="Remove from board", exact=True
            ).click()
            await expect(board.locator('[data-node="calculate"]')).to_have_count(0)
            await page.keyboard.press("Control+z")
            await expect(board.locator('[data-node="calculate"]')).to_have_count(1)
            await expect(
                board.locator('path[data-parent="thinking"][data-child="calculate"]')
            ).to_have_count(1)
            await board.locator('[data-node="root"]').click()
            for label in (
                "HDS · free-worker",
                "SKILL · nested",
                "TOOL · calculate",
                "TOOL · date-time",
            ):
                await connect(page, board, label)
            await expect(board.locator(".q-select.q-field--disabled")).to_have_count(1)
            await expect(
                board.get_by_role("button", name="Connect", exact=True)
            ).to_be_disabled()
            await page.keyboard.press("Control+z")
            await expect(board.get_by_label("Connect to", exact=True)).to_be_enabled()
            assert await candidates(page, board) == {"TOOL · date-time"}
        finally:
            await browser.close()


async def test_undo_hds_settings_root_creation_copy_and_disconnect(regression_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 1000})
            board = await open_board(page, regression_server)
            worker = board.locator('[data-node="worker"]')
            await worker.click()
            history_toggle = board.get_by_role(
                "switch", name="Preserve sub-agent history"
            )
            await expect(history_toggle).to_be_checked()
            await history_toggle.click()
            await expect(history_toggle).not_to_be_checked()
            await page.keyboard.press("Control+z")
            await expect(history_toggle).to_be_checked()
            await board.get_by_role("button", name="Use as root", exact=True).click()
            await expect(worker.locator("small")).to_contain_text("ROOT")
            await page.keyboard.press("Control+z")
            await expect(board.locator('[data-node="root"] small')).to_contain_text(
                "ROOT"
            )
            await expect(
                board.locator('path[data-parent="root"][data-child="worker"]')
            ).to_have_count(1)
            await worker.click()
            await board.get_by_role(
                "button", name="worker → nested", exact=True
            ).click()
            await expect(
                board.locator('path[data-parent="worker"][data-child="nested"]')
            ).to_have_count(0)
            await page.keyboard.press("Control+z")
            await expect(
                board.locator('path[data-parent="worker"][data-child="nested"]')
            ).to_have_count(1)
            await board.get_by_role("button", name="New HDS", exact=True).click()
            dialog = (
                page.get_by_role("dialog").filter(has_text="New sub-agent HDS").last
            )
            name = dialog.get_by_label("HDS name", exact=True)
            await name.fill("added-worker")
            await name.press("Control+z")
            await expect(dialog).to_be_visible()
            await expect(
                board.get_by_role("button", name="Undo", exact=True)
            ).to_be_disabled()
            await name.fill("added-worker")
            await dialog.get_by_role("button", name="Add sub-agent", exact=True).click()
            await expect(
                board.locator(".agent-board-node").filter(has_text="added-worker")
            ).to_have_count(1)
            await page.keyboard.press("Control+z")
            await expect(
                board.locator(".agent-board-node").filter(has_text="added-worker")
            ).to_have_count(0)
            # Palette additions are a single edit, including their generated dependencies.
            await (
                board.locator(".agent-palette-item")
                .filter(has_text="bash")
                .first.click()
            )
            await expect(
                board.locator(".agent-board-node").filter(has_text="bash")
            ).to_have_count(1)
            await page.keyboard.press("Control+z")
            await expect(
                board.locator(".agent-board-node").filter(has_text="bash")
            ).to_have_count(0)
            await board.locator('[data-node="root"]').click()
            await board.get_by_role("button", name="Edit a copy", exact=True).click()
            editor = (
                page.get_by_role("dialog").filter(has_text="Edit capability copy").last
            )
            await editor.get_by_label("New capability name", exact=True).fill(
                "edited-root"
            )
            await editor.get_by_label("Instructions", exact=True).fill(
                "Changed root instructions"
            )
            await editor.get_by_role("button", name="Save copy", exact=True).click()
            await expect(editor).not_to_be_visible()
            await expect(board.locator('[data-node="root"]')).to_contain_text(
                "edited-root"
            )
            await page.keyboard.press("Control+z")
            await expect(board.locator('[data-node="root"] strong')).to_have_text(
                "root"
            )
            await expect(
                board.locator(".agent-palette-item").filter(has_text="edited-root")
            ).to_have_count(0)
        finally:
            await browser.close()
