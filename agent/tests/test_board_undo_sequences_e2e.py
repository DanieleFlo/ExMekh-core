"""Long browser workflows exercise both Control and Apple's Command/Meta undo."""

from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, expect

from tests.board_support import (
    board_snapshot,
    connect,
    expect_board_snapshot,
    move_node,
    open_board,
    select_node,
    start_node_drag,
)
from tests.browser_support import get_edge_executable, start_mock_server


@pytest.fixture(scope="module")
def board_server():
    yield from start_mock_server(0, "tests.run_board_agent")


@pytest.fixture(params=["Control+z", "Meta+z"], ids=["ctrl-z", "cmd-z"])
def undo_shortcut(request):
    return request.param


@pytest.fixture
async def page():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            yield await browser.new_page(viewport={"width": 1680, "height": 1100})
        finally:
            await browser.close()


async def test_seven_mixed_edits_unwind_exactly_in_reverse_order(
    page, board_server, undo_shortcut
):
    board = await open_board(page, board_server)
    checkpoints = []

    checkpoints.append(await board_snapshot(board))
    await connect(page, board, "TOOL · date-time")
    await expect(
        board.locator('path[data-parent="root"][data-child="date-time"]')
    ).to_have_count(1)

    await select_node(board, "worker")
    checkpoints.append(await board_snapshot(board))
    toggle = board.get_by_role("switch", name="Preserve sub-agent history")
    await toggle.click()
    await expect(toggle).not_to_be_checked()

    checkpoints.append(await board_snapshot(board))
    await move_node(page, board, "worker")

    checkpoints.append(await board_snapshot(board))
    await board.get_by_role("button", name="Use as root", exact=True).click()
    await expect(board.locator('[data-node="worker"] small')).to_contain_text("ROOT")
    await expect(
        board.locator('path[data-parent="root"][data-child="worker"]')
    ).to_have_count(0)

    await select_node(board, "calculate")
    checkpoints.append(await board_snapshot(board))
    await board.get_by_role("button", name="Remove from board", exact=True).click()
    await expect(board.locator('[data-node="calculate"]')).to_have_count(0)

    checkpoints.append(await board_snapshot(board))
    await board.get_by_role("button", name="New HDS", exact=True).click()
    draft = page.get_by_role("dialog").filter(has_text="New sub-agent HDS").last
    await draft.get_by_label("HDS name", exact=True).fill("sequence-worker")
    await draft.get_by_role("button", name="Add sub-agent", exact=True).click()
    await expect(
        board.locator(".agent-board-node").filter(has_text="sequence-worker")
    ).to_have_count(1)

    await select_node(board, "worker")
    checkpoints.append(await board_snapshot(board))
    await board.get_by_role("button", name="Edit a copy", exact=True).click()
    editor = page.get_by_role("dialog").filter(has_text="Edit capability copy").last
    copied_name = "sequence-copy-" + undo_shortcut.split("+")[0].lower()
    await editor.get_by_label("New capability name", exact=True).fill(copied_name)
    await editor.get_by_label("Instructions", exact=True).fill(
        "Edited instructions after six board edits."
    )
    await editor.get_by_role("button", name="Save copy", exact=True).click()
    await expect(board.locator('[data-node="worker"] strong')).to_have_text(copied_name)
    await expect(editor).not_to_be_visible()

    assert len(checkpoints) == 7
    for checkpoint in reversed(checkpoints):
        await page.keyboard.press(undo_shortcut)
        await expect_board_snapshot(board, checkpoint)
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_disabled()
    # Repeated undo at the beginning must not remove the initial configuration.
    for _ in range(2):
        await page.keyboard.press(undo_shortcut)
        await expect_board_snapshot(board, checkpoints[0])

    await select_node(board, "worker")
    await board.get_by_role("button", name="Edit a copy", exact=True).click()
    restored_editor = (
        page.get_by_role("dialog").filter(has_text="Edit capability copy").last
    )
    await expect(
        restored_editor.get_by_label("Instructions", exact=True)
    ).to_have_value("Worker instructions")
    await restored_editor.get_by_role("button", name="Cancel", exact=True).click()
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_disabled()


async def test_undo_then_new_branch_and_cancel_drag_preserve_previous_edits(
    page, board_server, undo_shortcut
):
    board = await open_board(page, board_server)
    initial = await board_snapshot(board)
    await connect(page, board, "TOOL · date-time")
    await expect(
        board.locator('path[data-parent="root"][data-child="date-time"]')
    ).to_have_count(1)
    await select_node(board, "thinking")
    before_connection = await board_snapshot(board)
    await connect(page, board, "SKILL · nested")
    await expect(
        board.locator('path[data-parent="thinking"][data-child="nested"]')
    ).to_have_count(1)
    await select_node(board, "nested")
    before_move = await board_snapshot(board)
    await move_node(page, board, "nested")
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, before_move)
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, before_connection)

    # Make a different edit after undoing two edits. It includes shared dependencies.
    await select_node(board, "worker")
    before_addition = await board_snapshot(board)
    await board.get_by_role("button", name="mock-skill SKILL", exact=True).click()
    for name in (
        "mock-skill",
        "nested-skill",
        "mock-sub-tool",
        "nested-sub-tool",
        "shared-tool",
    ):
        await expect(
            board.locator(".agent-board-node strong").get_by_text(name, exact=True)
        ).to_have_count(1)
    added = await board_snapshot(board)
    assert len(added["nodes"]) == len(before_addition["nodes"]) + 5
    shared_id = next(
        node["id"] for node in added["nodes"] if node["name"] == "shared-tool"
    )
    assert sum(link["child"] == shared_id for link in added["links"]) == 2

    # Cancelling a drag must not consume the earlier skill addition.
    await select_node(board, "thinking")
    before_drag = await board_snapshot(board)
    handle = await start_node_drag(page, board, "thinking")
    assert (await board_snapshot(board))["nodes"] != before_drag["nodes"]
    await page.keyboard.press(undo_shortcut)
    await page.mouse.up()
    await expect_board_snapshot(board, before_drag)
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_enabled()
    await handle.dispose()

    # A whole skill subtree is removed by one undo, while the earlier connection stays.
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, before_addition)
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, initial)
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_disabled()
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, initial)


async def test_dialog_focus_rejected_changes_navigation_and_reopening_do_not_corrupt_undo(
    page, board_server, undo_shortcut
):
    board = await open_board(page, board_server)
    initial = await board_snapshot(board)
    await connect(page, board, "TOOL · date-time")
    await expect(
        board.locator('path[data-parent="root"][data-child="date-time"]')
    ).to_have_count(1)
    await select_node(board, "worker")
    viewport = board.locator(".agent-board-viewport")
    scroll_before = await viewport.evaluate("element => element.scrollLeft")
    scroll_limit = await viewport.evaluate(
        "element => element.scrollWidth - element.clientWidth"
    )
    expected_scroll = min(scroll_before + 48, scroll_limit)
    assert expected_scroll > scroll_before
    await board.get_by_role("button", name="Pan right", exact=True).click()
    await page.wait_for_function(
        "({viewport, value}) => Math.abs(viewport.scrollLeft - value) <= 1",
        arg={"viewport": await viewport.element_handle(), "value": expected_scroll},
    )
    before_draft = await board_snapshot(board)

    await board.get_by_role("button", name="New HDS", exact=True).click()
    draft = page.get_by_role("dialog").filter(has_text="New sub-agent HDS").last
    name = draft.get_by_label("HDS name", exact=True)
    await name.fill("draft-worker")
    await name.press(undo_shortcut)
    await expect_board_snapshot(board, before_draft)
    await expect(draft).to_be_visible()
    # The dialog protects the underlying board even when focus is on a button.
    add = draft.get_by_role("button", name="Add sub-agent", exact=True)
    await add.focus()
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, before_draft)

    await name.fill("worker")
    await add.click()
    await expect(
        page.locator(".q-notification").filter(has_text="unique")
    ).to_be_visible()
    await expect(draft).to_be_visible()
    await expect_board_snapshot(board, before_draft)
    await page.keyboard.press("Escape")
    await expect(draft).not_to_be_visible()

    await board.get_by_role("button", name="Edit a copy", exact=True).click()
    editor = page.get_by_role("dialog").filter(has_text="Edit capability copy").last
    await editor.get_by_label("New capability name", exact=True).fill("main")
    await editor.get_by_role("button", name="Save copy", exact=True).click()
    await expect(
        page.locator(".q-notification").filter(has_text="Unable to save capability")
    ).to_be_visible()
    await expect_board_snapshot(board, before_draft)
    cancel = editor.get_by_role("button", name="Cancel", exact=True)
    await cancel.focus()
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, before_draft)
    await cancel.click()
    await expect(editor).not_to_be_visible()

    await select_node(board, "root")
    await board.get_by_label("Connect to", exact=True).click()
    await expect(page.get_by_role("option").first).to_be_visible()
    # Shift+undo is not a request to pop another entry.
    await page.keyboard.press(undo_shortcut.replace("+z", "+Shift+z"))
    selected_root = dict(before_draft, selected="root")
    await expect_board_snapshot(board, selected_root)
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, initial)
    await expect(page.get_by_role("option")).to_have_count(0)
    assert (
        abs(await viewport.evaluate("element => element.scrollLeft") - expected_scroll)
        <= 1
    )
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_disabled()

    # Unsaved edits are discarded, and reopening starts a fresh undo history.
    await connect(page, board, "TOOL · calculate")
    await expect(
        board.locator('path[data-parent="root"][data-child="calculate"]')
    ).to_have_count(1)
    await board.get_by_role("button", name="Cancel", exact=True).click()
    await expect(board).not_to_be_visible()
    await page.keyboard.press(undo_shortcut)
    settings = page.get_by_role("dialog").filter(has_text="Save chat agent").last
    await settings.get_by_role("button", name="Custom", exact=True).click()
    await board.locator('[data-node="root"]').click(trial=True)
    await expect_board_snapshot(board, initial)
    await expect(board.get_by_role("button", name="Undo", exact=True)).to_be_disabled()
    await page.keyboard.press(undo_shortcut)
    await expect_board_snapshot(board, initial)
