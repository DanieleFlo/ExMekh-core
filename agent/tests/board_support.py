"""Browser actions and observable graph state shared by agent-board scenarios."""

from __future__ import annotations

import asyncio
import time

from playwright.async_api import Locator, Page, expect


async def open_board(page, server):
    await page.goto(server, wait_until="networkidle")
    await page.get_by_role("button", name="Chat agent settings", exact=True).click()
    settings = page.get_by_role("dialog").filter(has_text="Save chat agent").last
    await settings.get_by_role("button", name="Custom", exact=True).click()
    board = (
        page.locator(".agent-designer-dialog")
        .filter(has=page.locator(".agent-board-node"))
        .last
    )
    await expect(board.locator(".agent-board-viewport")).to_be_visible()
    # The dialog takes focus after its entrance animation, which can close a
    # select menu opened before that transition finishes.
    await board.evaluate(
        "async board => await Promise.all(board.closest('.q-dialog')"
        ".getAnimations({subtree: true}).map(animation => animation.finished))"
    )
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


# Read rendered state only: no application mutation or private Python state.
BOARD_SNAPSHOT = r"""board => ({
    nodes: Array.from(board.querySelectorAll('.agent-board-node'), node => ({
        id: node.dataset.node, name: node.querySelector('strong').textContent,
        root: node.querySelector('small').textContent.includes('ROOT'),
        x: parseFloat(node.style.left), y: parseFloat(node.style.top),
    })).sort((a, b) => a.id.localeCompare(b.id)),
    links: Array.from(board.querySelectorAll('.agent-board-links path'), path => ({
        parent: path.dataset.parent, child: path.dataset.child,
        curve: path.getAttribute('d').match(/-?\d+(?:\.\d+)?/g).map(Number),
        dashed: path.hasAttribute('stroke-dasharray'),
    })).sort((a, b) => (a.parent + '/' + a.child).localeCompare(b.parent + '/' + b.child)),
    catalog: Array.from(board.querySelectorAll('.agent-palette-item'),
        item => item.textContent.trim().replace(/\s+/g, ' ')),
    selected: board.querySelector('.agent-board-node.selected')?.dataset.node ?? null,
    history: board.querySelector('.agent-node-details [role="switch"]')?.getAttribute('aria-checked') ?? null,
})"""


async def board_snapshot(board: Locator) -> dict:
    return await board.evaluate(BOARD_SNAPSHOT)


async def expect_board_snapshot(board: Locator, expected: dict) -> None:
    """Wait for the server's render and compare all nodes, edges, catalog and selection."""
    deadline = time.monotonic() + 10
    actual = await board_snapshot(board)
    while actual != expected and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        actual = await board_snapshot(board)
    assert actual == expected


async def select_node(board: Locator, node_id: str) -> None:
    node = board.locator(f'[data-node="{node_id}"]')
    await node.click()
    await expect(node).to_have_class("agent-board-node selected")
    await expect(board.locator(".agent-node-details > :first-child")).to_have_text(
        await node.locator("strong").inner_text()
    )


async def start_node_drag(page: Page, board: Locator, node_id: str, dx=60, dy=35):
    node = board.locator(f'[data-node="{node_id}"]')
    await node.click(trial=True)
    handle = await node.element_handle()
    box = await node.bounding_box()
    await page.mouse.move(box["x"] + 25, box["y"] + 20)
    await page.mouse.down()
    await page.mouse.move(box["x"] + 25 + dx, box["y"] + 20 + dy, steps=6)
    return handle


async def move_node(page: Page, board: Locator, node_id: str) -> None:
    before = await board_snapshot(board)
    handle = await start_node_drag(page, board, node_id)
    moving = await board_snapshot(board)
    assert moving["nodes"] != before["nodes"]
    for old, new in zip(before["links"], moving["links"], strict=True):
        affected = node_id in (old["parent"], old["child"])
        assert (old["curve"] != new["curve"]) == affected
    await page.mouse.up()
    # Wait for persistence of the move: a DOM-only drag is not enough.
    await page.wait_for_function("node => !node.isConnected", arg=handle)
    assert (await board_snapshot(board))["nodes"] == moving["nodes"]
    await handle.dispose()
