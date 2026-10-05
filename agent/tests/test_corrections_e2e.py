from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def live_server():
    yield from start_mock_server(0, "tests.run_corrections_agent")


async def test_tool_thinking_and_tokens_are_visible_before_final_answer(live_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            await page.goto(live_server, wait_until="networkidle")
            await send_command(page, "Check live tools")
            tool = page.locator(".tool-event-card").filter(has_text="Tool: date-time")
            await expect(tool).to_have_attribute(
                "aria-expanded", "false", timeout=20000
            )
            await tool.click()
            panel = page.locator(".tool-output-panel").filter(has_text="total_seconds")
            await expect(panel).to_be_visible()
            await expect(panel.locator(".output-thinking")).to_contain_text(
                "THINKING SUMMARY"
            )
            assert (await (await page.request.get(live_server + "/test/live")).json())[
                "summaries"
            ] == 1
            await expect(
                page.get_by_text("FINAL ANSWER", exact=True)
            ).not_to_be_visible()
            counter = page.locator(".input-bar-context-tokens")
            await expect(counter).to_be_visible()
            first_count = await counter.inner_text()
            await expect(counter).not_to_have_text(first_count, timeout=5000)

            # Reconnect to the same unfinished turn and continue counting its stream.
            await page.reload(wait_until="networkidle")
            await expect(tool).to_have_attribute("aria-expanded", "false")
            await tool.click()
            await expect(panel.locator(".output-thinking")).to_contain_text(
                "THINKING SUMMARY"
            )
            first_count = await counter.inner_text()
            await expect(counter).not_to_have_text(first_count, timeout=5000)

            await page.request.post(live_server + "/test/release-skill")
            skill = page.locator(".skill-compact-header").filter(
                has_text="deep-thinking"
            )
            await expect(skill).to_be_visible(timeout=10000)
            await skill.click()
            await page.locator(".skill-output-expansion .q-item").first.click()
            await expect(page.locator(".skill-output-panel")).to_contain_text(
                "Verified 24 elapsed hours"
            )
            await expect(
                page.get_by_text("FINAL ANSWER", exact=True)
            ).not_to_be_visible()
            await page.request.post(live_server + "/test/release-final")
            await wait_turn_completion(page)
            await expect(counter).to_contain_text("6.1k/")
        finally:
            await page.request.post(live_server + "/test/release-skill")
            await page.request.post(live_server + "/test/release-final")
            await browser.close()


@pytest.fixture(scope="module")
def board_server():
    yield from start_mock_server(0, "tests.run_todo_agent")


@pytest.mark.parametrize("touch", [False, True], ids=["desktop", "phone-touch"])
async def test_board_pan_controls_drag_and_rerender_preserve_node_positions(
    board_server, touch
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(
            viewport={"width": 390 if touch else 1440, "height": 1000},
            has_touch=touch,
        )
        try:
            await page.goto(board_server, wait_until="networkidle")
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
            viewport = canvas.locator(".agent-board-viewport")
            await expect(viewport).to_be_visible()
            positions = await canvas.locator(".agent-board-node").evaluate_all(
                "nodes => nodes.map(node => node.getAttribute('style'))"
            )
            await canvas.get_by_role("button", name="Pan right", exact=True).click()
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollLeft > 0"
            )
            await canvas.get_by_role("button", name="Pan left", exact=True).click()
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollLeft === 0"
            )
            await canvas.get_by_role("button", name="Pan down", exact=True).click()
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollTop > 0"
            )
            await canvas.get_by_role("button", name="Pan up", exact=True).click()
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollTop === 0"
            )
            await viewport.focus()
            await viewport.press("ArrowRight")
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollLeft > 0"
            )
            await viewport.press("ArrowLeft")
            await page.wait_for_function(
                "document.querySelector('.agent-board-viewport').scrollLeft === 0"
            )
            point = await viewport.evaluate("""element => {
                const bounds = element.getBoundingClientRect();
                for (let y = bounds.bottom - 30; y > bounds.top + 80; y -= 30) {
                    for (let x = bounds.right - 30; x > bounds.left + 140; x -= 30) {
                        const target = document.elementFromPoint(x, y);
                        if (target?.closest('.agent-board') && !target.closest('.agent-board-node')) return {x, y};
                    }
                }
                throw new Error('No empty board surface found');
            }""")
            if touch:
                cdp = await page.context.new_cdp_session(page)
                await cdp.send(
                    "Input.dispatchTouchEvent",
                    {"type": "touchStart", "touchPoints": [point]},
                )
                await cdp.send(
                    "Input.dispatchTouchEvent",
                    {
                        "type": "touchMove",
                        "touchPoints": [{"x": point["x"] - 100, "y": point["y"] - 60}],
                    },
                )
                await cdp.send(
                    "Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []}
                )
            else:
                await page.mouse.move(point["x"], point["y"])
                await page.mouse.down()
                await page.mouse.move(point["x"] - 100, point["y"] - 60, steps=5)
                await page.mouse.up()
            await page.wait_for_function("""() => {
                const viewport = document.querySelector('.agent-board-viewport');
                return viewport.scrollLeft >= 99 && viewport.scrollTop >= 59;
            }""")
            assert (
                await canvas.locator(".agent-board-node").evaluate_all(
                    "nodes => nodes.map(node => node.getAttribute('style'))"
                )
                == positions
            )
            scroll = await viewport.evaluate(
                "element => [element.scrollLeft, element.scrollTop]"
            )
            count = await canvas.locator(".agent-board-node").count()
            await canvas.locator(".agent-palette-item").first.click()
            await expect(canvas.locator(".agent-board-node")).to_have_count(count + 1)
            assert (
                await viewport.evaluate(
                    "element => [element.scrollLeft, element.scrollTop]"
                )
                == scroll
            )
            await canvas.get_by_role("button", name="Cancel", exact=True).click()
        finally:
            await browser.close()
