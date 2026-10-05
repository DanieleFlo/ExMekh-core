"""Resize an open designer while keeping catalog, board and connections usable."""

from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import get_edge_executable, start_mock_server


@pytest.fixture(scope="module")
def responsive_designer_server():
    yield from start_mock_server(0, "tests.run_message_edit_agent")


async def test_designer_resizes_without_losing_catalog_or_wrapping_connections(
    responsive_designer_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            await page.goto(responsive_designer_server, wait_until="networkidle")
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
            await canvas.get_by_role("button", name="Edit a copy", exact=True).click()
            editor = (
                page.get_by_role("dialog").filter(has_text="Edit capability copy").last
            )
            root_name = "responsive-root-with-a-very-long-capability-name"
            await editor.get_by_label("New capability name", exact=True).fill(root_name)
            await editor.get_by_role("button", name="Save copy", exact=True).click()
            await expect(editor).not_to_be_visible()
            palette = canvas.locator(".agent-palette")
            board = canvas.locator(".agent-board-viewport")
            details = canvas.locator(".agent-node-details")
            links = details.locator(".agent-connection-button")
            assert await links.count() > 0
            selected = canvas.locator(".agent-board-node.selected")
            selected_id = await selected.get_attribute("data-node")
            clipped_widths = []

            for width, height in [
                (1440, 900),
                (1100, 800),
                (900, 900),
                (768, 1024),
                (520, 844),
                (390, 844),
                (320, 640),
                (844, 390),
                (1440, 900),
            ]:
                await page.set_viewport_size({"width": width, "height": height})
                await palette.scroll_into_view_if_needed()
                await expect(palette).to_be_in_viewport()
                if width <= 600:
                    assert await palette.evaluate(
                        """element => {
                            const bottom = element.getBoundingClientRect().bottom;
                            const headingElement = element.querySelector('h3');
                            const heading = headingElement.getBoundingClientRect();
                            const lineHeight = parseFloat(getComputedStyle(headingElement).lineHeight);
                            const first = element.querySelector('.agent-palette-item').getBoundingClientRect();
                            return lineHeight <= heading.height && heading.bottom <= first.top && [...element.querySelectorAll('.agent-palette-item')].every(
                                item => item.getBoundingClientRect().bottom <= bottom
                            );
                        }"""
                    )
                actions = canvas.locator(".agent-designer-actions")
                await expect(
                    actions.get_by_role("button", name="Save agent", exact=True)
                ).to_be_in_viewport()
                await expect(selected).to_have_attribute("data-node", selected_id)
                for panel in [palette, board, details]:
                    await panel.scroll_into_view_if_needed()
                    bounds = await panel.bounding_box()
                    assert bounds is not None
                    assert bounds["width"] > 0
                    assert bounds["x"] >= -1
                    assert bounds["x"] + bounds["width"] <= width + 1
                assert await canvas.locator(".agent-designer-shell").evaluate(
                    "element => element.scrollWidth <= element.clientWidth"
                )
                assert await canvas.locator(".agent-designer-layout").evaluate(
                    "element => element.scrollWidth <= element.clientWidth"
                )
                link = links.filter(has_text="control_tool_execution").first
                await link.scroll_into_view_if_needed()
                await expect(link).to_contain_text(
                    f"{root_name} → control_tool_execution"
                )
                content = link.locator(".q-btn__content")
                label = content.locator(":scope > .block")
                await expect(content).to_have_css("flex-wrap", "nowrap")
                await expect(content).to_have_css("justify-content", "flex-start")
                await expect(label).to_have_css("white-space", "nowrap")
                await expect(label).to_have_css("text-overflow", "ellipsis")
                await expect(label).to_have_css("text-align", "left")
                metrics = await label.evaluate(
                    "element => ({width: element.clientWidth, total: element.scrollWidth})"
                )
                if metrics["total"] > metrics["width"]:
                    clipped_widths.append(width)
                icon_box = await content.locator(".q-icon").bounding_box()
                label_box = await label.bounding_box()
                assert (
                    abs(
                        icon_box["y"]
                        + icon_box["height"] / 2
                        - label_box["y"]
                        - label_box["height"] / 2
                    )
                    <= 1
                )
            assert 1440 in clipped_widths and 320 in clipped_widths

            # The compact catalog scrolls horizontally and still supports changes.
            await page.set_viewport_size({"width": 390, "height": 844})
            item = palette.locator(".agent-palette-item").filter(has_text="shared-tool")
            await item.scroll_into_view_if_needed()
            await expect(item).to_be_visible()
            assert await palette.evaluate(
                "element => element.scrollWidth > element.clientWidth"
            )
            previous_nodes = await canvas.locator(".agent-board-node").count()
            await item.click()
            await expect(canvas.locator(".agent-board-node")).to_have_count(
                previous_nodes + 1
            )
            await canvas.get_by_role("button", name="Undo", exact=True).click()
            await expect(canvas.locator(".agent-board-node")).to_have_count(
                previous_nodes
            )

            # A clipped label still identifies the complete connection and can remove it.
            link = details.get_by_role(
                "button", name=f"{root_name} → control_tool_execution", exact=True
            )
            await link.scroll_into_view_if_needed()
            await link.hover()
            await expect(
                page.get_by_role("tooltip").filter(has_text=root_name)
            ).to_contain_text(f"{root_name} → control_tool_execution")
            previous_links = await links.count()
            await link.click()
            await expect(links).to_have_count(previous_links - 1)
            await canvas.get_by_role("button", name="Undo", exact=True).click()
            await expect(links).to_have_count(previous_links)
            await canvas.locator(".agent-designer-header").hover()
            await expect(page.get_by_role("tooltip")).to_have_count(0)
            await details.scroll_into_view_if_needed()
            await page.screenshot(
                path="screenshots/agent-designer-connections-phone.png",
                animations="disabled",
            )
            await palette.scroll_into_view_if_needed()
            await page.screenshot(
                path="screenshots/agent-designer-catalog-phone.png",
                animations="disabled",
            )
            await page.set_viewport_size({"width": 1440, "height": 900})
            await page.screenshot(
                path="screenshots/agent-designer-connections-desktop.png",
                animations="disabled",
            )
        finally:
            await browser.close()
