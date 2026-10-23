from __future__ import annotations

import json

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import get_edge_executable, start_mock_server


@pytest.fixture(scope="module")
def output_server():
    yield from start_mock_server(0, "tests.run_todo_agent")


async def test_user_opens_tool_and_skill_outputs_scrolls_and_closes(output_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            await page.goto(output_server, wait_until="networkidle")
            rows = page.locator(".tool-event-card").filter(has_text="Tool: bash")
            panels = page.locator(".tool-output-panel")
            await expect(panels.filter(has_text="FIRST RESULT")).not_to_be_visible()
            await rows.nth(0).click()
            await expect(panels.filter(has_text="FIRST RESULT")).to_be_visible()
            await expect(panels.filter(has_text="FIRST RESULT")).to_contain_text(
                "FIRST RESULT"
            )
            metrics = await panels.filter(has_text="FIRST RESULT").evaluate(
                "element => ({height: element.clientHeight, total: element.scrollHeight})"
            )
            assert metrics["height"] <= 240 and metrics["total"] > metrics["height"]
            await panels.filter(has_text="FIRST RESULT").evaluate(
                "element => element.scrollTop = element.scrollHeight"
            )
            assert await panels.filter(has_text="FIRST RESULT").evaluate(
                "element => element.scrollTop > 0"
            )
            await rows.nth(1).focus()
            await rows.nth(1).press("Enter")
            await expect(panels.filter(has_text="SECOND RESULT")).to_be_visible()
            await expect(panels.filter(has_text="SECOND RESULT")).to_contain_text(
                '"result": "SECOND RESULT"'
            )
            await panels.filter(has_text="FIRST RESULT").evaluate(
                "element => element.scrollTop = 0"
            )
            await page.screenshot(
                path="screenshots/tool-outputs-desktop.png",
                full_page=True,
                animations="disabled",
            )
            await rows.nth(0).click()
            await expect(panels.filter(has_text="FIRST RESULT")).not_to_be_visible()
            skill_header = page.locator(".skill-compact-header").filter(
                has_text="analysis"
            )
            await expect(skill_header).to_have_attribute("aria-expanded", "false")
            await skill_header.focus()
            await skill_header.press("Enter")
            await expect(skill_header).to_have_attribute("aria-expanded", "true")
            await skill_header.press("Space")
            await expect(skill_header).to_have_attribute("aria-expanded", "false")
            await skill_header.click()
            await page.locator(".skill-output-expansion .q-item").first.click()
            await expect(page.locator(".skill-output-panel")).to_contain_text(
                "SKILL FINAL RESULT"
            )
            await (
                page.locator(".tool-event-card")
                .filter(has_text="Tool: calculate")
                .click()
            )
            await expect(panels.filter(has_text='"answer": 42')).to_contain_text(
                '"answer": 42'
            )
            await page.locator(".sidebar-item-selected").first.click()
            await expect(skill_header).to_have_attribute("aria-expanded", "true")
            await expect(page.locator(".skill-output-panel")).to_be_visible()
            await expect(panels.filter(has_text='"answer": 42')).to_be_visible()
            await expect(panels.filter(has_text="FIRST RESULT")).not_to_be_visible()
            state = await (await page.request.get(output_server + "/test/state")).json()
            assert state["session"]["messages"][-1]["skills"][0]["is_collapsed"] is True
            await page.screenshot(
                path="screenshots/skill-output-desktop.png",
                full_page=True,
                animations="disabled",
            )
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".tool-output-panel").first).not_to_be_visible()
        finally:
            await browser.close()


async def test_raw_history_multiline_and_json_output_are_readable_and_copy_stays_original(
    output_server,
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            context = await browser.new_context(
                permissions=["clipboard-read", "clipboard-write"]
            )
            page = await context.new_page()
            await page.goto(output_server, wait_until="networkidle")
            await page.get_by_role("tab", name="Raw History", exact=True).click()
            readable = page.locator(
                ".raw-history-container:visible .json-readable-string"
            ).filter(has_text="FIRST RESULT")
            await expect(readable).to_be_visible()
            await expect(readable).to_contain_text(
                "Log line 001: completed\nLog line 002: completed"
            )
            await expect(readable).to_contain_text(
                "<header>\n  <name>bash</name>\n  <type>Tool</type>"
            )
            await expect(
                page.locator(".raw-history-container:visible .output-header")
            ).to_have_count(0)
            await readable.scroll_into_view_if_needed()
            await page.screenshot(
                path="screenshots/raw-history-desktop.png",
                full_page=True,
                animations="disabled",
            )
            await page.get_by_role("button", name="Copy full JSON", exact=True).click()
            copied = await page.evaluate("navigator.clipboard.readText()")
            assert "<header>" in copied and "FIRST RESULT" in copied
        finally:
            await browser.close()


@pytest.mark.parametrize(
    "viewport",
    [
        {"width": 390, "height": 844},
        {"width": 768, "height": 1024},
        {"width": 1366, "height": 900},
    ],
    ids=["phone", "tablet", "desktop"],
)
async def test_all_tabs_responsive_and_composer_remains_usable(output_server, viewport):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(
                viewport=viewport, has_touch=viewport["width"] < 900
            )
            await page.goto(output_server, wait_until="networkidle")
            for tab in ["Chat", "Raw History", "Graph"]:
                await page.get_by_role("tab", name=tab, exact=True).click()
                await expect(page.locator(".chat-input-textarea")).to_be_visible()
                box = await page.locator(".chat-composer").bounding_box()
                assert (
                    box["x"] >= 0 and box["x"] + box["width"] <= viewport["width"] + 1
                )
                assert box["y"] + box["height"] <= viewport["height"] + 1
                assert await page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                await page.screenshot(
                    path=f"screenshots/{tab.lower().replace(' ', '-')}-{viewport['width']}.png",
                    full_page=True,
                    animations="disabled",
                )
            await page.locator(".graph-node").first.click()
            await expect(page.locator(".graph-bottom-menu")).to_be_visible()
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            )
            if viewport["width"] == 390:
                await page.get_by_role(
                    "button", name="Open projects menu", exact=True
                ).click()
                await expect(
                    page.get_by_role("button", name="New project", exact=True)
                ).to_be_visible()
                await page.get_by_role("button", name="New project", exact=True).click()
                dialog = (
                    page.get_by_role("dialog").filter(has_text="Create project").last
                )
                await expect(
                    dialog.get_by_label("Project folder", exact=True)
                ).to_be_visible()
                await page.screenshot(
                    path="screenshots/new-project-phone.png",
                    full_page=True,
                    animations="disabled",
                )
                await dialog.get_by_role("button", name="Cancel", exact=True).click()
                await page.get_by_role(
                    "button", name="Close projects menu", exact=True
                ).click()
                await page.get_by_role(
                    "button", name="Chat agent settings", exact=True
                ).click()
                chat_dialog = (
                    page.get_by_role("dialog").filter(has_text="Save chat agent").last
                )
                await chat_dialog.get_by_role(
                    "button", name="Custom", exact=True
                ).click()
                canvas = (
                    page.locator(".agent-designer-dialog")
                    .filter(has=page.locator(".agent-board-node"))
                    .last
                )
                await expect(canvas.locator(".agent-board-viewport")).to_be_visible()
                await expect(canvas.locator(".agent-designer-shell")).to_have_css(
                    "width", "390px"
                )
                bounds = await canvas.locator(".agent-board-viewport").bounding_box()
                assert bounds["x"] + bounds["width"] <= viewport["width"]
                await page.screenshot(
                    path="screenshots/agent-designer-phone.png",
                    full_page=True,
                    animations="disabled",
                )
                await canvas.get_by_role("button", name="Cancel", exact=True).click()
                await chat_dialog.get_by_role(
                    "button", name="Cancel", exact=True
                ).click()
        finally:
            await browser.close()


@pytest.mark.serial(reason="Reads and writes the shared system clipboard")
async def test_raw_history_image_data_buttons_copy_original_values(
    output_server, tmp_path
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            context = await browser.new_context(
                permissions=["clipboard-read", "clipboard-write"],
                viewport={"width": 1360, "height": 900},
            )
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(output_server, wait_until="networkidle")
            await page.get_by_role("tab", name="Raw History", exact=True).click()
            raw = page.locator(".raw-history-container:visible")
            buttons = raw.get_by_role("button", name="Copy image data", exact=True)
            await expect(buttons).to_have_count(2)
            assert "data:image/png;base64," not in await raw.inner_text()
            await expect(raw).to_contain_text(
                "These images are previews from generate-image."
            )
            originals = await buttons.evaluate_all(
                "elements => elements.map(el => el.dataset.imageData)"
            )
            assert all(len(value) > 10000 for value in originals)
            assert originals[0] != originals[1]
            await page.evaluate("navigator.clipboard.writeText('before image copy')")
            await buttons.first.click()
            await page.wait_for_function(
                "expected => navigator.clipboard.readText().then(value => value === expected)",
                arg=originals[0],
            )
            await expect(
                page.get_by_text("Image data copied", exact=True)
            ).to_be_visible()
            await buttons.nth(1).focus()
            await buttons.nth(1).press("Enter")
            await page.wait_for_function(
                "expected => navigator.clipboard.readText().then(value => value === expected)",
                arg=originals[1],
            )
            await page.get_by_role("button", name="Copy full JSON", exact=True).click()
            await page.wait_for_function(
                "navigator.clipboard.readText().then(value => value.startsWith('['))"
            )
            copied = json.loads(await page.evaluate("navigator.clipboard.readText()"))
            attachment = next(
                item["attachments"] for item in copied if item.get("attachments")
            )
            assert [
                part["data"] for part in attachment["parts"] if part["type"] == "image"
            ] == originals
            await buttons.first.scroll_into_view_if_needed()
            await page.screenshot(
                path=str(tmp_path / "raw-history-copy-image-data.png")
            )
            await page.reload(wait_until="networkidle")
            await page.get_by_role("tab", name="Raw History", exact=True).click()
            await expect(buttons).to_have_count(2)
            assert "data:image/png;base64," not in await raw.inner_text()
            await buttons.first.click()
            await page.wait_for_function(
                "expected => navigator.clipboard.readText().then(value => value === expected)",
                arg=originals[0],
            )
            assert not errors
        finally:
            await browser.close()
