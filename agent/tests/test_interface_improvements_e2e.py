from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import get_edge_executable, start_mock_server

pytestmark = pytest.mark.serial(reason="Reads and writes the shared system clipboard")


@pytest.fixture(scope="module")
def interface_server():
    yield from start_mock_server(0.0, "tests.run_interface_agent")


async def test_copy_menu_hover_and_single_line_tool_arguments(
    interface_server, tmp_path
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            context = await browser.new_context(
                viewport={"width": 1000, "height": 900},
                permissions=["clipboard-read", "clipboard-write"],
            )
            page = await context.new_page()
            await page.goto(interface_server, wait_until="networkidle")
            user = page.locator(".chat-message-user").first
            copy = user.get_by_role("button", name="Copy message")
            await page.mouse.move(0, 0)
            await expect(copy).to_have_css("opacity", "0")
            await user.hover()
            await expect(copy).to_have_css("opacity", "1")
            await copy.click()
            await page.wait_for_function(
                "navigator.clipboard.readText().then(t => t === 'Primo messaggio **utente**')"
            )
            preview = page.locator(".tool-arguments-preview").first
            await expect(preview).to_contain_text("command")
            bounds = await preview.evaluate(
                "el => ({overflow: el.scrollWidth > el.clientWidth, ellipsis: getComputedStyle(el).textOverflow, nowrap: getComputedStyle(el).whiteSpace, height: el.getBoundingClientRect().height})"
            )
            assert (
                bounds["overflow"]
                and bounds["ellipsis"] == "ellipsis"
                and bounds["nowrap"] == "nowrap"
            )
            assert bounds["height"] < 24
            assert (
                await page.locator(".tool-event-card").first.evaluate(
                    "el => el.getBoundingClientRect().height"
                )
                < 30
            )
            for history in [False, True]:
                for format in ["JSON", "Markdown"]:
                    await page.get_by_role("button", name="Chat menu").click()
                    toggle = page.get_by_role("switch", name="Include subskill history")
                    if (
                        await toggle.get_attribute("aria-checked")
                        != str(history).lower()
                    ):
                        await toggle.click()
                    await expect(toggle).to_have_attribute(
                        "aria-checked", str(history).lower()
                    )
                    await page.evaluate(
                        "navigator.clipboard.writeText('pending export')"
                    )
                    await page.get_by_text(f"Copy as {format}", exact=True).click()
                    await page.wait_for_function(
                        "navigator.clipboard.readText().then(t => t !== 'pending export')"
                    )
                    await expect(
                        page.get_by_text(
                            f"Chat copied as {format.upper()}", exact=True
                        ).last
                    ).to_be_visible()
                    copied = await page.evaluate("navigator.clipboard.readText()")
                    assert "Seconda risposta" in copied
                    assert ("Nested transcript" in copied) == history
                    if format == "JSON":
                        assert len(json.loads(copied)["messages"]) == 4
            await page.screenshot(path=str(Path(tmp_path) / "interface.png"))
            await page.get_by_role("button", name="Show child").click()
            await page.get_by_role("button", name="Chat menu").click()
            await page.evaluate(
                "navigator.clipboard.writeText('pending child-view export')"
            )
            await page.get_by_text("Copy as JSON", exact=True).click()
            await page.wait_for_function(
                "navigator.clipboard.readText().then(t => t.includes('Seconda risposta'))"
            )
            assert (
                len(
                    json.loads(await page.evaluate("navigator.clipboard.readText()"))[
                        "messages"
                    ]
                )
                == 4
            )
        finally:
            await browser.close()


async def test_export_includes_the_response_still_streaming(interface_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            context = await browser.new_context(
                permissions=["clipboard-read", "clipboard-write"]
            )
            page = await context.new_page()
            await page.goto(interface_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="Start partial response", exact=True
            ).click()
            await expect(page.locator(".chat-message-streaming")).to_contain_text(
                "Partial response"
            )
            await page.get_by_role("button", name="Chat menu").click()
            await page.evaluate("navigator.clipboard.writeText('pending live export')")
            await page.get_by_text("Copy as JSON", exact=True).click()
            await page.wait_for_function(
                "navigator.clipboard.readText().then(t => t !== 'pending live export')"
            )
            data = json.loads(await page.evaluate("navigator.clipboard.readText()"))
            assert data["messages"][-1]["content"] == "Partial response"
            assert data["messages"][-1]["is_streaming"]
        finally:
            await browser.close()
