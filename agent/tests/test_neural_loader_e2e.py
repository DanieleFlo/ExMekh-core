from __future__ import annotations

from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from exmekh_agent.exmekh_bridge.providers.context_usage import get_qwen_tokenizer
from tests.browser_support import BrowserTestServer, get_edge_executable

TEXT_CAPTURE = """
window.loaderTexts = new Set();
window.loaderDraws = 0;
const original = CanvasRenderingContext2D.prototype.fillText;
CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
    if (this.canvas.classList.contains('input-neural-loader')) {
        window.loaderTexts.add(text);
        window.loaderDraws++;
    }
    return original.call(this, text, ...args);
};
"""


@pytest.fixture(scope="module")
def loader_server(tmp_path_factory):
    with BrowserTestServer(
        tmp_path_factory.mktemp("neural-loader"), module="tests.run_neural_loader_agent"
    ) as url:
        yield url


async def test_neural_loader_animates_library_tokens_resizes_and_stops(loader_server):
    tokenizer = get_qwen_tokenizer()
    vocabulary = set(tokenizer.get_id_vocabulary())
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1100, "height": 600})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.add_init_script(TEXT_CAPTURE)
            await page.goto(loader_server, wait_until="networkidle")
            canvas = page.locator(".input-neural-loader")
            await expect(canvas).to_be_hidden()
            await page.get_by_role("button", name="Start generation").click()
            await expect(canvas).to_be_visible()
            await expect(canvas).to_have_attribute("aria-label", "Generating response")
            await page.wait_for_function("window.loaderTexts.size > 20")
            tokens = await page.evaluate(
                "Array.from(window.loaderTexts).filter(text => !/^-?\\d+$/.test(text))"
            )
            assert tokens and set(tokens) <= vocabulary
            image = await canvas.evaluate("el => el.toDataURL()")
            await page.wait_for_function(
                "image => document.querySelector('.input-neural-loader').toDataURL() !== image",
                arg=image,
            )
            screenshots = Path("screenshots")
            screenshots.mkdir(exist_ok=True)
            await page.locator(".chat-composer").screenshot(
                path=str(screenshots / "neural-loader-desktop.png")
            )
            await page.set_viewport_size({"width": 390, "height": 700})
            await page.wait_for_function(
                "() => {const c = document.querySelector('.input-neural-loader'); return c.width === Math.round(c.clientWidth * devicePixelRatio); }"
            )
            bounds = await canvas.bounding_box()
            assert bounds["height"] == 34 and bounds["width"] <= 390
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            )
            await page.locator(".chat-composer").screenshot(
                path=str(screenshots / "neural-loader-mobile.png")
            )
            await page.get_by_role("button", name="Complete generation").click()
            await expect(canvas).to_be_hidden()
            await page.wait_for_timeout(100)
            draws = await page.evaluate("window.loaderDraws")
            await page.wait_for_timeout(200)
            assert await page.evaluate("window.loaderDraws") == draws
            await page.get_by_role("button", name="Start generation").click()
            await expect(canvas).to_be_visible()
            await page.wait_for_function(
                "draws => window.loaderDraws > draws", arg=draws
            )
            await page.locator("button:has(i:has-text('stop'))").click()
            await expect(canvas).to_be_hidden()
            await page.get_by_role("button", name="Start generation").click()
            await expect(canvas).to_be_visible()
            await page.get_by_role("button", name="Remove composer").click()
            await expect(canvas).to_have_count(0)
            await page.wait_for_timeout(100)
            draws = await page.evaluate("window.loaderDraws")
            await page.wait_for_timeout(200)
            assert await page.evaluate("window.loaderDraws") == draws
            assert errors == []
        finally:
            await browser.close()


async def test_neural_loader_respects_reduced_motion(loader_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(reduced_motion="reduce")
            await page.add_init_script(TEXT_CAPTURE)
            await page.goto(loader_server, wait_until="networkidle")
            await page.get_by_role("button", name="Start generation").click()
            canvas = page.locator(".input-neural-loader")
            await expect(canvas).to_be_visible()
            await page.wait_for_function("window.loaderDraws > 0")
            await page.wait_for_timeout(100)
            image = await canvas.evaluate("el => el.toDataURL()")
            await page.wait_for_timeout(200)
            assert await canvas.evaluate("el => el.toDataURL()") == image
            await page.emulate_media(reduced_motion="no-preference")
            await page.wait_for_function(
                "image => document.querySelector('.input-neural-loader').toDataURL() !== image",
                arg=image,
            )
        finally:
            await browser.close()
