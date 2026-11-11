from __future__ import annotations

import re

from playwright.async_api import async_playwright, expect

from tests.audit_theme_contrast import PROBE
from tests.browser_support import (
    BrowserTestServer,
    get_edge_executable,
    wait_turn_completion,
)


async def test_contrast_probe_flags_white_on_white_and_warns_on_gradients():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable()
        )
        try:
            page = await browser.new_page()
            await page.set_content("""
                <main style="background: white; color: #555">
                  <p>Readable text</p>
                  <p style="color: white">White on white</p>
                  <p style="opacity: .3">Faded text</p>
                  <p style="background-image: linear-gradient(white, #999)">Gradient</p>
                </main>
            """)
            result = await page.locator("main").evaluate(PROBE)
            assert result["checked"] == 3
            assert [
                (entry["label"], entry["status"]) for entry in result["findings"]
            ] == [
                ("White on white", "NOT PASS"),
                ("Faded text", "NOT PASS"),
                ("Gradient", "WARNING"),
            ]
            assert result["findings"][0]["ratio"] == 1
        finally:
            await browser.close()


async def test_theme_switch_and_commands(tmp_path):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            with BrowserTestServer(tmp_path, module="tests.run_projects_agent") as url:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                switch = page.get_by_role("switch", name="Light mode")
                body = page.locator("body")
                textarea = page.locator(".chat-input-textarea textarea")
                sun = switch.locator(".theme-sun circle")
                moon = switch.locator(".theme-moon path")
                await expect(switch).to_have_attribute("aria-checked", "false")
                await expect(body).to_have_css("background-color", "rgb(9, 9, 11)")
                await expect(sun).to_have_css("fill", "none")
                await expect(moon).to_have_css("fill", re.compile("#theme-moon-fill"))

                await textarea.fill("A draft in progress")
                await switch.press("Space")
                await expect(switch).to_have_attribute("aria-checked", "true")
                await expect(body).to_have_css("background-color", "rgb(248, 250, 252)")
                await expect(textarea).to_have_css("color", "rgb(24, 35, 53)")
                send = page.locator(".exmekh-chat-input-box .q-btn.bg-dark")
                await expect(send).to_have_css("background-color", "rgb(255, 255, 255)")
                await expect(send).to_have_css("color", "rgb(24, 35, 53)")
                await expect(sun).to_have_css("fill", re.compile("#theme-sun-fill"))
                await expect(moon).to_have_css("fill", "none")
                await expect(textarea).to_have_value("A draft in progress")

                await page.reload(wait_until="networkidle")
                await expect(switch).to_have_attribute("aria-checked", "true")
                for command, light in (
                    ("  ChIuDi TuTtO  ", False),
                    ("  sMaRmElLa  ", True),
                    ("chiudi tutto", False),
                    ("  ApRi TuTtO  ", True),
                    ("chiudi tutto", False),
                ):
                    await textarea.fill(command)
                    await textarea.press("Enter")
                    await expect(switch).to_have_attribute(
                        "aria-checked", str(light).lower()
                    )
                    await expect(body).to_have_css(
                        "background-color",
                        "rgb(248, 250, 252)" if light else "rgb(9, 9, 11)",
                    )
                    await expect(textarea).to_have_value("")
                    await expect(page.locator(".msg-user")).to_have_count(0)
                calls = await (await page.request.get(url + "/test/state")).json()
                assert calls["calls"] == []

                await page.reload(wait_until="networkidle")
                await expect(switch).to_have_attribute("aria-checked", "false")
                await textarea.fill("Smarmella, apri tutto e chiudi tutto!")
                await textarea.press("Enter")
                await wait_turn_completion(page)
                await expect(page.locator(".msg-user")).to_contain_text(
                    "Smarmella, apri tutto e chiudi tutto!"
                )
                await expect(switch).to_have_attribute("aria-checked", "false")
        finally:
            await browser.close()
