from __future__ import annotations

import asyncio
import re

import httpx
import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def compaction_server():
    yield from start_mock_server(0, "tests.run_compaction_agent")


@pytest.fixture
def progress_server():
    yield from start_mock_server(0, "tests.run_compaction_agent")


async def state(url):
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.get(url + "/test/state")
        response.raise_for_status()
        return response.json()


@pytest.mark.parametrize("completion", ["resume", "stop"])
async def test_compaction_bar_is_in_response_and_survives_reload(
    progress_server, tmp_path, completion
):
    compaction_server = progress_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 920})
            await page.goto(compaction_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await page.request.post(compaction_server + "/test/pause-compaction")
            await send_command(page, "Show compaction progress")
            progress = page.locator(".msg-assistant .compaction-progress")
            await expect(progress).to_be_visible()
            await expect(progress).to_contain_text("Evaluating outputs")
            await expect(progress).to_contain_text(re.compile(r"Elapsed [1-9][0-9]*s"))
            await expect(progress.get_by_role("progressbar")).to_be_visible()
            await page.screenshot(
                path=str(tmp_path / "compaction-progress.png"), full_page=True
            )
            await page.reload(wait_until="networkidle")
            await expect(progress).to_be_visible()
            await page.set_viewport_size({"width": 390, "height": 844})
            close_menu = page.get_by_role(
                "button", name="Close projects menu", exact=True
            )
            await expect(close_menu).to_be_hidden()
            await expect(progress).to_be_visible()
            bounds = await progress.bounding_box()
            assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
            await page.screenshot(
                path=str(tmp_path / "compaction-progress-mobile.png"), full_page=True
            )
            if completion == "stop":
                await page.locator("button:has(i:has-text('stop'))").first.click()
            await page.request.post(compaction_server + "/test/resume-compaction")
            await wait_turn_completion(page)
            await expect(progress).to_have_count(0)
            if completion == "resume":
                await expect(page.locator(".msg-assistant").last).to_contain_text(
                    "COMPACTION_OK"
                )
            else:
                assert (await state(compaction_server))["session"]["messages"][-1][
                    "error"
                ] == "cancelled"
        finally:
            await page.request.post(compaction_server + "/test/resume-compaction")
            await browser.close()


async def test_compaction_preserves_raw_graph_and_reload(compaction_server, tmp_path):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 920})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(compaction_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await send_command(page, "First compaction")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant").last).to_contain_text(
                "COMPACTION_OK"
            )
            data = await state(compaction_server)
            assert data["compactions"] >= 2
            message = data["session"]["messages"][-1]
            assert "ORIGINAL_1" in str(message["raw_history"])
            assert "ORIGINAL_1" not in str(message["model_context"]["history"])
            await page.locator(".sub-tabs-bar .q-tab:has-text('Raw History')").click()
            await page.get_by_role("button", name="Expand all", exact=True).click()
            await expect(
                page.locator("code:visible").filter(has_text="ORIGINAL_1").first
            ).to_be_visible()
            await page.screenshot(
                path=str(tmp_path / "raw-archive.png"), full_page=True
            )
            view_switch = page.get_by_role(
                "switch", name="Show current model context", exact=True
            )
            await view_switch.click()
            await expect(view_switch).to_have_attribute("aria-checked", "true")
            await expect(
                page.locator("code:visible").filter(has_text="ORIGINAL_1")
            ).to_have_count(0)
            await page.get_by_role("button", name="Expand all", exact=True).click()
            await expect(
                page.locator("code:visible").filter(has_text="compact_history").first
            ).to_be_visible()
            await page.screenshot(
                path=str(tmp_path / "raw-context.png"), full_page=True
            )
            await page.set_viewport_size({"width": 390, "height": 844})
            close_menu = page.get_by_role(
                "button", name="Close projects menu", exact=True
            )
            if await close_menu.is_visible():
                await close_menu.click()
            await view_switch.click()
            await expect(view_switch).to_have_attribute("aria-checked", "false")
            await expect(
                page.locator("code:visible").filter(has_text="ORIGINAL_1").first
            ).to_be_visible()
            switch_bounds = await page.locator(
                ".raw-history-context-switch:visible"
            ).bounding_box()
            assert (
                switch_bounds["x"] >= 0
                and switch_bounds["x"] + switch_bounds["width"] <= 390
            )
            await view_switch.focus()
            await page.keyboard.press("Space")
            await expect(view_switch).to_have_attribute("aria-checked", "true")
            await page.screenshot(
                path=str(tmp_path / "raw-context-mobile.png"), full_page=True
            )
            await page.set_viewport_size({"width": 1360, "height": 920})
            await page.locator(".sub-tabs-bar .q-tab:has-text('Graph')").click()
            await expect(page.get_by_text("Graph View", exact=True)).to_be_visible()
            compaction_points = page.locator("button[data-kind='compaction']")
            expected_points = len(message["model_context"]["compactions"])
            assert expected_points > 0
            await expect(compaction_points).to_have_count(expected_points)
            await compaction_points.first.click()
            await expect(page.locator(".graph-bottom-menu-open")).to_contain_text(
                "History compaction"
            )
            await page.screenshot(path=str(tmp_path / "graph.png"), full_page=True)
            assert "ORIGINAL_1" in str(data["graph"])
            await page.reload(wait_until="networkidle")
            await send_command(page, "Continue after reload")
            await wait_turn_completion(page)
            data = await state(compaction_server)
            assert "ORIGINAL_1" in str(data["graph"])
            assert "ORIGINAL_1" not in str(data["calls"][-4:])
            await page.locator(".sub-tabs-bar .q-tab:has-text('Graph')").click()
            total_operations = sum(
                len(item["model_context"]["compactions"])
                for item in data["session"]["messages"]
                if item.get("model_context")
            )
            await expect(compaction_points).to_have_count(total_operations)
            assert not errors
        finally:
            await browser.close()


async def test_global_threshold_validation_and_save(compaction_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            await page.goto(compaction_server, wait_until="networkidle")
            await page.locator("button:has(i:has-text('tune'))").first.click()
            threshold = page.get_by_label(
                "Automatic context compaction (%)", exact=True
            )
            await expect(threshold).to_have_value("80")
            await threshold.fill("100")
            await page.get_by_role(
                "button", name="Save global settings", exact=True
            ).click()
            assert (await state(compaction_server))["threshold"] == 80
            await threshold.fill("65")
            await page.get_by_role(
                "button", name="Save global settings", exact=True
            ).click()
            await expect(
                page.get_by_text("Global settings saved", exact=True)
            ).to_be_visible()
            assert (await state(compaction_server))["threshold"] == 65
            await page.reload(wait_until="networkidle")
            await page.locator("button:has(i:has-text('tune'))").first.click()
            await expect(
                page.get_by_label("Automatic context compaction (%)", exact=True)
            ).to_have_value("65")
        finally:
            await browser.close()


async def test_failed_compaction_is_visible_and_can_be_retried(
    compaction_server, tmp_path
):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page()
        try:
            await page.goto(compaction_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await page.request.post(compaction_server + "/test/summary-error/true")
            before = await state(compaction_server)
            await send_command(page, "Fail compaction")
            await wait_turn_completion(page)
            await expect(
                page.get_by_text(
                    "Automatic history compaction failed", exact=False
                ).first
            ).to_be_visible()
            failed = await state(compaction_server)
            assert len(failed["calls"]) == len(before["calls"]) + 1
            assert failed["session"]["messages"][-1]["error"].startswith(
                "Automatic history compaction failed"
            )
            await expect(page.locator(".compaction-progress")).to_have_count(0)
            await expect(
                page.get_by_text("Mock compaction unavailable", exact=False).first
            ).to_be_visible()
            assert "ORIGINAL_" in str(failed["session"]["messages"][-1]["raw_history"])
            assert "ORIGINAL_" in str(failed["graph"])
            await page.screenshot(
                path=str(tmp_path / "compaction-error.png"), full_page=True
            )
            await page.request.post(compaction_server + "/test/summary-error/false")
            await page.reload(wait_until="networkidle")
            await send_command(page, "Retry compaction")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant").last).to_contain_text(
                "COMPACTION_OK"
            )
            resumed = await state(compaction_server)
            assert resumed["session"]["messages"][-1]["model_context"] is not None
            assert "ORIGINAL_" in str(resumed["graph"])
        finally:
            await page.request.post(compaction_server + "/test/summary-error/false")
            await browser.close()


async def test_switch_during_compaction_and_reconnected_stream(compaction_server):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(compaction_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await page.request.post(compaction_server + "/test/pause-compaction")
            await send_command(page, "Paused compaction")

            async def wait_for_summary():
                while not (await state(compaction_server))["compaction_waiting"]:
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(wait_for_summary(), 5)
            await page.locator(".sub-tabs-bar .q-tab:has-text('Raw History')").click()
            switch = page.get_by_role("switch", name="Show current model context")
            await switch.click()
            await expect(switch).to_have_attribute("aria-checked", "true")
            await page.reload(wait_until="networkidle")
            await page.locator(".sub-tabs-bar .q-tab:has-text('Raw History')").click()
            await page.request.post(compaction_server + "/test/resume-compaction")
            await wait_turn_completion(page)
            await expect(
                page.locator("code:visible").filter(has_text="ORIGINAL_").first
            ).to_be_visible()
            switch = page.get_by_role("switch", name="Show current model context")
            await switch.click()
            await expect(switch).to_have_attribute("aria-checked", "true")
            await expect(
                page.locator("code:visible").filter(has_text="ORIGINAL_")
            ).to_have_count(0)
            assert not errors
        finally:
            await page.request.post(compaction_server + "/test/resume-compaction")
            await browser.close()
