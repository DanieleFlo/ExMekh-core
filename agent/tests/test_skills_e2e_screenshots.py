from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from fe.qs_bridge.models import ChatMessage, ChatSession, SkillNode, ToolEvent
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.fixture(scope="module")
def fe_server(tmp_path_factory: pytest.TempPathFactory):
    """Sets up test sessions and starts NiceGUI server for Playwright E2E screenshots."""
    data_dir = tmp_path_factory.mktemp("skills-data")
    chats_dir = data_dir / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Active Nested Skills Session
    active_sub_skill = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="active",
        tool_events=[
            ToolEvent(
                tool_name="nested-sub-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
            )
        ],
        is_collapsed=False,
    )
    active_root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="active",
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill")
        ],
        sub_skills=[active_sub_skill],
        is_collapsed=False,
    )
    active_session = ChatSession(
        id="test-skills-active",
        title="Skill Attiva Annidata",
        updated_at=time.time() + 500,
        messages=[
            ChatMessage(
                role="user",
                content="Dimostra l'apertura e l'annidamento multi-livello delle skill gerarchiche.",
            ),
            ChatMessage(
                role="assistant",
                content="Esecuzione in corso all'interno della gerarchia delle skill...",
                reasoning_content="Inizializzo l'albero: apertura di 'mock-skill' (depth 1) e della sotto-skill 'nested-sub-skill' (depth 2).",
                skills=[active_root_skill],
            ),
        ],
    )
    file_active = chats_dir / "test-skills-active.json"
    file_active.write_text(active_session.model_dump_json(indent=2), encoding="utf-8")

    # 2. Prepare Completed / Compacted Skills Session
    comp_sub_skill = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="completed",
        tool_events=[
            ToolEvent(
                tool_name="nested-sub-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
            )
        ],
        is_collapsed=False,
    )
    comp_root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        is_collapsed=True,
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill")
        ],
        sub_skills=[comp_sub_skill],
    )
    comp_session = ChatSession(
        id="test-skills-completed",
        title="Skill Compattata",
        updated_at=time.time() + 400,
        messages=[
            ChatMessage(
                role="user",
                content="Mostrami il riepilogo delle skill completate.",
            ),
            ChatMessage(
                role="assistant",
                content="Tutte le operazioni all'interno delle skill sono state completate con successo.",
                reasoning_content="Esecuzione completata.",
                skills=[comp_root_skill],
            ),
        ],
    )
    file_comp = chats_dir / "test-skills-completed.json"
    file_comp.write_text(comp_session.model_dump_json(indent=2), encoding="utf-8")

    with BrowserTestServer(data_dir) as url:
        yield url


@pytest.mark.asyncio
async def test_nested_skills_animation_and_screenshots(fe_server: str) -> None:
    """E2E Test capturing 3 distinct screenshots:

    1. Active opening state with glowing border and nested sub-skill.
    2. Vertically compacted accordion summary header.
    3. Interactive click-to-expand details view.
    """
    fe_dir = Path(__file__).resolve().parents[1]
    screenshots_dir = fe_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=edge_bin,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 860})
        page = await context.new_page()

        # -------------------------------------------------------------
        # 1. SCREENSHOT 1: Active Skill Opening & Nested Structure
        # -------------------------------------------------------------
        await page.goto(fe_server, wait_until="networkidle")
        await page.get_by_text("Skill Attiva Annidata", exact=True).click()
        await page.wait_for_selector(".skill-container-active", timeout=10000)

        # Allow animation to complete and glowing badge to pulse
        await asyncio.sleep(1.0)

        screenshot_1 = screenshots_dir / "01_skill_opening_nested.png"
        await page.screenshot(path=str(screenshot_1))
        print(f"\n[E2E Screenshot 1]: Saved to {screenshot_1}")
        assert screenshot_1.exists() and screenshot_1.stat().st_size > 0

        # Verify active badge is present
        assert await page.locator(".skill-active-badge").first.is_visible()

        # -------------------------------------------------------------
        # 2. SCREENSHOT 2: Completed & Vertically Compacted State
        # -------------------------------------------------------------
        # Click on the completed session in the sidebar
        completed_chat_item = page.locator("text=Skill Compattata").first
        await completed_chat_item.click()

        # Wait for compacted accordion header
        await page.wait_for_selector(".skill-compact-header", timeout=8000)
        await asyncio.sleep(0.8)

        screenshot_2 = screenshots_dir / "02_skill_compacted.png"
        await page.screenshot(path=str(screenshot_2))
        print(f"[E2E Screenshot 2]: Saved to {screenshot_2}")
        assert screenshot_2.exists() and screenshot_2.stat().st_size > 0

        # -------------------------------------------------------------
        # 3. SCREENSHOT 3: Interactive Click to Expand Details
        # -------------------------------------------------------------
        # Click the compacted header to expand the skill body
        header_bar = page.locator(".skill-compact-header").first
        await header_bar.click()

        # Wait for vertical expansion animation
        await asyncio.sleep(0.8)

        # Verify the detailed actions and nested sub-skill are visible
        assert await page.locator("text=Actions executed in skill:").first.is_visible()

        screenshot_3 = screenshots_dir / "03_skill_expanded_detail.png"
        await page.screenshot(path=str(screenshot_3))
        print(f"[E2E Screenshot 3]: Saved to {screenshot_3}")
        assert screenshot_3.exists() and screenshot_3.stat().st_size > 0

        await browser.close()
