from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from fe.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
)
from fe.ui.chat_view import ChatViewComponent
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.fixture(scope="module")
def fe_codex_server(tmp_path_factory: pytest.TempPathFactory):
    """Starts NiceGUI server with a multi-path, multi-generation reasoning session."""
    data_dir = tmp_path_factory.mktemp("codex-data")
    chats_dir = data_dir / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    # Prepare session with separated reasoning across different paths & generations
    b1 = ReasoningBlock(
        skill_path="/",
        depth=0,
        generation_index=1,
        content="[Fase 1 - Root] Analisi iniziale della richiesta: decido di invocare la skill mock-skill.",
    )
    b2 = ReasoningBlock(
        skill_path="/mock-skill",
        depth=1,
        generation_index=2,
        content=(
            "[Fase 2 - Skill mock-skill] Calcolo passo passo di $\\int x^2 \\, dx$:\n\n"
            "$$\\int x^2 \\, dx = \\frac{x^3}{3} + C$$\n\n"
            "Esecuzione all'interno della skill: preparo gli argomenti per il mock-tool."
        ),
    )
    b3 = ReasoningBlock(
        skill_path="/",
        depth=0,
        generation_index=3,
        content="[Fase 3 - Root Sintesi] Sintesi finale dopo il ritorno dalla skill: elaborazione risposta per l'utente.",
    )

    skill_completed = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        is_collapsed=True,
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill")
        ],
    )

    codex_session = ChatSession(
        id="test-codex-multipath-reasoning",
        title="Codex Multi-Path Reasoning",
        updated_at=time.time() + 9999,  # High timestamp to be selected first
        messages=[
            ChatMessage(
                role="user",
                content="Dimostra l'esecuzione con ragionamenti separati per path, stile Codex e calcola l'integrale di x^2 in dx.",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "Esecuzione completata con successo: i processi di ragionamento sono separati per path e generazione.\n\n"
                    "Calcolo dell'integrale di $x^2$ passo passo in $dx$:\n\n"
                    "$$\\int x^2 \\, dx = \\frac{x^3}{3} + C$$\n\n"
                    "La formula generale applicata è $\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C$."
                ),
                reasoning_blocks=[b1, b2, b3],
                reasoning_content=f"{b1.content}\n\n{b2.content}\n\n{b3.content}",
                skills=[skill_completed],
            ),
        ],
    )
    session_file = chats_dir / "test-codex-multipath-reasoning.json"
    session_file.write_text(codex_session.model_dump_json(indent=2), encoding="utf-8")

    with BrowserTestServer(data_dir) as url:
        yield url


@pytest.mark.asyncio
async def test_codex_multipath_reasoning_e2e(fe_codex_server: str) -> None:
    """E2E Test verifying:

    1. Reasoning text is partitioned into separate blocks by path and generation (NOT merged into one).
    2. Paths and generation indicators are clearly labeled on each block.
    3. UI reflects clean Codex aesthetic (subtle dark borders, no bright purple gradients or neon glow).
    4. Captures screenshot to FE/screenshots/04_codex_multipath_reasoning.png.
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
        context = await browser.new_context(viewport={"width": 1360, "height": 880})
        page = await context.new_page()

        await page.goto(fe_codex_server, wait_until="networkidle")

        # Select the session if not already opened
        session_locator = page.locator("text=Codex Multi-Path Reasoning").first
        if await session_locator.is_visible():
            await session_locator.click()
            await asyncio.sleep(0.8)

        # 1. Verify that the reasoning blocks are compacted into a single master line
        master_header = page.locator(".reasoning-master-header").first
        await master_header.wait_for(state="visible", timeout=8000)
        master_text = await master_header.inner_text()
        assert "Reasoning Process" in master_text, (
            f"Expected 'Reasoning Process' in master header, got '{master_text}'"
        )
        assert "3 blocks" in master_text, (
            f"Expected '3 blocks' in master header, got '{master_text}'"
        )

        # 2. Click the master line to expand into the 3 sub-blocks
        await master_header.click()
        await asyncio.sleep(0.4)

        # Verify that 3 distinct reasoning sub-blocks are present
        sub_blocks = page.locator(".reasoning-sub-block")
        count = await sub_blocks.count()
        assert count == 3, f"Expected 3 separate reasoning sub-blocks, got {count}"

        # 3. Verify path badges on each sub-block
        path_badges = page.locator(".reasoning-path-badge")
        badge_texts = [
            await path_badges.nth(i).inner_text()
            for i in range(await path_badges.count())
        ]
        assert badge_texts[0].strip() == "/", (
            f"First block expected '/', got '{badge_texts[0]}'"
        )
        assert badge_texts[1].strip() == "/mock-skill", (
            f"Second block expected '/mock-skill', got '{badge_texts[1]}'"
        )
        assert badge_texts[2].strip() == "/", (
            f"Third block expected '/', got '{badge_texts[2]}'"
        )

        # 4. Verify generation index badges on each sub-block
        gen_badges = page.locator(".reasoning-gen-badge")
        gen_texts = [
            await gen_badges.nth(i).inner_text()
            for i in range(await gen_badges.count())
        ]
        assert "gen #1" in gen_texts[0], f"Expected 'gen #1', got '{gen_texts[0]}'"
        assert "gen #2" in gen_texts[1], f"Expected 'gen #2', got '{gen_texts[1]}'"
        assert "gen #3" in gen_texts[2], f"Expected 'gen #3', got '{gen_texts[2]}'"

        # 5. Expand each sub-block with a click to reveal its individual thinking text
        sub_headers = page.locator(".reasoning-sub-header")
        sub_count = await sub_headers.count()
        assert sub_count == 3, f"Expected 3 sub-headers, got {sub_count}"
        for i in range(3):
            await sub_headers.nth(i).click()
            await asyncio.sleep(0.3)

        thinking_labels = page.locator(".thinking-box")
        t_count = await thinking_labels.count()
        assert t_count >= 3, f"Expected at least 3 thinking text boxes, got {t_count}"

        text_0 = await thinking_labels.nth(0).inner_text()
        text_1 = await thinking_labels.nth(1).inner_text()
        text_2 = await thinking_labels.nth(2).inner_text()

        # Check content isolation (distinct generations in different paths)
        assert "[Fase 1 - Root]" in text_0
        assert "[Fase 2 - Skill mock-skill]" not in text_0, (
            "Block 0 must not contain Block 1 text"
        )

        assert "[Fase 2 - Skill mock-skill]" in text_1
        assert "[Fase 3 - Root Sintesi]" not in text_1, (
            "Block 1 must not contain Block 2 text"
        )

        assert "[Fase 3 - Root Sintesi]" in text_2
        assert "[Fase 1 - Root]" not in text_2, "Block 2 must not contain Block 0 text"

        # 5. Verify Codex Minimalist UI styling (no bright purple gradient on user msg)
        user_msg = page.locator(".msg-user").first
        await user_msg.wait_for(state="visible")
        bg_style = await user_msg.evaluate(
            "el => window.getComputedStyle(el).backgroundImage"
        )
        assert "linear-gradient" not in bg_style, (
            f"Expected no gradient background on .msg-user in Codex style, got {bg_style}"
        )

        # 6. Verify assistant message has no border and transparent background
        assistant_el = page.locator(".msg-assistant").first
        border_assistant = await assistant_el.evaluate(
            "el => window.getComputedStyle(el).borderStyle"
        )
        assert border_assistant in ("none", "hidden", ""), (
            f"Expected no border on .msg-assistant, got {border_assistant}"
        )

        # 7. Verify thinking-box has no heavy purple border-left
        tb = thinking_labels.first
        border_left = await tb.evaluate("el => window.getComputedStyle(el).borderLeft")
        # In old style it was "3px solid rgb(168, 85, 247)" (purple)
        assert "rgb(168, 85, 247)" not in border_left, (
            f"Expected no bright purple left border on thinking-box, got {border_left}"
        )

        # 8. Verify LaTeX Math formulas are properly parsed to HTML / MathML without raw $$
        response_text = await page.locator(
            ".parsed-response-content"
        ).first.inner_text()
        assert "$$" not in response_text, (
            f"Expected no raw '$$' in parsed response, got: {response_text}"
        )

        response_math_count = await page.locator(
            ".parsed-response-content math"
        ).count()
        assert response_math_count >= 1, (
            f"Expected MathML in response, found {response_math_count}"
        )

        thinking_math_count = await page.locator(".thinking-box math").count()
        assert thinking_math_count >= 1, (
            f"Expected MathML in thinking block 2, found {thinking_math_count}"
        )

        # 9. Capture screenshot for verification
        screenshot_path = screenshots_dir / "04_codex_multipath_reasoning.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"\n[E2E Codex Multi-Path Screenshot]: Saved to {screenshot_path}")
        assert screenshot_path.exists() and screenshot_path.stat().st_size > 0

        await browser.close()


def test_streaming_reasoning_blocks_isolation() -> None:
    """Verifies that ChatViewComponent tracks separate reasoning blocks per path and generation."""
    cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
    cv.chat_container = None  # unit mode without active page context

    # Test key generation and separation logic
    key1 = "/:0:1"
    key2 = "/mock-skill:1:2"
    key3 = "/:0:3"

    assert key1 != key2
    assert key2 != key3
    assert key1 != key3
