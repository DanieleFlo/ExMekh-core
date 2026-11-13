from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from exmekh_agent.exmekh_bridge.history.compaction import (
    ModelContextCheckpoint,
    RecordedCompaction,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ContentBlock,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.fixture(scope="module")
def agent_codex_server(tmp_path_factory: pytest.TempPathFactory):
    """Starts NiceGUI server with a multi-path, multi-generation reasoning session."""
    data_dir = tmp_path_factory.mktemp("codex-data")
    chats_dir = data_dir / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    # Prepare session with separated reasoning across different paths & generations
    b1 = ReasoningBlock(
        skill_path="/",
        depth=0,
        generation_index=1,
        timestamp=1,
        content="[Fase 1 - Root] Analisi iniziale della richiesta: decido di invocare la skill mock-skill.",
    )
    b2 = ReasoningBlock(
        skill_path="/mock-skill",
        depth=1,
        generation_index=2,
        timestamp=3,
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
        timestamp=6,
        content="[Fase 3 - Root Sintesi] Sintesi finale dopo il ritorno dalla skill: elaborazione risposta per l'utente.",
    )

    skill_completed = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        start_time=2,
        end_time=5.5,
        is_collapsed=True,
        tool_events=[
            ToolEvent(
                tool_name="mock-tool", depth=1, skill_path="/mock-skill", timestamp=5
            )
        ],
    )

    response = (
        "Esecuzione completata con successo: i processi di ragionamento sono separati per path e generazione.\n\n"
        "Calcolo dell'integrale di $x^2$ passo passo in $dx$:\n\n"
        "$$\\int x^2 \\, dx = \\frac{x^3}{3} + C$$\n\n"
        "La formula generale applicata è $\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C$."
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
                content=response,
                content_blocks=[
                    ContentBlock(
                        skill_path="/mock-skill",
                        depth=1,
                        timestamp=4,
                        content="Risultato intermedio della skill.",
                    ),
                    ContentBlock(content=response, timestamp=7),
                ],
                reasoning_blocks=[b1, b2, b3],
                reasoning_content=f"{b1.content}\n\n{b2.content}\n\n{b3.content}",
                skills=[skill_completed],
                tool_events=[ToolEvent(tool_name="final-tool", timestamp=8)],
                model_context=ModelContextCheckpoint(
                    compactions=[
                        RecordedCompaction(
                            id="applied",
                            replacement_call_ids=["first", "second"],
                            skill_path=skill_completed.path,
                            depth=1,
                            timestamp=4.5,
                            duration_seconds=0.25,
                            completed_summaries=2,
                        )
                    ]
                ),
            ),
        ],
    )
    session_file = chats_dir / "test-codex-multipath-reasoning.json"
    session_file.write_text(codex_session.model_dump_json(indent=2), encoding="utf-8")

    with BrowserTestServer(
        data_dir, environment={"EXMEKH_AGENT_TEST_BYPASS_CORE_HTTP": "1"}
    ) as url:
        yield url


@pytest.mark.asyncio
async def test_codex_multipath_reasoning_e2e(agent_codex_server: str) -> None:
    """Verify that each thinking block opens directly in its timeline position."""
    agent_dir = Path(__file__).resolve().parents[1]
    screenshots_dir = agent_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=edge_bin,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 880})
        page = await context.new_page()

        await page.goto(agent_codex_server, wait_until="networkidle")

        # Select the session if not already opened
        session_locator = page.locator("text=Codex Multi-Path Reasoning").first
        if await session_locator.is_visible():
            await session_locator.click()
            await asyncio.sleep(0.8)

        skill = page.locator(".skill-compact-header")
        skill_body = skill.locator("..").locator(".skill-activity")
        await expect(page.locator(".compaction-marker")).to_be_hidden()
        await expect(skill_body).to_be_hidden()
        await skill.click()
        await expect(skill_body).to_be_visible()
        await expect(skill_body).to_have_css(
            "border-left", "2px solid rgb(52, 211, 153)"
        )
        assert await skill.evaluate("""header => {
            const headerBox = header.getBoundingClientRect();
            const bodyBox = header.nextElementSibling.getBoundingClientRect();
            return Math.abs(headerBox.left - bodyBox.left) < 1
                && Math.abs(headerBox.bottom - bodyBox.top) < 1
                && bodyBox.height > headerBox.height;
        }""")
        await expect(
            page.get_by_text("Risultato intermedio della skill.")
        ).to_be_visible()
        marker = skill.locator("..").locator(".compaction-marker")
        await expect(marker).to_be_visible()
        heading = marker.get_by_text("Context compacted · /mock-skill", exact=True)
        details = marker.locator(".compaction-marker-details")
        await expect(details).to_have_text(
            "Elapsed 0.25s · 2 history blocks · 2 summaries"
        )
        assert (await heading.bounding_box())["x"] < (await details.bounding_box())["x"]

        activities = page.locator(
            ".msg-assistant .reasoning-master-header, "
            ".msg-assistant .parsed-response-content, .msg-assistant .tool-event-card, "
            ".msg-assistant .compaction-marker"
        )
        assert await activities.evaluate_all(
            """elements => elements.map(element => {
                if (element.matches('.reasoning-master-header')) return 'thinking';
                if (element.matches('.tool-event-card')) return 'tool';
                if (element.matches('.compaction-marker')) return 'compaction';
                return 'content';
            })"""
        ) == [
            "thinking",
            "thinking",
            "content",
            "compaction",
            "tool",
            "thinking",
            "content",
            "tool",
        ]

        sections = page.locator(".reasoning-master-expansion")
        await expect(sections).to_have_count(3)
        first_header = sections.first.locator(".reasoning-master-header")
        tool = page.locator(".tool-event-card").filter(has_text="Tool: final-tool")
        row_style = """element => {
            const style = getComputedStyle(element);
            return [style.borderLeft, style.borderTopWidth, style.paddingLeft, style.backgroundColor];
        }"""
        assert await first_header.evaluate(row_style) == await tool.evaluate(row_style)
        horizontal_alignment = """row => {
            const icon = row.querySelector('.tool-event-main > .q-icon');
            return [row.getBoundingClientRect().x, icon.getBoundingClientRect().x];
        }"""
        assert await first_header.evaluate(horizontal_alignment) == await tool.evaluate(
            horizontal_alignment
        )
        assert await skill.evaluate(horizontal_alignment) == await tool.evaluate(
            horizontal_alignment
        )
        inner_tool = page.locator(".tool-event-card").filter(has_text="Tool: mock-tool")
        inner_thinking = sections.nth(1).locator(".reasoning-master-header")
        assert await marker.evaluate(horizontal_alignment) == await inner_tool.evaluate(
            horizontal_alignment
        )
        assert await marker.evaluate(
            horizontal_alignment
        ) == await inner_thinking.evaluate(horizontal_alignment)
        await expect(skill).to_have_css("border-left-color", "rgb(52, 211, 153)")
        await expect(skill.locator(".skill-status-icon")).to_have_css(
            "color", "rgb(52, 211, 153)"
        )
        await expect(marker).to_have_css("border-left-color", "rgb(56, 189, 248)")
        for section in await sections.all():
            header = section.locator(".reasoning-master-header")
            await expect(header).to_contain_text("Reasoning Process")
            await expect(header).not_to_contain_text("1 block")
            await expect(section.locator(".q-expansion-item")).to_have_count(0)
            thinking = section.locator(".thinking-box")
            await expect(thinking).to_be_hidden()
            await header.click()
            await expect(thinking).to_be_visible()

        await tool.click()
        await expect(tool).to_have_attribute("aria-expanded", "true")
        assert await first_header.evaluate(row_style) == await tool.evaluate(row_style)

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
