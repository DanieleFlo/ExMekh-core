from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from tests.browser_support import BrowserTestServer, get_edge_executable


async def safe_screenshot(page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            await page.screenshot(path=str(path))
            return
        except OSError:
            await asyncio.sleep(0.4)
    await page.screenshot(path=str(path))


@pytest.fixture(scope="module")
def agent_scroll_server(tmp_path_factory):
    """Starts a NiceGUI server with a phased streaming mock to test smart scroll."""
    runner_dir = tmp_path_factory.mktemp("scroll_server")
    runner_script = runner_dir / "run_scroll_test_server.py"

    runner_script.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations
            import asyncio
            import time
            from nicegui import app, ui
            from exmekh_agent.config import config
            from exmekh_agent.state.session_manager import session_manager
            import exmekh_agent.ui.page_controller as pc_mod
            from exmekh_agent.ui.page_controller import ChatPageController
            from exmekh_agent.ui.styles import load_css

            phase2_event = asyncio.Event()
            phase3_event = asyncio.Event()

            @app.get("/_test/reset_events")
            async def reset_events():
                phase2_event.clear()
                phase3_event.clear()
                return {"ok": True}

            @app.get("/_test/trigger_phase/{phase_num}")
            async def trigger_phase(phase_num: int):
                if phase_num == 2:
                    phase2_event.set()
                elif phase_num == 3:
                    phase3_event.set()
                return {"triggered": phase_num}

            async def phased_run_agent_turn(
                user_prompt,
                model,
                on_token,
                on_reasoning,
                on_tool_event,
                agent_mode,
                on_skill_event,
                on_raw_history=None,
                history=None,
                agent_context_id=None,
                on_sub_agent_event=None,
                sub_agent_history=None,
                settings=None,
                on_context=None,
                **kwargs,
            ):
                full_chunks = []
                # Stream initial reasoning block
                for r_idx in range(1, 5):
                    on_reasoning(
                        f"Analisi preliminare dello scroll automatico - passo {r_idx}\\n",
                        skill_path="/",
                        depth=0,
                    )
                    await asyncio.sleep(0.02)

                # Phase 1: Stream lines 1 to 25 (enough to overflow the chat container well beyond viewport)
                for i in range(1, 26):
                    line = (
                        f"### Sezione {i:02d} — Streaming Automatico (Fase 1)\\n"
                        f"Riga {i:02d}: La vista della chat scende automaticamente seguendo il testo generato in tempo reale.\\n\\n"
                    )
                    full_chunks.append(line)
                    on_token(line)
                    await asyncio.sleep(0.015)

                # Wait for test to inspect Phase 1, scroll up, and trigger Phase 2
                await asyncio.wait_for(phase2_event.wait(), timeout=30.0)

                # Phase 2: Stream lines 26 to 40 while user is scrolled up (detached from bottom)
                for i in range(26, 41):
                    line = (
                        f"### Sezione {i:02d} — Testo Aggiunto Mentre Utente è in Alto (Fase 2)\\n"
                        f"Riga {i:02d}: Questo testo viene aggiunto sotto mentre la visuale resta ferma in alto.\\n\\n"
                    )
                    full_chunks.append(line)
                    on_token(line)
                    await asyncio.sleep(0.015)

                # Wait for test to inspect Phase 2, scroll back to bottom, and trigger Phase 3
                await asyncio.wait_for(phase3_event.wait(), timeout=30.0)

                # Phase 3: Stream lines 41 to 55 after user scrolled back to bottom (auto-scroll resumed)
                for i in range(41, 56):
                    suffix = " — FINE STREAM" if i == 55 else ""
                    line = (
                        f"### Sezione {i:02d} — Ripresa Auto-Scroll verso il Basso (Fase 3){suffix}\\n"
                        f"Riga {i:02d}: Dopo essere tornati in fondo, lo scroll riprende a seguire il testo{suffix}.\\n\\n"
                    )
                    full_chunks.append(line)
                    on_token(line)
                    await asyncio.sleep(0.015)

                return "".join(full_chunks)

            pc_mod.run_agent_turn = phased_run_agent_turn

            @ui.page("/")
            def index_page() -> None:
                load_css()
                ui.dark_mode().enable()
                s = session_manager.create_session()
                s.title = "Test Smart Auto-Scroll Chat"
                session_manager.save_session(s)
                ChatPageController(ui.context.client).build()

            ui.run(
                host="127.0.0.1",
                port=config.port,
                title="ExMekh-Lib Scroll E2E Test",
                dark=True,
                reload=False,
                show=False,
            )
            """
        ),
        encoding="utf-8",
    )

    with BrowserTestServer(runner_dir / "data", script=runner_script) as url:
        yield url


@pytest.mark.asyncio
async def test_smart_chat_scroll_lifecycle_and_screenshots(
    agent_scroll_server: str,
) -> None:
    """Verifies the complete smart auto-scroll lifecycle across 4 distinct moments:
    1. Default auto-scroll follows text downward as it grows.
    2. Scrolling upward detaches from the bottom (`data-scroll-pinned='false'`).
    3. Subsequent text growth does NOT move the viewport downward while detached.
    4. Scrolling all the way back down to the bottom re-attaches (`data-scroll-pinned='true'`)
       and resumes following new text downward.
    """
    agent_dir = Path(__file__).resolve().parents[1]
    screenshots_dir = agent_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 860})
        page = await context.new_page()

        await page.goto(f"{agent_scroll_server}/_test/reset_events")
        await page.goto(agent_scroll_server, wait_until="networkidle")

        chat_box = page.locator(".chat-scroll-container").first
        await chat_box.wait_for(state="visible", timeout=10000)

        # Send prompt to start the phased streaming response
        textarea = page.locator("textarea").first
        await textarea.fill(
            "Genera un lungo report multi-sezione per verificare lo scroll intelligente."
        )
        await textarea.press("Enter")

        # ---------------------------------------------------------------------
        # MOMENTO 1: Default Auto-Scroll Follows Text Downward (Phase 1: Lines 1..25)
        # ---------------------------------------------------------------------
        await chat_box.get_by_text("Riga 25:", exact=False).first.wait_for(
            state="visible", timeout=15000
        )
        await asyncio.sleep(0.25)

        metrics_1 = await chat_box.evaluate(
            """el => ({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                distFromBottom: el.scrollHeight - el.clientHeight - el.scrollTop,
                pinned: el.getAttribute('data-scroll-pinned'),
            })"""
        )
        print(f"\n[Momento 1 - Auto-scroll default]: {metrics_1}")
        assert metrics_1["scrollHeight"] > metrics_1["clientHeight"] + 400
        assert metrics_1["scrollTop"] > 400
        assert metrics_1["distFromBottom"] <= 24
        assert metrics_1["pinned"] == "true"

        ss_1 = screenshots_dir / "scroll_01_auto_follow_bottom.png"
        await safe_screenshot(page, ss_1)
        assert ss_1.exists() and ss_1.stat().st_size > 0

        # ---------------------------------------------------------------------
        # MOMENTO 2: User Scrolls Upward to Detach from Bottom
        # ---------------------------------------------------------------------
        box_rect = await chat_box.bounding_box()
        assert box_rect is not None
        await page.mouse.move(
            box_rect["x"] + box_rect["width"] / 2,
            box_rect["y"] + box_rect["height"] / 2,
        )
        # Scroll upward with mouse wheel
        await page.mouse.wheel(0, -650)
        await asyncio.sleep(0.35)

        metrics_2 = await chat_box.evaluate(
            """el => ({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                distFromBottom: el.scrollHeight - el.clientHeight - el.scrollTop,
                pinned: el.getAttribute('data-scroll-pinned'),
            })"""
        )
        print(f"[Momento 2 - Scrolled up / detached]: {metrics_2}")
        assert metrics_2["pinned"] == "false"
        assert metrics_2["distFromBottom"] > 300
        assert metrics_2["scrollTop"] < metrics_1["scrollTop"] - 300

        ss_2 = screenshots_dir / "scroll_02_scrolled_up_detached.png"
        await safe_screenshot(page, ss_2)
        assert ss_2.exists() and ss_2.stat().st_size > 0

        # ---------------------------------------------------------------------
        # MOMENTO 3: Text Continues Growing (Phase 2: Lines 26..40) While Detached
        # ---------------------------------------------------------------------
        await page.evaluate("fetch('/_test/trigger_phase/2')")
        await (
            page.locator(".parsed-response-content")
            .filter(has_text="Riga 40:")
            .wait_for(state="attached", timeout=15000)
        )
        await asyncio.sleep(0.35)

        metrics_3 = await chat_box.evaluate(
            """el => ({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                distFromBottom: el.scrollHeight - el.clientHeight - el.scrollTop,
                pinned: el.getAttribute('data-scroll-pinned'),
            })"""
        )
        print(f"[Momento 3 - Text grew while detached]: {metrics_3}")
        # Verify scrollHeight grew significantly, yet scrollTop remained fixed where the user left it!
        assert metrics_3["scrollHeight"] > metrics_2["scrollHeight"] + 400
        assert abs(metrics_3["scrollTop"] - metrics_2["scrollTop"]) <= 2
        assert metrics_3["pinned"] == "false"
        assert metrics_3["distFromBottom"] > metrics_2["distFromBottom"] + 400

        ss_3 = screenshots_dir / "scroll_03_text_grows_while_detached.png"
        await safe_screenshot(page, ss_3)
        assert ss_3.exists() and ss_3.stat().st_size > 0

        # ---------------------------------------------------------------------
        # MOMENTO 4: User Scrolls All the Way Down to Bottom -> Resumes Auto-Scroll (Phase 3: Lines 41..55)
        # ---------------------------------------------------------------------
        # Scroll all the way down to the bottom using mouse wheel
        await page.mouse.move(
            box_rect["x"] + box_rect["width"] / 2,
            box_rect["y"] + box_rect["height"] / 2,
        )
        await page.mouse.wheel(0, 5000)
        await asyncio.sleep(0.35)

        metrics_reattached = await chat_box.evaluate(
            """el => ({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                distFromBottom: el.scrollHeight - el.clientHeight - el.scrollTop,
                pinned: el.getAttribute('data-scroll-pinned'),
            })"""
        )
        print(f"[Momento 4a - Re-attached at bottom]: {metrics_reattached}")
        assert metrics_reattached["pinned"] == "true"
        assert metrics_reattached["distFromBottom"] <= 24

        # Now trigger Phase 3 (Lines 41..55 + stream finalization) and verify it automatically follows down!
        await page.evaluate("fetch('/_test/trigger_phase/3')")
        await chat_box.get_by_text("Riga 55:", exact=False).first.wait_for(
            state="visible", timeout=15000
        )
        await asyncio.sleep(0.4)

        metrics_4 = await chat_box.evaluate(
            """el => ({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                distFromBottom: el.scrollHeight - el.clientHeight - el.scrollTop,
                pinned: el.getAttribute('data-scroll-pinned'),
            })"""
        )
        print(f"[Momento 4b - Auto-scroll resumed to end of stream]: {metrics_4}")
        assert metrics_4["scrollHeight"] > metrics_reattached["scrollHeight"] + 300
        assert metrics_4["scrollTop"] > metrics_reattached["scrollTop"] + 300
        assert metrics_4["distFromBottom"] <= 24
        assert metrics_4["pinned"] == "true"

        ss_4 = screenshots_dir / "scroll_04_scrolled_back_to_bottom_resumed.png"
        await safe_screenshot(page, ss_4)
        assert ss_4.exists() and ss_4.stat().st_size > 0

        await browser.close()


@pytest.mark.asyncio
async def test_point_of_interest_targets_current_message_row(
    agent_scroll_server: str,
) -> None:
    """A replaced message row with identical text must receive the dot's click."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        await page.goto(agent_scroll_server, wait_until="networkidle")
        result = await page.evaluate(
            """() => {
                const chat = document.querySelector('.chat-scroll-container');
                const original = document.createElement('div');
                original.className = 'chat-message-row chat-message-user';
                original.textContent = 'Same message';
                original.style.height = '1600px';
                chat.appendChild(original);
                window.__exmekhChatScroll.refreshPointsOfInterest();
                const track = document.getElementById('exmekh-chat-poi-track');
                const oldDot = track.querySelector('.chat-poi-dot');
                if (!oldDot) return { initialized: false };

                const replacement = original.cloneNode(true);
                original.replaceWith(replacement);
                let target = null;
                replacement.scrollIntoView = () => { target = 'replacement'; };
                original.scrollIntoView = () => { target = 'original'; };
                window.__exmekhChatScroll.refreshPointsOfInterest();
                const newDot = track.querySelector('.chat-poi-dot');
                newDot.click();
                return { initialized: true, rebuilt: newDot !== oldDot, target };
            }"""
        )
        assert result == {
            "initialized": True,
            "rebuilt": True,
            "target": "replacement",
        }
        await browser.close()
