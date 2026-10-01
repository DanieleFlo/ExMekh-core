from __future__ import annotations

import asyncio
import re
from collections.abc import Generator
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.browser_support import (
    get_edge_executable,
    send_command,
    start_mock_server,
    wait_turn_completion,
)


@pytest.fixture(scope="module")
def fe_mock_server() -> Generator[str]:
    yield from start_mock_server(0.0)


@pytest.fixture(scope="module")
def fe_slow_mock_server() -> Generator[str]:
    yield from start_mock_server(0.8)


@pytest.fixture(scope="module")
def fe_refresh_mock_server() -> Generator[str]:
    yield from start_mock_server(0.8)


@pytest.fixture(scope="module")
def fe_reopen_mock_server() -> Generator[str]:
    yield from start_mock_server(5.0)


@pytest.fixture(scope="module")
def fe_approval_mock_server() -> Generator[str]:
    yield from start_mock_server(0.0, "tests.run_approval_fe")


@pytest.fixture(scope="module")
def fe_steering_mock_server() -> Generator[str]:
    yield from start_mock_server(0.0, "tests.run_steering_fe")


@pytest.fixture(scope="module")
def fe_nested_steering_mock_server() -> Generator[str]:
    yield from start_mock_server(
        0.0, "tests.run_steering_fe", steering_first_response="nested_skill"
    )


async def test_steering_button_updates_the_running_qs_turn_after_refresh(
    fe_steering_mock_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(fe_steering_mock_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            approval = page.locator(".input-bar-approval-select").first
            await approval.click()
            await page.get_by_role("option", name="Approve All", exact=True).click()
            await send_command(page, "Start steering test")
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await send_command(page, "Use the other file")
            first_row = page.locator(".qs-queue-item").filter(
                has_text="Use the other file"
            )
            await expect(first_row).to_be_visible()
            await send_command(page, "Also keep the existing API")
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await expect(
                first_row.get_by_role("button", name="Steering")
            ).to_be_enabled()
            await first_row.get_by_role("button", name="Steering").click()
            await expect(first_row).to_have_count(0)
            await expect(page.locator(".msg-user")).to_contain_text(
                ["Start steering test", "Use the other file"]
            )
            await expect(page.locator(".msg-user-steering")).to_contain_text(
                "Use the other file"
            )
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            stats = await (
                await page.request.get(f"{fe_steering_mock_server}/test/steering-stats")
            ).json()
            assert len(stats["requests"]) == 1
            await expect(page.locator(".steering-tool-marker")).to_have_count(0)

            # The second queued message also steers the same original run.
            row = page.locator(".qs-queue-item").filter(
                has_text="Also keep the existing API"
            )
            await row.get_by_role("button", name="Steering").click()
            await expect(page.locator(".qs-queue-box")).not_to_be_visible()
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".msg-user")).to_contain_text(
                [
                    "Start steering test",
                    "Use the other file",
                    "Also keep the existing API",
                ]
            )
            await page.request.post(f"{fe_steering_mock_server}/test/release-inference")
            # QS confirms insertion before the next inference completes.
            await expect(page.locator(".steering-tool-marker")).to_have_count(2)
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".steering-tool-marker")).to_have_count(2)
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await page.request.post(f"{fe_steering_mock_server}/test/release-reply")
            await wait_turn_completion(page)
            await expect(page.locator(".msg-assistant")).to_contain_text(
                "Reply following guidance: Also keep the existing API"
            )
            await expect(page.locator(".steering-tool-marker")).to_have_count(2)
            stats = await (
                await page.request.get(f"{fe_steering_mock_server}/test/steering-stats")
            ).json()
            request = stats["requests"][1]
            guidance_index = next(
                index
                for index, item in enumerate(request)
                if item.get("content") == "Use the other file"
            )
            preceding_tool = request[guidance_index - 1]
            assert preceding_tool["role"] == "tool"
            tool_call = request[guidance_index - 2]["tool_calls"][0]
            assert tool_call["id"] == preceding_tool["tool_call_id"]
            label = (
                f"Steering inserted after {tool_call['function']['name']} · skill root"
            )
            await expect(page.locator(".steering-tool-marker")).to_contain_text(
                [label, label]
            )
            await page.reload(wait_until="networkidle")
            await expect(page.locator(".msg-user")).to_have_count(3)
            await expect(page.locator(".msg-user-steering")).to_have_count(2)
            await expect(page.locator(".steering-tool-marker")).to_have_count(2)
            stats = await (
                await page.request.get(f"{fe_steering_mock_server}/test/steering-stats")
            ).json()
            # QS may retrieve asynchronous bash output before ending the first run.
            completed_request_count = len(stats["requests"])
            await send_command(page, "Continue")
            await wait_turn_completion(page)
            stats = await (
                await page.request.get(f"{fe_steering_mock_server}/test/steering-stats")
            ).json()
            assert len(stats["requests"]) == completed_request_count + 1
            assert len(stats["session"]["messages"][1]["steering_insertions"]) == 2
            request = stats["requests"][-1]
            assert (
                sum(item.get("content") == "Use the other file" for item in request)
                == 1
            )
            assert (
                sum(
                    item.get("content") == "Also keep the existing API"
                    for item in request
                )
                == 1
            )
            assert any(item.get("tool_call_id") == "steering-probe" for item in request)
        finally:
            await browser.close()


async def test_graph_keeps_steering_inside_nested_skill_after_reload(
    fe_nested_steering_mock_server: str,
) -> None:
    server = fe_nested_steering_mock_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await page.locator(".input-bar-approval-select").first.click()
            await page.get_by_role("option", name="Approve All", exact=True).click()
            await send_command(page, "Start nested steering test")
            await expect(page.locator(".chat-message-streaming")).to_be_visible()
            await page.request.post(f"{server}/test/release-inference")

            async def wait_for_nested_inference() -> None:
                while True:
                    stats = await (
                        await page.request.get(f"{server}/test/steering-stats")
                    ).json()
                    if len(stats["requests"]) == 2:
                        return
                    await asyncio.sleep(0.1)

            await asyncio.wait_for(wait_for_nested_inference(), 5)
            await send_command(page, "Use the nested route")
            row = page.locator(".qs-queue-item").filter(has_text="Use the nested route")
            await row.get_by_role("button", name="Steering").click()
            await expect(page.locator(".msg-user-steering")).to_contain_text(
                "Use the nested route"
            )
            await page.request.post(f"{server}/test/release-reply")
            await wait_turn_completion(page)

            stats = await (
                await page.request.get(f"{server}/test/steering-stats")
            ).json()
            nested_history = stats["session"]["messages"][1]["skill_history_tree"][0][
                "children"
            ][0]["history"]
            assert any(
                item.get("role") == "user"
                and item.get("content") == "Use the nested route"
                for item in nested_history
            )
            await page.locator(".sub-tabs-bar .q-tab:has-text('Graph')").click()
            for reload in (False, True):
                if reload:
                    await page.reload(wait_until="networkidle")
                    await page.locator(".sub-tabs-bar .q-tab:has-text('Graph')").click()
                steering = page.locator(".graph-node-steering")
                await expect(steering).to_have_count(1)
                await expect(steering).to_have_attribute("data-depth", "2")
                await expect(steering).to_have_attribute(
                    "data-skill-path", "/mock-skill/nested-skill"
                )
                labels = await page.locator(".graph-node").evaluate_all(
                    "nodes => nodes.map(node => node.getAttribute('aria-label'))"
                )
                index = labels.index("Steering: Use the nested route")
                assert labels[index - 1] == "Skill: nested-skill"
                assert labels[index + 1] == "Tool: nested-sub-tool"
                await steering.click()
                await expect(page.locator(".graph-event-card")).to_contain_text(
                    "Use the nested route"
                )
        finally:
            await page.request.post(f"{server}/test/release-inference")
            await page.request.post(f"{server}/test/release-reply")
            await browser.close()


@pytest.mark.asyncio
async def test_new_chat_reuses_existing_empty_chat(fe_mock_server: str) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(fe_mock_server, wait_until="networkidle")
            new_chat = page.get_by_role(
                "button", name="New project chat", exact=True
            ).first
            input_box = page.locator(".qs-chat-input-box")
            chat_cards = page.locator(".lm-sidebar .sidebar-item:not(.sub-agent-card)")
            await new_chat.click()
            await expect(input_box).to_have_attribute(
                "data-session-id", re.compile(r".+")
            )
            first_id = await input_box.get_attribute("data-session-id")
            assert first_id is not None
            initial_count = await chat_cards.count()

            await new_chat.click()
            await expect(input_box).to_have_attribute("data-session-id", first_id)
            await expect(chat_cards).to_have_count(initial_count)
            await expect(page.locator(".msg-user")).to_have_count(0)

            prompt = "Verify empty chat reuse"
            await send_command(page, prompt)
            await wait_turn_completion(page)
            await expect(chat_cards.filter(has_text=prompt).first).to_be_visible()
            await new_chat.click()
            await expect(input_box).not_to_have_attribute("data-session-id", first_id)
            await expect(page.locator(".msg-user")).to_have_count(0)
            empty_id = await input_box.get_attribute("data-session-id")
            assert empty_id is not None
            count_with_empty = await chat_cards.count()

            await chat_cards.filter(has_text=prompt).first.click()
            await expect(input_box).to_have_attribute("data-session-id", first_id)
            await expect(page.locator(".msg-user")).to_contain_text(prompt)
            await new_chat.click()

            await expect(input_box).to_have_attribute("data-session-id", empty_id)
            await expect(chat_cards).to_have_count(count_with_empty)
            await expect(page.locator(".msg-user")).to_have_count(0)
            await expect(
                page.locator(".lm-sidebar .sidebar-item-selected:not(.sub-agent-card)")
            ).to_have_count(1)
            await new_chat.click()
            await expect(input_box).to_have_attribute("data-session-id", empty_id)
            await expect(chat_cards).to_have_count(count_with_empty)
        finally:
            await browser.close()


async def test_stop_while_awaiting_approval_stops_inference_and_persists_in_chat(
    fe_approval_mock_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 860})
            await page.goto(fe_approval_mock_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            await send_command(page, "Run echo approval_probe")
            await expect(page.locator(".qs-approval-box")).to_be_visible()
            await page.locator("button:has(i:has-text('stop'))").first.click()
            await expect(page.locator(".generation-stopped")).to_have_text(
                "stop_circleGeneration stopped by user"
            )
            await expect(page.locator(".qs-approval-box")).not_to_be_visible()
            await expect(page.locator(".chat-message-streaming")).to_have_count(0)
            await expect(
                page.get_by_text("Output before approval", exact=True)
            ).to_be_visible()
            await wait_turn_completion(page)
            stats = await page.request.get(
                f"{fe_approval_mock_server}/test/approval-stats"
            )
            assert (await stats.json())["requests"] == 1

            await page.reload(wait_until="networkidle")
            await expect(page.locator(".generation-stopped")).to_be_visible()
            await send_command(page, "Continue with a new turn")
            await wait_turn_completion(page)
            await expect(
                page.get_by_text("Reply after declining the command", exact=True)
            ).to_be_visible()
            stats = await page.request.get(
                f"{fe_approval_mock_server}/test/approval-stats"
            )
            assert (await stats.json())["requests"] == 2
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_file_drop_stages_text_and_binary_files_in_the_current_chat(
    fe_mock_server: str,
) -> None:
    staged_paths: list[Path] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        try:
            await page.goto(fe_mock_server, wait_until="networkidle")
            await page.get_by_role(
                "button", name="New project chat", exact=True
            ).first.click()
            drop_zone = page.locator(".qs-chat-input-box")
            await expect(drop_zone).to_have_attribute(
                "data-session-id", re.compile(r".+")
            )
            transfer = await page.evaluate_handle("""() => {
                const transfer = new DataTransfer();
                transfer.items.add(new File(['print(42)'], 'review.py', {type: 'text/x-python'}));
                transfer.items.add(new File([new Uint8Array([0, 255, 1, 2])], 'review.bin', {type: 'application/octet-stream'}));
                return transfer;
            }""")
            await drop_zone.dispatch_event("dragenter", {"dataTransfer": transfer})
            await expect(page.locator(".qs-drop-overlay")).to_have_class(
                re.compile(r".*qs-active.*")
            )
            async with page.expect_response(
                lambda response: response.url.endswith("/_qs/upload")
            ) as pending_upload:
                await drop_zone.dispatch_event("drop", {"dataTransfer": transfer})
            response = await pending_upload.value
            assert response.status == 200
            descriptors = await response.json()
            project_root = Path(__file__).resolve().parents[1]
            staged_paths = [
                project_root / descriptor["relative_path"] for descriptor in descriptors
            ]
            assert [path.read_bytes() for path in staged_paths] == [
                b"print(42)",
                b"\x00\xff\x01\x02",
            ]
            await expect(page.locator(".qs-attachment-chip")).to_have_count(2)
            await expect(page.locator(".qs-drop-overlay")).not_to_have_class(
                re.compile(r".*qs-active.*")
            )
            await send_command(page, "Review the attachments")
            await wait_turn_completion(page)
            await expect(page.locator(".qs-message-attachment-chip")).to_have_count(2)
            await (
                page.locator(".qs-message-attachment-chip")
                .filter(has_text=".bin")
                .click()
            )
            await expect(
                page.get_by_text(
                    "Binary file (4 bytes). Text preview unavailable.",
                    exact=True,
                )
            ).to_be_visible()
        finally:
            await browser.close()
            for path in staged_paths:
                path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reselecting_active_chat_keeps_live_output(
    fe_slow_mock_server: str,
) -> None:
    """A sidebar click must not detach the assistant row during generation."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        await page.goto(fe_slow_mock_server, wait_until="networkidle")
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await send_command(
            page, "Inizializza pipeline ed esegui diagnostica preliminare"
        )

        thinking = page.locator(".reasoning-master-expansion:visible").first
        await thinking.wait_for(state="visible", timeout=8000)
        assistant = await page.locator(".msg-assistant:visible").first.element_handle()
        assert assistant is not None
        main_card = page.locator(".lm-sidebar .q-card:not(.sub-agent-card)").first
        await main_card.click()
        assert await assistant.is_visible()
        await expect(page.locator(".reasoning-master-expansion:visible")).to_have_count(
            1
        )

        await (
            page.locator(".msg-assistant:visible .parsed-response-content")
            .get_by_text("Diagnostica di sistema completata con successo.")
            .wait_for(state="visible", timeout=10000)
        )
        await main_card.click()
        assert await assistant.is_visible()
        await expect(
            page.locator(".msg-assistant:visible .parsed-response-content")
        ).to_contain_text("Diagnostica di sistema completata")

        await wait_turn_completion(page, timeout=20000)
        await expect(page.locator(".msg-assistant:visible")).to_contain_text(
            "SYS_DIAG_OK"
        )
        await browser.close()


@pytest.mark.asyncio
async def test_refresh_restores_active_generation(fe_refresh_mock_server: str) -> None:
    """A refreshed page must reconnect to the running turn and its final answer."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        await page.goto(fe_refresh_mock_server, wait_until="networkidle")
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await send_command(
            page, "Inizializza pipeline ed esegui diagnostica preliminare"
        )
        await page.locator(".chat-message-streaming:visible").wait_for(
            state="visible", timeout=8000
        )

        await page.reload(wait_until="networkidle")

        await expect(page.locator(".input-linear-progress:visible")).to_have_count(1)
        await expect(page.locator(".chat-message-streaming:visible")).to_have_count(1)
        await expect(
            page.locator("button:has(i:has-text('stop')):visible")
        ).to_have_count(1)
        await wait_turn_completion(page, timeout=20000)
        await expect(page.locator(".msg-assistant:visible")).to_contain_text(
            "SYS_DIAG_OK"
        )
        await expect(page.locator(".input-linear-progress:visible")).to_have_count(0)

        await browser.close()


@pytest.mark.asyncio
async def test_reopening_tab_restores_active_generation(
    fe_reopen_mock_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        await page.goto(fe_reopen_mock_server, wait_until="networkidle")
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await send_command(
            page, "Inizializza pipeline ed esegui diagnostica preliminare"
        )
        await page.locator(".chat-message-streaming:visible").wait_for(
            state="visible", timeout=8000
        )

        await page.close()
        page = await browser.new_page(viewport={"width": 1360, "height": 860})
        await page.goto(fe_reopen_mock_server, wait_until="domcontentloaded")
        await expect(page.locator(".input-linear-progress:visible")).to_have_count(1)
        await expect(page.locator(".chat-message-streaming:visible")).to_have_count(1)
        await wait_turn_completion(page, timeout=40000)
        await expect(page.locator(".msg-assistant:visible")).to_contain_text(
            "SYS_DIAG_OK"
        )
        await browser.close()


@pytest.mark.asyncio
async def test_ui_e2e_consecutive_commands_pipeline(fe_mock_server: str) -> None:
    """E2E Test verifying a full 3-step consecutive command pipeline in the UI:

    1. Step 1 (Diagnostic): Input command 1 -> thinking + tool mock-tool + SYS_DIAG_OK response.
    2. Step 2 (Nested Skills): Input command 2 -> hierarchical skill container + nested-sub-tool.
    3. Step 3 (Synthesis Report): Input command 3 -> compacted skill + consolidated markdown report.
    4. Captures screenshot to FE/screenshots/05_pipeline_consecutive_commands_e2e.png.
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
        context = await browser.new_context(viewport={"width": 1360, "height": 920})
        page = await context.new_page()

        await page.goto(fe_mock_server, wait_until="networkidle")
        await page.wait_for_selector(".lm-sidebar", timeout=15000)

        # Start a fresh conversation inside the test project.
        new_chat_btn = page.get_by_role(
            "button", name="New project chat", exact=True
        ).first
        await new_chat_btn.wait_for(state="visible", timeout=15000)
        await new_chat_btn.click()
        await asyncio.sleep(0.5)

        # -------------------------------------------------------------
        # 1. COMMAND 1: INIZIALIZZAZIONE & DIAGNOSTICA
        # -------------------------------------------------------------
        cmd1 = "Inizializza pipeline ed esegui diagnostica di sistema"
        await send_command(page, cmd1)

        # Wait for user message 1 to appear
        user_msg_1 = page.locator(".msg-user").first
        await user_msg_1.wait_for(state="visible", timeout=8000)
        assert cmd1 in await user_msg_1.inner_text()

        # Wait for assistant response to finish
        await wait_turn_completion(page)

        # Assert Step 1 output in UI:
        # - Thinking process block is rendered
        assert await page.locator(".reasoning-expansion").first.is_visible()
        # - Tool event mock-tool is displayed
        assert await page.locator("text=mock-tool").first.is_visible()
        # - Assistant message contains diagnostic token
        assistant_text_1 = await page.locator(".msg-assistant").first.inner_text()
        assert "SYS_DIAG_OK" in assistant_text_1

        # -------------------------------------------------------------
        # 2. COMMAND 2: APPROFONDIMENTO CON SOTTO-SKILL (LINKED COMMAND)
        # -------------------------------------------------------------
        cmd2 = "Approfondisci con la sotto-skill ed elabora i dati raccolti"
        await send_command(page, cmd2)

        # Wait for turn completion
        await wait_turn_completion(page)

        # Assert Step 2 UI hierarchical elements:
        # - Nested sub-tool is displayed
        assert await page.locator("text=nested-sub-tool").first.is_visible()
        # - Skill container / node is displayed
        assert await page.locator("text=mock-skill").first.is_visible()
        # - Second assistant message content
        assistants = page.locator(".msg-assistant")
        assert await assistants.count() >= 2
        assistant_text_2 = await assistants.nth(1).inner_text()
        assert "Elaborazione gerarchica completata" in assistant_text_2

        # -------------------------------------------------------------
        # 3. COMMAND 3: REPORT FINALE CONSOLIDATO DELLA PIPELINE
        # -------------------------------------------------------------
        cmd3 = "Genera il report finale consolidato della pipeline"
        await send_command(page, cmd3)

        # Wait for third turn to complete
        await wait_turn_completion(page)

        # Assert Step 3 UI outputs:
        # - 3 user messages and 3 assistant messages present in DOM
        assert await page.locator(".msg-user").count() == 3
        assert await page.locator(".msg-assistant").count() == 3

        # - Final assistant message has complete markdown report
        final_assistant = page.locator(".msg-assistant").nth(2)
        final_text = await final_assistant.inner_text()
        assert "Report Finale della Pipeline Consecutiva" in final_text
        assert "SYS_DIAG_OK" in final_text
        assert "mock-skill" in final_text

        # - Verify sidebar session title updated automatically from prompt 1
        sidebar_card = page.locator(".lm-sidebar .q-card").first
        await sidebar_card.wait_for(state="visible", timeout=8000)
        sidebar_title = await sidebar_card.inner_text()
        assert "Inizializza pipeline" in sidebar_title

        # Capture complete pipeline screenshot
        screenshot_path = screenshots_dir / "05_pipeline_consecutive_commands_e2e.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"\n[E2E Pipeline Screenshot]: Saved to {screenshot_path}")
        assert screenshot_path.exists() and screenshot_path.stat().st_size > 0

        await browser.close()


@pytest.mark.asyncio
async def test_ui_e2e_pipeline_session_switch_and_history_persistence(
    fe_mock_server: str,
) -> None:
    """E2E Test verifying multi-session switching with pipeline conversation preserved."""
    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=edge_bin,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 860})
        page = await context.new_page()

        await page.goto(fe_mock_server, wait_until="networkidle")
        await page.wait_for_selector(".lm-sidebar", timeout=15000)

        # Create new chat session A
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await asyncio.sleep(0.4)

        cmd_a = "Inizializza pipeline di test sessione A"
        await send_command(page, cmd_a)
        await wait_turn_completion(page)

        assert await page.locator(".msg-user").count() >= 1

        # Create new chat session B
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await asyncio.sleep(0.5)

        # In session B, chat view should be reset to welcome hero
        assert await page.locator(".msg-user").count() == 0
        assert await page.locator("text=QS Studio").first.is_visible()

        # Switch back to session A via sidebar
        session_a_item = page.locator(
            ".lm-sidebar .q-card:has-text('Inizializza pipeline')"
        ).first
        await session_a_item.wait_for(state="visible", timeout=8000)
        await session_a_item.click()
        await asyncio.sleep(0.5)

        # Verify session A history was reloaded cleanly
        assert await page.locator(".msg-user").count() >= 1
        assert (
            "Inizializza pipeline" in await page.locator(".msg-user").first.inner_text()
        )

        await browser.close()


@pytest.mark.asyncio
async def test_ui_e2e_pipeline_stop_generation_control_flow(
    fe_mock_server: str,
) -> None:
    """E2E Test verifying interactive control flow: user clicking Stop button during generation."""
    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=edge_bin,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 860})
        page = await context.new_page()

        await page.goto(fe_mock_server, wait_until="networkidle")
        await page.wait_for_selector(".lm-sidebar", timeout=15000)

        # 1. Start a fresh chat session
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await asyncio.sleep(0.4)

        # 2. Start a turn
        textarea = page.locator("textarea").first
        await textarea.fill("Inizializza pipeline ed esegui diagnostica di sistema")

        send_btn = page.locator("button:has(i:has-text('arrow_upward'))").first
        stop_btn = page.locator("button:has(i:has-text('stop'))").first

        await send_btn.click()

        # 3. If stop button becomes visible, test clicking it
        try:
            await stop_btn.wait_for(state="visible", timeout=3000)
            await stop_btn.click()
            await asyncio.sleep(0.5)
        except Exception:
            # Turn executed very fast
            pass

        # 4. Verify send button is restored and input bar is unlocked
        await send_btn.wait_for(state="visible", timeout=8000)
        await textarea.fill("Comando successivo dopo verifica")
        assert await textarea.input_value() == "Comando successivo dopo verifica"

        await browser.close()


@pytest.mark.asyncio
@pytest.mark.serial(reason="Copies code to the shared system clipboard")
async def test_ui_e2e_fenced_code_block_copy_button(
    fe_mock_server: str,
) -> None:
    """E2E Test verifying that fenced code and special text sections render a dedicated copy button that copies only that section."""
    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=edge_bin,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1360, "height": 860})
        page = await context.new_page()

        await page.goto(fe_mock_server, wait_until="networkidle")
        await page.wait_for_selector(".lm-sidebar", timeout=15000)

        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await asyncio.sleep(0.4)

        await send_command(page, "Mostra esempio codice")
        await wait_turn_completion(page)

        copy_buttons = page.locator(".parsed-response-content .code-copy-btn")
        await expect(copy_buttons).to_have_count(2, timeout=8000)

        # Verify both use the content_copy icon
        btn_0 = copy_buttons.nth(0)
        btn_1 = copy_buttons.nth(1)
        assert "content_copy" in (await btn_0.inner_text())
        assert "content_copy" in (await btn_1.inner_text())

        # Click first block copy button -> should copy only the python block
        await btn_0.click()
        copied_0 = await page.evaluate("() => window.__qsLastCopiedCode")
        assert copied_0 == "def calcola(x):\n    return x * 2"
        assert await btn_0.get_attribute("data-copied") == "true"

        # Click second block copy button -> should copy only the special text block
        await btn_1.click()
        copied_1 = await page.evaluate("() => window.__qsLastCopiedCode")
        assert copied_1 == "SEZIONE_TESTO_SPECIALE_SOLO_QUESTO"
        assert await btn_1.get_attribute("data-copied") == "true"

        await browser.close()
