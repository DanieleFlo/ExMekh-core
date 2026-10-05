from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    Project,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from tests.browser_support import BrowserTestServer, get_edge_executable


@pytest.fixture(scope="module")
def agent_tabs_server(tmp_path_factory: pytest.TempPathFactory):
    """Sets up a test session with hierarchical skills and starts NiceGUI server."""
    agent_dir = Path(__file__).resolve().parents[1]
    data_dir = tmp_path_factory.mktemp("tabs-data")
    chats_dir = data_dir / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    project = Project(id="test-tabs-project", work_root=str(agent_dir))
    project_dir = data_dir / "projects"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / f"{project.id}.json"
    project_file.write_text(project.model_dump_json(), encoding="utf-8")

    # Prepare hierarchical sub-skill session
    nested_sub = SkillNode(
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
    )
    root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="completed",
        tool_events=[
            ToolEvent(tool_name="mock-tool", depth=1, skill_path="/mock-skill")
        ],
        sub_skills=[nested_sub],
    )

    session = ChatSession(
        id="test-tabs-e2e-session",
        project_id=project.id,
        title="Test Sub-Tabs e Raw History",
        updated_at=time.time() + 1000,
        messages=[
            ChatMessage(role="user", content="quali tool hai a disposizione?"),
            ChatMessage(
                role="assistant",
                content="Ecco i tool a disposizione in questo ambiente con mock-skill.",
                skills=[root_skill],
            ),
        ],
        sub_agent_conversations=[
            SubAgentConversation(
                id="persistent-child",
                path="main/coding-agent",
                order=1,
                persistent_history=True,
                status="completed",
                messages=[
                    ChatMessage(role="user", content="Task persistente"),
                    ChatMessage(role="assistant", content="Risposta coding agent"),
                ],
                raw_history=[
                    {"role": "system", "content": "Coding agent"},
                    {"role": "user", "content": "Task persistente"},
                    {"role": "assistant", "content": "Risposta coding agent"},
                ],
            ),
            SubAgentConversation(
                id="fresh-child-1",
                path="main/sub/sub-sub",
                order=1,
                persistent_history=False,
                status="completed",
                messages=[ChatMessage(role="assistant", content="Risposta nuova 1")],
            ),
            SubAgentConversation(
                id="fresh-child-2",
                path="main/sub/sub-sub",
                order=2,
                persistent_history=False,
                status="completed",
                messages=[ChatMessage(role="assistant", content="Risposta nuova 2")],
            ),
        ],
    )

    session_file = chats_dir / f"{session.id}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        f.write(session.model_dump_json(indent=2))

    with BrowserTestServer(data_dir, module="tests.run_projects_agent") as url:
        yield url


async def safe_screenshot(page, path: Path) -> None:
    for attempt in range(5):
        try:
            await page.screenshot(path=str(path))
            return
        except OSError:
            await asyncio.sleep(0.4)
    await page.screenshot(path=str(path))


@pytest.mark.asyncio
async def test_sub_agent_cards_and_tabs_live(agent_tabs_server: str) -> None:
    """Open the running NiceGUI page and navigate all child tabs in a browser."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 920})
        await page.goto(agent_tabs_server, wait_until="networkidle")

        cards = page.locator(".sub-agent-card")
        await cards.first.wait_for(state="visible")
        assert await cards.count() == 3
        assert "border-sky" in await cards.nth(0).get_attribute("class")
        assert "border-amber" in await cards.nth(1).get_attribute("class")
        assert "main/sub/sub-sub #2" in await cards.nth(2).inner_text()
        assert await page.locator(".sub-agent-tab").count() == 0

        await cards.nth(0).click()
        await page.locator(".sub-agent-card-selected").first.wait_for(state="visible")
        assert "sidebar-item-selected" in await cards.nth(0).get_attribute("class")
        child_selection_animation = await cards.nth(0).evaluate(
            "element => getComputedStyle(element).animationName"
        )
        assert child_selection_animation == "sidebar-selection-pulse"
        await (
            page.locator(".msg-assistant")
            .get_by_text("Risposta coding agent", exact=True)
            .wait_for(state="visible", timeout=5000)
        )
        assert await page.get_by_text("main/coding-agent #1").count() >= 1

        await page.locator(".sub-tabs-bar .q-tab:has-text('Raw History')").click()
        await (
            page.locator(".raw-history-container:visible")
            .get_by_text("Raw History (JSON Inspector)")
            .wait_for(state="visible", timeout=5000)
        )
        assert await page.locator(
            ".raw-history-container:visible .json-key"
        ).first.is_visible()

        await cards.nth(2).click()
        await (
            page.locator(".sub-agent-card-selected")
            .filter(has_text="main/sub/sub-sub #2")
            .wait_for(state="visible", timeout=5000)
        )
        assert "sidebar-item-selected" in await cards.nth(2).get_attribute("class")
        await (
            page.locator(".raw-history-container:visible")
            .get_by_text("0 messages")
            .wait_for(state="visible", timeout=5000)
        )
        await page.locator(".sub-tabs-bar .q-tab:has-text('Chat')").click()
        await (
            page.locator(".msg-assistant")
            .get_by_text("Risposta nuova 2", exact=True)
            .wait_for(state="visible", timeout=5000)
        )

        await page.locator(".lm-sidebar .q-card:not(.sub-agent-card)").first.click()
        await expect(page.locator(".sub-agent-card-selected")).to_have_count(0)
        main_selection_animation = await page.locator(
            ".lm-sidebar .q-card.sidebar-item-selected"
        ).first.evaluate("element => getComputedStyle(element).animationName")
        assert main_selection_animation == child_selection_animation
        await (
            page.locator(".msg-assistant")
            .get_by_text(
                "Ecco i tool a disposizione in questo ambiente con mock-skill.",
                exact=True,
            )
            .wait_for(state="visible", timeout=5000)
        )

        await page.locator(".sub-tabs-bar .q-tab:has-text('Graph')").click()
        await cards.nth(0).click()
        await page.get_by_text("Graph View").wait_for(state="visible", timeout=5000)
        await browser.close()


@pytest.mark.asyncio
async def test_narrow_layout_keeps_navigation_accessible(
    agent_tabs_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 390, "height": 844})
        await page.goto(agent_tabs_server, wait_until="networkidle")
        await page.screenshot(
            path=str(
                Path(__file__).resolve().parents[1]
                / "screenshots"
                / "responsive_mobile.png"
            )
        )
        assert await page.locator("button:has(.q-icon:text-is('menu'))").count() == 1
        assert await page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        await page.locator("button:has(.q-icon:text-is('menu'))").click()
        assert await page.get_by_text("PROJECTS", exact=True).is_visible()
        await browser.close()


@pytest.mark.asyncio
async def test_global_and_project_settings_with_delete_confirmation(
    agent_tabs_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=get_edge_executable(), headless=True
        )
        page = await browser.new_page(viewport={"width": 1360, "height": 920})
        await page.goto(agent_tabs_server, wait_until="networkidle")

        await page.locator("button:has(.q-icon:text-is('tune'))").click()
        await page.get_by_text("LOCAL PROVIDER").wait_for(state="visible")
        assert await page.locator(".header-provider").is_visible()
        assert not await page.get_by_text(
            "ExMekh-Lib HIERARCHICAL CAPABILITIES"
        ).is_visible()
        await page.wait_for_timeout(350)
        await page.screenshot(
            path=str(
                Path(__file__).resolve().parents[1]
                / "screenshots"
                / "global_settings.png"
            )
        )
        await page.locator(
            ".q-drawer--right button:has(.q-icon:text-is('close'))"
        ).click()

        card = (
            page.locator(".sidebar-item")
            .filter(has_text="Test Sub-Tabs e Raw Hist")
            .first
        )
        proj = card.locator("xpath=ancestor::*[contains(@class, 'sidebar-project')]")
        await proj.get_by_role("button", name="Project actions", exact=True).click()
        await page.get_by_text("Project settings", exact=True).click()
        await page.get_by_role("group", name="Agent type", exact=True).wait_for(
            state="visible"
        )
        await page.wait_for_timeout(350)
        await page.screenshot(
            path=str(
                Path(__file__).resolve().parents[1]
                / "screenshots"
                / "chat_settings.png"
            )
        )
        await page.locator(".q-dialog button:has(.q-icon:text-is('close'))").click()

        await card.hover()
        await card.get_by_role("button", name="Delete chat", exact=True).click()
        await page.get_by_text("Delete this chat?").wait_for(state="visible")
        await page.get_by_role("button", name="Cancel").click()
        assert await card.count() == 1
        chat_cards = page.locator(".lm-sidebar .q-card:not(.sub-agent-card)")
        initial_count = await chat_cards.count()
        await page.get_by_role(
            "button", name="New project chat", exact=True
        ).first.click()
        await expect(chat_cards).to_have_count(initial_count + 1)
        disposable_card = page.locator(
            ".lm-sidebar .q-card.sidebar-item-selected:not(.sub-agent-card)"
        )
        await disposable_card.hover()
        await disposable_card.get_by_role(
            "button", name="Delete chat", exact=True
        ).click()
        await page.get_by_role("button", name="Yes, delete").click()
        await expect(chat_cards).to_have_count(initial_count)
        assert await card.count() == 1
        await browser.close()


@pytest.mark.asyncio
async def test_sub_tabs_and_raw_history_e2e_screenshots(agent_tabs_server: str) -> None:
    """Verifies the 3 sub-tabs in the UI, navigates between them, tests the colorized

    Raw History and vertical compactions with matryoshka nesting, and captures screenshots.
    """
    agent_dir = Path(__file__).resolve().parents[1]
    screenshots_dir = agent_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    edge_bin = get_edge_executable()

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=edge_bin, headless=True)
        context = await browser.new_context(viewport={"width": 1360, "height": 920})
        page = await context.new_page()

        await page.goto(agent_tabs_server, wait_until="networkidle")
        await page.wait_for_selector(".sub-tabs-bar", timeout=12000)
        conversation = (
            page.locator(".sidebar-item")
            .filter(has_text="Test Sub-Tabs e Raw Hist")
            .first
        )
        if not await conversation.is_visible():
            await (
                conversation.locator(
                    "xpath=ancestor::*[contains(@class, 'sidebar-project')]"
                )
                .locator(".sidebar-folder-icon, .q-item")
                .first.click()
            )
        await conversation.click()

        # -----------------------------------------------------------------
        # 1. TAB 1: Chat View
        # -----------------------------------------------------------------
        # Verify all 3 tabs are present in the bar
        tab_chat = page.locator(".sub-tabs-bar .q-tab:has-text('Chat')")
        tab_raw = page.locator(".sub-tabs-bar .q-tab:has-text('Raw History')")
        tab_graph = page.locator(".sub-tabs-bar .q-tab:has-text('Graph')")

        assert await tab_chat.is_visible()
        assert await tab_raw.is_visible()
        assert await tab_graph.is_visible()

        # Chat message elements are visible
        await page.wait_for_selector(".msg-user", timeout=6000)
        await asyncio.sleep(0.6)

        ss_chat = screenshots_dir / "06_sub_tabs_chat_view.png"
        await safe_screenshot(page, ss_chat)
        print(f"\n[Screenshot Chat View]: Saved to {ss_chat}")
        assert ss_chat.exists() and ss_chat.stat().st_size > 0

        # -----------------------------------------------------------------
        # 2. TAB 2: Raw History (Compacted)
        # -----------------------------------------------------------------
        await tab_raw.click()
        await page.wait_for_selector(".raw-history-container", timeout=6000)
        await asyncio.sleep(0.6)

        # Verify JSON syntax highlighting elements
        assert await page.locator(".json-key").first.is_visible()
        assert await page.locator(".json-val-str").first.is_visible()

        # Verify vertical compaction bar is present
        assert await page.locator(".subskill-compaction").first.is_visible()
        assert await page.locator("text=Sub-Skill: mock-skill").first.is_visible()

        ss_raw_compact = screenshots_dir / "07_sub_tabs_raw_history_compact.png"
        await safe_screenshot(page, ss_raw_compact)
        print(f"[Screenshot Raw History Compact]: Saved to {ss_raw_compact}")
        assert ss_raw_compact.exists() and ss_raw_compact.stat().st_size > 0

        # -----------------------------------------------------------------
        # 3. TAB 2: Raw History (Expanded with Matryoshka)
        # -----------------------------------------------------------------
        # Click "Expand all" button
        expand_all_btn = page.locator("button:has-text('Expand all')").first
        await expand_all_btn.click()
        await asyncio.sleep(0.8)

        # Verify nested sub-sub-skill is revealed inside (matryoshka)
        nested_elem = page.locator("text=Sub-Skill: nested-sub-skill").first
        assert await nested_elem.is_visible()
        await nested_elem.scroll_into_view_if_needed()
        await asyncio.sleep(0.5)

        ss_raw_expanded = screenshots_dir / "08_sub_tabs_raw_history_expanded.png"
        await safe_screenshot(page, ss_raw_expanded)
        print(
            f"[Screenshot Raw History Expanded Matrioska]: Saved to {ss_raw_expanded}"
        )
        assert ss_raw_expanded.exists() and ss_raw_expanded.stat().st_size > 0

        # -----------------------------------------------------------------
        # 4. TAB 3: Graph (Interactive Star Graph & Horizontal Bottom Menu)
        # -----------------------------------------------------------------
        await tab_graph.click()
        await page.wait_for_selector("text=Graph View", timeout=6000)
        await page.wait_for_selector(".graph-node", timeout=6000)
        await asyncio.sleep(0.4)

        # Verify main history nodes and counterclockwise-offset skill nodes exist
        assert await page.locator(".graph-node-main-line").count() >= 2
        assert await page.locator(".graph-node-skill-offset").count() >= 2

        # Click a skill/tool node to open the bottom horizontal events menu
        skill_node = page.locator(".graph-node-skill-offset").first
        await skill_node.click()
        await page.wait_for_selector(".graph-bottom-menu-open", timeout=6000)
        await page.wait_for_selector(
            ".graph-horizontal-events-row .graph-event-card", timeout=6000
        )
        assert (
            await page.locator(".graph-horizontal-events-row .graph-event-card").count()
            == 3
        )
        await asyncio.sleep(0.4)

        ss_graph = screenshots_dir / "09_sub_tabs_grafico_empty.png"
        await safe_screenshot(page, ss_graph)
        print(f"[Screenshot Grafico Star & Menu]: Saved to {ss_graph}")
        assert ss_graph.exists() and ss_graph.stat().st_size > 0

        # -----------------------------------------------------------------
        # 5. TAB 2: Live Progressive Streaming in Raw History ("man mano")
        # -----------------------------------------------------------------
        await tab_raw.click()
        await page.wait_for_selector(".raw-history-container", timeout=6000)
        await asyncio.sleep(0.4)

        # Enter message in input textarea and press Enter
        input_area = page.locator("textarea").first
        await input_area.fill("Dimostra lo streaming man mano della raw history")
        await input_area.press("Enter")

        # Wait for message and live streaming elements in Raw History
        await asyncio.sleep(1.0)

        ss_live = screenshots_dir / "10_sub_tabs_raw_history_live_streaming.png"
        await safe_screenshot(page, ss_live)
        print(f"[Screenshot Raw History Live Streaming]: Saved to {ss_live}")
        assert ss_live.exists() and ss_live.stat().st_size > 0

        await browser.close()
