from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.models import (
    ChatMessage,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SteeringInsertion,
    SubAgentConversation,
    ToolEvent,
)
from fe.state.active_turns import active_turns
from fe.state.session_manager import session_manager
from fe.ui.chat_view import (
    CHAT_SCROLL_JS_PATH,
    ChatViewComponent,
    get_chat_scroll_js,
    load_chat_scroll_js,
)
from fe.ui.header import HeaderComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.new_project_dialog import NewProjectDialogComponent, pick_folder_dialog
from fe.ui.page_controller import ChatPageController
from fe.ui.project_settings_dialog import ProjectSettingsDialogComponent
from fe.ui.settings_drawer import SettingsDrawerComponent
from fe.ui.sidebar import SidebarComponent


@pytest.fixture
def ui_context():
    c = Client(page("/test-context"))
    with c:
        yield c


# =============================================================================
# INPUT BAR TESTS
# =============================================================================


def test_input_bar_component(ui_context: Client) -> None:
    sent_messages = []
    stopped = False

    def on_send(msg: str) -> None:
        sent_messages.append(msg)

    def on_stop() -> None:
        nonlocal stopped
        stopped = True

    ib = InputBarComponent(on_send=on_send, on_stop=on_stop)
    bar = ib.build()
    assert "items-center" in bar._classes
    assert ib.textarea is not None and "chat-input-textarea" in ib.textarea._classes
    assert ib.send_button is not None and "self-center" in ib.send_button._classes
    assert ib.stop_button is not None and "self-center" in ib.stop_button._classes

    # Trigger with empty/whitespace text -> should not send
    ib.set_text("   ")
    ib._trigger_send()
    assert len(sent_messages) == 0

    # Set text and trigger send
    ib.set_text("Hello NiceGUI")
    ib._trigger_send()
    assert sent_messages == ["Hello NiceGUI"]
    assert ib.textarea.value == ""

    # Test shift+enter (inserts newline)
    mock_shift_event = MagicMock()
    mock_shift_event.args = {"shiftKey": True}
    ib.set_text("Line 1")
    ib._handle_enter(mock_shift_event)
    assert ib.textarea.value == "Line 1\n"

    # Test regular enter (triggers send, strips whitespace)
    mock_enter_event = MagicMock()
    mock_enter_event.args = {"shiftKey": False}
    ib._handle_enter(mock_enter_event)
    assert sent_messages == ["Hello NiceGUI", "Line 1"]

    # Test set_busy
    assert ib.progress_container is not None
    assert ib.progress_container.visible is False

    ib.set_busy(True)
    assert ib.is_busy is True
    assert ib.progress_container.visible is True
    # When busy, _trigger_send is allowed so messages can be queued
    ib.set_text("Queued Message")
    ib._trigger_send()
    assert sent_messages == ["Hello NiceGUI", "Line 1", "Queued Message"]

    ib.set_busy(False)
    assert ib.is_busy is False
    assert ib.progress_container.visible is False

    # Test queue rendering and interactions
    edited = []
    deleted = []
    reordered = []
    resumed = []
    ib.on_edit_queued = edited.append
    ib.on_delete_queued = deleted.append
    ib.on_reorder_queued = lambda s, t, p: reordered.append((s, t, p))
    ib.on_resume_queue = lambda: resumed.append(True)

    # 1. Empty queue -> hidden
    ib.render_queue([])
    assert ib.queue_container.visible is False

    # 2. 1 item -> visible, no drag handle
    q1 = QueuedMessage(text="First queued line\nSecond line ignored")
    ib.render_queue([q1])
    assert ib.queue_container.visible is True
    assert "1" in ib.queue_count_label.text
    assert len(ib.queue_list.default_slot.children) == 1

    # 3. 2 items -> visible with drag handle
    q2 = QueuedMessage(text="Second queued message")
    ib.render_queue([q1, q2])
    assert ib.queue_container.visible is True
    assert "2" in ib.queue_count_label.text
    assert len(ib.queue_list.default_slot.children) == 2

    # Test trigger edit, delete, reorder, resume
    ib._trigger_edit(q1.id)
    assert edited == [q1.id]

    ib._trigger_delete(q2.id)
    assert deleted == [q2.id]

    ib._trigger_reorder(q1.id, q2.id, "after")
    assert reordered == [(q1.id, q2.id, "after")]

    ib._trigger_resume_queue()
    assert resumed == [True]

    # Test focus
    ib.textarea.run_method = MagicMock()
    ib.focus()
    ib.textarea.run_method.assert_called_with("focus")


# =============================================================================
# HEADER TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_header_component(ui_context: Client) -> None:
    drawer_toggled = False

    def toggle_drawer():
        nonlocal drawer_toggled
        drawer_toggled = True

    with ui_context:
        header = HeaderComponent(
            on_toggle_drawer=toggle_drawer,
        )
        header.build()

        assert not hasattr(header, "mode_toggle")


# =============================================================================
# SETTINGS DRAWER TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_settings_drawer_component(ui_context: Client) -> None:
    saved = False

    async def on_saved():
        nonlocal saved
        saved = True

    with ui_context:
        drawer = SettingsDrawerComponent(on_settings_saved=on_saved)
        drawer.build()

        assert drawer.select_reasoning is not None
        assert drawer.select_reasoning.value == "xhigh"
        assert drawer.select_reasoning.options == ["none", "low", "medium", "xhigh"]

        with patch(
            "fe.ui.settings_drawer.check_provider_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = (True, ["alpha-model", "beta-model"])
            await drawer.refresh_models()
            assert drawer.status_badge.text == "Online (2 models)"
            assert "alpha-model" in drawer.model_select.options
            assert "beta-model" in drawer.model_select.options

        # Test toggle
        drawer.toggle()

        # Test save settings
        with patch("fe.config.AppConfig.save_to_env"):
            await drawer._save_settings()
        assert saved is True


def test_project_settings_update_all_chats_and_keep_local_approvals(
    ui_context: Client, tmp_path: Path
) -> None:
    project = session_manager.create_project(tmp_path)
    first = session_manager.create_session(project_id=project.id)
    second = session_manager.create_session(project_id=project.id)
    first.settings.tool_approval_mode = "Approve All"
    with ui_context:
        dialog = ProjectSettingsDialogComponent()
        dialog.build()
        dialog.open(project.id)
        dialog.input_hds.value = "shared-prompt"
        dialog.switch_coding_history.value = False
        dialog.approval_select.value = "Approve via LLM"
        dialog._save_settings()
        assert dialog.root_label.text == str(tmp_path)
        assert not hasattr(dialog, "input_work_root")
    assert first.settings.name_hds == second.settings.name_hds == "shared-prompt"
    assert first.settings.tool_approval_mode == "Approve All"
    assert second.settings.tool_approval_mode == "Manual"
    new = session_manager.create_session(project_id=project.id)
    assert new.settings.tool_approval_mode == "Approve via LLM"
    assert new.settings.coding_agent_persistent_history is False


# =============================================================================
# SIDEBAR TESTS
# =============================================================================


def test_sidebar_component(ui_context: Client) -> None:
    switched_ids = []
    new_chat_called = False

    def on_new_chat():
        nonlocal new_chat_called
        new_chat_called = True

    with ui_context:
        sidebar = SidebarComponent(
            on_session_switched=lambda sid: switched_ids.append(sid),
            on_new_chat=on_new_chat,
        )
        sidebar.build()

    # New chat
    sidebar._handle_new_chat()
    assert new_chat_called is True

    # Select session
    curr = session_manager.get_current_session()
    sidebar._select_session(curr.id)
    assert curr.id in switched_ids

    # Delete session
    sidebar._delete_session(curr.id)
    assert len(switched_ids) >= 2

    # Project toggle tracking
    sidebar._on_project_toggle("test-proj", True)
    assert sidebar.is_project_expanded("test-proj")
    sidebar._on_project_toggle("test-proj", False)
    assert not sidebar.is_project_expanded("test-proj")


def test_sidebar_preserves_expanded_projects_when_switching_sessions(
    ui_context: Client, tmp_path: Path
) -> None:
    folder1 = tmp_path / "Proj1"
    folder1.mkdir()
    folder2 = tmp_path / "Proj2"
    folder2.mkdir()
    p1 = session_manager.create_project(str(folder1))
    p2 = session_manager.create_project(str(folder2))
    s1 = session_manager.create_session(project_id=p1.id)
    s2 = session_manager.create_session(project_id=p2.id)
    quick = session_manager.create_session(project_id=None)

    with ui_context:
        sidebar = SidebarComponent(
            on_session_switched=lambda _: None, on_new_chat=lambda: None
        )
        sidebar.build()

    # Select project 1 session -> p1 expanded
    sidebar._select_session(s1.id)
    assert sidebar.is_project_expanded(p1.id)

    # Select project 2 session -> both p1 and p2 remain expanded (p1 does not close!)
    sidebar._select_session(s2.id)
    assert sidebar.is_project_expanded(p1.id)
    assert sidebar.is_project_expanded(p2.id)

    # Select quick chat -> neither p1 nor p2 close!
    sidebar._select_session(quick.id)
    assert sidebar.is_project_expanded(p1.id)
    assert sidebar.is_project_expanded(p2.id)

    # Explicit toggle collapse p1 -> p1 collapses, p2 stays open
    sidebar._on_project_toggle(p1.id, False)
    assert not sidebar.is_project_expanded(p1.id)
    assert sidebar.is_project_expanded(p2.id)


# =============================================================================
# CHAT VIEW TESTS
# =============================================================================


def test_chat_view_component(ui_context: Client) -> None:
    suggested = []
    cv = ChatViewComponent(on_suggested_prompt=lambda p: suggested.append(p))
    cv.build()

    # Session with diverse messages: user, assistant with reasoning, skills, tools
    b1 = ReasoningBlock(
        skill_path="/", depth=0, generation_index=1, content="Root reasoning"
    )
    b2 = ReasoningBlock(
        skill_path="/mock-skill", depth=1, generation_index=2, content="Skill reasoning"
    )

    tool_ev = ToolEvent(tool_name="test_cmd", depth=1, skill_path="/mock-skill")

    sub_skill = SkillNode(
        name="sub",
        path="/mock-skill/sub",
        depth=2,
        status="completed",
        is_collapsed=False,
    )

    skill_active = SkillNode(
        name="active-skill",
        path="/active",
        depth=1,
        status="active",
        reasoning_blocks=[b2],
        tool_events=[tool_ev],
        sub_skills=[sub_skill],
    )

    skill_completed = SkillNode(
        name="completed-skill",
        path="/completed",
        depth=1,
        status="completed",
        is_collapsed=True,
        reasoning_blocks=[b2],
        tool_events=[tool_ev],
        sub_skills=[sub_skill],
    )

    sess = session_manager.get_current_session()
    sess.messages = [
        ChatMessage(role="user", content="Hello test"),
        ChatMessage(
            role="assistant",
            content="Reply message",
            reasoning_blocks=[b1],
            skills=[skill_active, skill_completed],
        ),
        # Assistant message with tool events fallback
        ChatMessage(
            role="assistant",
            content="Tool only message",
            tool_events=[tool_ev],
        ),
        # Assistant message with flat reasoning_content fallback
        ChatMessage(
            role="assistant",
            content="Fallback reasoning",
            reasoning_content="Flat reasoning fallback",
        ),
    ]

    # Render session with populated messages
    cv.render_current_session()

    # Streaming workflow
    cv.start_assistant_stream()
    cv.stream_reasoning(
        "Reasoning token 1 ", skill_path="/", depth=0, generation_index=1
    )
    cv.stream_reasoning(
        "Reasoning token 2", skill_path="/", depth=0, generation_index=1
    )
    assert cv.current_live_reasoning_container is not None
    assert cv.current_live_reasoning_container.visible is True
    assert cv.current_master_reasoning_expansion is None
    assert len(cv.streaming_reasoning_blocks) == 1
    assert cv.streaming_reasoning_blocks[0].content == (
        "Reasoning token 1 Reasoning token 2"
    )

    # Different block arrives -> block 1 auto-compacts while block 2 becomes active
    cv.stream_reasoning(
        "Skill reasoning", skill_path="/sub", depth=1, generation_index=2
    )
    assert len(cv.streaming_reasoning_blocks) == 2
    assert cv.current_master_reasoning_expansion is not None
    assert cv.current_master_reasoning_expansion.value is False
    assert cv.current_live_reasoning_block is cv.streaming_reasoning_blocks[1]
    assert cv.current_reasoning_container.default_slot.children == [
        cv.current_master_reasoning_expansion,
        cv.current_live_reasoning_container,
    ]
    assert cv.streaming_reasoning_blocks[1].content == "Skill reasoning"
    assert "Skill reasoning" in cv.current_thinking_label.content
    assert "Reasoning token 1" not in cv.current_thinking_label.content

    # A tool finishes the second step; another step appears below the summary.
    cv.add_tool_event(ToolEvent(tool_name="root_tool", skill_path="/", depth=0))
    assert cv.current_live_reasoning_container is None
    assert cv.current_live_reasoning_block is None
    cv.stream_reasoning("Final thought", skill_path="/", depth=0, generation_index=3)
    assert cv.current_master_reasoning_expansion is not None
    assert cv.current_master_reasoning_expansion.value is False
    assert cv.current_live_reasoning_container is not None
    assert cv.current_live_reasoning_block is cv.streaming_reasoning_blocks[2]
    assert len(cv.streaming_reasoning_blocks) == 3
    assert cv.current_reasoning_container.default_slot.children == [
        cv.current_master_reasoning_expansion,
        cv.current_live_reasoning_container,
    ]
    assert "Final thought" in cv.current_thinking_label.content

    # Response content finishes the last live step.
    cv.stream_token("Chunk 1 ")
    assert cv.current_live_reasoning_container is None
    assert cv.current_live_reasoning_block is None
    assert cv.current_master_reasoning_expansion is not None
    assert cv.current_master_reasoning_expansion.value is False
    assert cv.current_reasoning_container.default_slot.children == [
        cv.current_master_reasoning_expansion
    ]

    cv.handle_live_skill_event(skill_active, "opened")

    cv.finalize_stream()
    assert cv.current_stream_card is None


def test_chat_view_welcome_hero(ui_context: Client) -> None:
    suggested = []
    cv = ChatViewComponent(on_suggested_prompt=lambda p: suggested.append(p))
    cv.build()

    sess = session_manager.get_current_session()
    sess.messages.clear()
    cv.render_current_session()

    # Chat container should show welcome hero
    assert cv.chat_container is not None


def test_reselecting_chat_preserves_live_thinking_and_response(
    ui_context: Client,
) -> None:
    with ui_context:
        session = session_manager.get_current_session()
        session.messages = [ChatMessage(role="user", content="Explain the code")]
        child = SubAgentConversation(path="main/coding-agent", status="completed")
        child.messages = [ChatMessage(role="assistant", content="Child result")]
        session.sub_agent_conversations = [child]

        view = ChatViewComponent(on_suggested_prompt=lambda _: None)
        view.build()
        view.start_assistant_stream()
        view.stream_reasoning("Thinking now", generation_index=1)
        view.stream_token("Partial answer")
        stream_row = view.current_stream_card
        assert stream_row is not None
        assert stream_row.id in ui_context.elements

        view.show_main()
        assert stream_row.id in ui_context.elements
        assert view.current_response_text == "Partial answer"
        assert view.streaming_reasoning_blocks[0].content == "Thinking now"

        view.show_sub_agent(child.id)
        assert stream_row.id in ui_context.elements
        assert view.child_chat_container is not None
        assert view.child_chat_container.visible is True
        view.show_main()
        assert stream_row.id in ui_context.elements
        assert view.chat_container is not None
        assert view.chat_container.visible is True

        other_session = session_manager.create_session()
        view.show_main()
        assert stream_row.id in ui_context.elements
        assert view.other_chat_container is not None
        assert view.other_chat_container.visible is True
        session_manager.switch_session(session.id)
        view.show_main()
        view.stream_token(" continued")
        assert stream_row.id in ui_context.elements
        assert view.current_response_text == "Partial answer continued"

        view.finalize_stream()
        assert view.stream_session_id is None
        session_manager.switch_session(other_session.id)


def test_sidebar_edge_cases(ui_context: Client) -> None:
    with ui_context:
        sidebar = SidebarComponent(
            on_session_switched=lambda _: None, on_new_chat=lambda: None
        )
        # 1. render_sessions when container is None (early return)
        sidebar.render_sessions()

        sidebar.build()
        # 2. render_sessions when sessions is empty
        with patch.object(session_manager, "list_sessions_sorted", return_value=[]):
            sidebar.render_sessions()

        # 3. Verify delete button on session item has sidebar-chat-delete-btn class
        session_manager.create_session()
        sidebar.render_sessions()
        assert sidebar.container is not None
        delete_btn = next(
            (
                el
                for el in sidebar.container.descendants()
                if isinstance(el, ui.button)
                and "sidebar-chat-delete-btn" in el._classes
            ),
            None,
        )
        assert delete_btn is not None
        assert "sidebar-action-btn" in delete_btn._classes


def test_chat_view_edge_cases(ui_context: Client) -> None:
    with ui_context:
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)

        # 1. Unbuilt container guards
        cv.render_current_session()
        cv.start_assistant_stream()
        cv.stream_reasoning("test", "/", 0, 1)
        cv.handle_live_skill_event(SkillNode(), "opened")

        cv.build()

        # 2. current_reasoning_container is None guard
        cv.current_reasoning_container = None
        cv.stream_reasoning("test", "/", 0, 1)

        # 3. current_skills_container is None guard
        cv.current_skills_container = None
        cv.handle_live_skill_event(SkillNode(), "opened")

        # 4. Finalize a partially built stream
        cv.finalize_stream()

        # 5. Toggle collapse on completed skill
        node = SkillNode(
            name="test-skill",
            path="/test-skill",
            depth=1,
            status="completed",
            is_collapsed=True,
        )
        with ui.column() as parent_col:
            cv._render_skill_node(node)
            # Find the header row with click listener
            for child in parent_col.descendants():
                if hasattr(child, "_event_listeners"):
                    for listener in child._event_listeners.values():
                        if getattr(listener, "type", "") == "click":
                            listener.handler()
                            assert node.is_collapsed is False
                            listener.handler()
                            assert node.is_collapsed is True


def test_chat_scroll_js_and_reset_scroll(ui_context: Client) -> None:
    assert CHAT_SCROLL_JS_PATH.is_file()
    js_code = get_chat_scroll_js()
    assert "__exmekhChatScroll" in js_code
    assert "chat-scroll-container" in js_code

    with patch("fe.ui.chat_view.CHAT_SCROLL_JS_PATH", Path("nonexistent_scroll.js")):
        assert get_chat_scroll_js() == ""
        with patch("fe.ui.chat_view.ui.add_head_html") as mock_add_head:
            load_chat_scroll_js()
            mock_add_head.assert_not_called()

    with ui_context:
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
        cv.build()
        assert cv.chat_container is not None
        assert "chat-scroll-container" in cv.chat_container._classes
        assert cv.poi_track is not None
        assert "chat-poi-track" in cv.poi_track._classes
        initial_seq = cv.scroll_reset_seq
        cv.reset_scroll_to_bottom()
        assert cv.scroll_reset_seq == initial_seq + 1
        assert cv.chat_container._props["data-scroll-reset-seq"] == str(
            cv.scroll_reset_seq
        )
        assert cv.chat_container._props["data-scroll-pinned"] == "true"


def test_chat_view_timers_rendering_and_live_updates(ui_context: Client) -> None:
    with ui_context:
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
        cv.build()

        # 1. Verify static rendering of timers for general turn, thinking blocks, skills, and tools
        b1 = ReasoningBlock(
            skill_path="/",
            depth=0,
            generation_index=1,
            content="Step 1 thought",
            duration_seconds=30.0,
        )
        b2 = ReasoningBlock(
            skill_path="/skill-a",
            depth=1,
            generation_index=2,
            content="Step 2 thought",
            duration_seconds=40.0,
        )
        tool_completed = ToolEvent(
            tool_name="fast_tool",
            depth=0,
            skill_path="/",
            duration_seconds=59.0,
        )
        skill_completed = SkillNode(
            name="skill-a",
            path="/skill-a",
            depth=1,
            status="completed",
            start_time=100.0,
            end_time=7560.0,
            duration_seconds=7460.0,  # 2h 4m 20s
            reasoning_blocks=[b2],
            tool_events=[tool_completed],
            is_collapsed=False,
        )

        sess = session_manager.get_current_session()
        sess.messages = [
            ChatMessage(
                role="assistant",
                content="Completed turn with timers",
                reasoning_blocks=[b1, b2],
                skills=[skill_completed],
                duration_seconds=70.0,  # 1m 10s
            )
        ]
        cv.render_current_session()

        label_texts = [
            getattr(el, "text", "")
            for el in cv.chat_container.descendants()
            if hasattr(el, "text")
        ]
        # General duration: 70s -> 1m 10s
        assert any("1m 10s" in t for t in label_texts)
        # Individual thinking blocks: 30s and 40s, plus master sum (70s -> 1m 10s)
        assert any("30s" in t for t in label_texts)
        assert any("40s" in t for t in label_texts)
        # Completed tool duration: 59s
        assert any("59s" in t for t in label_texts)
        # Completed skill (open-to-close) duration: 7460s -> 2h 4m 20s
        assert any("2h 4m 20s" in t for t in label_texts)

        # 2. Verify live ticking timers during streaming
        cv.start_assistant_stream()
        cv.stream_reasoning(
            "Live thinking...", skill_path="/", depth=0, generation_index=1
        )
        assert cv.current_live_reasoning_timer_label is not None
        assert cv.current_live_reasoning_block is not None
        wall_time = 10_000.0
        cv.current_live_reasoning_block.timestamp = wall_time - 70.0

        active_skill = SkillNode(
            name="live-skill",
            path="/live-skill",
            depth=1,
            status="active",
            start_time=wall_time - 7460.0,  # 2h 4m 20s
        )
        cv.handle_live_skill_event(active_skill, "opened")

        running_tool = ToolEvent(
            tool_call_id="call_live_1",
            event_type="tool_start",
            status="running",
            tool_name="live_tool",
            skill_path="/",
            depth=0,
        )
        running_tool.timestamp = wall_time - 59.0
        cv.add_tool_event(running_tool)

        assert cv.stream_started_at is not None
        cv.stream_started_at = 930.0
        with patch("fe.ui.chat_view.time") as clock:
            clock.time.return_value = wall_time
            clock.monotonic.return_value = 1000.0
            cv._update_stream_duration()

        assert cv.stream_duration_label is not None
        assert "1m 10s" in cv.stream_duration_label.text
        assert len(cv.live_skill_timer_labels) == 1
        assert "2h 4m 20s" in cv.live_skill_timer_labels[0][1].text
        assert len(cv.live_tool_timer_labels) == 1
        assert "59s" in cv.live_tool_timer_labels[0][1].text

        # Complete the running tool -> removes from live_tool_timer_labels and renders completed timer
        end_tool = ToolEvent(
            tool_call_id="call_live_1",
            event_type="tool_end",
            status="completed",
            tool_name="live_tool",
            skill_path="/",
            depth=0,
            duration_seconds=70.0,
        )
        cv.add_tool_event(end_tool)
        assert len(cv.live_tool_timer_labels) == 0

        cv.finalize_stream()


@pytest.mark.asyncio
async def test_page_controller_message_queue_flow(ui_context: Client) -> None:
    with ui_context:
        controller = ChatPageController(ui_context)
        controller.build()
        session = session_manager.get_current_session()
        session.queued_messages.clear()
        controller._render_queue_ui(session)

        assert controller.input_bar is not None
        assert controller.input_bar.queue_container.visible is False

        # Simulate busy agent turn
        fake_task = asyncio.create_task(asyncio.sleep(10))
        controller._active_task = fake_task
        active_turns.start(session.id, fake_task, TurnStreamCollector(), started_at=0)

        # Send while busy -> message queued
        await controller.send("First message queued")
        assert len(session.queued_messages) == 1
        assert session.queued_messages[0].text == "First message queued"
        assert controller.input_bar.queue_container.visible is True

        # Send second message while busy -> message queued
        await controller.send("Second message queued")
        assert len(session.queued_messages) == 2
        m1 = session.queued_messages[0]
        m2 = session.queued_messages[1]

        # Reorder via event
        controller._handle_queue_reorder_event(
            SimpleNamespace(
                args={"source_id": m1.id, "target_id": m2.id, "position": "after"}
            )
        )
        assert [m.id for m in session.queued_messages] == [m2.id, m1.id]

        # Edit queued message: removes from queue and puts in textarea
        controller.edit_queued_message(m2.id)
        assert len(session.queued_messages) == 1
        assert controller.input_bar.textarea.value == m2.text

        # Delete queued message: removes from queue
        controller.delete_queued_message(m1.id)
        assert len(session.queued_messages) == 0
        assert controller.input_bar.queue_container.visible is False

        # Sequential dispatch test:
        # Add two messages to queue
        session.add_queued_message("Next 1")
        session.add_queued_message("Next 2")
        fake_task.cancel()
        active_turns.finish(session.id, fake_task)
        await asyncio.gather(fake_task, return_exceptions=True)
        controller._active_task = None

        with patch.object(controller, "_send_prepared") as mock_send_prepared:
            controller._check_and_dispatch_next_queued(session)
            assert mock_send_prepared.call_args.args[1].prompt == "Next 1"
            assert len(session.queued_messages) == 1
            assert session.queued_messages[0].text == "Next 2"

        # Resume queue test
        with patch.object(controller, "_send_prepared") as mock_send_prepared:
            controller.resume_queue()
            assert mock_send_prepared.call_args.args[1].prompt == "Next 2"
            assert len(session.queued_messages) == 0

        # Stop while active task running
        fake_task2 = asyncio.create_task(asyncio.sleep(10))
        controller._active_task = fake_task2
        active_turns.start(session.id, fake_task2, TurnStreamCollector(), started_at=0)
        session.add_queued_message("Kept in queue")
        controller.stop()
        await asyncio.sleep(0)
        assert fake_task2.cancelled()
        active_turns.finish(session.id, fake_task2)
        # Queued messages remain preserved
        assert len(session.queued_messages) == 1
        session.queued_messages.clear()


async def test_pick_folder_dialog_and_browse(
    ui_context: Client, tmp_path: Path
) -> None:
    folder = tmp_path / "TestPickFolder"
    folder.mkdir()

    with (
        patch("fe.ui.new_project_dialog.tkinter.Tk") as tk,
        patch(
            "fe.ui.new_project_dialog.filedialog.askdirectory", return_value=str(folder)
        ),
    ):
        assert pick_folder_dialog(folder) == str(folder)
        tk.return_value.destroy.assert_called_once()

    with (
        patch("fe.ui.new_project_dialog.tkinter.Tk") as tk,
        patch(
            "fe.ui.new_project_dialog.filedialog.askdirectory",
            side_effect=OSError("picker failed"),
        ),
    ):
        with pytest.raises(OSError, match="picker failed"):
            pick_folder_dialog(folder)
        tk.return_value.destroy.assert_called_once()

    # 2. Test NewProjectDialogComponent.browse with mock
    with ui_context:
        dialog = NewProjectDialogComponent(
            on_created=lambda _: None, settings_dialog=MagicMock()
        )
        dialog.folder_input = MagicMock()
        dialog.folder_input.value = "initial"

        with patch(
            "fe.ui.new_project_dialog.pick_folder_dialog", return_value=str(folder)
        ):
            await dialog.browse()
            assert dialog.folder_input.value == str(folder)


def test_chat_view_steering_loading_spinner(ui_context: Client) -> None:
    with ui_context:
        cv = ChatViewComponent(on_suggested_prompt=lambda _: None)
        cv.build()

        # 1. Start streaming turn
        cv.start_assistant_stream()
        assert len(cv.streaming_steering_spinners) == 0

        # 2. Receive live steering message
        msg1 = QueuedMessage(id="steering-1", text="Redirect to approach B")
        cv.show_steering_messages([msg1])

        # Spinner should be created and registered
        assert "steering-1" in cv.streaming_steering_spinners
        spinner1 = cv.streaming_steering_spinners["steering-1"]
        assert spinner1.visible is True

        # 3. Add second steering message
        msg2 = QueuedMessage(id="steering-2", text="Also check edge cases")
        cv.show_steering_messages([msg2])
        assert "steering-2" in cv.streaming_steering_spinners
        spinner2 = cv.streaming_steering_spinners["steering-2"]
        assert spinner2.visible is True

        # 4. First steering message gets confirmed and inserted into history
        turn_msg = ChatMessage(
            role="assistant",
            content="Working on approach B...",
            steering_messages=[msg1, msg2],
            steering_insertions=[
                SteeringInsertion(
                    message_id="steering-1",
                    after_tool_call_id=None,
                    tool_name=None,
                    skill_path="/",
                )
            ],
        )
        cv.show_steering_insertions(turn_msg)

        # spinner1 should now be popped from dict and hidden
        assert "steering-1" not in cv.streaming_steering_spinners
        assert spinner1.visible is False

        # spinner2 should still be pending
        assert "steering-2" in cv.streaming_steering_spinners
        assert spinner2.visible is True

        # 5. Finalize stream hides all remaining spinners and clears dict
        cv.finalize_stream()
        assert len(cv.streaming_steering_spinners) == 0
        assert spinner2.visible is False

        # 6. Test restore_stream_snapshot with mixed inserted and pending messages
        cv.start_assistant_stream()
        cv.restore_stream_snapshot(turn_msg)
        # msg1 was already in steering_insertions, so only msg2 should get a pending spinner
        assert "steering-1" not in cv.streaming_steering_spinners
        assert "steering-2" in cv.streaming_steering_spinners
        cv.finalize_stream()

        # 7. Static historical message should render without pending spinner
        historical_msg = ChatMessage(
            role="assistant",
            content="Done",
            steering_messages=[msg1],
            steering_insertions=[
                SteeringInsertion(
                    message_id="steering-1",
                    after_tool_call_id=None,
                    tool_name=None,
                    skill_path="/",
                )
            ],
        )
        cv._render_message(historical_msg)
        assert len(cv.streaming_steering_spinners) == 0
