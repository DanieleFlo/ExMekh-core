from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fe.config import AppConfig
from fe.exmekh_bridge.models import (
    ChatSession,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from fe.exmekh_bridge.sub_agents import SubAgentEvent
from fe.state.json_store import JsonRecordStore
from fe.state.session_manager import SessionManager
from fe.state.sub_agent_conversations import apply_sub_agent_event
from fe.ui.input_bar import InputBarComponent
from fe.ui.project_settings_dialog import ProjectSettingsDialogComponent
from fe.ui.sidebar import SidebarComponent


def test_config_saves_new_env_file(tmp_path) -> None:
    path = tmp_path / ".env"
    AppConfig(base_url="http://provider.test").save_to_env(path)
    assert "BASE_URL=http://provider.test" in path.read_text(encoding="utf-8")


def test_duration_uses_end_time_when_precalculated_duration_is_absent() -> None:
    assert ToolEvent(tool_name="read", timestamp=2, end_time=5).elapsed_seconds() == 3
    assert ReasoningBlock(timestamp=2, end_time=5).elapsed_seconds() == 3
    assert SkillNode(start_time=2, end_time=5).elapsed_seconds() == 3


def test_duration_uses_live_and_precalculated_values() -> None:
    assert ToolEvent(tool_name="read", duration_seconds=-2).elapsed_seconds() == 0
    assert (
        ToolEvent(tool_name="read", status="running", timestamp=2).elapsed_seconds(
            now=5
        )
        == 3
    )
    assert (
        ToolEvent(
            tool_name="read", event_type="tool_start", timestamp=2
        ).elapsed_seconds(now=5)
        == 3
    )
    assert ToolEvent(tool_name="read", timestamp=2).elapsed_seconds(now=5) == 0
    assert ReasoningBlock(duration_seconds=-2).elapsed_seconds() == 0
    assert ReasoningBlock(timestamp=2).elapsed_seconds(now=5) == 3
    assert SkillNode(duration_seconds=-2).elapsed_seconds() == 0
    assert SkillNode(status="active", start_time=2).elapsed_seconds(now=5) == 3
    assert SkillNode(status="completed", start_time=2).elapsed_seconds(now=5) == 0


def test_reasoning_keeps_precalculated_duration_when_closed() -> None:
    conversation = SubAgentConversation(path="main/child")
    conversation.start_call("task")
    block = ReasoningBlock(content="thought", duration_seconds=2)
    conversation.messages[-1].reasoning_blocks.append(block)
    conversation.append_token("answer")
    assert block.end_time is not None
    assert block.duration_seconds == 2


def test_tool_event_without_payload_does_not_change_child() -> None:
    session = ChatSession()
    session.start_sub_agent_conversation("child", "main/child", False, "task")
    event = SubAgentEvent(
        action="tool", id="child", path="main/child", persistent_history=False
    )
    conversation = apply_sub_agent_event(session, event)
    assert conversation.messages[-1].tool_events == []


def test_session_deletion_handles_missing_file_and_noncurrent_session(tmp_path) -> None:
    manager = object.__new__(SessionManager)
    manager.storage_dir = tmp_path
    manager._session_store = JsonRecordStore(tmp_path, ChatSession)
    current = ChatSession()
    other = ChatSession()
    manager.sessions = {current.id: current, other.id: other}
    manager.current_session_id = current.id
    manager.delete_session(other.id)
    assert manager.current_session_id == current.id
    assert other.id not in manager.sessions


def test_input_bar_without_elements_ignores_actions() -> None:
    sent: list[str] = []
    bar = InputBarComponent(sent.append, lambda: None)
    bar._handle_enter(SimpleNamespace())
    bar._trigger_send()
    bar.set_busy(True)
    bar.set_text("ignored")
    assert sent == []


def test_sidebar_optional_callbacks_may_be_absent() -> None:
    sidebar = SidebarComponent(lambda _: None, lambda: None)
    sidebar.render_sessions = MagicMock()
    sidebar._select_sub_agent("child")
    sidebar._open_project_settings("project")
    sidebar._new_project_chat("project")
    assert sidebar.selected_sub_agent_id == "child"


def test_project_settings_dialog_handles_unbuilt_dialog() -> None:
    dialog = ProjectSettingsDialogComponent()
    dialog.open("missing")
    dialog.close()
    dialog._save_settings()
    dialog.project_id = "missing"
    with patch(
        "fe.ui.project_settings_dialog.session_manager.save_project_settings"
    ) as save:
        dialog._save_settings()
    save.assert_not_called()
