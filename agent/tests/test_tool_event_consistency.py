from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.deep_agent import DeepAgentExecutionEvent
from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentExecutionEventType

from exmekh_agent.exmekh_bridge.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.converter import reconstruct_execution_history_and_cache
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.tool_events import tool_event_from_execution
from exmekh_agent.ui.chat_view import ChatViewComponent
from exmekh_agent.ui.input_bar import InputBarComponent


@pytest.mark.parametrize("status", ["completed", "error"])
@pytest.mark.parametrize("call_id", [None, "invocation"])
def test_tool_results_are_consistent_in_root_child_and_live_view(
    status: str, call_id: str | None
) -> None:
    collector = TurnStreamCollector()
    conversation = SubAgentConversation(path="main/worker")
    conversation.start_call("Run a tool")
    view = ChatViewComponent(lambda _: None)
    view.current_skills_container = MagicMock()
    view._refresh_streaming_skills_and_tools = MagicMock()
    recorders = [
        collector.append_tool,
        conversation.add_tool_event,
        view.add_tool_event,
    ]
    start = ToolEvent(
        tool_name="read",
        tool_call_id=call_id,
        event_type="tool_start",
        timestamp=0,
    )
    end = ToolEvent(
        tool_name="read",
        tool_call_id=call_id,
        event_type="tool_end",
        status=status,
        timestamp=3,
        end_time=0,
    )
    for record in recorders:
        record(start)
        record(end)
    root = collector.tools[0]
    child = conversation.messages[-1].tool_events[0]
    live = view.streaming_root_tools[0]
    assert root == child == live
    assert root.status == status
    assert root.end_time == root.duration_seconds == 0
    assert not root.is_running
    start.tool_name = "Changed by publisher"
    end.status = "running"
    assert root.tool_name == child.tool_name == live.tool_name == "read"
    assert root.status == child.status == live.status == status


@pytest.mark.parametrize("call_id", [None, "call"])
def test_replayed_callbacks_preserve_the_first_terminal_result(
    call_id: str | None,
) -> None:
    collector = TurnStreamCollector()
    start = ToolEvent(tool_name="read", tool_call_id=call_id, event_type="tool_start")
    failed = ToolEvent(
        id=start.id,
        tool_name="read",
        tool_call_id=call_id,
        event_type="tool_end",
        status="error",
    )
    for event in [start, start, failed, failed, start]:
        collector.append_tool(event)
    assert len(collector.tools) == 1
    assert collector.tools[0].status == "error"
    assert collector.tools[0].event_type == "tool_end"


@pytest.mark.parametrize(("path", "depth"), [("/other", 1), ("/worker", 2)])
def test_legacy_callbacks_without_ids_match_their_execution_scope(
    path: str, depth: int
) -> None:
    collector = TurnStreamCollector()
    first = ToolEvent(
        tool_name="read", event_type="tool_start", skill_path="/worker", depth=1
    )
    other = ToolEvent(
        tool_name="read", event_type="tool_start", skill_path=path, depth=depth
    )
    collector.append_tool(first)
    collector.append_tool(other)
    collector.append_tool(
        ToolEvent(
            tool_name="read", event_type="tool_end", skill_path="/worker", depth=1
        )
    )
    assert len(collector.tools) == 2
    assert not collector.tools[0].is_running
    assert collector.tools[1].is_running


def test_reconstruction_keeps_completed_invocation_ids_without_duplicate_records() -> (
    None
):
    history = MagicMock()
    message = ChatMessage(
        role="assistant",
        tool_events=[
            ToolEvent(
                tool_name="read", tool_call_id="read-call", event_type="tool_start"
            ),
            ToolEvent(
                tool_name="read", tool_call_id="read-call", event_type="tool_end"
            ),
            ToolEvent(
                tool_name="write", tool_call_id="write-call", event_type="tool_end"
            ),
        ],
    )
    reconstruct_execution_history_and_cache([message], history)
    assert [call.args for call in history.record.call_args_list] == [
        ("read", "read-call"),
        ("write", "write-call"),
    ]


def test_live_and_saved_activity_share_invocation_order_without_duplicate_skill_tools() -> (
    None
):
    view = ChatViewComponent(lambda _: None)
    skill = SkillNode(start_time=2, opening_tool_call_id="skill-call")
    before = ToolEvent(tool_name="read", timestamp=1)
    after = ToolEvent(tool_name="write", timestamp=3)
    opening = ToolEvent(tool_name="skill", timestamp=2, tool_call_id="skill-call")
    rendered = []
    view._render_skill_node = rendered.append
    view._render_tool_event = rendered.append
    message = ChatMessage(
        role="assistant", skills=[skill], tool_events=[after, opening, before]
    )
    view._render_turn_activity(message)
    assert rendered == [before.normalized_copy(), skill, after.normalized_copy()]


def test_graph_waits_while_a_child_conversation_is_active() -> None:
    session = ChatSession(
        sub_agent_conversations=[SubAgentConversation(path="main/worker")]
    )
    view = ChatViewComponent(lambda _: None)
    view.graph_view = MagicMock()
    with patch(
        "exmekh_agent.ui.chat_view.session_manager.get_current_session",
        return_value=session,
    ):
        view._render_selected_graph()
    assert view.graph_view.render_session.call_args.kwargs["is_generating"]


def test_skill_owns_lookup_and_attachment_to_the_deepest_known_parent() -> None:
    older = SkillNode(path="/root/worker", opening_tool_call_id="old")
    current = SkillNode(path="/root/worker", opening_tool_call_id="new")
    root = SkillNode(path="/root", sub_skills=[older, current])
    nested = SkillNode(path="/root/worker/nested")
    assert root.attach_descendant(nested)
    assert current.sub_skills == [nested]
    assert not older.sub_skills
    assert root.find_invocation("/root/worker") is current
    assert root.find_invocation("/root/worker", "old") is older
    assert root.find_invocation(nested.path) is nested
    assert root.find_invocation("/missing") is None
    sibling = SkillNode(path="/root/other")
    assert root.attach_descendant(sibling)
    assert root.sub_skills == [older, current, sibling]
    assert current.sub_skills == [nested]
    assert not root.attach_descendant(SkillNode(path="/unrelated"))


def test_queue_sync_detects_in_place_edits_without_rebuilding_unchanged_rows() -> None:
    bar = InputBarComponent(lambda _: None, lambda: None)
    message = QueuedMessage(text="Original guidance")
    bar.render_queue([message])
    with patch.object(bar, "render_queue", wraps=bar.render_queue) as render:
        bar.sync_queue([message], can_steer=False)
        render.assert_not_called()
        message.text = "Edited guidance"
        bar.sync_queue([message], can_steer=False)
        render.assert_called_once()
    assert bar.queued_messages[0].text == "Edited guidance"


@pytest.mark.parametrize("failed", [True, False])
def test_framework_failure_status_is_preserved_in_root_child_and_ui(
    failed: bool,
) -> None:
    start = tool_event_from_execution(
        DeepAgentExecutionEvent(
            type=DeepAgentExecutionEventType.TOOL_START,
            depth=0,
            skill_path="/",
            history=[],
            history_tree=[],
            tool_name="work",
            tool_call_id="invocation",
        ),
        now=1,
    )
    end = tool_event_from_execution(
        DeepAgentExecutionEvent(
            type=DeepAgentExecutionEventType.TOOL_END,
            depth=0,
            skill_path="/",
            history=[],
            history_tree=[],
            tool_name="work",
            tool_call_id="invocation",
            failed=failed,
        ),
        now=3,
    )
    collector = TurnStreamCollector()
    child = SubAgentConversation(path="main/worker")
    child.start_call("Run work")
    view = ChatViewComponent(lambda _: None)
    view.current_skills_container = MagicMock()
    view._refresh_streaming_skills_and_tools = MagicMock()
    for record in (collector.append_tool, child.add_tool_event, view.add_tool_event):
        record(start)
        record(end)
    root = collector.tools[0]
    assert root == child.messages[-1].tool_events[0] == view.streaming_root_tools[0]
    assert root.status == ("error" if failed else "completed")
    assert root.duration_seconds == 2
