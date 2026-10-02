from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
)
from exmekh_lib.core.models.assistant_message import AssistantMessage
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.converter import chat_message_to_exmekh_messages
from exmekh_agent.exmekh_bridge.event_handlers import DeepAgentTurnEventHandler
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.sub_agents import SubAgentEventRouter
from exmekh_agent.exmekh_bridge.tool_calls import (
    extract_tool_arguments,
    parse_tool_calls,
)
from exmekh_agent.exmekh_bridge.tool_events import tool_event_from_execution
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat_export import export_chat
from exmekh_agent.ui.graph_data import build_graph_data_from_session
from exmekh_agent.ui.message_text import message_display_text
from exmekh_agent.ui.raw_history_view import RawHistoryComponent


def recorded_call(call_id, arguments):
    return {"id": call_id, "function": {"name": "bash", "arguments": arguments}}


def execution_event(kind, history):
    return DeepAgentExecutionEvent(
        type=kind,
        depth=1,
        skill_path="/worker",
        history=history,
        history_tree=[],
        tool_name="bash",
        tool_call_id="call",
    )


def test_every_tool_owner_keeps_one_result_and_late_arguments():
    collector = TurnStreamCollector()
    reasoning = ReasoningBlock(content="Before tool execution")
    skill = SkillNode(path="/worker", reasoning_blocks=[reasoning])
    child = SubAgentConversation(path="main/worker")
    child.start_call("Run")
    client = Client(page("/tools-review"))
    with client:
        raw = RawHistoryComponent()
        raw.build()
        raw.start_live_turn(ChatSession())
        start = ToolEvent(
            tool_name="bash", tool_call_id="call", event_type="tool_start", timestamp=10
        )
        end = ToolEvent(
            tool_name="bash",
            tool_call_id="call",
            event_type="tool_end",
            timestamp=15,
            arguments={"command": {"text": "echo hello"}},
        )
        for record in [
            collector.append_tool,
            skill.add_tool_event,
            child.add_tool_event,
            raw.add_tool_event,
        ]:
            for event in [start, start, end, end, start]:
                record(event)
        records = [
            collector.tools,
            skill.tool_events,
            child.message_history()[-1].tool_events,
            raw.live_tool_events,
        ]
        for events in records:
            assert len(events) == 1
            assert not events[0].is_running
            assert events[0].elapsed_seconds() == 5
            assert events[0].arguments == end.arguments
        summaries = [
            entry
            for entry in raw.raw_json_data
            if entry.get("source") == "live_tool_event"
        ]
        assert reasoning.end_time is not None
        assert len(summaries) == 1
        assert summaries[0]["status"] == "completed"
        assert summaries[0]["arguments"] == end.arguments
        end.arguments["command"]["text"] = "Publisher changed"
        assert all(
            events[0].arguments["command"]["text"] == "echo hello" for events in records
        )
        skill.merge_snapshot(SkillNode(path="/worker", tool_events=[start]))
        assert not skill.tool_events[0].is_running


def test_execution_adapter_and_handler_keep_actual_elapsed_time():
    history = [{"tool_calls": [recorded_call("call", {"command": "echo hello"})]}]
    start = execution_event(DeepAgentExecutionEventType.TOOL_START, [])
    end = execution_event(DeepAgentExecutionEventType.TOOL_END, history)
    mapped = tool_event_from_execution(start, now=0)
    assert mapped.timestamp == 0 and mapped.is_running
    mapped.complete_from(tool_event_from_execution(end, now=5))
    assert mapped.elapsed_seconds() == 5 and mapped.arguments == {
        "command": "echo hello"
    }
    snapshots = []
    tools = []
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None),
        lambda _: None,
        lambda *_: None,
        lambda event: tools.append(event.model_copy(deep=True)),
        lambda node, _: snapshots.append(node.model_copy(deep=True)),
        None,
    )
    with patch(
        "exmekh_agent.exmekh_bridge.event_handlers.time.time",
        side_effect=[10, 10, 15, 15],
    ):
        handler.handle(start)
        handler.handle(end)
    assert tools[-1].elapsed_seconds() == 5
    assert snapshots[-1].tool_events[0].arguments == {"command": "echo hello"}
    assert not snapshots[-1].tool_events[0].is_running


def test_model_resolves_legacy_nested_arguments_without_ui_traversal():
    event = ToolEvent(tool_name="bash", tool_call_id="call")
    message = ChatMessage(
        role="assistant",
        skill_history_tree=[
            {
                "history": [],
                "children": [
                    {
                        "history": [
                            {
                                "tool_calls": [
                                    recorded_call("call", {"command": "nested"})
                                ]
                            }
                        ]
                    }
                ],
            }
        ],
    )
    assert message.arguments_for_tool(event) == {"command": "nested"}
    assert message.arguments_for_tool(ToolEvent(tool_name="unknown")) == {}
    assert message.arguments_for_tool(
        ToolEvent(tool_name="bash", arguments={"command": "recorded"})
    ) == {"command": "recorded"}
    message.raw_history = [{"tool_calls": [recorded_call("call", {"command": "root"})]}]
    assert message.arguments_for_tool(event) == {"command": "root"}
    assert SkillNode(tool_events=[event]).export_data(include_history=True)[
        "tool_events"
    ]


def test_merging_historical_tools_does_not_close_new_reasoning():
    tool = ToolEvent(tool_name="bash", tool_call_id="call")
    skill = SkillNode(path="/worker", tool_events=[tool])
    snapshot = SkillNode(
        path="/worker",
        tool_events=[tool],
        reasoning_blocks=[ReasoningBlock(content="Reasoning after the tool")],
    )
    skill.merge_snapshot(snapshot)
    assert skill.reasoning_blocks[0].end_time is None


def test_skill_snapshots_preserve_closed_invocations_and_do_not_share_children() -> (
    None
):
    failed = SkillNode(
        path="/root/worker",
        opening_tool_call_id="first",
        status="error",
        start_time=2,
        end_time=4,
    )
    active = SkillNode(path="/root/worker", opening_tool_call_id="second", start_time=5)
    root = SkillNode(path="/root", start_time=1, sub_skills=[failed, active])
    closed = SkillNode(
        path="/root",
        status="completed",
        start_time=1,
        end_time=10,
        sub_skills=[
            SkillNode(
                path="/root/worker",
                opening_tool_call_id="second",
                status="completed",
                start_time=5,
                end_time=8,
            ),
            SkillNode(
                path="/root/other",
                start_time=6,
                reasoning_blocks=[ReasoningBlock(content="Last thought", timestamp=6)],
            ),
        ],
    )
    root.merge_snapshot(closed)
    assert len(root.sub_skills) == 3
    assert root.sub_skills[0].status == "error"
    assert root.sub_skills[1].end_time == 8
    assert root.sub_skills[2].status == "completed"
    closed.sub_skills[1].reasoning_blocks[0].content = "Publisher changed"
    assert root.sub_skills[2].reasoning_blocks[0].content == "Last thought"
    stale = SkillNode(
        path="/root",
        start_time=1,
        sub_skills=[SkillNode(path="/root/worker", opening_tool_call_id="first")],
    )
    root.merge_snapshot(stale)
    assert root.status == "completed" and root.end_time == 10
    assert root.sub_skills[0].status == "error" and root.sub_skills[0].end_time == 4
    assert root.sub_skills[1].status == "completed"
    assert len(root.sub_skills) == 3


@pytest.mark.parametrize("error", [None, "Failed execution"])
def test_collector_finalizes_owned_work_and_exports_independent_snapshots(
    error,
) -> None:
    collector = TurnStreamCollector()
    skill = SkillNode(
        path="/root",
        depth=1,
        sub_skills=[SkillNode(path="/root/child", depth=2)],
        reasoning_blocks=[ReasoningBlock(content="Skill thought")],
    )
    raw = [{"role": "assistant", "metadata": {"value": "Original"}}]
    tree = [{"skill": "root", "history": [{"content": "Original"}]}]
    collector.append_skill(skill)
    collector.set_raw_history(raw, tree)
    skill.sub_skills[0].name = "Publisher changed"
    raw[0]["metadata"]["value"] = "Publisher changed"
    tree[0]["history"][0]["content"] = "Publisher changed"
    collector.append_tool(ToolEvent(tool_name="read", status="running"))
    collector.append_reasoning("Root thought")
    snapshot = collector.snapshot_message()
    assert snapshot.skills[0].sub_skills[0].name == "root"
    assert snapshot.raw_history[0]["metadata"]["value"] == "Original"
    assert snapshot.skill_history_tree[0]["history"][0]["content"] == "Original"
    snapshot.raw_history[0]["metadata"]["value"] = "Observer changed"
    snapshot.skill_history_tree[0]["history"][0]["content"] = "Observer changed"
    snapshot.skills[0].sub_skills[0].status = "error"
    final = collector.build_message(error=error)
    expected = "completed" if error is None else "error"
    assert final.skills[0].status == final.skills[0].sub_skills[0].status == expected
    assert final.tool_events[0].execution_status == expected
    assert final.reasoning_blocks[0].end_time is not None
    assert snapshot.tool_events[0].is_running
    assert final.raw_history[0]["metadata"]["value"] == "Original"
    assert final.skill_history_tree[0]["history"][0]["content"] == "Original"
    final.skills[0].sub_skills[0].name = "Final message changed"
    assert collector.skill_snapshot("/root/child").name == "root"


def test_running_tool_without_start_type_uses_the_same_lifecycle_in_all_owners() -> (
    None
):
    collector = TurnStreamCollector()
    message = ChatMessage(role="assistant")
    skill = SkillNode()
    start = ToolEvent(
        tool_name="read", tool_call_id="call", status="running", timestamp=10
    )
    end = ToolEvent(
        tool_name="read",
        tool_call_id="call",
        event_type="tool_end",
        timestamp=15,
    )
    for record in (collector.append_tool, message.add_tool_event, skill.add_tool_event):
        record(start)
    for event in (collector.tools[0], message.tool_events[0], skill.tool_events[0]):
        assert event.is_running
        assert event.end_time is None and event.duration_seconds is None
        assert event.elapsed_seconds(now=12) == 2
    for record in (collector.append_tool, message.add_tool_event, skill.add_tool_event):
        record(end)
    for event in (collector.tools[0], message.tool_events[0], skill.tool_events[0]):
        assert event.execution_status == "completed"
        assert event.duration_seconds == 5
    assert start.event_type == "execution" and start.status == "running"


@pytest.mark.parametrize(
    ("event_type", "status", "expected"),
    [
        ("tool_start", "completed", "running"),
        ("tool_start", "error", "error"),
        ("tool_end", "running", "completed"),
    ],
)
def test_graph_interprets_legacy_and_terminal_tool_events_like_chat(
    event_type,
    status,
    expected,
) -> None:
    tool = ToolEvent(tool_name="read", event_type=event_type, status=status)
    graph = build_graph_data_from_session(
        ChatSession(messages=[ChatMessage(role="assistant", tool_events=[tool])])
    )
    execution = next(
        event
        for history in graph.histories
        for node in history.nodes
        for event in node.events
        if event.get("event_stage") == "tool_execution"
    )
    assert execution["status"] == tool.execution_status == expected
    assert tool.normalized_copy().execution_status == expected


def test_skill_counts_actions_at_every_depth() -> None:
    deepest = SkillNode(
        path="/root/child/grandchild", tool_events=[ToolEvent(tool_name="read")]
    )
    child = SkillNode(path="/root/child", sub_skills=[deepest])
    root = SkillNode(
        path="/root", tool_events=[ToolEvent(tool_name="write")], sub_skills=[child]
    )
    assert root.action_count == 2
    assert child.action_count == deepest.action_count == 1


@pytest.mark.parametrize("arguments", [None, "null", "[]", '"string"', 3])
def test_parser_rejects_non_object_arguments_consistently(arguments):
    history = [{"tool_calls": [recorded_call("call", arguments)]}]
    assert parse_tool_calls(history[0]["tool_calls"])[0].get_arguments() == {}
    assert extract_tool_arguments(history, "call") == {}


def test_parser_accepts_flat_calls_but_argument_lookup_never_invents_ids():
    flat = {"id": "call", "name": "bash", "arguments": '{"command": "echo hello"}'}
    assert extract_tool_arguments([{"tool_calls": [flat]}], "call") == {
        "command": "echo hello"
    }
    no_id = {"name": "bash", "arguments": {"command": "unmatched"}}
    assert parse_tool_calls([no_id])[0].get_name() == "bash"
    assert extract_tool_arguments([{"tool_calls": [no_id]}], "call_1") == {}


@pytest.mark.parametrize("display", [None, ""])
def test_presentation_respects_explicit_empty_text(display):
    message = ChatMessage(role="user", content="Fallback text", display_content=display)
    expected = "Fallback text" if display is None else ""
    assert message_display_text(message) == expected
    markdown = export_chat(
        ChatSession(messages=[message]), "markdown", include_history=False
    )
    assert ("Fallback text" in markdown) == (display is None)


def test_export_rejects_unknown_formats():
    with pytest.raises(ValueError, match="Unsupported chat export format"):
        export_chat(ChatSession(), "html", include_history=False)


@pytest.mark.asyncio
async def test_session_snapshot_exports_live_response_without_changing_saved_messages(
    tmp_path,
):
    manager = SessionManager(storage_dir=tmp_path)
    session = manager.get_current_session()
    session.add_message(ChatMessage(role="user", content="Question"))
    assert len(manager.snapshot_current_session().message_history()) == 1
    collector = TurnStreamCollector()
    collector.append_token("Partial response")
    task = asyncio.create_task(asyncio.sleep(10))
    active_turns.start(session.id, task, collector, started_at=0)
    try:
        snapshot = manager.snapshot_current_session()
        data = json.loads(export_chat(snapshot, "json", include_history=False))
        assert data["messages"][-1]["content"] == "Partial response"
        assert data["messages"][-1]["is_streaming"]
        assert len(session.message_history()) == 1
        snapshot.messages[0].content = "Snapshot changed"
        assert session.message_history()[0].content == "Question"
    finally:
        active_turns.cancel_and_forget(session.id)
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("position", ["before", "after", "invalid"])
def test_invalid_queue_reordering_does_not_move_the_message(position):
    session = ChatSession()
    first = session.add_queued_message("First")
    second = session.add_queued_message("Second")
    assert not session.reorder_queued_message(first.id, "missing", position)
    assert session.queued_messages == [first, second]


def test_graph_matches_parallel_tool_results_by_invocation_id():
    calls = [
        recorded_call("first", {"command": "first"}),
        recorded_call("second", {"command": "second"}),
    ]
    session = ChatSession(
        messages=[
            ChatMessage(
                role="assistant",
                content="Done",
                raw_history=[
                    {"role": "assistant", "tool_calls": calls},
                    {
                        "role": "tool",
                        "tool_call_id": "second",
                        "content": "Second result",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "first",
                        "content": "First result",
                    },
                ],
            )
        ]
    )
    graph = build_graph_data_from_session(session)
    tools = [
        node
        for history in graph.histories
        for node in history.nodes
        if node.kind == "tool_bundle"
    ]
    assert [node.events[-1]["content"] for node in tools] == [
        "First result",
        "Second result",
    ]


@pytest.mark.parametrize(
    "result_id,expected", [(None, "Legacy result"), ("unrelated", "")]
)
def test_graph_uses_positional_fallback_only_when_result_has_no_id(result_id, expected):
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "assistant", "tool_calls": [recorded_call("call", {})]},
            {"role": "tool", "tool_call_id": result_id, "content": "Legacy result"},
        ],
    )
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    tool = next(
        node
        for history in graph.histories
        for node in history.nodes
        if node.kind == "tool_bundle"
    )
    assert tool.events[-1]["content"] == expected


def test_conversion_and_graph_preserve_recorded_arguments_without_raw_history():
    event = ToolEvent(
        tool_name="bash", tool_call_id="original", arguments={"command": "echo hello"}
    )
    message = ChatMessage(role="assistant", tool_events=[event])
    calls = [
        call
        for exmekh_message in chat_message_to_exmekh_messages(message)
        if isinstance(exmekh_message, AssistantMessage)
        for call in exmekh_message.get_tool_calls() or []
    ]
    assert calls[0].get_id() == "original"
    assert calls[0].get_arguments() == event.arguments
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    tool = next(
        node
        for history in graph.histories
        for node in history.nodes
        if node.kind == "tool_bundle"
    )
    assert tool.events[0]["tool_calls"][0]["function"]["arguments"] == event.arguments
    nested = ChatMessage(
        role="assistant", skills=[SkillNode(path="/worker", tool_events=[event])]
    )
    graph = build_graph_data_from_session(ChatSession(messages=[nested]))
    tool = next(
        node
        for history in graph.histories
        for node in history.nodes
        if node.kind == "tool_bundle"
    )
    assert tool.events[0]["tool_calls"][0]["function"]["arguments"] == event.arguments
