from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    SteeringInsertion,
    SubAgentConversation,
    ToolEvent,
)
from fe.ui.chat_view import ChatViewComponent
from fe.ui.graph_data import (
    _extract_nodes_from_raw_scope,
    build_graph_data_from_session,
)
from fe.ui.graph_view import GraphViewComponent
from fe.ui.new_project_dialog import NewProjectDialogComponent, pick_folder_dialog
from fe.ui.raw_history_view import RawHistoryComponent


@pytest.mark.parametrize("status", ["completed", "error"])
@pytest.mark.parametrize("call_id", [None, "call"])
def test_generic_tool_results_and_late_arguments_are_consistent(status, call_id):
    collector = TurnStreamCollector()
    message = ChatMessage(role="assistant")
    skill = SkillNode()
    child = SubAgentConversation(path="main/child")
    child.start_call("Run")
    client = Client(page("/review-tools"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            view.build()
            view.start_assistant_stream()
            raw = RawHistoryComponent()
            raw.build()
            raw.start_live_turn(ChatSession())
            start = ToolEvent(
                tool_name="read", tool_call_id=call_id, status="running", timestamp=1
            )
            end = ToolEvent(
                tool_name="read", tool_call_id=call_id, status=status, timestamp=3
            )
            late = end.model_copy(
                deep=True, update={"arguments": {"file": "notes.txt"}}
            )
            for record in (
                collector.append_tool,
                message.add_tool_event,
                skill.add_tool_event,
                child.add_tool_event,
                view.add_tool_event,
                raw.add_tool_event,
            ):
                for event in (start, end, late, start):
                    record(event)
            for events in (
                collector.tools,
                message.tool_events,
                skill.tool_events,
                child.message_history()[-1].tool_events,
                view.streaming_root_tools,
                raw.live_tool_events,
            ):
                assert len(events) == 1
                assert events[0].execution_status == status
                assert events[0].duration_seconds == 2
                assert events[0].arguments == {"file": "notes.txt"}
            view.finalize_stream()
    finally:
        client.delete()


def test_legacy_replays_preserve_distinct_invocations_and_the_first_result():
    collector = TurnStreamCollector()
    first_start = ToolEvent(tool_name="read", status="running", timestamp=1)
    first_end = ToolEvent(tool_name="read", status="error", timestamp=2)
    second_start = ToolEvent(tool_name="read", status="running", timestamp=3)
    second_end = ToolEvent(tool_name="read", timestamp=4)
    for event in (first_start, first_end, first_end, second_start, second_end):
        collector.append_tool(event)
    assert len(collector.tools) == 2
    assert [event.execution_status for event in collector.tools] == [
        "error",
        "completed",
    ]
    assert [event.timestamp for event in collector.tools] == [1, 3]


def test_steering_matching_keeps_identity_and_fifo_for_repeated_text():
    first = QueuedMessage(text="Same guidance")
    second = QueuedMessage(text=first.text)
    message = ChatMessage(role="assistant", steering_messages=[first, second])
    assert message.find_steering_message(first.text) is first
    assert message.find_steering_message(first.text, excluded_ids={first.id}) is second
    assert (
        message.find_steering_message("Different content", message_id=second.id)
        is second
    )
    assert message.find_steering_message(first.text, message_id="legacy-id") is first
    assert (
        message.find_steering_message(first.text, excluded_ids={first.id, second.id})
        is None
    )


def test_explicit_steering_confirmations_control_live_spinners_after_restore():
    client = Client(page("/review-steering"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            view.build()
            view.start_assistant_stream()
            first = QueuedMessage(text="Same guidance")
            second = QueuedMessage(text=first.text)
            view.show_steering_messages([first, second])
            snapshot = ChatMessage(
                role="assistant",
                steering_messages=[first, second],
                raw_history_start=0,
                raw_history=[{"role": "user", "content": first.text}],
                steering_insertions=[],
            )
            view.show_steering_insertions(snapshot)
            assert set(view.streaming_steering_spinners) == {first.id, second.id}
            snapshot.steering_insertions = [
                SteeringInsertion(
                    message_id=first.id,
                    after_tool_call_id=None,
                    tool_name=None,
                    skill_path="/",
                )
            ]
            first_spinner = view.streaming_steering_spinners[first.id]
            view.restore_stream_snapshot(snapshot)
            assert not first_spinner.visible
            assert set(view.streaming_steering_spinners) == {second.id}
            view.streaming_steering_spinners[second.id].delete()
            view.finalize_stream()
    finally:
        client.delete()


def test_graph_preserves_repeated_guidance_and_pending_confirmation():
    first = QueuedMessage(text="Same guidance", display_content="")
    second = QueuedMessage(text=first.text, display_content="Second copy")
    message = ChatMessage(
        role="assistant",
        raw_history_start=0,
        steering_messages=[first, second],
        raw_history=[
            {"role": "user", "content": first.text, "message_id": first.id},
            {"role": "user", "content": first.text, "message_id": first.id},
            {"role": "user", "content": second.text},
        ],
        steering_insertions=[
            SteeringInsertion(
                message_id=first.id,
                after_tool_call_id=None,
                tool_name=None,
                skill_path="/",
            )
        ],
        tool_events=[ToolEvent(tool_name="read", tool_call_id="read-call")],
    )
    session = ChatSession(messages=[message])
    graph = build_graph_data_from_session(session)
    nodes = graph.histories[0].nodes
    guidance = [node.events[0] for node in nodes if node.kind == "steering"]
    assert [event["message_id"] for event in guidance] == [first.id, second.id]
    assert [event["pending"] for event in guidance] == [False, True]
    assert guidance[0]["display_content"] == ""
    assert guidance[1]["display_content"] == "Second copy"
    assert any(node.kind == "tool_bundle" for node in nodes)
    assert len({node.id for node in nodes}) == len(nodes)
    client = Client(page("/review-pending-graph"))
    try:
        with client:
            view = GraphViewComponent()
            view.build()
            view.render_session(session)
            pending_node = next(
                node
                for node in nodes
                if node.kind == "steering" and node.events[0]["pending"]
            )
            view.select_node(pending_node.id)
            assert any(
                isinstance(element, ui.label) and element.text == "Pending"
                for element in client.elements.values()
            )
    finally:
        client.delete()
    raw_nodes = []
    _extract_nodes_from_raw_scope(
        [{"role": "user", "content": "Legacy guidance"}],
        skill_tree=[],
        structured_skills=[],
        tool_events=[],
        history_id="legacy",
        history_label="Legacy",
        depth=0,
        skill_path="/",
        skill_name=None,
        nodes=raw_nodes,
    )
    assert raw_nodes[0].events[0]["content"] == "Legacy guidance"


def test_graph_and_chat_share_root_and_nested_invocation_order():
    before = ToolEvent(tool_name="read", timestamp=1)
    nested = SkillNode(
        name="nested",
        path="/worker/nested",
        depth=2,
        start_time=3,
        opening_tool_call_id="nested-call",
    )
    child_tool = ToolEvent(tool_name="inside", timestamp=4, skill_path="/worker")
    opening = ToolEvent(tool_name="worker", tool_call_id="worker-call", timestamp=2)
    skill = SkillNode(
        name="worker",
        path="/worker",
        depth=1,
        start_time=2,
        opening_tool_call_id="worker-call",
        sub_skills=[nested],
        tool_events=[
            child_tool,
            ToolEvent(tool_name="nested", tool_call_id="nested-call"),
        ],
    )
    after = ToolEvent(tool_name="write", timestamp=5)
    message = ChatMessage(
        role="assistant",
        skills=[skill],
        tool_events=[after, child_tool, opening, before],
    )
    assert message.ordered_activity() == [
        before.normalized_copy(),
        skill,
        after.normalized_copy(),
    ]
    assert skill.ordered_activity() == [nested, child_tool.normalized_copy()]
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    activity = [
        node
        for node in graph.histories[0].nodes
        if node.kind in ("tool_bundle", "skill_bundle")
    ]
    assert [node.events[0]["tool_name"] for node in activity] == [
        "read",
        "worker",
        "nested",
        "inside",
        "write",
    ]
    assert activity[1].events[0]["tool_call_id"] == "worker-call"
    assert activity[2].events[0]["tool_call_id"] == "nested-call"


async def test_folder_picker_failure_is_reported_and_cancellation_preserves_input():
    client = Client(page("/review-picker"))
    try:
        with client:
            dialog = NewProjectDialogComponent(lambda _: None, MagicMock())
            dialog.build()
            dialog.folder_input.value = "original"
            with patch(
                "fe.ui.new_project_dialog.pick_folder_dialog",
                side_effect=OSError("picker failed"),
            ):
                await dialog.browse()
            assert (
                client.outbox.messages[-1][2]["message"]
                == "Unable to open folder picker: picker failed"
            )
            with patch(
                "fe.ui.new_project_dialog.pick_folder_dialog", return_value=None
            ):
                await dialog.browse()
            assert dialog.folder_input.value == "original"
            with (
                patch("fe.ui.new_project_dialog.tkinter.Tk"),
                patch(
                    "fe.ui.new_project_dialog.filedialog.askdirectory", return_value=""
                ),
            ):
                assert pick_folder_dialog() is None
    finally:
        client.delete()
