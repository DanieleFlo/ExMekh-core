import json

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.history.compaction import (
    HistoryArchive,
    ModelContextCheckpoint,
    RecordedCompaction,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage, ChatSession
from exmekh_agent.ui.graph.graph_data import (
    _extract_nodes_from_raw_scope,
    build_graph_data_from_session,
)
from exmekh_agent.ui.graph.graph_view import GraphViewComponent


def call(name, call_id):
    return [
        {
            "role": "assistant",
            "tool_calls": json.dumps(
                [{"id": call_id, "function": {"name": name, "arguments": "{}"}}]
            ),
        },
        {"role": "tool", "tool_call_id": call_id, "name": name, "content": "result"},
    ]


def session_with(raw, records=()):
    return ChatSession(
        messages=[
            ChatMessage(role="user", content="task"),
            ChatMessage(
                role="assistant",
                content="done",
                raw_history=raw,
                raw_history_start=0,
                model_context=ModelContextCheckpoint(compactions=list(records)),
            ),
        ]
    )


def sequence(session):
    nodes = build_graph_data_from_session(session).histories[0].nodes
    return nodes, [node.events[0].get("tool_call_id", node.kind) for node in nodes]


def test_one_operation_replacing_several_blocks_is_placed_at_its_actual_boundary():
    raw = [
        *call("work", "one"),
        *call("compact_history", "first-block"),
        *call("work", "two"),
        *call("compact_history", "second-block"),
        *call("work", "active-call"),
        *call("work", "resumed"),
    ]
    record = RecordedCompaction(
        id="first-block",
        replacement_call_ids=["first-block", "second-block"],
        after_tool_call_id="active-call",
        skill_path="/skill",
        depth=1,
    )
    session = session_with(raw, [record])
    restored = ChatSession.model_validate_json(session.model_dump_json())
    nodes, order = sequence(restored)
    assert order[-6:] == [
        "one",
        "two",
        "active-call",
        "compaction",
        "resumed",
        "assistant",
    ]
    compacted = [node for node in nodes if node.kind == "compaction"]
    assert len(compacted) == 1
    assert compacted[0].depth == 1 and compacted[0].skill_path == "/skill"
    assert compacted[0].events[0]["replacement_call_ids"] == [
        "first-block",
        "second-block",
    ]
    assert [node.step_index for node in nodes] == list(range(len(nodes)))
    assert len({node.id for node in nodes}) == len(nodes)


def test_marker_anchors_keep_successive_operations_between_the_correct_tools():
    raw = [
        *call("work", "one"),
        *call("compact_history", "a"),
        *call("work", "two"),
        *call("compact_history", "b"),
        *call("work", "three"),
    ]
    records = [
        RecordedCompaction(id=key, replacement_call_ids=[key], after_tool_call_id=key)
        for key in ("a", "b")
    ]
    _, order = sequence(session_with(raw, records))
    assert order[-6:] == [
        "one",
        "compaction",
        "two",
        "compaction",
        "three",
        "assistant",
    ]


@pytest.mark.parametrize("anchor", [None, "previous-turn-call"])
def test_compaction_of_previous_context_precedes_new_work(anchor):
    record = RecordedCompaction(
        id="operation", replacement_call_ids=["old-block"], after_tool_call_id=anchor
    )
    _, order = sequence(session_with(call("work", "new-work"), [record]))
    assert order[-3:] == ["compaction", "new-work", "assistant"]


def test_accepted_guidance_before_compaction_stays_before_its_graph_point():
    raw = [
        *call("work", "one"),
        {"role": "user", "content": "updated constraint"},
        *call("work", "two"),
    ]
    record = RecordedCompaction(
        id="operation",
        replacement_call_ids=["block"],
        after_tool_call_id="one",
        following_user_count=1,
    )
    _, order = sequence(session_with(raw, [record]))
    assert order[-5:] == ["one", "steering", "compaction", "two", "assistant"]


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_legacy_marker_becomes_one_runtime_event_without_invented_execution_or_time(
    with_checkpoint,
):
    session = session_with(
        [*call("work", "one"), *call("compact_history", "old-marker")]
    )
    if not with_checkpoint:
        session.messages[-1].model_context = None
    nodes, order = sequence(session)
    assert order[-3:] == ["one", "compaction", "assistant"]
    event = next(node for node in nodes if node.kind == "compaction").events
    assert len(event) == 1 and event[0]["event_stage"] == "history_compaction"
    assert event[0]["timestamp"] is None


def test_raw_scope_without_a_saved_message_can_display_a_legacy_compaction():
    nodes = []
    _extract_nodes_from_raw_scope(
        call("compact_history", "legacy"),
        skill_tree=[],
        structured_skills=[],
        tool_events=[],
        history_id="main",
        history_label="Main",
        depth=0,
        skill_path="/",
        skill_name=None,
        nodes=nodes,
    )
    assert len(nodes) == 1 and nodes[0].kind == "compaction"


def test_archive_records_each_applied_batch_once_and_resets_only_turn_events():
    archive = HistoryArchive()
    archive.begin_turn()
    archive.record_compaction({})
    assert archive.checkpoint() is None
    replacements = {"a": [], "b": []}
    history = [*call("work", "active"), {"role": "user", "content": "guidance"}]
    archive.record_compaction(replacements, history=history)
    archive.record_compaction(replacements, history=history)
    snapshot = archive.checkpoint()
    assert len(snapshot.compactions) == 1
    assert snapshot.compactions[0].after_tool_call_id == "active"
    assert snapshot.compactions[0].following_user_count == 1
    archive.begin_turn()
    assert archive.checkpoint().compactions == []
    assert archive.checkpoint().replacements == replacements
    assert len(snapshot.compactions) == 1


def test_compaction_point_opens_its_own_event_details():
    client = Client(page("/graph-compaction"))
    try:
        with client:
            view = GraphViewComponent()
            view.build()
            view.render_session(session_with(call("compact_history", "legacy")))
            node = next(
                node
                for node in view.current_graph_data.histories[0].nodes
                if node.kind == "compaction"
            )
            view.select_node(node.id)
            assert view.get_selected_node_id(view.current_session.id) == node.id
    finally:
        client.delete()
