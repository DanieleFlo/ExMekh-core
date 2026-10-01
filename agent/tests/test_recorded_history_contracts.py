from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.converter import reconstruct_execution_history_and_cache
from fe.qs_bridge.models import ChatMessage, ChatSession, SkillNode, ToolEvent
from fe.ui.graph_data import build_graph_data_from_session
from fe.ui.raw_history_view import RawHistoryComponent


def tool_call(call_id: str, name: str = "read") -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": {}}}


def tool_result(call_id: str | None, output_id: str) -> dict:
    result = {
        "role": "tool",
        "content": f"<header><name>read</name><type>Tool</type><id>{output_id}</id></header>\n{output_id}",
    }
    if call_id is not None:
        result["tool_call_id"] = call_id
    return result


@pytest.mark.parametrize("legacy", [False, True])
def test_graph_and_reconstruction_pair_results_with_the_same_tool_invocation(
    legacy: bool,
) -> None:
    first = tool_result(None if legacy else "first", "first-output")
    second = tool_result(None if legacy else "second", "second-output")
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "user", "content": "Read both"},
            {
                "role": "assistant",
                "tool_calls": [tool_call("first"), tool_call("second")],
            },
            *([first, second] if legacy else [second, first]),
        ],
    )
    cache = MagicMock()
    cache.session_dir = Path("cache")
    reconstruct_execution_history_and_cache([message], MagicMock(), cache)
    restored = [call.args[0].to_dict() for call in cache.store.call_args_list]
    assert [(record["tool_call_id"], record["id"]) for record in restored] == [
        ("first", "first-output"),
        ("second", "second-output"),
    ]
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    nodes = [node for node in graph.histories[0].nodes if node.kind == "tool_bundle"]
    assert [node.events[-1]["content"] for node in nodes] == [
        first["content"],
        second["content"],
    ]


def test_result_for_another_call_is_never_assigned_by_position() -> None:
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "assistant", "tool_calls": [tool_call("missing-result")]},
            tool_result("other-call", "other-output"),
        ],
    )
    cache = MagicMock()
    reconstruct_execution_history_and_cache([message], MagicMock(), cache)
    cache.store.assert_not_called()
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    tool = next(node for node in graph.histories[0].nodes if node.kind == "tool_bundle")
    assert tool.events[-1]["content"] == ""


def test_graph_uses_legacy_events_without_borrowing_another_invocations_timing() -> (
    None
):
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "assistant",
                "tool_calls": [tool_call("first"), tool_call("second")],
            }
        ],
        tool_events=[
            ToolEvent(tool_name="read", tool_call_id="unrelated", duration_seconds=99),
            ToolEvent(tool_name="read", duration_seconds=3),
        ],
    )
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    tools = [node for node in graph.histories[0].nodes if node.kind == "tool_bundle"]
    assert [node.events[1]["duration_seconds"] for node in tools] == [3, 0]


@pytest.mark.parametrize("legacy_first_skill", [False, True])
def test_graph_matches_repeated_structured_skills_by_their_invocation(
    legacy_first_skill: bool,
) -> None:
    histories = []
    skills = []
    for index, call_id in enumerate(("first", "second"), start=1):
        inner_id = f"{call_id}-inner"
        histories.append(
            {
                "skill": "worker",
                "opening_tool_call_id": call_id,
                "history": [
                    {"role": "assistant", "tool_calls": [tool_call(call_id, "worker")]},
                    {"role": "tool", "tool_call_id": call_id, "content": "catalog"},
                    {"role": "assistant", "tool_calls": [tool_call(inner_id)]},
                    tool_result(inner_id, f"{call_id}-output"),
                ],
            }
        )
        skills.append(
            SkillNode(
                name="worker",
                path="/worker",
                depth=1,
                opening_tool_call_id=None
                if legacy_first_skill and index == 1
                else call_id,
                tool_events=[
                    ToolEvent(
                        tool_name="read", tool_call_id=inner_id, duration_seconds=index
                    )
                ],
            )
        )
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("first", "worker"),
                    tool_call("second", "worker"),
                ],
            }
        ],
        skill_history_tree=histories,
        skills=[
            SkillNode(name="other", path="/other", opening_tool_call_id="unrelated"),
            skills[1],
            skills[0],
        ],
    )
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    tools = [node for node in graph.histories[0].nodes if node.kind == "tool_bundle"]
    assert [node.events[0]["tool_call_id"] for node in tools] == [
        "first-inner",
        "second-inner",
    ]
    assert [node.events[1]["duration_seconds"] for node in tools] == [1, 2]


@pytest.mark.parametrize("identity", ["explicit", "inferred", "legacy"])
def test_raw_history_renders_each_repeated_skill_once_at_its_own_call(
    identity: str,
) -> None:
    calls = [tool_call("first", "worker"), tool_call("second", "worker")]
    skills = []
    for call in calls:
        skill = {"skill": "worker", "history": [], "output": call["id"]}
        if identity == "explicit":
            skill["opening_tool_call_id"] = call["id"]
        if identity == "inferred":
            skill["history"] = [{"role": "assistant", "tool_calls": [call]}]
        skills.append(skill)
    if identity != "legacy":
        skills.reverse()
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "assistant", "tool_calls": [calls[0]]},
            {"role": "assistant", "tool_calls": [calls[0]]},
            {"role": "assistant", "tool_calls": [calls[1]]},
        ],
        skill_history_tree=skills,
    )
    with Client(page("/recorded-skill-contracts")):
        view = RawHistoryComponent()
        view.build()
        with patch.object(view, "_render_skill_compaction") as render:
            view.render_session(ChatSession(messages=[message]))
    assert [call.args[0]["output"] for call in render.call_args_list] == [
        "first",
        "second",
    ]
    assert [
        call.kwargs["prefix_messages"][-1]["tool_calls"][0]["id"]
        for call in render.call_args_list
    ] == ["first", "second"]


def test_graph_uses_the_skill_catalog_instead_of_a_sibling_tool_result() -> None:
    opening = {
        "role": "assistant",
        "tool_calls": [tool_call("sibling"), tool_call("open", "worker")],
    }
    skill = {
        "skill": "worker",
        "opening_tool_call_id": "open",
        "history": [
            opening,
            {"role": "tool", "tool_call_id": "sibling", "content": "Sibling result"},
            {"role": "tool", "tool_call_id": "open", "content": "Skill catalog"},
            {"role": "user", "content": "Early guidance"},
        ],
    }
    graph = build_graph_data_from_session(
        ChatSession(
            messages=[
                ChatMessage(
                    role="assistant",
                    raw_history=[opening],
                    skill_history_tree=[skill],
                )
            ]
        )
    )
    worker = next(
        node for node in graph.histories[0].nodes if node.kind == "skill_bundle"
    )
    assert worker.events[-1]["content"] == "Skill catalog"
    following = graph.histories[0].nodes[graph.histories[0].nodes.index(worker) + 1]
    assert following.label == "Steering: Early guidance"
    assert following.depth == 1
    assert following.skill_path == "/worker"


@pytest.mark.parametrize(
    "calls",
    [
        "invalid JSON",
        [],
        [None, tool_call("other", "other")],
        json.dumps([tool_call("open", "worker")]),
    ],
)
def test_collector_interprets_legacy_opening_calls_through_the_typed_parser(
    calls,
) -> None:
    collector = TurnStreamCollector()
    collector.append_skill(
        SkillNode(
            name="worker",
            depth=1,
            opening_tool_call_id="open",
            tokens_before_close=10,
            tokens_after_close=8,
        )
    )
    collector.set_raw_history(
        [],
        [
            {
                "skill": "worker",
                "history": [{"role": "assistant", "tool_calls": calls}],
            }
        ],
    )
    entry = collector.snapshot_message().skill_history_tree[0]
    if isinstance(calls, str) and calls.startswith("["):
        assert entry["tokens_before_close"] == 10
        assert entry["tokens_after_close"] == 8
    else:
        assert "tokens_before_close" not in entry
