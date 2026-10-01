from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.interfaces.message import Message

from fe.qs_bridge.converter import (
    chat_message_to_qs_messages,
    parse_tool_calls,
    qs_message_to_raw_dict,
    qs_messages_to_chat_messages,
    qs_messages_to_session,
    raw_dict_to_qs_message,
    reconstruct_execution_history_and_cache,
    restore_node_history,
)
from fe.qs_bridge.models import ChatMessage, ToolEvent


def _tool_call(call_id: str, name: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": {}}}


def _header(name: str, source: str, output_id: str, body: str) -> str:
    return (
        f"<header><name>{name}</name><type>{source}</type>"
        f"<id>{output_id}</id></header>\n{body}"
    )


def test_tool_call_parser_accepts_flat_valid_json_arguments() -> None:
    calls = parse_tool_calls(
        [{"function": None, "name": "read", "arguments": '{"path":"a"}'}]
    )
    assert calls[0].function.arguments == {"path": "a"}


def test_tool_reply_fills_only_missing_metadata() -> None:
    pending = [ToolCall(id="one", function=FunctionCall(name="read", arguments={}))]
    reply = raw_dict_to_qs_message({"role": "tool", "name": "override"}, pending)
    assert reply.get_tool_call_id() == "one"
    assert reply.get_name() == "override"
    pending = [ToolCall(id="two", function=FunctionCall(name="write", arguments={}))]
    reply = raw_dict_to_qs_message(
        {"role": "tool", "tool_call_id": "override"}, pending
    )
    assert reply.get_tool_call_id() == "override"
    assert reply.get_name() == "tool_default"
    assert [call.id for call in pending] == ["two"]
    pending = [ToolCall(id="three", function=FunctionCall(name="run", arguments={}))]
    raw_dict_to_qs_message(
        {"role": "tool", "tool_call_id": "unknown", "name": "other"}, pending
    )
    assert len(pending) == 1


def test_system_snapshot_round_trips_and_session_can_generate_id() -> None:
    original = ChatMessage(role="system", content="custom rules")
    restored = qs_messages_to_chat_messages(chat_message_to_qs_messages(original))
    assert restored == [original]
    session = qs_messages_to_session([SystemMessage("system")])
    assert session.id
    assert session.system_prompt == "system"


def test_unknown_qs_message_is_preserved_as_plain_chat_message() -> None:
    unknown = MagicMock(spec=Message)
    unknown.get_role_name.return_value = "user"
    unknown.get_message.return_value = "plain"
    assert qs_message_to_raw_dict(unknown) == {"role": "user", "content": "plain"}
    result = qs_messages_to_chat_messages([unknown])
    assert result[0].content == "plain"


def test_unmatched_tool_reply_is_recorded_but_matching_one_is_not_duplicated() -> None:
    matched = ToolCall(id="call", function=FunctionCall(name="read", arguments={}))
    messages = qs_messages_to_chat_messages(
        [
            AssistantMessage(content=None, tool_calls=[matched]),
            ToolMessage(content="result", name="read", tool_call_id="call"),
            ToolMessage(content="other", name="unmatched", tool_call_id="other"),
        ]
    )
    assert [event.tool_name for event in messages[0].tool_events] == [
        "read",
        "unmatched",
    ]


def test_reconstruction_records_child_skill_and_inner_tool_output() -> None:
    history = MagicMock()
    history.record.return_value = MagicMock()
    cache = MagicMock()
    cache.session_dir = Path("/cache")
    child = {
        "skill": "analysis",
        "opening_tool_call_id": "open",
        "history": [
            {"role": "assistant", "tool_calls": [_tool_call("open", "analysis")]},
            {"role": "tool", "content": "opened"},
            {"role": "user", "content": "ignored"},
            {"role": "assistant", "tool_calls": [_tool_call("nested", "read")]},
            {
                "role": "tool",
                "content": _header("read", "Tool", "nested-output", "inside"),
            },
        ],
        "children": [],
        "output": _header("analysis", "DeepSkill", "skill-output", "done"),
    }
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "tool_calls": [_tool_call("open", "analysis")]},
            {"role": "tool", "content": "skill result"},
        ],
        skill_history_tree=[child],
    )
    reconstruct_execution_history_and_cache([message], history, cache)
    assert [call.args[0] for call in history.record.call_args_list] == [
        "analysis",
        "read",
    ]
    assert cache.store.call_count == 2
    assert history.attach_output.call_args.args[0] == "open"


def test_reconstruction_matches_child_by_name_and_handles_missing_output() -> None:
    history = MagicMock()
    child = {"skill": "analysis", "history": [{"role": "assistant"}], "children": []}
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "tool_calls": [_tool_call("open", "analysis")]},
        ],
        skill_history_tree=[child],
    )
    reconstruct_execution_history_and_cache([message], history)
    history.record.assert_called_once_with("analysis", "open")


def test_reconstruction_handles_cumulative_raw_history_and_legacy_events() -> None:
    history = MagicMock()
    first = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "tool_calls": [_tool_call("one", "read")]},
    ]
    second = first + [
        {"role": "user", "content": "second"},
        {"role": "assistant", "tool_calls": [_tool_call("two", "write")]},
    ]
    messages = [
        ChatMessage(role="assistant", raw_history=first),
        ChatMessage(role="assistant", raw_history=second),
        ChatMessage(
            role="assistant",
            tool_events=[ToolEvent(tool_name="old", event_type="tool_start")],
        ),
    ]
    reconstruct_execution_history_and_cache(messages, history)
    assert [call.args[0] for call in history.record.call_args_list] == [
        "read",
        "write",
        "old",
    ]


def test_restore_node_history_ignores_legacy_reconstruction_errors() -> None:
    node = MagicMock()

    def restore(prior, callback):
        callback(MagicMock(), MagicMock())

    node.restore_history.side_effect = restore
    with patch(
        "fe.qs_bridge.converter.reconstruct_execution_history_and_cache",
        side_effect=ValueError("bad"),
    ):
        restore_node_history(
            node, [ChatMessage(role="user", content="old")], prior_qs=[]
        )
    node.restore_history.assert_called_once()
    node.restore_history.side_effect = RuntimeError("restore failed")
    restore_node_history(node, [ChatMessage(role="user", content="old")], prior_qs=[])
    restore_node_history(node, [], prior_qs=[])


def test_flat_tool_call_parser_keeps_object_arguments() -> None:
    calls = parse_tool_calls(
        [{"function": None, "name": "read", "arguments": {"path": "a"}}]
    )
    assert calls[0].function.arguments == {"path": "a"}


def test_reconstruction_skips_unmatched_child_before_name_match() -> None:
    history = MagicMock()
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "assistant", "tool_calls": [_tool_call("open", "analysis")]}
        ],
        skill_history_tree=[
            {"skill": "other", "history": []},
            {"skill": "analysis", "history": []},
        ],
    )
    reconstruct_execution_history_and_cache([message], history)
    history.record.assert_called_once_with("analysis", "open")


def test_reconstruction_restores_completed_and_running_legacy_tool_events() -> None:
    history = MagicMock()
    message = ChatMessage(
        role="assistant",
        tool_events=[
            ToolEvent(tool_name="done", event_type="execution"),
            ToolEvent(tool_name="start", event_type="tool_start"),
        ],
    )
    reconstruct_execution_history_and_cache([message], history)
    assert [call.args[0] for call in history.record.call_args_list] == ["done", "start"]
    assert history.record.call_args.args[0] == "start"
