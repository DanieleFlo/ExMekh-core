from __future__ import annotations

import pytest
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import UserMessage

from fe.qs_bridge.converter import (
    _parse_xml_header_output,
    chat_message_to_qs_messages,
    parse_tool_calls,
    qs_message_to_raw_dict,
    qs_messages_to_chat_messages,
    qs_messages_to_session,
    raw_dict_to_qs_message,
)
from fe.qs_bridge.models import ChatMessage


@pytest.mark.parametrize("invalid", ["", "null", "{broken", 12, [], ["bad"]])
def test_invalid_tool_call_data_has_no_calls(invalid: object) -> None:
    assert parse_tool_calls(invalid) is None


def test_parse_tool_calls_accepts_objects_and_both_dict_shapes() -> None:
    existing = ToolCall(
        id="original", function=FunctionCall(name="read", arguments={"file": "a"})
    )
    calls = parse_tool_calls(
        [
            existing,
            {"function": {"name": "write", "arguments": '{"file":"b"}'}},
            {"function": None, "name": "fallback", "arguments": "{broken"},
            {"function": {"name": "fallback2", "arguments": "{broken"}},
        ]
    )
    assert [call.function.name for call in calls] == [
        "read",
        "write",
        "fallback",
        "fallback2",
    ]
    assert calls[0] is existing
    assert calls[1].function.arguments == {"file": "b"}
    assert calls[2].function.arguments == {}
    assert calls[3].function.arguments == {}
    assert len(parse_tool_calls('[{"id":"a","function":{"name":"search"}}]')) == 1


def test_tool_replies_use_pending_call_metadata_when_missing() -> None:
    pending = [
        ToolCall(id="call_1", function=FunctionCall(name="search", arguments={}))
    ]
    reply = raw_dict_to_qs_message({"role": "tool", "content": "result"}, pending)
    assert isinstance(reply, ToolMessage)
    assert reply.get_tool_call_id() == "call_1"
    assert reply.get_name() == "search"
    assert pending == []


def test_tool_replies_use_defaults_without_pending_call() -> None:
    reply = raw_dict_to_qs_message({"role": "function", "content": "ok"})
    assert reply.get_tool_call_id() == "call_default"
    assert reply.get_name() == "tool_default"


def test_tool_replies_remove_matching_pending_call() -> None:
    pending = [
        ToolCall(id="first", function=FunctionCall(name="one", arguments={})),
        ToolCall(id="second", function=FunctionCall(name="two", arguments={})),
    ]
    raw_dict_to_qs_message(
        {"role": "tool", "tool_call_id": "second", "name": "two"}, pending
    )
    assert [call.id for call in pending] == ["first"]


def test_unknown_role_is_restored_as_user_and_snapshots_round_trip() -> None:
    message = raw_dict_to_qs_message(
        {
            "role": "custom",
            "content": "hello",
            "_fe_chat_message_snapshot": {"id": "one"},
            "_fe_session_metadata": {"id": "session"},
        }
    )
    assert isinstance(message, UserMessage)
    assert qs_message_to_raw_dict(message) == {"role": "user", "content": "hello"}
    assert qs_message_to_raw_dict(message, include_snapshots=True)[
        "_fe_session_metadata"
    ] == {"id": "session"}


def test_assistant_raw_history_synthesizes_tool_call_for_orphan_reply() -> None:
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "tool",
                "content": "result",
                "name": "run",
                "tool_call_id": "orphan",
            }
        ],
    )
    restored = chat_message_to_qs_messages(message)
    assert isinstance(restored[0], AssistantMessage)
    assert restored[0].get_tool_calls()[0].id == "orphan"
    assert isinstance(restored[1], ToolMessage)


def test_system_message_without_snapshot_uses_plain_content() -> None:

    result = qs_messages_to_chat_messages(
        [SystemMessage("rules"), UserMessage("question")]
    )
    assert [(message.role, message.content) for message in result] == [
        ("system", "rules"),
        ("user", "question"),
    ]


def test_session_without_metadata_uses_requested_defaults() -> None:

    session = qs_messages_to_session(
        [SystemMessage("instructions"), UserMessage("hi")],
        session_id="requested",
        title="Title",
        model="model",
        agent_mode="unknown",
    )
    assert session.id == "requested"
    assert session.title == "Title"
    assert session.model == "model"
    assert session.agent_mode == "deep_agent"
    assert session.system_prompt == "instructions"
    assert session.messages[0].content == "hi"


@pytest.mark.parametrize(
    "value",
    ["hello", "<header>", "<header><id></id></header>", "<header><id>bad</header>"],
)
def test_invalid_xml_tool_output_is_ignored(value: str) -> None:

    assert _parse_xml_header_output(value) is None


def test_xml_tool_output_parses_metadata_and_body() -> None:

    assert _parse_xml_header_output(
        "<header><name>search</name><type>Tool</type><id>stored</id></header>\nresult"
    ) == ("search", "Tool", "stored", "result")


def test_raw_metadata_snapshots_are_owned_and_never_sent_to_provider() -> None:
    raw = {
        "role": "user",
        "content": "hello",
        "_fe_chat_message_snapshot": {"content": "original"},
        "_fe_steering_message": False,
        "_fe_session_metadata": None,
    }
    message = raw_dict_to_qs_message(raw)
    raw["_fe_chat_message_snapshot"]["content"] = "changed"
    exported = qs_message_to_raw_dict(message, include_snapshots=True)
    assert exported["_fe_chat_message_snapshot"] == {"content": "original"}
    assert exported["_fe_steering_message"] is False
    assert exported["_fe_session_metadata"] is None
    assert message.message_dump() == {"role": "user", "content": "hello"}
    exported["_fe_chat_message_snapshot"]["content"] = "changed again"
    assert message.get_metadata()["_fe_chat_message_snapshot"] == {
        "content": "original"
    }
