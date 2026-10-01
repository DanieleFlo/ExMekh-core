from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page
from qs.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
)
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall

from fe.qs_bridge.converter import qs_messages_to_chat_messages
from fe.qs_bridge.event_handlers import DeepAgentTurnEventHandler
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    ReasoningBlock,
    ToolEvent,
)
from fe.qs_bridge.sub_agents import ObservedNode, SubAgentEventRouter
from fe.qs_bridge.tool_calls import extract_tool_arguments
from fe.ui.chat_export import export_chat
from fe.ui.chat_view import ChatViewComponent
from tests.interface_sample import interface_session


@pytest.mark.parametrize("include_history", [False, True])
@pytest.mark.parametrize("format", ["json", "markdown"])
def test_export_preserves_full_chat_and_optional_nested_transcripts(
    format, include_history
):
    session = interface_session()
    before = session.model_dump()
    text = export_chat(session, format, include_history=include_history)
    for expected in [
        "Prima risposta",
        "Secondo messaggio",
        "Seconda risposta",
        "Root reasoning",
        "Legacy reasoning",
        "Visible steering",
    ]:
        assert expected in text
    for nested in [
        "Nested reasoning",
        "Nested transcript",
        "Child transcript",
        "nested-private.txt",
    ]:
        assert (nested in text) == include_history
    assert session.model_dump() == before
    if format == "json":
        data = json.loads(text)
        assert len(data["messages"]) == 4
        assert "settings" not in data
        assert ("sub_agent_conversations" in data) == include_history
        assert ("skill_history_tree" in data["messages"][1]) == include_history
        assert (
            data["messages"][1]["tool_events"][0]["arguments"]
            == session.messages[1].tool_events[0].arguments
        )
    else:
        assert "# Interfaccia · test" in text
        assert "Primo messaggio **utente**" in text
        assert "Internal attachment markup" not in text
        assert "## System" in text
        assert "Status: cancelled" in text
        if include_history:
            assert "````json" in text


def test_empty_and_child_only_reasoning_exports():
    assert (
        export_chat(ChatSession(title="Empty"), "markdown", include_history=False)
        == "# Empty\n"
    )
    session = ChatSession(
        messages=[
            ChatMessage(
                role="assistant",
                reasoning_content="Nested only",
                reasoning_blocks=[ReasoningBlock(content="Nested only", depth=1)],
            )
        ]
    )
    session.messages[0].steering_messages = [QueuedMessage(text="Plain guidance")]
    assert "Plain guidance" in export_chat(session, "markdown", include_history=False)
    assert "Nested only" not in export_chat(session, "markdown", include_history=False)


@pytest.mark.parametrize("encoded_calls", [False, True])
@pytest.mark.parametrize("encoded_args", [False, True])
def test_arguments_match_invocation_id_and_decode_json(encoded_calls, encoded_args):
    arguments = {"command": "echo à <tag>", "nested": {"number": 2}}
    calls = [
        {
            "id": "other",
            "function": {"name": "bash", "arguments": {"command": "wrong"}},
        },
        {
            "id": "target",
            "function": {
                "name": "bash",
                "arguments": json.dumps(arguments) if encoded_args else arguments,
            },
        },
    ]
    history = [{"tool_calls": json.dumps(calls) if encoded_calls else calls}]
    assert extract_tool_arguments(history, "target") == arguments
    assert extract_tool_arguments(history, "missing") == {}


@pytest.mark.parametrize(
    "history,call_id",
    [
        (None, "call"),
        ([], None),
        ([None], "call"),
        ([{"tool_calls": "broken"}], "call"),
        ([{"tool_calls": [None, {"id": "call", "function": None}]}], "call"),
        (
            [{"tool_calls": [{"id": "call", "function": {"arguments": "broken"}}]}],
            "call",
        ),
        ([{"tool_calls": [{"id": "call", "function": {"arguments": []}}]}], "call"),
    ],
)
def test_unavailable_arguments_do_not_break_legacy_sessions(history, call_id):
    assert extract_tool_arguments(history, call_id) == {}


def test_arguments_survive_root_and_child_streaming_and_conversion():
    arguments = {"command": "echo hello"}
    call = ToolCall(id="call", function=FunctionCall(name="bash", arguments=arguments))
    converted = qs_messages_to_chat_messages(
        [AssistantMessage(content="", tool_calls=[call])]
    )
    assert converted[0].tool_events[0].arguments == arguments
    assert converted[0].tool_events[0].tool_call_id == "call"
    history = [
        {
            "tool_calls": [
                {"id": "call", "function": {"name": "bash", "arguments": arguments}}
            ]
        }
    ]
    tools = []
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None),
        lambda _: None,
        lambda *_: None,
        tools.append,
        None,
        None,
    )
    events = [
        DeepAgentExecutionEvent(
            type=kind,
            depth=0,
            skill_path="/",
            history=history,
            history_tree=[],
            tool_name="bash",
            tool_call_id="call",
        )
        for kind in DeepAgentExecutionEventType
    ]
    for event in events:
        handler.handle(event)
    assert tools[-1].arguments == arguments
    assert not tools[-1].is_running
    child_events = []
    node = ObservedNode(
        path="main/worker",
        event_router=SubAgentEventRouter(child_events.append),
        name_hds="worker",
    )
    node._active_call_id = "child"
    for event in events:
        node.forward_stream_event(event)
    assert all(event.tool_event.arguments == arguments for event in child_events)


@pytest.mark.asyncio
async def test_copy_actions_use_visible_user_text_and_current_root_chat():
    session = interface_session()
    client = Client(page("/interface-test"))
    with (
        client,
        patch(
            "fe.ui.chat_view.session_manager.get_current_session", return_value=session
        ),
        patch.object(ui.clipboard, "write") as write,
        patch.object(ui, "notify"),
    ):
        view = ChatViewComponent(lambda _: None)
        view.build()
        view.show_sub_agent("child")
        view.include_subskill_history.value = True
        await view._copy_chat("json")
        assert len(json.loads(write.call_args.args[0])["messages"]) == 4
        user = ChatMessage(
            role="user", content="hidden", display_content="Visible text"
        )
        with patch.object(view, "_render_copy_button") as copy:
            view._render_message_body(user)
        copy.assert_called_once_with("Visible text", user=True)
        view._rendering_message = ChatMessage(
            role="assistant",
            raw_history=[
                {
                    "tool_calls": [
                        {
                            "id": "old",
                            "function": {"arguments": {"command": "<script>"}},
                        }
                    ]
                }
            ],
        )
        view._render_tool_event(ToolEvent(tool_name="legacy", tool_call_id="old"))
        previews = [
            element
            for element in client.elements.values()
            if "tool-arguments-preview" in element._classes
        ]
        assert "&lt;script&gt;" in previews[-1].content
