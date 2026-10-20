from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)

from exmekh_agent.exmekh_bridge.agents.agent_runner import execute_agent_turn_sync
from exmekh_agent.exmekh_bridge.agents.sub_agents import (
    ObservedNode,
    SubAgentEvent,
    SubAgentEventRouter,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.state.sub_agent_conversations import apply_sub_agent_event
from tests.mock_client import MockProvider


def test_out_of_order_sub_agent_exit_preserves_the_active_routing_stack() -> None:
    router = SubAgentEventRouter(None)
    parent = MagicMock(spec=ObservedNode)
    child = MagicMock(spec=ObservedNode)
    event = MagicMock()
    router.enter(parent)
    router.enter(child)
    with pytest.raises(RuntimeError, match="out of order"):
        router.leave(parent)
    assert router.route(event)
    child.forward_stream_event.assert_called_once_with(event)
    parent.forward_stream_event.assert_not_called()

    router.leave(child)
    assert router.route(event)
    parent.forward_stream_event.assert_called_once_with(event)
    router.leave(parent)
    assert not router.route(event)
    with pytest.raises(RuntimeError, match="out of order"):
        router.leave(child)


def test_persistent_and_fresh_calls_create_the_right_blocks() -> None:
    session = ChatSession()
    for action, call_id, persistent, text in (
        ("started", "stable", True, "first"),
        ("completed", "stable", True, "answer 1"),
        ("started", "stable", True, "second"),
        ("completed", "stable", True, "answer 2"),
        ("started", "fresh-1", False, "third"),
        ("completed", "fresh-1", False, "answer 3"),
        ("started", "fresh-2", False, "fourth"),
        ("completed", "fresh-2", False, "answer 4"),
    ):
        apply_sub_agent_event(
            session,
            SubAgentEvent(
                action=action,
                id=call_id,
                path="main/coding-agent",
                persistent_history=persistent,
                user_message=text if action == "started" else "",
                response=text if action == "completed" else "",
            ),
        )

    assert len(session.sub_agent_conversations) == 3
    assert [item.order for item in session.sub_agent_conversations] == [1, 2, 3]
    assert [len(item.messages) for item in session.sub_agent_conversations] == [4, 2, 2]
    assert session.sub_agent_conversations[0].persistent_history is True
    assert all(
        not item.persistent_history for item in session.sub_agent_conversations[1:]
    )
    restored = ChatSession.model_validate_json(session.model_dump_json())
    assert len(restored.sub_agent_conversations) == 3
    assert restored.sub_agent_conversations[0].messages[-1].content == "answer 2"


def test_child_stream_updates_the_active_chat_block() -> None:
    session = ChatSession()
    base = {"id": "live", "path": "main/sub/sub-sub", "persistent_history": False}
    apply_sub_agent_event(
        session, SubAgentEvent(action="started", user_message="Write code", **base)
    )
    apply_sub_agent_event(
        session, SubAgentEvent(action="reasoning", token="Inspecting", **base)
    )
    apply_sub_agent_event(session, SubAgentEvent(action="token", token="Done", **base))
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="tool", tool_event=ToolEvent(tool_name="read_file"), **base
        ),
    )
    child = session.sub_agent_conversations[0]
    assert child.status == "active"
    assert child.messages[-1].is_streaming is True
    assert child.messages[-1].content == "Done"
    assert child.messages[-1].reasoning_content == "Inspecting"
    assert child.messages[-1].tool_events[0].tool_name == "read_file"
    apply_sub_agent_event(
        session, SubAgentEvent(action="completed", response="Done", **base)
    )
    assert child.messages[-1].is_streaming is False


def test_sub_agent_call_rejects_out_of_order_events() -> None:
    session = ChatSession()
    base = {"id": "child", "path": "main/child", "persistent_history": True}
    with pytest.raises(ValueError, match="Unknown sub-agent conversation"):
        apply_sub_agent_event(
            session, SubAgentEvent(action="token", token="early", **base)
        )

    apply_sub_agent_event(session, SubAgentEvent(action="started", **base))
    with pytest.raises(ValueError, match="already active"):
        apply_sub_agent_event(session, SubAgentEvent(action="started", **base))

    apply_sub_agent_event(session, SubAgentEvent(action="error", **base))
    with pytest.raises(ValueError, match="no active call"):
        apply_sub_agent_event(
            session, SubAgentEvent(action="token", token="late", **base)
        )


def test_child_reasoning_blocks_follow_skill_boundaries() -> None:
    session = ChatSession()
    base = {"id": "child", "path": "main/child", "persistent_history": False}
    apply_sub_agent_event(session, SubAgentEvent(action="started", **base))
    for token, path, depth in (
        ("a", "/", 0),
        ("b", "/", 0),
        ("c", "/skill", 1),
    ):
        apply_sub_agent_event(
            session,
            SubAgentEvent(
                action="reasoning", token=token, skill_path=path, depth=depth, **base
            ),
        )

    blocks = session.sub_agent_conversations[0].messages[-1].reasoning_blocks
    assert [(block.content, block.skill_path, block.depth) for block in blocks] == [
        ("ab", "/", 0),
        ("c", "/skill", 1),
    ]


def test_persistent_child_reuses_identity_after_call_without_raw_history() -> None:
    events: list[SubAgentEvent] = []
    node = ObservedNode(
        name_hds="child",
        path="main/child",
        persistent_history=True,
        event_router=SubAgentEventRouter(events.append),
    )
    node.restore_conversation(
        SubAgentConversation(id="previous-call", path="main/child"),
        prior_exmekh=[],
    )
    response = MagicMock()
    response.get_message.return_value = "done"
    response.history = []

    with patch.object(Node, "run", return_value=response):
        node.run("continue")

    assert [event.id for event in events] == ["previous-call", "previous-call"]


class TwoCallProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.root_calls = 0
        self.child_user_counts: list[int] = []

    def call(self, request):
        messages = request.get_messages()
        system = str(messages[0].get_message() or "")
        if "You are a coding agent" in system:
            self.child_user_counts.append(
                sum(message.get_role_name() == "user" for message in messages)
            )
            if request.on_event:
                request.on_event.handle(
                    ProviderStreamEvent(
                        type=ProviderStreamEventType.RESPONSE,
                        content="Child answer",
                    )
                )
            return ProviderResponse(AssistantMessage("Child answer"))

        self.root_calls += 1
        if self.root_calls <= 2:
            call = ToolCall(
                id=f"call_child_{self.root_calls}",
                function=FunctionCall(
                    name="call_sub_agent",
                    arguments={
                        "sub_agent": "coding-agent",
                        "text": f"Task {self.root_calls}",
                    },
                ),
            )
            return ProviderResponse(AssistantMessage(None, tool_calls=[call]))
        return ProviderResponse(AssistantMessage("Main answer"))


class OneCallProvider(TwoCallProvider):
    def call(self, request):
        messages = request.get_messages()
        system = str(messages[0].get_message() or "")
        if "You are a coding agent" in system:
            self.child_user_counts.append(
                sum(message.get_role_name() == "user" for message in messages)
            )
            return ProviderResponse(AssistantMessage("Child answer"))
        self.root_calls += 1
        if self.root_calls == 1:
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            id="call_child_once",
                            function=FunctionCall(
                                name="call_sub_agent",
                                arguments={
                                    "sub_agent": "coding-agent",
                                    "text": "Continue work",
                                },
                            ),
                        )
                    ],
                )
            )
        return ProviderResponse(AssistantMessage("Main answer"))


@pytest.mark.parametrize(
    ("persistent", "expected_counts"),
    [(True, [1, 2]), (False, [1, 1])],
)
def test_live_child_calls_follow_history_policy(
    persistent: bool, expected_counts: list[int]
) -> None:
    """Exercise the real ExMekh-Lib node tree and call_sub_agent with a scripted provider."""
    provider = TwoCallProvider()
    events: list[SubAgentEvent] = []
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.config.coding_agent_persistent_history",
            persistent,
        ),
    ):
        answer = execute_agent_turn_sync(
            user_prompt="Delegate twice",
            model="scripted",
            base_url="http://localhost:8080/v1",
            api_key="",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="deep_agent",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
            on_sub_agent_event=events.append,
        )

    assert answer == "Main answer"
    assert provider.child_user_counts == expected_counts
    started = [event for event in events if event.action == "started"]
    assert len(started) == 2
    assert all(event.path == "main/coding-agent" for event in started)
    assert (started[0].id == started[1].id) is persistent
    assert len([event for event in events if event.action == "completed"]) == 2
    assert len([event for event in events if event.action == "token"]) == 2


@pytest.mark.parametrize("legacy_history", [False, True])
def test_live_persistent_child_reuses_block_and_history_across_root_turns(
    legacy_history: bool,
) -> None:
    session = ChatSession()
    for turn in (1, 2):
        if turn == 2 and legacy_history:
            for child in session.sub_agent_conversations:
                child.raw_history = []
                for message in child.messages:
                    message.raw_history = []
        provider = OneCallProvider()
        raw_history: list[dict] = []
        with (
            patch(
                "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
                return_value=provider,
            ),
            patch(
                "exmekh_agent.exmekh_bridge.agents.agent_runner.config.coding_agent_persistent_history",
                True,
            ),
        ):
            answer = execute_agent_turn_sync(
                user_prompt=f"Turn {turn}",
                model="scripted",
                base_url="http://localhost:8080/v1",
                api_key="",
                api_type="responses",
                reasoning_effort_str="none",
                agent_mode="deep_agent",
                capabilities_path_str="",
                name_hds="main",
                on_token=lambda _: None,
                on_reasoning=lambda _: None,
                on_tool_event=lambda _: None,
                on_sub_agent_event=lambda event: apply_sub_agent_event(session, event),
                on_raw_history=lambda history, _: raw_history.extend(history),
                history=list(session.messages),
                sub_agent_history=[
                    *session.sub_agent_conversations,
                    SubAgentConversation(path="main/removed-agent"),
                ],
                agent_context_id=session.id,
            )
        session.messages.extend(
            [
                ChatMessage(role="user", content=f"Turn {turn}"),
                ChatMessage(role="assistant", content=answer, raw_history=raw_history),
            ]
        )
        assert provider.child_user_counts == [turn]

    assert len(session.sub_agent_conversations) == 1
    child = session.sub_agent_conversations[0]
    assert child.order == 1
    assert [message.content for message in child.messages] == [
        "Continue work",
        "Child answer",
        "Continue work",
        "Child answer",
    ]
