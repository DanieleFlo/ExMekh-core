from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.user_message import UserMessage
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    convert_to_exmekh_messages,
    execute_agent_turn_sync,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.event_handlers import DeepAgentTurnEventHandler
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEventRouter
from exmekh_agent.exmekh_bridge.history.converter import (
    exmekh_message_to_raw_dict,
    exmekh_messages_to_session,
    raw_dict_to_exmekh_message,
    session_to_exmekh_messages,
)
from exmekh_agent.exmekh_bridge.history.pasted_input import PreparedUserInput
from exmekh_agent.exmekh_bridge.history.steering_position import (
    find_steering_insertions,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    QueuedMessage,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    build_compacted_context_messages,
)
from exmekh_agent.state.active_turns import ActiveTurnRegistry, active_turns
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.graph.graph_data import build_graph_data_from_session
from exmekh_agent.ui.history.raw_history_view import build_raw_history_from_session
from tests.capability_fixtures import TEST_CAPABILITIES
from tests.steering_probe import SteeringProbeProvider


def _mock_turn_options(
    tmp_path: Path,
    steering: TurnSteering,
    *,
    agent_mode: str = "deep_agent",
    user_prompt: str = "Start",
    on_raw_history: Callable[..., None] | None = None,
) -> dict[str, Any]:
    return dict(
        user_prompt=user_prompt,
        model="steering-probe",
        agent_mode=agent_mode,
        on_token=lambda _: None,
        on_reasoning=lambda *_: None,
        on_tool_event=lambda _: None,
        on_raw_history=on_raw_history,
        settings=ChatSettings(
            work_root=str(tmp_path),
            capabilities_path=str(TEST_CAPABILITIES),
            tool_approval_mode="Approve All",
        ),
        steering=steering,
    )


async def _wait_for_probe(event: Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 5), 6)


def test_steering_records_only_accepted_guidance_and_copies_it() -> None:
    control = TurnSteering()
    guidance = QueuedMessage(text="Use another file", display_content="Correction")
    assert not control.available
    assert not control.send(guidance)
    control.apply_to(ChatMessage(role="assistant"))

    received: list[str] = []
    control.bind(received.append, history_start=4)
    assert control.available
    assert control.send(guidance)
    assert received == ["Use another file"]
    guidance.text = "Edited afterwards"
    snapshot = ChatMessage(role="assistant")
    control.apply_to(snapshot)
    assert snapshot.steering_messages[0].text == "Use another file"
    assert snapshot.steering_messages[0].display_content == "Correction"
    assert snapshot.raw_history_start == 4
    snapshot.steering_messages[0].text = "Changed snapshot"
    other_snapshot = ChatMessage(role="assistant")
    control.apply_to(other_snapshot)
    assert other_snapshot.steering_messages[0].text == "Use another file"

    control.bind(MagicMock(side_effect=RuntimeError("run closed")), history_start=4)
    assert not control.send(guidance)
    assert control.available
    # An early rejection must not disable steering if the run starts just after it.
    control.bind(received.append, history_start=4)
    assert control.send(QueuedMessage(text="Accepted on retry"))
    control.close()
    assert not control.available
    assert not control.send(guidance)


def test_closed_steering_cannot_be_rebound_by_a_late_worker() -> None:
    control = TurnSteering()
    handler = MagicMock()
    control.close()
    control.bind(handler, history_start=2)
    assert not control.available
    assert not control.send(QueuedMessage(text="Late guidance"))
    handler.assert_not_called()


def test_steering_events_confirm_fifo_positions_without_history_matching() -> None:
    collector = TurnStreamCollector()
    steering = collector.steering
    observer = MagicMock()
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None),
        collector.append_token,
        collector.append_reasoning,
        collector.append_tool,
        collector.append_skill,
        None,
        steering,
        observer,
    )

    def added(text: str, path: str = "/research") -> DeepAgentStreamEvent:
        return DeepAgentStreamEvent(
            ProviderStreamEvent(
                ProviderStreamEventType.STEERING_ADDED, message=UserMessage(text)
            ),
            depth=1 if path != "/" else 0,
            skill_path=path,
        )

    handler.handle(added("Unexpected"))
    steering.bind(lambda _: None, history_start=2)
    first = QueuedMessage(text="Same correction", display_content="First")
    second = QueuedMessage(text="Same correction", display_content="Second")
    third = QueuedMessage(text="Later correction")
    for message in (first, second, third):
        assert steering.send(message)
    # Even identical text in an old snapshot cannot falsely confirm insertion.
    collector.set_raw_history([{"role": "user", "content": first.text}], [])
    assert find_steering_insertions(collector.snapshot_message()) == []
    handler.handle(added("Unrelated"))
    observer.assert_not_called()
    for call_id in ("first-tool", "last-tool"):
        handler.handle(
            DeepAgentExecutionEvent(
                type=DeepAgentExecutionEventType.TOOL_END,
                depth=1,
                skill_path="/research",
                history=[],
                history_tree=[],
                tool_name="read",
                tool_call_id=call_id,
            )
        )
    handler.handle(added(first.text))
    handler.handle(added(second.text))
    snapshot = collector.snapshot_message()
    positions = find_steering_insertions(snapshot)
    assert [item.message_id for item in positions] == [first.id, second.id]
    assert [item.after_tool_call_id for item in positions] == ["last-tool"] * 2
    assert [item.skill_path for item in positions] == ["/research"] * 2
    assert [item.tool_name for item in positions] == ["read"] * 2
    assert observer.call_count == 2
    # A later inference with no tools must not reuse the previous batch's anchor.
    handler.handle(
        DeepAgentStreamEvent(ProviderStreamEvent(ProviderStreamEventType.START), 0, "/")
    )
    handler.on_steering_added = None
    handler.handle(added(third.text, "/"))
    handler.handle(added(third.text, "/"))  # No more accepted input.
    final_message = collector.build_message()
    assert final_message.steering_insertions[-1].after_tool_call_id is None
    assert final_message.steering_insertions[-1].tool_name is None
    assert len(final_message.steering_insertions) == 3
    assert len(snapshot.steering_insertions) == 2
    positions[0].skill_path = "/changed"
    assert snapshot.steering_insertions[0].skill_path == "/research"
    # Compaction or a later history update cannot move or drop confirmed positions.
    final_message.raw_history = []
    restored = ChatMessage.model_validate_json(final_message.model_dump_json())
    assert find_steering_insertions(restored) == final_message.steering_insertions
    handler.steering = None
    handler.handle(added("No steering control", "/"))


def test_steering_position_does_not_assign_an_unknown_result_to_a_pending_call() -> (
    None
):
    guidance = QueuedMessage(text="Correction")
    message = ChatMessage(
        role="assistant",
        raw_history_start=0,
        raw_history=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "read-call", "function": {"name": "read", "arguments": {}}}
                ],
            },
            {"role": "tool", "tool_call_id": "unknown"},
            {"role": "tool", "tool_call_id": "read-call"},
            {"role": "user", "content": guidance.text},
        ],
        steering_messages=[guidance],
    )
    insertion = find_steering_insertions(message)[0]
    assert insertion.after_tool_call_id == "read-call"
    assert insertion.tool_name == "read"


def test_steering_positions_follow_exmekh_delivery_and_repeated_messages() -> None:
    first = QueuedMessage(text="Same correction")
    second = QueuedMessage(text="Same correction")
    later = QueuedMessage(text="Later correction")
    message = ChatMessage(
        role="assistant",
        raw_history_start=1,
        raw_history=[
            {"role": "user", "content": "Start"},
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "name": "bash", "tool_call_id": "first"},
            {"role": "user", "content": first.text},
            {"role": "user", "content": second.text},
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "name": "read_file", "tool_call_id": "second"},
            {"role": "user", "content": later.text},
        ],
        steering_messages=[first, second, later],
        tool_events=[
            ToolEvent(tool_name="bash", tool_call_id="first"),
            ToolEvent(tool_name="read_file", tool_call_id="second"),
        ],
    )
    positions = find_steering_insertions(message)
    assert [item.message_id for item in positions] == [
        first.id,
        second.id,
        later.id,
    ]
    assert [item.after_tool_call_id for item in positions] == [
        "first",
        "first",
        "second",
    ]
    assert [item.tool_name for item in positions] == ["bash", "bash", "read_file"]


def test_steering_position_uses_skill_invocation_and_ignores_pending() -> None:
    delivered = QueuedMessage(text="Use another route")
    pending = QueuedMessage(text="Pending")
    message = ChatMessage(
        role="assistant",
        raw_history_start=1,
        raw_history=[
            {"role": "user", "content": "Start"},
            {"role": "tool", "name": "research", "tool_call_id": "skill-call"},
            {"role": "user", "content": delivered.text},
        ],
        steering_messages=[delivered, pending],
        skills=[
            SkillNode(
                name="research",
                path="/research",
                opening_tool_call_id="skill-call",
            )
        ],
    )
    positions = find_steering_insertions(message)
    assert len(positions) == 1
    assert positions[0].tool_name == "research"
    assert positions[0].skill_path == "/research"


def test_steering_after_plain_answer_has_no_tool_anchor() -> None:
    guidance = QueuedMessage(text="Change answer")
    message = ChatMessage(
        role="assistant",
        raw_history_start=1,
        raw_history=[
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "Draft"},
            {"role": "user", "content": guidance.text},
        ],
        steering_messages=[guidance],
    )
    positions = find_steering_insertions(message)
    assert len(positions) == 1
    assert positions[0].after_tool_call_id is None
    assert positions[0].tool_name is None


def test_steering_position_recovers_tool_identity_from_assistant_call() -> None:
    guidance = QueuedMessage(text="Correction")
    message = ChatMessage(
        role="assistant",
        raw_history_start=1,
        raw_history=[
            {"role": "user", "content": "Start"},
            {
                "role": "assistant",
                "tool_calls": '[{"id":"call-1","function":{"name":"bash","arguments":{}}}]',
            },
            {"role": "tool", "content": "Tool output without identity"},
            {"role": "user", "content": guidance.text},
        ],
        steering_messages=[guidance],
        tool_events=[ToolEvent(tool_name="bash", tool_call_id="call-1")],
    )
    positions = find_steering_insertions(message)
    assert len(positions) == 1
    assert positions[0].after_tool_call_id == "call-1"
    assert positions[0].tool_name == "bash"


def test_steering_position_does_not_attach_to_an_unrelated_history_event() -> None:
    guidance = QueuedMessage(text="Correction")
    message = ChatMessage(
        role="assistant",
        raw_history_start=1,
        raw_history=[
            {"role": "user", "content": "Start"},
            {"role": "tool", "name": "bash", "tool_call_id": "call-1"},
            {"role": "user", "content": "Unrelated history entry"},
            {"role": "developer", "content": "Context changed"},
            {"role": "user", "content": guidance.text},
        ],
        steering_messages=[guidance],
    )
    positions = find_steering_insertions(message)
    assert len(positions) == 1
    assert positions[0].after_tool_call_id is None


def test_steering_close_waits_for_inflight_acceptance() -> None:
    steering = TurnSteering()
    entered = Event()
    release = Event()
    close_started = Event()
    message = QueuedMessage(text="Accepted before close")

    def accept(_: str) -> None:
        entered.set()
        assert release.wait(5)

    def close() -> None:
        close_started.set()
        steering.close()

    steering.bind(accept, history_start=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        sending = executor.submit(steering.send, message)
        assert entered.wait(5)
        closing = executor.submit(close)
        assert close_started.wait(5)
        assert not closing.done()
        release.set()
        assert sending.result(timeout=5)
        closing.result(timeout=5)

    assert not steering.available
    assert not steering.send(QueuedMessage(text="Rejected after close"))
    snapshot = ChatMessage(role="assistant")
    steering.apply_to(snapshot)
    assert snapshot.steering_messages == [message]
    assert snapshot.raw_history_start == 2


async def test_mock_exmekh_accepts_burst_and_later_guidance_during_inference(
    tmp_path: Path,
) -> None:
    provider = SteeringProbeProvider(block_second=True)
    steering = TurnSteering()
    observed_history: list[list[dict[str, Any]]] = []
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
    ):
        task = asyncio.create_task(
            run_agent_turn(
                **_mock_turn_options(
                    tmp_path,
                    steering,
                    on_raw_history=lambda raw, _: observed_history.append(raw),
                )
            )
        )
        try:
            await _wait_for_probe(provider.started)
            assert steering.send(QueuedMessage(text="First correction"))
            assert steering.send(QueuedMessage(text="Second correction"))
            assert not task.done()
            provider.release.set()
            await _wait_for_probe(provider.second_started)
            second_request = provider.requests[1]
            confirmed = ChatMessage(role="assistant")
            steering.apply_to(confirmed)
            assert len(find_steering_insertions(confirmed)) == 2
            guidance_index = next(
                index
                for index, item in enumerate(second_request)
                if item.get("content") == "First correction"
            )
            # ExMekh-Lib may read asynchronous bash output before accepting guidance.
            preceding_tool = second_request[guidance_index - 1]
            assert preceding_tool["role"] == "tool"
            assert all(
                item.after_tool_call_id == preceding_tool["tool_call_id"]
                for item in confirmed.steering_insertions
            )
            assert [
                item["content"] for item in second_request if item["role"] == "user"
            ][-3:] == ["Start", "First correction", "Second correction"]
            assert steering.send(QueuedMessage(text="Third correction"))
            provider.release_second.set()
            assert await asyncio.wait_for(task, 5) == (
                "Reply following guidance: Third correction"
            )
        finally:
            provider.release.set()
            provider.release_second.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert len(provider.requests) == 3
    assert [item["content"] for item in provider.requests[2] if item["role"] == "user"][
        -4:
    ] == [
        "Start",
        "First correction",
        "Second correction",
        "Third correction",
    ]
    assert observed_history
    assert [item["content"] for item in observed_history[-1] if item["role"] == "user"][
        -4:
    ] == [
        "Start",
        "First correction",
        "Second correction",
        "Third correction",
    ]
    assert not steering.available
    assert not steering.send(QueuedMessage(text="Too late"))
    steering.apply_to(confirmed)
    assert len(confirmed.steering_insertions) == 3
    # ExMekh-Lib may deliver completed asynchronous output after the intermediate answer.
    # Check the actual insertion boundary instead of assuming that no tool ran.
    third_request = provider.requests[2]
    third_index = next(
        index
        for index, item in enumerate(third_request)
        if item.get("content") == "Third correction"
    )
    preceding = third_request[third_index - 1]
    expected_call_id = (
        preceding.get("tool_call_id") if preceding["role"] == "tool" else None
    )
    assert confirmed.steering_insertions[-1].after_tool_call_id == expected_call_id


@pytest.mark.parametrize("nested", [False, True], ids=["skill", "nested-skill"])
async def test_exmekh_steering_in_skill_survives_compaction_and_export(
    tmp_path: Path,
    nested: bool,
) -> None:
    provider = SteeringProbeProvider(
        first_response="nested_skill" if nested else "skill", block_reply=True
    )
    collector = TurnStreamCollector()
    notifications: list[ChatMessage] = []
    guidance = QueuedMessage(text="Use another route")
    options = _mock_turn_options(
        tmp_path,
        collector.steering,
        on_raw_history=collector.set_raw_history,
    )
    options.update(
        on_tool_event=collector.append_tool,
        on_skill_event=collector.append_skill,
        on_steering_added=lambda: notifications.append(collector.snapshot_message()),
    )
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
    ):
        task = asyncio.create_task(run_agent_turn(**options))
        try:
            await _wait_for_probe(provider.started)
            if nested:
                provider.release.set()
                await _wait_for_probe(provider.second_started)
            assert collector.steering.send(guidance)
            assert collector.snapshot_message().steering_insertions == []
            provider.release.set()
            if nested:
                provider.release_second.set()
                result = await asyncio.wait_for(task, 5)
            else:
                await _wait_for_probe(provider.second_started)
                assert not task.done()
            assert len(notifications) == 1
            position = notifications[0].steering_insertions[0]
            assert position.message_id == guidance.id
            assert position.after_tool_call_id == (
                "open-nested-steering-skill" if nested else "steering-probe"
            )
            assert position.tool_name == ("nested-skill" if nested else "mock-skill")
            assert position.skill_path == (
                "/mock-skill/nested-skill" if nested else "/mock-skill"
            )
            if not nested:
                provider.release_second.set()
                result = await asyncio.wait_for(task, 5)
        finally:
            provider.release.set()
            provider.release_second.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(notifications) == 1  # Compaction does not emit a second confirmation.
    message = collector.build_message(fallback_content=result)
    assert message.steering_insertions == [position]
    session = ChatSession(messages=[ChatMessage(role="user", content="Start"), message])
    exported = session_to_exmekh_messages(session)
    restored = exmekh_messages_to_session(
        [
            raw_dict_to_exmekh_message(
                exmekh_message_to_raw_dict(item, include_snapshots=True)
            )
            for item in exported
        ]
    )
    assert find_steering_insertions(restored.messages[-1]) == [position]
    assert message.skills[0].status == "completed"
    assert len(provider.requests) == (6 if nested else 3)
    # ExMekh-Lib compacts the user entry out of root history into the skill transcript.
    assert not any(
        item.get("content") == guidance.text for item in provider.requests[-1]
    )
    skill_history = message.skill_history_tree[0]
    if nested:
        skill_history = skill_history["children"][0]
    assert any(
        item.get("content") == guidance.text for item in skill_history["history"]
    )
    for graph_session in (session, restored):
        nodes = build_graph_data_from_session(graph_session).histories[0].nodes
        steering_nodes = [node for node in nodes if node.kind == "steering"]
        assert len(steering_nodes) == 1
        steering_node = steering_nodes[0]
        assert not steering_node.events[0]["pending"]
        assert steering_node.events[0]["message_id"] == guidance.id
        assert steering_node.skill_path == position.skill_path
        assert steering_node.depth == (2 if nested else 1)
        assert nodes[steering_node.step_index - 1].events[0]["tool_call_id"] == (
            position.after_tool_call_id
        )
        assert nodes[steering_node.step_index + 1].events[0]["tool_call_id"] == (
            "nested-steering-tool" if nested else "close-steering-skill-2"
        )
    assert (
        find_steering_insertions(
            message.model_copy(update={"steering_insertions": None})
        )
        == []
    )


async def test_mock_exmekh_continues_after_plain_answer_when_steered(
    tmp_path: Path,
) -> None:
    provider = SteeringProbeProvider(first_response="answer")
    steering = TurnSteering()
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
    ):
        task = asyncio.create_task(
            run_agent_turn(**_mock_turn_options(tmp_path, steering))
        )
        try:
            await _wait_for_probe(provider.started)
            assert steering.send(QueuedMessage(text="Change the answer"))
            provider.release.set()
            assert await asyncio.wait_for(task, 5) == (
                "Reply following guidance: Change the answer"
            )
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert len(provider.requests) == 2
    assert [item["role"] for item in provider.requests[1]][-3:] == [
        "user",
        "assistant",
        "user",
    ]
    assert not steering.available


async def test_mock_exmekh_failure_does_not_transfer_guidance_to_a_new_run(
    tmp_path: Path,
) -> None:
    provider = SteeringProbeProvider(first_response="error")
    steering = TurnSteering()
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
    ):
        task = asyncio.create_task(
            run_agent_turn(**_mock_turn_options(tmp_path, steering))
        )
        try:
            await _wait_for_probe(provider.started)
            assert steering.send(QueuedMessage(text="Guidance for failed turn"))
            provider.release.set()
            with pytest.raises(RuntimeError, match="Mock provider failed"):
                await asyncio.wait_for(task, 5)
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert not steering.available
        assert not steering.send(QueuedMessage(text="Late guidance"))
        failed_snapshot = ChatMessage(role="assistant")
        steering.apply_to(failed_snapshot)
        assert failed_snapshot.steering_messages
        assert find_steering_insertions(failed_snapshot) == []
        response = await run_agent_turn(
            **_mock_turn_options(
                tmp_path, TurnSteering(), user_prompt="Independent request"
            )
        )

    assert response == "Reply following guidance: Independent request"
    assert len(provider.requests) == 2
    assert all(
        item.get("content") != "Guidance for failed turn"
        for item in provider.requests[1]
    )


async def test_mock_exmekh_cancellation_rejects_late_steering_and_stops_inference(
    tmp_path: Path,
) -> None:
    provider = SteeringProbeProvider()
    steering = TurnSteering()
    worker_finished = Event()

    def execute(**kwargs: Any) -> str:
        try:
            return execute_agent_turn_sync(**kwargs)
        finally:
            worker_finished.set()

    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            side_effect=execute,
        ),
    ):
        task = asyncio.create_task(
            run_agent_turn(**_mock_turn_options(tmp_path, steering))
        )
        try:
            await _wait_for_probe(provider.started)
            assert steering.send(QueuedMessage(text="Guidance before cancellation"))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not steering.available
            assert not steering.send(QueuedMessage(text="Guidance after cancellation"))
        finally:
            provider.release.set()
            await _wait_for_probe(worker_finished)

    assert len(provider.requests) == 1


async def test_quick_chat_never_offers_deep_agent_steering(tmp_path: Path) -> None:
    provider = SteeringProbeProvider(first_response="answer")
    steering = TurnSteering()
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
    ):
        task = asyncio.create_task(
            run_agent_turn(
                **_mock_turn_options(tmp_path, steering, agent_mode="quick_chat")
            )
        )
        try:
            await _wait_for_probe(provider.started)
            assert not steering.available
            assert not steering.send(QueuedMessage(text="Must stay queued"))
            provider.release.set()
            assert await asyncio.wait_for(task, 5) == "Initial answer"
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_registry_routes_steering_only_to_an_active_session() -> None:
    registry = ActiveTurnRegistry()
    guidance = QueuedMessage(text="Correction")
    assert not registry.can_steer("missing")
    assert not registry.steer("missing", guidance)
    collector = TurnStreamCollector()
    task = asyncio.create_task(asyncio.sleep(10))
    registry.start("chat", task, collector, started_at=1)
    assert not registry.can_steer("chat")
    received: list[str] = []
    collector.steering.bind(received.append, history_start=2)
    assert registry.can_steer("chat")
    assert not registry.steer("another-chat", guidance)
    assert registry.steer("chat", guidance)
    assert received == [guidance.text]
    assert registry.snapshot("chat", after_revision=0).message.steering_messages == [
        guidance
    ]
    collector.steering.bind(MagicMock(side_effect=RuntimeError()), history_start=2)
    assert not registry.steer("chat", guidance)
    registry.stop("chat")
    assert not registry.can_steer("chat")
    assert not registry.steer("chat", guidance)
    await asyncio.gather(task, return_exceptions=True)
    assert not registry.can_steer("chat")
    registry.cancel_all()


async def test_controller_steers_only_the_selected_item_and_keeps_other_sessions(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "chats")
    active_session = manager.get_current_session()
    other_session = manager.create_session("Other session")
    manager.switch_session(active_session.id)
    first = active_session.add_queued_message("Send after the turn")
    selected = active_session.add_queued_message("Steer this one", "Visible guidance")
    last = active_session.add_queued_message("Also send after the turn")
    other_message = other_session.add_queued_message("Other session guidance")
    manager.save_session(active_session)
    manager.save_session(other_session)

    collector = TurnStreamCollector()
    accepted: list[str] = []
    collector.steering.bind(accepted.append, history_start=2)
    task = asyncio.create_task(asyncio.sleep(10))
    active_turns.start(active_session.id, task, collector, started_at=1)
    client = Client(page("/test-steering-queue-selection"))
    with (
        client,
        patch("exmekh_agent.ui.chat.page_controller.session_manager", manager),
        patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
    ):
        controller = ChatPageController(client)
        controller.input_bar = InputBarComponent(
            lambda _: None,
            lambda: None,
            on_steer_queued=controller.steer_queued_message,
        )
        controller.input_bar.build()
        controller._render_queue_ui(active_session)
        assert len(controller.input_bar.steering_buttons) == 3
        assert all(button.enabled for button in controller.input_bar.steering_buttons)
        controller.input_bar._trigger_steer(selected.id)
        assert accepted == [selected.text]
        assert [item.id for item in active_session.queued_messages] == [
            first.id,
            last.id,
        ]
        assert [item.id for item in other_session.queued_messages] == [other_message.id]
        persisted = ChatSession.model_validate_json(
            (manager.storage_dir / f"{active_session.id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert [item.id for item in persisted.queued_messages] == [first.id, last.id]
        assert collector.snapshot_message().steering_messages == [selected]

        manager.switch_session(other_session.id)
        controller._sync_active_turn()
        assert not controller.input_bar.steering_enabled
        controller.steer_queued_message(selected.id)
        assert other_session.queued_messages == [other_message]

        manager.switch_session(active_session.id)
        collector.steering.bind(
            MagicMock(side_effect=RuntimeError("run closed")), history_start=2
        )
        controller.steer_queued_message(first.id)
        assert [item.id for item in active_session.queued_messages] == [
            first.id,
            last.id,
        ]
        assert accepted == [selected.text]
        assert controller.input_bar.steering_enabled
        controller.steer_queued_message(first.id)
        assert [item.id for item in active_session.queued_messages] == [
            first.id,
            last.id,
        ]

    active_turns.cancel_and_forget(active_session.id)
    await asyncio.gather(task, return_exceptions=True)
    client.delete()


@pytest.mark.parametrize("fails", [False, True])
def test_sync_runner_closes_steering_when_the_run_exits(fails: bool) -> None:
    control = TurnSteering()
    agent = MagicMock()
    agent.run.return_value.get_message.return_value = "Done"
    agent.run.return_value.history = []
    agent.run.return_value.skill_executions = []

    def run(**_):
        assert control.available
        assert control.send(QueuedMessage(text="Guidance"))
        if fails:
            raise ValueError("Failed turn")
        return agent.run.return_value

    agent.run.side_effect = run
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=SteeringProbeProvider(),
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.DeepAgent",
            return_value=agent,
        ),
    ):
        arguments = dict(
            user_prompt="Start",
            model="test",
            base_url="http://localhost:8080/v1",
            api_key="",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="deep_agent",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda *_: None,
            on_tool_event=lambda _: None,
            steering=control,
        )
        if fails:
            with pytest.raises(ValueError, match="Failed turn"):
                execute_agent_turn_sync(**arguments)
        else:
            assert execute_agent_turn_sync(**arguments) == "Done"
    assert not control.available
    agent.steer.assert_called_once_with("Guidance")


def test_steered_turn_keeps_tool_pairs_and_guidance_on_history_reload() -> None:
    raw_history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "before-guidance",
                    "function": {"name": "bash", "arguments": {"command": "echo hi"}},
                }
            ],
        },
        {
            "role": "tool",
            "name": "bash",
            "tool_call_id": "before-guidance",
            "content": "hi",
        },
        {"role": "user", "content": "Correction"},
        {"role": "assistant", "content": "Corrected answer"},
    ]
    session = ChatSession(
        messages=[
            ChatMessage(role="user", content="Start"),
            ChatMessage(
                role="assistant",
                content="Corrected answer",
                raw_history=raw_history,
                raw_history_start=2,
                steering_messages=[QueuedMessage(text="Correction")],
            ),
        ]
    )
    restored = ChatSession.model_validate_json(session.model_dump_json())
    exported = session_to_exmekh_messages(restored)
    assert exmekh_messages_to_session(exported).messages == restored.messages
    serialized = [
        exmekh_message_to_raw_dict(item, include_snapshots=True) for item in exported
    ]
    assert (
        exmekh_messages_to_session(
            [raw_dict_to_exmekh_message(item) for item in serialized]
        ).messages
        == restored.messages
    )
    messages = convert_to_exmekh_messages(restored.messages)
    assert [item.get_role_name() for item in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert [
        item.get_message() for item in messages if item.get_role_name() == "user"
    ] == ["Start", "Correction"]
    assert build_compacted_context_messages(restored) == raw_history
    assert build_raw_history_from_session(restored)[0] == raw_history
    graph_nodes = build_graph_data_from_session(restored).histories[0].nodes
    assert any(
        event.get("tool_name") == "bash"
        for node in graph_nodes
        for event in node.events
    )
    st_node = next(n for n in graph_nodes if n.kind == "steering")
    assert st_node.label == "Steering: Correction"
    assert st_node.events[0]["content"] == "Correction"
    assert st_node.events[0]["event_stage"] == "steering_message"


def test_multiple_steering_messages_preserve_both_tool_batches_on_export() -> None:
    raw_history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "first-tool", "function": {"name": "read", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "name": "read",
            "tool_call_id": "first-tool",
            "content": "source",
        },
        {"role": "user", "content": "Correction A"},
        {"role": "user", "content": "Correction B"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "second-tool", "function": {"name": "write", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "name": "write",
            "tool_call_id": "second-tool",
            "content": "saved",
        },
        {"role": "user", "content": "Correction C"},
        {"role": "assistant", "content": "Final"},
    ]
    guidance = [
        QueuedMessage(text="Correction A"),
        QueuedMessage(text="Correction B"),
        QueuedMessage(text="Correction C"),
    ]
    session = ChatSession(
        messages=[
            ChatMessage(role="user", content="Start"),
            ChatMessage(
                role="assistant",
                content="Final",
                raw_history=raw_history,
                raw_history_start=2,
                steering_messages=guidance,
            ),
        ]
    )
    exmekh_messages = session_to_exmekh_messages(session)
    assert [item.get_role_name() for item in exmekh_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    reconstructed = exmekh_messages_to_session(
        [
            raw_dict_to_exmekh_message(
                exmekh_message_to_raw_dict(item, include_snapshots=True)
            )
            for item in exmekh_messages
        ]
    )
    assert reconstructed.messages == session.messages
    assert build_compacted_context_messages(reconstructed) == raw_history
    assert build_raw_history_from_session(reconstructed)[0] == raw_history


async def test_controller_steers_real_exmekh_after_refresh_and_preserves_next_turn(
    tmp_path: Path,
) -> None:
    provider = SteeringProbeProvider(block_reply=True)
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    session.agent_mode = "deep_agent"
    session.settings.work_root = str(tmp_path)
    session.settings.capabilities_path = str(TEST_CAPABILITIES)
    session.settings.tool_approval_mode = "Approve All"
    session.messages = [
        ChatMessage(role="user", content="Previous request"),
        ChatMessage(role="assistant", content="Previous answer"),
    ]
    clients = [Client(page(f"/test-steering-{i}")) for i in range(2)]
    controllers = []
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            execute_agent_turn_sync,
        ),
        patch("exmekh_agent.ui.chat.page_controller.session_manager", manager),
        patch("exmekh_agent.ui.chat.chat_view.session_manager", manager),
        patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
    ):
        for client in clients:
            with client:
                controller = ChatPageController(client)
                controller.chat_view = ChatViewComponent(lambda _: None)
                controller.chat_view.build()
                controller.input_bar = InputBarComponent(
                    controller.start_send,
                    controller.stop,
                    on_steer_queued=controller.steer_queued_message,
                )
                controller.input_bar.build()
                controllers.append(controller)
        original, refreshed = controllers
        original._send_prepared(session, PreparedUserInput("Start", None, "Start"))
        turn = original._active_task
        try:
            async with asyncio.timeout(5):
                while not provider.started.is_set():
                    await asyncio.sleep(0.01)
            guidance = session.add_queued_message("Use the other file", "Correction")
            refreshed._sync_active_turn()
            assert refreshed.input_bar.steering_buttons[0].enabled
            refreshed.input_bar._trigger_steer(guidance.id)
            refreshed.steer_queued_message(guidance.id)  # A repeated click is harmless.
            assert not session.queued_messages
            assert original._active_task is turn
            assert len(provider.requests) == 1
            assert original.input_bar.is_busy
            assert refreshed.chat_view.current_steering_container is not None
            assert len(refreshed.chat_view.streaming_steering_ids) == 1
            provider.release.set()
            await _wait_for_probe(provider.second_started)
            assert not turn.done()
            refreshed._sync_active_turn()
            assert len(refreshed.chat_view.streaming_steering_insertions) == 1
            provider.release_second.set()
            await asyncio.wait_for(turn, 5)
            refreshed._sync_active_turn()
            assert not refreshed.input_bar.steering_enabled
            assert len(session.messages) == 4
            answer = session.messages[-1]
            assert answer.raw_history_start == 4
            assert answer.steering_messages == [guidance]
            assert "Use the other file" in answer.content
            request = provider.requests[1]
            guidance_index = next(
                index
                for index, item in enumerate(request)
                if item.get("content") == guidance.text
            )
            assert request[guidance_index - 1]["role"] == "tool"
            assert (
                find_steering_insertions(answer)[0].after_tool_call_id
                == (request[guidance_index - 1]["tool_call_id"])
            ), repr(answer.turn_raw_history())
            assert not active_turns.can_steer(session.id)

            # Reload saved JSON before sending the next independent turn.
            manager.load_all()
            session = manager.get_current_session()
            await refreshed.send("Continue")
            await asyncio.wait_for(refreshed._active_task, 5)
            follow_up = provider.requests[2]
            assert sum(item.get("content") == guidance.text for item in follow_up) == 1
            assert any(
                item.get("tool_call_id") == "steering-probe" for item in follow_up
            )
            assert len(session.messages) == 6
        finally:
            provider.release.set()
            provider.release_second.set()
            for controller in controllers:
                if controller._active_task and not controller._active_task.done():
                    controller.stop()
            await asyncio.gather(turn, return_exceptions=True)
            for client in clients:
                client.delete()


def test_unavailable_steering_keeps_the_queue_and_disables_the_button(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    guidance = session.add_queued_message("Keep this queued")
    client = Client(page("/test-steering-unavailable"))
    with (
        client,
        patch("exmekh_agent.ui.chat.page_controller.session_manager", manager),
        patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
    ):
        controller = ChatPageController(client)
        controller.input_bar = InputBarComponent(
            lambda _: None,
            lambda: None,
            on_steer_queued=controller.steer_queued_message,
        )
        controller.input_bar.build()
        controller._render_queue_ui(session)
        assert not controller.input_bar.steering_buttons[0].enabled
        controller.steer_queued_message(guidance.id)
        assert session.queued_messages == [guidance]
    client.delete()
