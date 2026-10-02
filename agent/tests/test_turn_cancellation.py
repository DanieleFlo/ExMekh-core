from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from exmekh_lib.core.agents.node import DefaultCapabilities
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from nicegui.client import Client as NiceGuiClient
from nicegui.page import page

from fe.config import config
from fe.exmekh_bridge.agent_runner import (
    check_provider_health,
    execute_agent_turn_sync,
    run_agent_turn,
)
from fe.exmekh_bridge.cancellation import CancellableProvider, CancellationEventHandler
from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.event_handlers import QuickChatEventHandler
from fe.exmekh_bridge.models import ChatSession, ChatTitle
from fe.state.active_turns import active_turns
from fe.state.session_manager import SessionManager
from fe.ui.chat_view import ChatViewComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.page_controller import ChatPageController
from tests.approval_probe import ApprovalProbeProvider
from tests.mock_client import MockProvider


def test_cancellable_provider_preserves_public_request_options_and_streams() -> None:
    provider = MockProvider()
    cancellation = Event()
    guarded = CancellableProvider(provider, cancellation)
    tokens: list[str] = []
    handler = QuickChatEventHandler(tokens.append, lambda *_: None)
    request = ProviderRequest(
        [UserMessage("hello")],
        extra={"temperature": 0.2},
        tool_choice="none",
        stream=True,
        on_event=handler,
        reasoning_effort_override="none",
    )
    assert guarded.call(request).get_message()
    forwarded = provider.last_request
    assert forwarded is not request
    assert forwarded.get_extra() == request.get_extra()
    assert forwarded.get_tool_choice() == "none"
    assert forwarded.get_tools() is request.get_tools()
    assert forwarded.get_reasoning_effort_override() == "none"
    assert forwarded.get_streaming()
    assert tokens
    assert request.on_event is handler
    silent_request = ProviderRequest([UserMessage("no stream callback")])
    CancellationEventHandler(
        cancellation, silent_request.execute_on_stream_event
    ).handle(ProviderStreamEvent(ProviderStreamEventType.START))

    cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        guarded.call(request)
    with pytest.raises(asyncio.CancelledError):
        forwarded.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, "late")
        )


def test_structured_requests_keep_the_schema_and_stop_after_cancellation() -> None:
    cancellation = Event()
    provider = MockProvider()
    guarded = CancellableProvider(provider, cancellation)
    request = StructuredProviderRequest(
        [UserMessage("Choose a title")], ChatTitle, strict=False
    )
    original_call = provider.call_structured

    def respond(forwarded):
        assert forwarded.get_response_model() is ChatTitle
        assert forwarded.get_strict() is False
        assert forwarded.get_messages() == request.get_messages()
        return original_call(forwarded)

    with patch.object(provider, "call_structured", side_effect=respond):
        assert guarded.call_structured(request).get_message().title
    with patch.object(
        provider, "call_structured", side_effect=lambda _: cancellation.set()
    ):
        with pytest.raises(asyncio.CancelledError):
            guarded.call_structured(request)
    with pytest.raises(asyncio.CancelledError):
        guarded.call_structured(request)


def test_non_streaming_response_is_discarded_when_cancelled() -> None:
    cancellation = Event()
    provider = MockProvider()
    guarded = CancellableProvider(provider, cancellation)
    with patch.object(provider, "call", side_effect=lambda _: cancellation.set()):
        with pytest.raises(asyncio.CancelledError):
            guarded.call(ProviderRequest([UserMessage("hello")]))


@pytest.mark.parametrize("action", ["stop", "decline"])
async def test_real_exmekh_turn_stops_before_another_inference_when_approval_is_cancelled(
    tmp_path: Path, action: str
) -> None:
    provider = ApprovalProbeProvider()
    client = NiceGuiClient(page(f"/test-real-approval-{action}"))
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    session.agent_mode = "deep_agent"
    session.settings.work_root = str(tmp_path)
    session.settings.capabilities_path = str(
        DefaultCapabilities(mocks=True).capabilities_path
    )
    collector = TurnStreamCollector()
    worker_finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    with client:
        controller = ChatPageController(client)
        controller.input_bar = InputBarComponent(lambda _: None, controller.stop)
        controller.input_bar.build()
        controller.chat_view = ChatViewComponent(lambda _: None)
        controller.chat_view.build()

    def execute(**kwargs):
        try:
            return execute_agent_turn_sync(**kwargs)
        finally:
            loop.call_soon_threadsafe(worker_finished.set)

    with (
        patch("fe.exmekh_bridge.agent_runner.LocalProvider", return_value=provider),
        patch(
            "fe.exmekh_bridge.agent_runner.execute_agent_turn_sync", side_effect=execute
        ),
        patch("fe.ui.page_controller.session_manager", manager),
        patch("fe.ui.chat_view.session_manager", manager),
        patch("fe.ui.page_controller.ui.notify"),
    ):
        turn = asyncio.create_task(
            controller._run_turn(
                session=session,
                user_text="Run echo approval_probe",
                prior_history=[],
                collector=collector,
                generate_title=False,
            )
        )
        controller._active_task = turn
        active_turns.start(session.id, turn, collector, started_at=0)
        try:
            async with asyncio.timeout(5):
                while not controller.input_bar.approval_container.visible:
                    await asyncio.sleep(0.01)
            with client:
                if action == "stop":
                    controller.stop()
                else:
                    controller.input_bar._handle_approval_decline()
            await asyncio.wait_for(turn, 3)
            await asyncio.wait_for(worker_finished.wait(), 3)
            assert provider.request_count == (1 if action == "stop" else 2)
            assert session.messages[-1].error == (
                "cancelled" if action == "stop" else None
            )
            assert session.messages[-1].content.startswith("Output before approval")
            if action == "decline":
                assert (
                    "Reply after declining the command" in session.messages[-1].content
                )
            labels = [
                element.text
                for element in controller.chat_view.chat_container.descendants()
                if hasattr(element, "text")
            ]
            assert ("Generation stopped by user" in labels) is (action == "stop")
            assert session.settings.approved_commands_history == []
        finally:
            controller._cancel_pending_approval(interrupt_turn=True)
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
            client.delete()


async def test_cancelled_turn_waiting_for_the_provider_lock_never_starts() -> None:
    cancellation = Event()
    cancellation.set()
    with patch("fe.exmekh_bridge.agent_runner.LocalProvider") as provider:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: execute_agent_turn_sync(
                    "hello",
                    "model",
                    "url",
                    "key",
                    "responses",
                    "none",
                    "quick_chat",
                    "capabilities",
                    "main",
                    lambda _: None,
                    lambda *_: None,
                    lambda _: None,
                    cancellation_event=cancellation,
                ),
            )
        provider.assert_not_called()


@pytest.mark.live
async def test_live_stop_pending_approval_prevents_another_model_request(
    tmp_path: Path,
) -> None:
    alive, models = await check_provider_health()
    if not alive or not models:
        pytest.skip("Configured provider is unavailable")
    model = config.default_model if config.default_model in models else models[0]
    session = ChatSession(agent_mode="deep_agent")
    session.settings.work_root = str(tmp_path)
    session.settings.capabilities_path = str(
        DefaultCapabilities(mocks=True).capabilities_path
    )
    approval_requested = asyncio.Event()
    worker_finished = asyncio.Event()
    decision: Future[bool] = Future()
    requests: list[ProviderRequest] = []
    loop = asyncio.get_running_loop()
    original_call = LocalProvider.call

    def call(provider: LocalProvider, request: ProviderRequest):
        requests.append(request)
        return original_call(provider, request)

    def execute(**kwargs):
        try:
            return execute_agent_turn_sync(**kwargs)
        finally:
            loop.call_soon_threadsafe(worker_finished.set)

    def request_approval(command: str) -> bool:
        loop.call_soon_threadsafe(approval_requested.set)
        return decision.result()

    with (
        patch.object(LocalProvider, "call", call),
        patch(
            "fe.exmekh_bridge.agent_runner.execute_agent_turn_sync", side_effect=execute
        ),
    ):
        turn = asyncio.create_task(
            run_agent_turn(
                user_prompt="Call bash once with command 'echo approval_probe' and wait_seconds=60. "
                "Wait for command approval. Do not delegate or use other tools.",
                model=model,
                agent_mode="deep_agent",
                agent_context_id=session.id,
                settings=session.settings,
                on_token=lambda _: None,
                on_reasoning=lambda *_: None,
                on_tool_event=lambda _: None,
                on_request_manual_approval=request_approval,
            )
        )
        try:
            await asyncio.wait_for(approval_requested.wait(), 90)
            request_count = len(requests)
            turn.cancel()
            decision.set_exception(asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                await turn
            await asyncio.wait_for(worker_finished.wait(), 5)
            assert len(requests) == request_count
        finally:
            if not decision.done():
                decision.set_exception(asyncio.CancelledError())
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
