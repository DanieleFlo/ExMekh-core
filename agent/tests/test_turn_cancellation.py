from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest
from exmekh_lib.core.agents.node import DefaultCapabilities
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.interactive import InteractiveHandler
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from nicegui.client import Client as NiceGuiClient
from nicegui.page import page

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    check_provider_health,
    execute_agent_turn_sync,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.agents.cancellation import (
    CancellableProvider,
    CancellationEventHandler,
)
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.event_handlers import QuickChatEventHandler
from exmekh_agent.exmekh_bridge.models import ChatSession, ChatSettings, ChatTitle
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController
from tests.approval_probe import ApprovalProbeProvider
from tests.mock_client import MockProvider


@pytest.mark.parametrize("structured", [False, True])
def test_cancellation_wrapper_does_not_duplicate_tool_attachments(structured):
    provider = MockProvider()
    companion = UserMessage("tool attachment")
    messages = [
        UserMessage("task"),
        AssistantMessage(None, tool_calls=[ToolCall("call", FunctionCall("work", {}))]),
        ToolMessage("result", "call", "work", attachments=companion),
    ]
    request = (
        StructuredProviderRequest(messages, ChatTitle)
        if structured
        else ProviderRequest(messages)
    )
    method = "call_structured" if structured else "call"
    guarded = CancellableProvider(provider, Event())
    with patch.object(provider, method, return_value=Mock()) as call:
        getattr(guarded, method)(request)
    forwarded = call.call_args.args[0]
    assert forwarded.get_messages() == request.get_messages()
    assert forwarded.get_messages().count(companion) == 1


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
    request.interactive_handler = Mock(spec=InteractiveHandler)
    assert guarded.call(request).get_message()
    forwarded = provider.last_request
    assert forwarded is not request
    assert forwarded.get_extra() == request.get_extra()
    assert forwarded.get_tool_choice() == "none"
    assert forwarded.get_tools() is request.get_tools()
    assert forwarded.get_reasoning_effort_override() == "none"
    assert forwarded.require_interactive_handler() is request.interactive_handler
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
        [UserMessage("Choose a title")],
        ChatTitle,
        strict=False,
        reasoning_effort_override="none",
    )
    request.interactive_handler = Mock(spec=InteractiveHandler)
    original_call = provider.call_structured

    def respond(forwarded):
        assert forwarded.get_response_model() is ChatTitle
        assert forwarded.get_strict() is False
        assert forwarded.get_messages() == request.get_messages()
        assert forwarded.get_reasoning_effort_override() == "none"
        assert forwarded.require_interactive_handler() is request.interactive_handler
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


@pytest.mark.parametrize("track_context", [False, True])
async def test_cancellable_deep_agent_keeps_runtime_handler_when_entering_file_skill(
    tmp_path: Path, track_context: bool
) -> None:
    provider = MockProvider()
    requests: list[ProviderRequest] = []

    def respond(request: ProviderRequest) -> ProviderResponse:
        requests.append(request)
        if len(requests) == 1:
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall("open-files", FunctionCall("file-processing", {}))
                    ],
                )
            )
        if len(requests) == 2:
            # Match the real LocalProvider boundary even when no edit is generated.
            assert LocalProvider._has_interactive_tools(request)
            handler = request.require_interactive_handler()
            handler.begin()
            assert not handler.cancellation_requested()
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            "exit-files",
                            FunctionCall(
                                "exit_agent", {"text": "Files ready", "items": []}
                            ),
                        )
                    ],
                )
            )
        assert len(requests) == 3
        return ProviderResponse(AssistantMessage("Files ready"))

    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch.object(provider, "call", side_effect=respond),
    ):
        response = await run_agent_turn(
            user_prompt="Open the file-processing skill",
            model="mock-model",
            agent_mode="deep_agent",
            settings=ChatSettings(work_root=str(tmp_path)),
            on_token=lambda _: None,
            on_reasoning=lambda *_: None,
            on_tool_event=lambda _: None,
            on_context=(lambda _: None) if track_context else None,
        )

    assert response == "Files ready"
    assert len(requests) == 3


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
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            side_effect=execute,
        ),
        patch("exmekh_agent.ui.chat.page_controller.session_manager", manager),
        patch("exmekh_agent.ui.chat.chat_view.session_manager", manager),
        patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
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
    with patch(
        "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"
    ) as provider:
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
            "exmekh_agent.exmekh_bridge.agents.agent_runner.execute_agent_turn_sync",
            side_effect=execute,
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
