from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.interfaces.tool_hook import ToolContext, ToolResponse
from nicegui.client import Client as NiceGuiClient
from nicegui.page import page

from fe.config import config
from fe.exmekh_bridge.agent_runner import check_provider_health
from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.models import ChatSession
from fe.exmekh_bridge.tool_approval import (
    TOOL_APPROVAL_ALL,
    TOOL_APPROVAL_LLM,
    CommandApprovalDecision,
    CommandApprovalEvaluator,
    CommandApprovalHook,
    build_approval_message_history,
)
from fe.state.active_turns import active_turns
from fe.state.session_manager import SessionManager
from fe.ui.input_bar import InputBarComponent
from fe.ui.page_controller import ChatPageController
from tests.mock_client import MockProvider


def test_evaluator_preserves_only_approved_turns_using_the_real_exmekh_client() -> None:
    provider = MockProvider()
    decisions = [
        CommandApprovalDecision(approved=True, reason="Read only"),
        CommandApprovalDecision(approved=False, reason="Destructive"),
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return StructuredProviderResponse(StructuredAssistantMessage(decisions.pop(0)))

    evaluator = CommandApprovalEvaluator(history=["pwd"], api_type="responses")
    with (
        patch("fe.exmekh_bridge.tool_approval.LocalProvider", return_value=provider),
        patch.object(provider, "call_structured", side_effect=respond),
    ):
        assert evaluator.evaluate("git status").approved
        approved_history = evaluator.dump_history()
        assert evaluator.approved_commands() == ["pwd", "git status"]
        assert not evaluator.evaluate("rm -rf /").approved
        assert evaluator.dump_history() == approved_history
    assert len(requests[1].get_messages()) == len(approved_history) + 1
    assert requests[1].get_response_model() is CommandApprovalDecision

    with patch(
        "fe.exmekh_bridge.tool_approval.LocalProvider",
        side_effect=RuntimeError("offline"),
    ):
        assert not evaluator.evaluate("echo missing provider").approved
    assert evaluator.dump_history() == approved_history


def test_approval_callback_type_error_is_not_retried() -> None:
    callback = MagicMock(side_effect=TypeError("error inside callback"))
    hook = CommandApprovalHook(mode=TOOL_APPROVAL_ALL, on_command_approved=callback)
    assert (
        hook.handle(
            ToolContext(arguments={"command": "pwd"}, work_root=Path.cwd(), history=())
        )
        is None
    )
    callback.assert_called_once()
    assert callback.call_args.args[0] == "pwd"
    assert len(callback.call_args.args[1]) == 3


def test_command_approval_context_and_settings_remain_consistent() -> None:
    settings = ChatSession().settings
    history = [{"role": "system", "content": "Approval instructions"}]
    settings.record_command_approval("git status", history)
    settings.record_command_approval("git status", history)
    history[0]["content"] = "modified outside settings"
    assert settings.approved_commands_history == ["git status"]
    assert settings.tool_approval_raw_history[0]["content"] == "Approval instructions"
    settings.set_tool_approval_mode("Approva tutto")
    assert settings.tool_approval_mode == TOOL_APPROVAL_ALL


def test_unrecognized_history_roles_restore_an_initial_system_message() -> None:
    restored = build_approval_message_history(
        [{"role": "unknown", "content": "ignored"}]
    )
    assert len(restored.get_history()) == 1
    assert restored.get_history()[0].get_role_name() == "system"


def test_llm_decline_without_a_reason_still_blocks_execution() -> None:
    evaluator = CommandApprovalEvaluator()
    with patch.object(
        evaluator, "evaluate", return_value=CommandApprovalDecision(approved=False)
    ):
        hook = CommandApprovalHook(mode=TOOL_APPROVAL_LLM, evaluator=evaluator)
        response = hook.handle(
            ToolContext(
                arguments={"command": "unsafe"}, work_root=Path.cwd(), history=()
            )
        )
    assert isinstance(response, ToolResponse)
    assert "not approved" in response.result
    assert hook.history == []


@pytest.mark.parametrize("action", ["approve", "decline", "stop", "switch", "delete"])
async def test_manual_approval_releases_the_worker_on_every_ui_exit(
    tmp_path: Path,
    action: str,
) -> None:
    client = NiceGuiClient(page(f"/test-pending-approval-{action}"))
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    worker_done = asyncio.Event()
    decisions: list[bool] = []
    interrupted: list[bool] = []
    loop = asyncio.get_running_loop()
    with client:
        controller = ChatPageController(client)
        bar = InputBarComponent(on_send=lambda _: None, on_stop=controller.stop)
        bar.build()
        controller.input_bar = bar

    async def fake_turn(**kwargs):
        def request_approval() -> str:
            try:
                approved = kwargs["on_request_manual_approval"]("git status")
                decisions.append(approved)
                if approved:
                    kwargs["on_command_approved"](
                        "git status", [{"role": "user", "content": "git status"}]
                    )
                return "finished"
            except asyncio.CancelledError:
                interrupted.append(True)
                raise
            finally:
                loop.call_soon_threadsafe(worker_done.set)

        return await loop.run_in_executor(None, request_approval)

    with (
        patch("fe.ui.page_controller.session_manager", manager),
        patch("fe.ui.page_controller.run_agent_turn", side_effect=fake_turn),
        patch("fe.ui.page_controller.ui.notify"),
    ):
        collector = TurnStreamCollector()
        turn = asyncio.create_task(
            controller._run_turn(
                session=session,
                user_text="Inspect repository",
                prior_history=[],
                collector=collector,
                generate_title=False,
            )
        )
        controller._active_task = turn
        active_turns.start(session.id, turn, collector, started_at=0)
        try:
            for _ in range(100):
                if bar.approval_container.visible:
                    break
                await asyncio.sleep(0.01)
            assert bar.approval_container.visible
            with client:
                if action == "approve":
                    bar._handle_approval_accept()
                elif action == "decline":
                    bar._handle_approval_decline()
                elif action == "stop":
                    controller.stop()
                elif action == "switch":
                    other_session = manager.create_session("Another conversation")
                    controller.session_switched(other_session.id)
                else:
                    client.delete()
            await asyncio.wait_for(worker_done.wait(), timeout=2)
            await asyncio.wait_for(turn, timeout=2)
            assert decisions == ([] if action == "stop" else [action == "approve"])
            assert interrupted == ([True] if action == "stop" else [])
            assert controller._pending_approval is None
            assert session.settings.approved_commands_history == (
                ["git status"] if action == "approve" else []
            )
        finally:
            controller._cancel_pending_approval()
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)


@pytest.mark.live
async def test_live_command_approval_retains_accepted_context_and_rejects_destruction() -> (
    None
):
    alive, models = await check_provider_health()
    if not alive or not models:
        pytest.skip("Configured provider is unavailable")
    evaluator = CommandApprovalEvaluator(
        model=config.default_model if config.default_model in models else models[0]
    )
    loop = asyncio.get_running_loop()
    accepted = await loop.run_in_executor(
        None, partial(evaluator.evaluate, "git status")
    )
    assert accepted.approved, accepted.reason
    checkpoint = evaluator.dump_history()
    declined = await loop.run_in_executor(None, partial(evaluator.evaluate, "rm -rf /"))
    assert not declined.approved
    assert evaluator.dump_history() == checkpoint
