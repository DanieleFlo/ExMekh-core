from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nicegui.client import Client as NiceGuiClient
from nicegui.page import page
from qs.core.models.history import MessageHistory
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.interfaces.tool_hook import ToolContext, ToolResponse

from fe.qs_bridge.models import ChatSession
from fe.qs_bridge.tool_approval import (
    COMMAND_APPROVAL_SYSTEM_PROMPT,
    DEFAULT_TOOL_APPROVAL_MODE,
    TOOL_APPROVAL_ALL,
    TOOL_APPROVAL_LLM,
    TOOL_APPROVAL_MANUAL,
    TOOL_APPROVAL_MODES,
    CommandApprovalDecision,
    CommandApprovalEvaluator,
    CommandApprovalHook,
    build_approval_message_history,
    evaluate_command_approval_llm,
    load_command_approval_hds,
    normalize_tool_approval_mode,
)
from fe.state.session_manager import SessionManager
from fe.ui.input_bar import InputBarComponent
from fe.ui.page_controller import ChatPageController


def test_approval_mode_constants() -> None:
    assert TOOL_APPROVAL_MANUAL == "Manual"
    assert TOOL_APPROVAL_ALL == "Approve All"
    assert TOOL_APPROVAL_LLM == "Approve via LLM"
    assert TOOL_APPROVAL_MODES == ["Manual", "Approve All", "Approve via LLM"]
    assert DEFAULT_TOOL_APPROVAL_MODE == "Manual"
    assert "safety" in COMMAND_APPROVAL_SYSTEM_PROMPT.lower()

    # Test normalization helper
    assert normalize_tool_approval_mode("Manuale") == "Manual"
    assert normalize_tool_approval_mode("Approva tutto") == "Approve All"
    assert normalize_tool_approval_mode("Approva da un LLM") == "Approve via LLM"
    assert normalize_tool_approval_mode("Manual") == "Manual"
    assert normalize_tool_approval_mode("Approve All") == "Approve All"
    assert normalize_tool_approval_mode("Approve via LLM") == "Approve via LLM"
    assert normalize_tool_approval_mode(None) == "Manual"
    assert normalize_tool_approval_mode("invalid") == "Manual"


def test_command_approval_hook_manual_approved() -> None:
    approved_commands: list[str] = []
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_MANUAL,
        history=[],
        request_manual_approval=lambda cmd: True,
        on_command_approved=lambda cmd, history: approved_commands.append(cmd),
    )

    ctx = ToolContext(
        arguments={"command": "ls -la"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert result is None  # Pass through unchanged
    assert approved_commands == ["ls -la"]
    assert hook.history == ["ls -la"]


def test_command_approval_hook_manual_declined() -> None:
    approved_commands: list[str] = []
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_MANUAL,
        history=[],
        request_manual_approval=lambda cmd: False,
        on_command_approved=lambda cmd, history: approved_commands.append(cmd),
    )

    ctx = ToolContext(
        arguments={"command": "rm -rf /"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert isinstance(result, ToolResponse)
    assert "declined" in result.result.lower()
    assert approved_commands == []
    assert hook.history == []


def test_command_approval_hook_manual_no_callback_falls_back() -> None:
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_MANUAL,
        history=[],
        request_manual_approval=None,
    )
    ctx = ToolContext(
        arguments={"command": "echo test"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert isinstance(result, ToolResponse)
    assert "not handled" in result.result.lower()


def test_command_approval_hook_no_command_in_args() -> None:
    hook = CommandApprovalHook(mode=TOOL_APPROVAL_MANUAL)
    ctx = ToolContext(
        arguments={"other_arg": 123},
        work_root=Path.cwd(),
        history=(),
    )
    assert hook.handle(ctx) is None


def test_command_approval_hook_approve_all() -> None:
    approved_commands: list[str] = []
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_ALL,
        history=["git status"],
        request_manual_approval=lambda cmd: False,  # Should not be called
        on_command_approved=lambda cmd, history: approved_commands.append(cmd),
    )

    ctx = ToolContext(
        arguments={"command": "npm run build"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert result is None
    assert approved_commands == ["npm run build"]
    assert hook.history == ["git status", "npm run build"]


def test_command_approval_record_approval_no_callback() -> None:
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_ALL,
        history=[],
        on_command_approved=None,
    )
    ctx = ToolContext(
        arguments={"command": "pwd"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert result is None
    assert hook.history == ["pwd"]


def test_command_approval_record_approval_callback_raises() -> None:
    def bad_callback(cmd: str, history: list[dict[str, str]]) -> None:
        raise ValueError("Callback boom")

    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_ALL,
        history=[],
        on_command_approved=bad_callback,
    )
    ctx = ToolContext(
        arguments={"command": "pwd"},
        work_root=Path.cwd(),
        history=(),
    )
    result = hook.handle(ctx)
    assert result is None
    assert hook.history == ["pwd"]


def test_command_approval_hook_llm_approved() -> None:
    approved_commands: list[str] = []
    mock_decision = CommandApprovalDecision(approved=True, reason="Safe command")

    with patch.object(
        CommandApprovalEvaluator,
        "evaluate",
        return_value=mock_decision,
    ) as mock_eval:
        hook = CommandApprovalHook(
            mode=TOOL_APPROVAL_LLM,
            history=["echo 1"],
            on_command_approved=lambda cmd, history: approved_commands.append(cmd),
            model="test-model",
            base_url="http://mock-provider",
            api_key="mock-key",
            api_type="chat_completions",
        )

        ctx = ToolContext(
            arguments={"command": "cat README.md"},
            work_root=Path.cwd(),
            history=(),
        )
        result = hook.handle(ctx)
        assert result is None
        assert approved_commands == ["cat README.md"]
        assert "cat README.md" in hook.history
        mock_eval.assert_called_once_with(
            command="cat README.md",
            context=ctx,
        )


def test_command_approval_hook_llm_declined() -> None:
    approved_commands: list[str] = []
    mock_decision = CommandApprovalDecision(
        approved=False, reason="Destructive directory removal"
    )

    with patch.object(
        CommandApprovalEvaluator,
        "evaluate",
        return_value=mock_decision,
    ):
        hook = CommandApprovalHook(
            mode=TOOL_APPROVAL_LLM,
            history=[],
            on_command_approved=lambda cmd, history: approved_commands.append(cmd),
        )

        ctx = ToolContext(
            arguments={"command": "rm -rf src"},
            work_root=Path.cwd(),
            history=(),
        )
        result = hook.handle(ctx)
        assert isinstance(result, ToolResponse)
        assert "Destructive directory removal" in str(result.result)
        assert approved_commands == []
        assert hook.history == []


def test_evaluate_command_approval_llm_calls_client_structured() -> None:
    mock_response = MagicMock()
    mock_response.get_message.return_value = CommandApprovalDecision(
        approved=True, reason="Valid tests"
    )

    with (
        patch("fe.qs_bridge.tool_approval.LocalProvider") as mock_provider_cls,
        patch("fe.qs_bridge.tool_approval.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.call_structured.return_value = mock_response
        mock_client.create_history_checkpoint.return_value = MagicMock()
        mock_client_cls.return_value = mock_client

        decision = evaluate_command_approval_llm(
            command="pytest tests",
            history=["git status"],
            model="qwen-model",
            base_url="http://local-provider:1234",
            api_key="key",
            api_type="responses",
        )

        assert decision.approved is True
        assert decision.reason == "Valid tests"
        mock_provider_cls.assert_called_once_with(
            model="qwen-model",
            api="responses",
            base_url="http://local-provider:1234",
            api_key="key",
        )
        mock_client_cls.assert_called_once()
        args, kwargs = mock_client_cls.call_args
        assert (
            kwargs["system_message"].get_message()
            == load_command_approval_hds().get_hds()
        )


def test_evaluate_command_approval_llm_handles_exception() -> None:
    with patch(
        "fe.qs_bridge.tool_approval.LocalProvider",
        side_effect=RuntimeError("Provider down"),
    ):
        decision = evaluate_command_approval_llm("echo test")
        assert decision.approved is False
        assert "Error during LLM approval evaluation" in decision.reason


def test_evaluator_rolls_back_history_when_declined() -> None:
    mock_response = MagicMock()
    mock_response.get_message.return_value = CommandApprovalDecision(
        approved=False, reason="Unsafe"
    )

    with (
        patch("fe.qs_bridge.tool_approval.LocalProvider"),
        patch("fe.qs_bridge.tool_approval.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.call_structured.return_value = mock_response
        initial_checkpoint = MagicMock()
        mock_client.create_history_checkpoint.return_value = initial_checkpoint
        mock_client_cls.return_value = mock_client

        evaluator = CommandApprovalEvaluator()
        initial_history = evaluator.dump_history()
        decision = evaluator.evaluate("rm -rf /")

        assert decision.approved is False
        assert evaluator.dump_history() == initial_history


def test_load_command_approval_hds_and_history_building() -> None:
    hds = load_command_approval_hds()
    assert hds.name == "command-approval"
    assert hds.get_agent_context_id() == "command-approval"

    # From MessageHistory directly
    mh = MessageHistory([SystemMessage("sys")])
    assert build_approval_message_history(mh) is mh

    # From raw history dump
    dump = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "Evaluate command:\n```bash\nls\n```"},
        {"role": "assistant", "content": '{"approved": true, "reason": "ok"}'},
    ]
    restored = build_approval_message_history(dump)
    assert len(restored.get_history()) == 3

    # Fallback when HDS file does not exist
    with patch(
        "fe.qs_bridge.tool_approval.COMMAND_APPROVAL_HDS_PATH", Path("non_existent.md")
    ):
        fallback_hds = load_command_approval_hds()
        assert fallback_hds.name == "command-approval"


def test_chat_settings_approval_mode_defaults_and_persistence(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()

        session = sm.create_session("Test Approval Session")
        assert session.settings.tool_approval_mode == "Manual"
        assert session.settings.approved_commands_history == []

        session.settings.set_tool_approval_mode("Approve via LLM")
        session.settings.record_command_approval("ls", [])
        sm.save_session(session)

        # Reload sessions from disk
        sm.load_all()
        reloaded = sm.get_session(session.id)
        assert reloaded is not None
        assert reloaded.settings.tool_approval_mode == "Approve via LLM"
        assert reloaded.settings.approved_commands_history == ["ls"]


def test_input_bar_approval_ui_interaction() -> None:
    mode_changes: list[str] = []
    with NiceGuiClient(page("/test-approval-bar")):
        bar = InputBarComponent(
            on_send=lambda text: None,
            on_stop=lambda: None,
            on_approval_mode_changed=lambda m: mode_changes.append(m),
        )
        bar.build()

    assert bar.approval_select is not None
    assert bar.approval_select.value == "Manual"
    assert bar.approval_container is not None
    assert bar.approval_container.visible is False

    # Simulate changing the mode via select value
    bar.approval_select.value = "Approve All"
    assert bar.approval_mode == "Approve All"
    assert mode_changes == ["Approve All"]

    # External update via set_approval_mode with English and legacy Italian
    bar.set_approval_mode("Approve via LLM")
    assert bar.approval_mode == "Approve via LLM"
    assert bar.approval_select.value == "Approve via LLM"

    bar.set_approval_mode("Manuale")
    assert bar.approval_mode == "Manual"
    assert bar.approval_select.value == "Manual"

    # Show approval prompt
    accepted = False
    declined = False

    def on_accept() -> None:
        nonlocal accepted
        accepted = True

    def on_decline() -> None:
        nonlocal declined
        declined = True

    bar.show_approval_prompt(
        command="git status",
        on_accept=on_accept,
        on_decline=on_decline,
    )
    assert bar.approval_command_label.text == "git status"
    assert bar.approval_container.visible is True

    # Accept
    bar._handle_approval_accept()
    assert accepted is True
    assert bar.approval_container.visible is False

    # Show again and decline
    bar.show_approval_prompt(
        command="rm -rf /",
        on_accept=on_accept,
        on_decline=on_decline,
    )
    assert bar.approval_container.visible is True
    bar._handle_approval_decline()
    assert declined is True
    assert bar.approval_container.visible is False

    # Calling set_busy(False) hides any open prompt
    bar.show_approval_prompt("test", on_accept, on_decline)
    assert bar.approval_container.visible is True
    bar.set_busy(False)
    assert bar.approval_container.visible is False


def test_page_controller_approval_mode_changed(tmp_path: Path) -> None:
    with (
        patch("fe.state.session_manager.config") as mock_cfg,
        patch("fe.ui.page_controller.ui.notify") as mock_notify,
    ):
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        session = sm.create_session("PageController Test")

        controller = ChatPageController.__new__(ChatPageController)
        controller._client = MagicMock()
        controller._client.is_deleted = False

        with patch("fe.ui.page_controller.session_manager", sm):
            controller.approval_mode_changed("Approve via LLM")
            assert session.settings.tool_approval_mode == "Approve via LLM"
            mock_notify.assert_called_with(
                "Approval mode set to: Approve via LLM", type="info"
            )


def test_page_controller_stop_cleans_pending_approval(tmp_path: Path) -> None:
    client = NiceGuiClient(page("/test-stop-pending-approval"))
    controller = ChatPageController(client)
    manager = SessionManager(tmp_path / "chats")
    decision: Future[bool] = Future()
    controller._pending_approval = decision
    controller._pending_approval_session_id = manager.get_current_session().id
    bar = MagicMock()
    controller.input_bar = bar

    with (
        patch("fe.ui.page_controller.session_manager", manager),
        patch("fe.ui.page_controller.ui.notify"),
        patch("fe.ui.page_controller.active_turns.stop", return_value=True),
        patch.object(controller, "update_context_usage"),
    ):
        controller.stop()

    with pytest.raises(asyncio.CancelledError):
        decision.result(timeout=1)
    assert controller._pending_approval is None
    bar.hide_approval_prompt.assert_called()
    client.delete()


def test_build_approval_message_history_edge_cases() -> None:
    raw = [
        {"role": "system", "content": "Custom System"},
        {"role": "system", "content": "Duplicate System Ignored"},
        {"role": "unknown_role", "content": "Ignored"},
        {"role": "user", "content": "User prompt"},
        {"role": "assistant", "content": "not a valid json"},
    ]
    mh = build_approval_message_history(raw)
    msgs = mh.get_history()
    assert len(msgs) == 3
    assert msgs[0].get_role_name() == "system"
    assert msgs[0].get_message() == "Custom System"
    assert msgs[1].get_role_name() == "user"
    assert msgs[2].get_role_name() == "assistant"

    # No system message in raw list -> prepends default system message
    raw_no_sys = [{"role": "user", "content": "Just user"}]
    mh_no_sys = build_approval_message_history(raw_no_sys)
    msgs_no_sys = mh_no_sys.get_history()
    assert msgs_no_sys[0].get_role_name() == "system"
    assert msgs_no_sys[1].get_role_name() == "user"

    # Exception during check_history falls back to base_history
    with patch(
        "fe.qs_bridge.tool_approval.MessageHistory.check_history",
        side_effect=ValueError("bad history"),
    ):
        mh_fallback = build_approval_message_history(
            [{"role": "user", "content": "test"}]
        )
        assert len(mh_fallback.get_history()) == 1
        assert mh_fallback.get_history()[0].get_role_name() == "system"


def test_command_approval_evaluator_with_work_root() -> None:
    evaluator = CommandApprovalEvaluator(model="test-model")
    mock_resp = MagicMock()
    mock_resp.get_message.return_value = CommandApprovalDecision(
        approved=True, reason="Safe"
    )

    with (
        patch("fe.qs_bridge.tool_approval.LocalProvider"),
        patch("fe.qs_bridge.tool_approval.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.call_structured.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        test_path = Path("/custom/work/path")
        ctx = ToolContext(
            arguments={"command": "ls"},
            work_root=test_path,
            history=(),
        )
        dec = evaluator.evaluate("ls", context=ctx)
        assert dec.approved is True

        args, _ = mock_client.call_structured.call_args
        request = args[0]
        assert f"Working directory: `{test_path}`" in request.message.get_message()


def test_command_approval_hook_edge_cases() -> None:
    hook = CommandApprovalHook(
        mode=TOOL_APPROVAL_MANUAL,
        history=MessageHistory(
            [SystemMessage("sys"), UserMessage("plain user text without code block")]
        ),
        request_manual_approval=lambda cmd: True,
    )

    # User message in history without code block markdown
    assert "plain user text without code block" in hook.history

    # Context without command argument returns None
    no_cmd_ctx = ToolContext(
        arguments={},
        work_root=Path.cwd(),
        history=(),
    )
    assert hook.handle(no_cmd_ctx) is None

    # Duplicate approval does not record twice
    hook._record_approval("cmd_once")
    count_before = len(hook.history)
    hook._record_approval("cmd_once")
    assert len(hook.history) == count_before


def test_chat_session_tool_approval_raw_history_property() -> None:
    session = ChatSession()
    assert session.settings.tool_approval_raw_history == []
    session.settings.tool_approval_raw_history = [{"role": "system", "content": "sys"}]
    assert session.settings.tool_approval_raw_history == [
        {"role": "system", "content": "sys"}
    ]
