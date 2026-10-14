from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.metadata import HDSMetadata
from exmekh_lib.core.agents.hds.parser import HDSMarkdownParser
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.tool_hook import (
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
)
from exmekh_lib.providers.local import LocalProvider
from pydantic import BaseModel, Field, ValidationError

from exmekh_agent.config import config
from exmekh_agent.config.approvals import (
    DEFAULT_TOOL_APPROVAL_MODE as DEFAULT_TOOL_APPROVAL_MODE,
)
from exmekh_agent.config.approvals import (
    LEGACY_TOOL_APPROVAL_MODES as LEGACY_TOOL_APPROVAL_MODES,
)
from exmekh_agent.config.approvals import TOOL_APPROVAL_ALL as TOOL_APPROVAL_ALL
from exmekh_agent.config.approvals import TOOL_APPROVAL_LLM as TOOL_APPROVAL_LLM
from exmekh_agent.config.approvals import TOOL_APPROVAL_MANUAL as TOOL_APPROVAL_MANUAL
from exmekh_agent.config.approvals import TOOL_APPROVAL_MODES as TOOL_APPROVAL_MODES
from exmekh_agent.config.approvals import (
    normalize_tool_approval_mode as normalize_tool_approval_mode,
)
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.paths import BUNDLED_CAPABILITIES_DIR
from exmekh_agent.exmekh_bridge.resources import load_prompt

logger = logging.getLogger(__name__)


type CommandApprovedCallback = Callable[[str, list[dict[str, str]]], None]


COMMAND_APPROVAL_SYSTEM_PROMPT = load_prompt("command_approval")

COMMAND_APPROVAL_HDS_PATH = BUNDLED_CAPABILITIES_DIR / "hds" / "command-approval.md"


def load_command_approval_hds() -> HDS:
    """Load and return the command approval HDS capability."""
    if COMMAND_APPROVAL_HDS_PATH.exists():
        parsed = HDSMarkdownParser().parse(
            COMMAND_APPROVAL_HDS_PATH.read_text(encoding="utf-8")
        )
        return HDS(
            name="command-approval",
            markdown_path=COMMAND_APPROVAL_HDS_PATH,
            metadata=parsed.metadata,
            body=parsed.body,
            references=parsed.references,
        )
    return HDS(
        name="command-approval",
        markdown_path=Path("command-approval.md"),
        metadata=HDSMetadata(
            description=app_text("approvals.command_approval.evaluator_description"),
            sub_skills=(),
            sub_tools=(),
            exit_tool="exit_agent",
            agent_context_id="command-approval",
        ),
        body=COMMAND_APPROVAL_SYSTEM_PROMPT,
        references=(),
    )


class CommandApprovalDecision(BaseModel):
    """Structured decision returned by the command approval LLM."""

    approved: bool = Field(
        ...,
        description=app_text("approvals.command_approval.approved_description"),
    )
    reason: str = Field(
        default="",
        description=app_text("approvals.command_approval.reason_description"),
    )


class ApprovalModeChanged(Exception):
    """A pending manual request must be evaluated again under the current policy."""


def create_initial_approval_history(system_prompt: str | None = None) -> MessageHistory:
    """Create a fresh MessageHistory starting with the SystemMessage from HDS."""
    prompt = system_prompt or load_command_approval_hds().get_hds()
    return MessageHistory([SystemMessage(prompt)])


def build_approval_message_history(
    history: MessageHistory | Sequence[dict[str, Any]] | Sequence[str] | None = None,
    system_prompt: str | None = None,
) -> MessageHistory:
    """Construct or restore a MessageHistory instance from existing data or command strings."""
    if isinstance(history, MessageHistory):
        return history

    base_history = create_initial_approval_history(system_prompt)
    if not history:
        return base_history

    first_item = history[0]
    if isinstance(first_item, dict):
        messages: list[Message[Any]] = []
        for item in history:
            role = str(item.get("role", "")).lower()
            content = str(item.get("content", ""))
            if role == "system":
                if not messages:
                    messages.append(SystemMessage(content))
            elif role == "user":
                messages.append(UserMessage(content))
            elif role == "assistant":
                try:
                    decision = CommandApprovalDecision.model_validate_json(content)
                    messages.append(StructuredAssistantMessage(decision))
                except ValidationError:
                    messages.append(AssistantMessage(content))
        if not messages or messages[0].get_role_name() != "system":
            messages.insert(
                0,
                SystemMessage(system_prompt or load_command_approval_hds().get_hds()),
            )
        try:
            restored = MessageHistory(messages)
            restored.check_history()
            return restored
        except (TypeError, ValueError) as error:
            logger.warning(
                app_text(
                    "errors_tools.tool_approval.could_not_restore_command_approval_history_s"
                ),
                error,
            )
            return base_history

    for cmd in history:
        cmd_str = str(cmd)
        base_history.add(
            [
                UserMessage(
                    app_text(
                        "approvals.tool_approval.evaluate_proposed_command_bash_cmd_str",
                        cmd_str=cmd_str,
                    )
                ),
                StructuredAssistantMessage(
                    CommandApprovalDecision(
                        approved=True,
                        reason=app_text(
                            "approvals.tool_approval.command_approved_in_session"
                        ),
                    )
                ),
            ]
        )
    return base_history


class CommandApprovalEvaluator:
    """Evaluates candidate shell commands using an HDS system prompt, a ExMekh-Lib Client, and MessageHistory."""

    def __init__(
        self,
        history: MessageHistory
        | Sequence[dict[str, Any]]
        | Sequence[str]
        | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        hds: HDS | None = None,
    ) -> None:
        self.hds = hds or load_command_approval_hds()
        self.system_message = SystemMessage(self.hds.get_hds())
        self._history = build_approval_message_history(
            history, system_prompt=self.hds.get_hds()
        )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type

    def dump_history(self) -> list[dict[str, str]]:
        """Return the approval conversation through the history's public API."""
        return self._history.dump_history()

    def approved_commands(self) -> list[str]:
        """Read command strings without exposing history storage to the hook."""
        commands: list[str] = []
        for message in self._history.get_history():
            if message.get_role_name() != "user":
                continue
            content = str(message.get_message())
            if "```bash\n" in content:
                content = content.split("```bash\n", 1)[1].split("\n```", 1)[0]
            commands.append(content)
        return commands

    def record_approved_command(
        self,
        command: str,
        reason: str = app_text("approvals.tool_approval.approved_in_session"),
    ) -> None:
        """Add an approved command to the MessageHistory so future evaluations retain context."""
        if command in self.approved_commands():
            return
        self._history.add(
            [
                UserMessage(
                    app_text(
                        "approvals.tool_approval.evaluate_proposed_command_bash_command",
                        command=command,
                    )
                ),
                StructuredAssistantMessage(
                    CommandApprovalDecision(approved=True, reason=reason)
                ),
            ]
        )

    def evaluate(
        self,
        command: str,
        context: ToolContext | None = None,
    ) -> CommandApprovalDecision:
        """Evaluate a command using Client.call_structured with multi-turn MessageHistory."""
        effective_model = self.model or config.default_model
        effective_api = self.api_type or config.api_type
        effective_base_url = self.base_url or config.base_url
        effective_api_key = self.api_key or config.api_key

        try:
            provider = LocalProvider(
                model=effective_model,
                api="responses" if effective_api == "responses" else "chat_completions",
                base_url=effective_base_url,
                api_key=effective_api_key,
            )

            client = Client(
                provider=provider,
                system_message=self.system_message,
            )

            client.replace_history_from(0, self._history.get_history())

            prompt_parts = [
                app_text(
                    "approvals.tool_approval.please_evaluate_whether_this_bash_command_should_be_approved_for_execu",
                    command=command,
                )
            ]
            if context and context.work_root:
                prompt_parts.append(
                    app_text(
                        "approvals.tool_approval.working_directory",
                        path=context.work_root,
                    )
                )

            request = ClientRequest(
                message=UserMessage("\n".join(prompt_parts)),
                extra={
                    "temperature": 0.0,
                    **(
                        {"max_output_tokens": 256, "reasoning": {"effort": "none"}}
                        if effective_api == "responses"
                        else {"max_tokens": 256, "reasoning_effort": "none"}
                    ),
                },
            )

            response = client.call_structured(
                request, CommandApprovalDecision, strict=True
            )
            # Evaluation can become obsolete while a user changes approval mode.
            # Only record_approved_command commits a decision accepted by the hook.
            return response.get_message()
        except Exception as error:  # noqa: BLE001 - fallback on LLM evaluation failure
            logger.warning(
                app_text(
                    "errors_tools.tool_approval.error_during_llm_command_approval_evaluation_s"
                ),
                error,
            )
            return CommandApprovalDecision(
                approved=False,
                reason=app_text(
                    "approvals.tool_approval.error_during_llm_approval_evaluation_error",
                    error=error,
                ),
            )


def evaluate_command_approval_llm(
    command: str,
    history: MessageHistory | Sequence[dict[str, Any]] | Sequence[str] = (),
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    context: ToolContext | None = None,
) -> CommandApprovalDecision:
    """Evaluate a candidate bash command using CommandApprovalEvaluator."""
    evaluator = CommandApprovalEvaluator(
        history=history,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_type=api_type,
    )
    return evaluator.evaluate(command=command, context=context)


class CommandApprovalHook(BeforeTool):
    """BeforeTool hook that intercepts bash executions and enforces approval rules."""

    def __init__(
        self,
        mode: str = DEFAULT_TOOL_APPROVAL_MODE,
        history: MessageHistory
        | Sequence[dict[str, Any]]
        | Sequence[str]
        | None = None,
        request_manual_approval: Callable[[str], bool] | None = None,
        on_command_approved: CommandApprovedCallback | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        evaluator: CommandApprovalEvaluator | None = None,
        get_mode: Callable[[], str] | None = None,
    ) -> None:
        self.mode = normalize_tool_approval_mode(mode)
        self.get_mode = get_mode
        self.evaluator = evaluator or CommandApprovalEvaluator(
            history=history,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_type=api_type,
        )
        self.request_manual_approval = request_manual_approval
        self.on_command_approved = on_command_approved

    @property
    def history(self) -> list[str]:
        """Return approved commands from the evaluator's public interface."""
        return self.evaluator.approved_commands()

    def handle(self, context: ToolContext) -> ToolArguments | ToolResponse | None:
        command = context.arguments.get("command")
        if command is None:
            return None

        command_str = str(command)

        while True:
            mode = self._current_mode()
            try:
                decision = self._evaluate_command(command_str, context, mode)
            except ApprovalModeChanged:
                continue
            if self._current_mode() != mode:
                continue
            if not decision.approved:
                return ToolResponse(result=decision.reason)
            self._record_approval(command_str, reason=decision.reason)
            return None

    def _evaluate_command(
        self, command: str, context: ToolContext, mode: str
    ) -> CommandApprovalDecision:
        if mode == TOOL_APPROVAL_ALL:
            return CommandApprovalDecision(
                approved=True,
                reason=app_text(
                    "approvals.tool_approval.approved_automatically_approve_all"
                ),
            )
        if mode == TOOL_APPROVAL_LLM:
            decision = self.evaluator.evaluate(command=command, context=context)
            if decision.approved:
                return decision
            reason = decision.reason or app_text(
                "approvals.tool_approval.command_was_not_approved_by_the_evaluation_model"
            )
            return CommandApprovalDecision(
                approved=False,
                reason=app_text(
                    "approvals.tool_approval.command_declined_by_evaluation_model_reason",
                    reason=reason,
                ),
            )
        if self.request_manual_approval is None:
            return CommandApprovalDecision(
                approved=False,
                reason=app_text(
                    "approvals.tool_approval.approval_request_was_not_handled"
                ),
            )
        approved = self.request_manual_approval(command)
        return CommandApprovalDecision(
            approved=approved,
            reason=app_text(
                "approvals.tool_approval.approved_manually_by_user"
                if approved
                else "approvals.tool_approval.command_declined_by_user"
            ),
        )

    def _current_mode(self) -> str:
        return normalize_tool_approval_mode(
            self.get_mode() if self.get_mode else self.mode
        )

    def _record_approval(
        self,
        command: str,
        reason: str = app_text("approvals.tool_approval.approved_in_session"),
    ) -> None:
        self.evaluator.record_approved_command(command, reason=reason)
        if self.on_command_approved:
            try:
                self.on_command_approved(command, self.evaluator.dump_history())
            except Exception as error:  # noqa: BLE001 - optional persistence observer
                logger.warning(
                    app_text(
                        "errors_tools.tool_approval.error_in_on_command_approved_callback_s"
                    ),
                    error,
                )
