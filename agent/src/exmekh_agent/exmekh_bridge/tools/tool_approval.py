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

logger = logging.getLogger(__name__)

TOOL_APPROVAL_MANUAL = "Manual"
TOOL_APPROVAL_ALL = "Approve All"
TOOL_APPROVAL_LLM = "Approve via LLM"
TOOL_APPROVAL_MODES: list[str] = [
    TOOL_APPROVAL_MANUAL,
    TOOL_APPROVAL_ALL,
    TOOL_APPROVAL_LLM,
]
DEFAULT_TOOL_APPROVAL_MODE = TOOL_APPROVAL_MANUAL

type CommandApprovedCallback = Callable[[str, list[dict[str, str]]], None]

LEGACY_TOOL_APPROVAL_MODES: dict[str, str] = {
    "Manuale": TOOL_APPROVAL_MANUAL,
    "Approva tutto": TOOL_APPROVAL_ALL,
    "Approva da un LLM": TOOL_APPROVAL_LLM,
}

COMMAND_APPROVAL_SYSTEM_PROMPT = (
    "You are a command security and safety evaluator in an autonomous development system. "
    "Your sole purpose is to evaluate whether a proposed bash/shell command should be approved for execution or declined. "
    "Carefully analyze if the command is safe, constructive, and appropriate for the user's task. "
    "Decline commands that are destructive, harmful, malicious, or unauthorized. "
    "Maintain the history of approved commands across the session to maintain context. "
    "Respond with approved=True if the command is safe to execute, or approved=False and provide a concise explanation in reason. "
    "Write explanations in the user's conversation language."
)

COMMAND_APPROVAL_HDS_PATH = (
    Path(__file__).resolve().parents[2] / "capabilities" / "hds" / "command-approval.md"
)


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
            description="Autonomous evaluator for bash command execution safety.",
            sub_skills=(),
            sub_tools=(),
            exit_tool="exit_agent",
            agent_context_id="command-approval",
        ),
        body=COMMAND_APPROVAL_SYSTEM_PROMPT,
        references=(),
    )


def normalize_tool_approval_mode(mode: str | None) -> str:
    """Normalize legacy Italian mode names or missing values to English standard modes."""
    if not mode:
        return DEFAULT_TOOL_APPROVAL_MODE
    if mode in LEGACY_TOOL_APPROVAL_MODES:
        return LEGACY_TOOL_APPROVAL_MODES[mode]
    if mode in TOOL_APPROVAL_MODES:
        return mode
    return DEFAULT_TOOL_APPROVAL_MODE


class CommandApprovalDecision(BaseModel):
    """Structured decision returned by the command approval LLM."""

    approved: bool = Field(
        ...,
        description="Whether the requested shell command is approved to execute.",
    )
    reason: str = Field(
        default="",
        description="Explanation for why the command was approved or declined.",
    )


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
            logger.warning("Could not restore command approval history: %s", error)
            return base_history

    for cmd in history:
        cmd_str = str(cmd)
        base_history.add(
            [
                UserMessage(f"Evaluate proposed command:\n```bash\n{cmd_str}\n```"),
                StructuredAssistantMessage(
                    CommandApprovalDecision(
                        approved=True, reason="Command approved in session"
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
        self, command: str, reason: str = "Approved in session"
    ) -> None:
        """Add an approved command to the MessageHistory so future evaluations retain context."""
        if command in self.approved_commands():
            return
        self._history.add(
            [
                UserMessage(f"Evaluate proposed command:\n```bash\n{command}\n```"),
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
                f"Please evaluate whether this bash command should be approved for execution:\n\n```bash\n{command}\n```"
            ]
            if context and context.work_root:
                prompt_parts.append(f"\nWorking directory: `{context.work_root}`")

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
            decision = response.get_message()

            if decision.approved:
                self._history = client.create_history_checkpoint()

            return decision
        except Exception as error:  # noqa: BLE001 - fallback on LLM evaluation failure
            logger.warning("Error during LLM command approval evaluation: %s", error)
            return CommandApprovalDecision(
                approved=False,
                reason=f"Error during LLM approval evaluation: {error}",
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
    ) -> None:
        self.mode = normalize_tool_approval_mode(mode)
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

        if self.mode == TOOL_APPROVAL_ALL:
            self._record_approval(
                command_str, reason="Approved automatically (Approve All)"
            )
            return None

        if self.mode == TOOL_APPROVAL_LLM:
            decision = self.evaluator.evaluate(command=command_str, context=context)
            if decision.approved:
                self._record_approval(command_str, reason=decision.reason)
                return None
            reason = (
                decision.reason or "Command was not approved by the evaluation model."
            )
            return ToolResponse(
                result=f"Command declined by evaluation model: {reason}"
            )

        # Default mode: "Manual"
        if self.request_manual_approval:
            approved = self.request_manual_approval(command_str)
            if approved:
                self._record_approval(command_str, reason="Approved manually by user")
                return None
            return ToolResponse(result="Command declined by user.")

        return ToolResponse(result="Approval request was not handled.")

    def _record_approval(
        self,
        command: str,
        reason: str = "Approved in session",
    ) -> None:
        self.evaluator.record_approved_command(command, reason=reason)
        if self.on_command_approved:
            try:
                self.on_command_approved(command, self.evaluator.dump_history())
            except Exception as error:  # noqa: BLE001 - optional persistence observer
                logger.warning("Error in on_command_approved callback: %s", error)
