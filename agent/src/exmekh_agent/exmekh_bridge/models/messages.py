"""Message content, guidance and child-agent conversation lifecycle."""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Any, Literal

from exmekh_lib.core.models.assistant_message import ToolCall
from pydantic import BaseModel, Field

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.recorded_history import (
    RecordedHistory,
    extract_turn_raw_history,
)
from exmekh_agent.exmekh_bridge.history.tool_calls import (
    parse_tool_calls,
)
from exmekh_agent.exmekh_bridge.models.execution import (
    ProviderContextUsage,
    ReasoningBlock,
    SkillNode,
    TerminalStatus,
    ToolEvent,
    ordered_execution_activity,
    record_tool_event,
)
from exmekh_agent.exmekh_bridge.models.timeline import (
    ExecutionTimeline,
    SteeringInsertion,
)

MessageRole = Literal["user", "assistant", "system"]


class QueuedMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    image_urls: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    attachment_paths: list[str] | None = None
    display_content: str | None = None
    created_at: float = Field(default_factory=time.time)

    @property
    def visible_text(self) -> str:
        return self.display_content if self.display_content is not None else self.text

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(
            id=self.id,
            role="user",
            content=self.text,
            image_urls=list(self.image_urls),
            image_paths=list(self.image_paths),
            attachment_paths=(
                list(self.attachment_paths)
                if self.attachment_paths is not None
                else None
            ),
            display_content=self.display_content,
            timestamp=self.created_at,
        )


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str = ""
    image_urls: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    attachment_paths: list[str] | None = None
    display_content: str | None = None
    reasoning_content: str = ""
    reasoning_blocks: list[ReasoningBlock] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    skills: list[SkillNode] = Field(default_factory=list)
    raw_history: list[dict[str, Any]] = Field(default_factory=list)
    raw_history_start: int | None = Field(default=None, ge=0)
    steering_messages: list[QueuedMessage] = Field(default_factory=list)
    # None identifies older chats whose positions still need history reconstruction.
    steering_insertions: list[SteeringInsertion] | None = None
    skill_history_tree: list[dict[str, Any]] = Field(default_factory=list)
    context_usage: ProviderContextUsage | None = None
    timestamp: float = Field(default_factory=time.time)
    duration_seconds: float | None = None
    is_streaming: bool = False
    error: str | None = None

    @property
    def visible_text(self) -> str:
        return (
            self.display_content if self.display_content is not None else self.content
        )

    def find_user_message(self, message_id: str) -> ChatMessage | None:
        """Return an independent user input, including guidance owned by this turn."""
        if self.role == "user" and self.id == message_id:
            return self.model_copy(deep=True)
        guidance = next(
            (item for item in self.steering_messages if item.id == message_id), None
        )
        return guidance.to_chat_message() if guidance is not None else None

    def system_prompt_message(self) -> dict[str, Any] | None:
        """Expose the retained prompt through the transcript's public API."""
        return RecordedHistory(
            self.raw_history, self.skill_history_tree
        ).system_prompt_message()

    def has_same_execution_details(self, other: ChatMessage) -> bool:
        """Compare the execution panels independently of root text and reasoning."""
        return (
            self.raw_history == other.raw_history
            and self.skill_history_tree == other.skill_history_tree
            and self.skills == other.skills
            and self.tool_events == other.tool_events
            and self.steering_insertions == other.steering_insertions
        )

    def arguments_for_tool(self, event: ToolEvent) -> dict[str, Any]:
        """Resolve arguments through the recorded invocation's public API."""
        if event.arguments:
            return event.arguments
        return RecordedHistory(
            self.raw_history, self.skill_history_tree
        ).arguments_for_call(event.tool_call_id)

    def output_for_call(self, call_id: str | None) -> str | None:
        """Resolve one invocation without exposing how its transcripts are stored."""
        return RecordedHistory(
            self.raw_history, self.skill_history_tree
        ).output_for_call(call_id)

    def find_steering_message(
        self,
        content: str | None = None,
        *,
        message_id: str | None = None,
        excluded_ids: set[str] | None = None,
    ) -> QueuedMessage | None:
        """Match history by identity, falling back to FIFO text for legacy entries."""
        excluded_ids = excluded_ids or set()
        candidates = [
            message
            for message in self.steering_messages
            if message.id not in excluded_ids
        ]
        if message_id:
            for message in candidates:
                if message.id == message_id:
                    return message
        return next(
            (message for message in candidates if message.text == content), None
        )

    def confirmed_steering_insertions(self) -> list[SteeringInsertion]:
        """Use explicit confirmations, reconstructing positions only for older chats."""
        if self.steering_insertions is not None:
            return [item.model_copy(deep=True) for item in self.steering_insertions]

        pending = iter(self.steering_messages)
        current = next(pending, None)
        if current is None:
            return []

        tools_by_id = {
            event.tool_call_id: event
            for event in self.tool_events
            if event.tool_call_id
        }
        skill_paths = {
            node.opening_tool_call_id: node.path
            for skill in self.skills
            for node in skill.iter_nodes()
            if node.opening_tool_call_id
        }
        insertions: list[SteeringInsertion] = []
        preceding_tool: tuple[str | None, str | None] | None = None
        pending_calls: list[ToolCall] = []
        for item in self.turn_raw_history():
            role = str(item.get("role", "")).lower()
            if role in ("tool", "function"):
                call_id = str(item.get("tool_call_id") or "")
                call = next(
                    (
                        candidate
                        for candidate in pending_calls
                        if candidate.get_id() == call_id
                    ),
                    None,
                )
                if call is None and not call_id and pending_calls:
                    call = pending_calls[0]
                if call is not None:
                    pending_calls.remove(call)
                preceding_tool = (
                    call_id or (call.get_id() if call else None),
                    str(item.get("name") or "") or (call.get_name() if call else None),
                )
            elif role == "user":
                if current is not None and item.get("content") == current.text:
                    call_id = preceding_tool[0] if preceding_tool else None
                    event = tools_by_id.get(call_id)
                    insertions.append(
                        SteeringInsertion(
                            message_id=current.id,
                            after_tool_call_id=call_id,
                            tool_name=(preceding_tool[1] if preceding_tool else None)
                            or (event.tool_name if event else None),
                            skill_path=skill_paths.get(
                                call_id, event.skill_path if event else "/"
                            ),
                        )
                    )
                    current = next(pending, None)
            elif role == "assistant":
                pending_calls = parse_tool_calls(item.get("tool_calls")) or []
                preceding_tool = None
            else:
                preceding_tool = None
        return insertions

    def export_data(self, *, include_history: bool) -> dict[str, Any]:
        """Select this message's root or complete execution history."""
        if include_history:
            return self.model_dump(mode="json")
        data = self.model_dump(
            mode="json",
            exclude={"skill_history_tree", "skills", "tool_events", "reasoning_blocks"},
        )
        data["skills"] = [
            skill.export_data(include_history=False) for skill in self.skills
        ]
        data["tool_events"] = [
            tool.model_dump(mode="json")
            for tool in self.tool_events
            if tool.skill_path == "/"
        ]
        root_reasoning = [block for block in self.reasoning_blocks if block.depth == 0]
        data["reasoning_blocks"] = [
            block.model_dump(mode="json") for block in root_reasoning
        ]
        if self.reasoning_blocks:
            data["reasoning_content"] = "\n\n".join(
                block.content for block in root_reasoning
            )
        return data

    def ordered_activity(self) -> list[SkillNode | ToolEvent]:
        """Expose root invocations through the same timeline used for child skills."""
        return ordered_execution_activity(self.skills, self.tool_events, scope_path="/")

    def execution_timeline(self) -> ExecutionTimeline:
        """Expose execution order without requiring views to correlate confirmations."""
        return ExecutionTimeline(
            self.ordered_activity(), self.confirmed_steering_insertions()
        )

    def turn_raw_history(
        self,
        *,
        previous_history: list[dict[str, Any]] | None = None,
        fallback_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Keep the entire turn, including guidance received after tool results."""
        return extract_turn_raw_history(
            self.raw_history or fallback_history or [],
            start=self.raw_history_start,
            previous_history=previous_history,
        )

    def append_token(self, token: str) -> None:
        """End the active reasoning phase before adding visible answer content."""
        self.close_reasoning()
        self.content += token

    def append_reasoning(
        self, token: str, skill_path: str = "/", depth: int = 0
    ) -> None:
        """Own reasoning segmentation instead of exposing block storage to callers."""
        self.reasoning_content += token
        if (
            self.reasoning_blocks
            and self.reasoning_blocks[-1].end_time is None
            and self.reasoning_blocks[-1].skill_path == skill_path
            and self.reasoning_blocks[-1].depth == depth
        ):
            self.reasoning_blocks[-1].content += token
            return
        self.close_reasoning()
        self.reasoning_blocks.append(
            ReasoningBlock(
                skill_path=skill_path,
                depth=depth,
                generation_index=len(self.reasoning_blocks) + 1,
                content=token,
            )
        )

    def add_tool_event(self, event: ToolEvent) -> None:
        self.close_reasoning()
        record_tool_event(self.tool_events, event)

    def close_reasoning(self, *, now: float | None = None) -> None:
        for block in self.reasoning_blocks:
            block.finalize_timing(now=now)

    def finish(self, status: TerminalStatus, *, now: float | None = None) -> None:
        """Close the streamed message and all pending nested executions."""
        current = now if now is not None else time.time()
        self.finalize_execution(status, now=current)
        self.duration_seconds = max(0.0, current - self.timestamp)
        self.is_streaming = False

    def finalize_execution(
        self, status: TerminalStatus, *, now: float | None = None
    ) -> None:
        """Finalize owned reasoning, tools and skills through their lifecycle APIs."""
        current = now if now is not None else time.time()
        self.close_reasoning(now=current)
        for event in self.tool_events:
            event.finalize_timing(status, now=current)
        for skill in self.skills:
            skill.finalize_timing(status, now=current)


class SubAgentConversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    path: str
    order: int = 1
    persistent_history: bool = True
    status: Literal["active", "completed", "error"] = "active"
    messages: list[ChatMessage] = Field(default_factory=list)
    raw_history: list[dict[str, Any]] = Field(default_factory=list)

    def message_history(self) -> list[ChatMessage]:
        return list(self.messages)

    def replay_history(self) -> list[dict[str, Any]] | list[ChatMessage]:
        """Replay the retained ExMekh-Lib snapshot, with a safe fallback for legacy calls."""
        return (
            deepcopy(self.raw_history) if self.raw_history else deepcopy(self.messages)
        )

    def discard_calls_from(self, timestamp: float) -> bool:
        """Retain completed calls that began before a revised parent turn."""
        self.messages = [
            message for message in self.messages if message.timestamp < timestamp
        ]
        if not self.messages:
            self.raw_history = []
            return False
        last = self.messages[-1]
        self.raw_history = deepcopy(last.raw_history)
        self.status = "error" if last.error else "completed"
        return True

    def start_call(self, user_message: str) -> None:
        if self.messages and self.messages[-1].is_streaming:
            raise ValueError(
                app_text(
                    "errors_exmekh_bridge.models.sub_agent_conversation_is_already_active",
                    value_1=self.id,
                )
            )
        self.status = "active"
        self.messages.append(ChatMessage(role="user", content=user_message))
        self.messages.append(ChatMessage(role="assistant", is_streaming=True))

    def append_token(self, token: str) -> None:
        self._active_assistant().append_token(token)

    def append_reasoning(self, token: str, skill_path: str, depth: int) -> None:
        self._active_assistant().append_reasoning(token, skill_path, depth)

    def add_tool_event(self, event: ToolEvent) -> None:
        self._active_assistant().add_tool_event(event)

    def complete_call(self, response: str, raw_history: list[dict[str, Any]]) -> None:
        message = self._active_assistant()
        message.content = response or message.content
        message.finish("completed")
        self.raw_history = deepcopy(raw_history)
        message.raw_history = deepcopy(raw_history)
        self.status = "completed"

    def fail_call(self) -> None:
        message = self._active_assistant()
        message.error = app_text("execution.models.sub_agent_call_failed")
        message.finish("error")
        self.status = "error"

    def interrupt_call(self) -> None:
        """Fail an unfinished call; repeated cleanup leaves terminal calls unchanged."""
        if self.messages and self.messages[-1].is_streaming:
            self.fail_call()
        elif self.status == "active":
            self.status = "error"

    def _active_assistant(self) -> ChatMessage:
        if not self.messages or not self.messages[-1].is_streaming:
            raise ValueError(
                app_text(
                    "errors_exmekh_bridge.models.sub_agent_conversation_has_no_active_call",
                    value_1=self.id,
                )
            )
        return self.messages[-1]


class ChatTitle(BaseModel):
    """Structured output schema for generating a concise chat session title."""

    title: str = Field(
        description=app_text("execution.agent_runner.title_schema_description")
    )
