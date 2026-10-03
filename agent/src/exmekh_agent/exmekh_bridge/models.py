from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from exmekh_lib.core.models.assistant_message import ToolCall
from pydantic import BaseModel, Field

from exmekh_agent.exmekh_bridge.tool_approval import normalize_tool_approval_mode
from exmekh_agent.exmekh_bridge.tool_calls import (
    extract_tool_arguments,
    parse_tool_calls,
)

MessageRole = Literal["user", "assistant", "system"]
TerminalStatus = Literal["completed", "error"]


def format_duration(seconds: float | int | None) -> str:
    """Format a duration in seconds into 'Xs', 'Ym Xs', or 'Zh Ym Xs'.

    Examples:
        0 -> '0s'
        30 -> '30s'
        59 -> '59s'
        70 -> '1m 10s'
        7460 -> '2h 4m 20s'
    """
    if seconds is None:
        return "0s"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ToolEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_call_id: str | None = None
    event_type: str = "execution"  # tool_start, tool_end, execution
    status: Literal["running", "completed", "error"] = "completed"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depth: int = 0
    skill_path: str = "/"
    timestamp: float = Field(default_factory=time.time)
    end_time: float | None = None
    duration_seconds: float | None = None

    @property
    def is_running(self) -> bool:
        return self.execution_status == "running"

    @property
    def execution_status(self) -> Literal["running", "completed", "error"]:
        """Interpret terminal events and legacy starts consistently for all views."""
        if self.event_type == "tool_end":
            return "error" if self.status == "error" else "completed"
        if self.status == "running" or (
            self.event_type == "tool_start"
            and self.status == "completed"
            and self.end_time is None
            and self.duration_seconds is None
        ):
            return "running"
        return self.status

    def matches_call(self, incoming: ToolEvent) -> bool:
        """Use invocation IDs when available; otherwise match the execution scope."""
        if incoming.tool_call_id is not None:
            return self.tool_call_id == incoming.tool_call_id
        return (
            self.tool_name == incoming.tool_name
            and self.skill_path == incoming.skill_path
            and self.depth == incoming.depth
        )

    def complete_from(self, incoming: ToolEvent) -> None:
        """Apply a terminal update while retaining this invocation's opening time."""
        self.event_type = "tool_end"
        self.status = "error" if incoming.status == "error" else "completed"
        self.end_time = (
            incoming.end_time if incoming.end_time is not None else incoming.timestamp
        )
        self.duration_seconds = (
            incoming.duration_seconds
            if incoming.duration_seconds is not None
            else max(0.0, self.end_time - self.timestamp)
        )
        if not self.arguments:
            self.arguments = deepcopy(incoming.arguments)

    def matches_update(self, incoming: ToolEvent) -> bool:
        """Correlate callbacks, including replayed legacy results without call IDs."""
        if self.id == incoming.id:
            return True
        if not self.matches_call(incoming):
            return False
        if incoming.tool_call_id is not None or self.is_running:
            return True
        result_time = (
            incoming.end_time if incoming.end_time is not None else incoming.timestamp
        )
        return not incoming.is_running and self.end_time == result_time

    def merge_update(self, incoming: ToolEvent) -> None:
        """Keep the first terminal result while accepting missing arguments later."""
        if self.is_running and not incoming.is_running:
            self.complete_from(incoming)
        elif not self.arguments:
            self.arguments = deepcopy(incoming.arguments)

    def normalized_copy(self) -> ToolEvent:
        """Normalize an incoming event without mutating the publisher's instance."""
        event = self.model_copy(deep=True)
        event.status = self.execution_status
        if event.is_running:
            event.event_type = "tool_start"
        else:
            if event.event_type == "tool_start":
                event.event_type = "tool_end"
            if event.event_type == "tool_end":
                if event.duration_seconds is None:
                    event.duration_seconds = 0.0
            if event.end_time is None:
                event.end_time = event.timestamp
            event.finalize_timing()
        return event

    def finalize_timing(
        self, status: TerminalStatus | None = None, *, now: float | None = None
    ) -> None:
        """Close a running tool when requested, preserving recorded terminal results."""
        if self.is_running and status is not None:
            self.status = status
            self.event_type = "tool_end"
            self.end_time = now if now is not None else time.time()
        if self.duration_seconds is None:
            if self.end_time is not None:
                self.duration_seconds = max(0.0, self.end_time - self.timestamp)
            elif not self.is_running:
                self.duration_seconds = 0.0

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Return the tool execution duration or live elapsed time in seconds."""
        if self.duration_seconds is not None:
            return max(0.0, self.duration_seconds)
        if self.end_time is not None:
            return max(0.0, self.end_time - self.timestamp)
        if self.is_running:
            current = now if now is not None else time.time()
            return max(0.0, current - self.timestamp)
        return 0.0


def record_tool_event(events: list[ToolEvent], incoming: ToolEvent) -> ToolEvent:
    """Keep one independent entry per invocation, preserving its first result."""
    existing = next(
        (event for event in reversed(events) if event.matches_update(incoming)),
        None,
    )
    if existing is not None:
        existing.merge_update(incoming)
        return existing
    recorded = incoming.normalized_copy()
    events.append(recorded)
    return recorded


def extract_turn_raw_history(
    raw_history: list[dict[str, Any]],
    *,
    start: int | None = None,
    previous_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Isolate a turn using its explicit boundary or a legacy cumulative prefix."""
    if start is None:
        if (
            previous_history
            and raw_history[: len(previous_history)] == previous_history
        ):
            start = len(previous_history)
        else:
            start = 1 + max(
                (
                    index
                    for index, item in enumerate(raw_history)
                    if str(item.get("role", "")).lower() == "user"
                ),
                default=-1,
            )
    return [
        item
        for item in raw_history[start:]
        if str(item.get("role", "")).lower() != "system"
    ]


class ReasoningBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    skill_path: str = "/"
    depth: int = 0
    generation_index: int = 1
    content: str = ""
    timestamp: float = Field(default_factory=time.time)
    end_time: float | None = None
    duration_seconds: float | None = None

    def finalize_timing(self, *, now: float | None = None) -> None:
        """Close this reasoning block without replacing previously recorded timing."""
        if self.end_time is None:
            self.end_time = now if now is not None else time.time()
        if self.duration_seconds is None:
            self.duration_seconds = max(0.0, self.end_time - self.timestamp)

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Return the thinking block duration or live elapsed time in seconds."""
        if self.duration_seconds is not None:
            return max(0.0, self.duration_seconds)
        if self.end_time is not None:
            return max(0.0, self.end_time - self.timestamp)
        current = now if now is not None else time.time()
        return max(0.0, current - self.timestamp)


class SkillNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "root"
    path: str = "/"
    depth: int = 0
    status: Literal["active", "completed", "error"] = "active"
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    duration_seconds: float | None = None
    reasoning_blocks: list[ReasoningBlock] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    sub_skills: list[SkillNode] = Field(default_factory=list)
    is_collapsed: bool = False
    tokens_before_close: int | None = None
    tokens_after_close: int | None = None
    opening_tool_call_id: str | None = None
    tokens_pending: bool = False

    def add_tool_event(self, event: ToolEvent) -> None:
        """Apply the same invocation lifecycle used by chat messages."""
        for block in self.reasoning_blocks:
            block.finalize_timing()
        record_tool_event(self.tool_events, event)

    def ordered_activity(self) -> list[SkillNode | ToolEvent]:
        """Expose child invocations and tools once, in their opening order."""
        return _ordered_activity(self.sub_skills, self.tool_events)

    def export_data(self, *, include_history: bool) -> dict[str, Any]:
        """Retain invocation metadata when nested execution details are omitted."""
        if include_history:
            return self.model_dump(mode="json")
        data = self.model_dump(
            mode="json", exclude={"reasoning_blocks", "tool_events", "sub_skills"}
        )
        data["sub_skills"] = [
            child.export_data(include_history=False) for child in self.sub_skills
        ]
        return data

    @property
    def has_closure_tokens(self) -> bool:
        return (
            self.tokens_before_close is not None and self.tokens_after_close is not None
        )

    def record_closure_tokens(self, before: int, after: int) -> None:
        """Freeze this invocation's closure totals, preserving the first measurement."""
        if not self.has_closure_tokens:
            self.tokens_before_close = before
            self.tokens_after_close = after
        self.tokens_pending = False

    def await_closure_tokens(self) -> None:
        if not self.has_closure_tokens:
            self.tokens_pending = True

    def matches_invocation(self, path: str, call_id: str | None = None) -> bool:
        return self.path == path and (
            call_id is None or self.opening_tool_call_id == call_id
        )

    def find_invocation(
        self, path: str, call_id: str | None = None
    ) -> SkillNode | None:
        """Find this invocation or its most recent matching descendant."""
        if self.matches_invocation(path, call_id):
            return self
        for child in reversed(self.sub_skills):
            found = child.find_invocation(path, call_id)
            if found is not None:
                return found
        return None

    def attach_descendant(self, node: SkillNode) -> bool:
        """Attach a new invocation to its deepest known parent in this tree."""
        if not node.path.startswith(self.path.rstrip("/") + "/"):
            return False
        for child in reversed(self.sub_skills):
            if child.attach_descendant(node):
                return True
        self.sub_skills.append(node)
        return True

    def iter_nodes(self) -> Iterator[SkillNode]:
        """Visit this skill and its descendants without exposing traversal details."""
        yield self
        for child in self.sub_skills:
            yield from child.iter_nodes()

    @property
    def action_count(self) -> int:
        """Count actions across the whole invocation, including nested skills."""
        return sum(len(node.tool_events) for node in self.iter_nodes())

    def merge_snapshot(self, snapshot: SkillNode) -> None:
        """Apply a stream update while retaining opening time and frozen totals."""
        self.start_time = min(self.start_time, snapshot.start_time)
        if self.status == "active":
            self.status = snapshot.status
            self.end_time = snapshot.end_time
            self.duration_seconds = snapshot.duration_seconds
            self.is_collapsed = snapshot.is_collapsed
        if self.end_time is not None:
            self.duration_seconds = max(0.0, self.end_time - self.start_time)
        self.opening_tool_call_id = (
            snapshot.opening_tool_call_id or self.opening_tool_call_id
        )
        if snapshot.has_closure_tokens:
            self.record_closure_tokens(
                snapshot.tokens_before_close, snapshot.tokens_after_close
            )
        elif snapshot.tokens_pending:
            self.await_closure_tokens()
        if snapshot.reasoning_blocks:
            self.reasoning_blocks = [
                block.model_copy(deep=True) for block in snapshot.reasoning_blocks
            ]
        if snapshot.tool_events:
            for event in snapshot.tool_events:
                record_tool_event(self.tool_events, event)
        for child in snapshot.sub_skills:
            existing = next(
                (
                    node
                    for node in reversed(self.sub_skills)
                    if node.matches_invocation(child.path, child.opening_tool_call_id)
                ),
                None,
            )
            if existing is not None:
                existing.merge_snapshot(child)
            else:
                self.sub_skills.append(child.model_copy(deep=True))
        self.finalize_timing()

    def finalize_timing(
        self, status: TerminalStatus | None = None, *, now: float | None = None
    ) -> None:
        """Finalize this skill and its descendants, preserving completed/error states."""
        current = now if now is not None else time.time()
        if self.status == "active" and status is not None:
            self.status = status
        terminal_status = None if self.status == "active" else self.status
        if terminal_status is not None:
            if self.end_time is None:
                self.end_time = current
            if self.duration_seconds is None:
                self.duration_seconds = max(0.0, self.end_time - self.start_time)
        closed_at = self.end_time if self.end_time is not None else current
        for block in self.reasoning_blocks:
            if terminal_status is not None or block.end_time is not None:
                block.finalize_timing(now=closed_at)
        for event in self.tool_events:
            event.finalize_timing(terminal_status, now=closed_at)
        for child in self.sub_skills:
            child.finalize_timing(terminal_status, now=closed_at)

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Return the skill open-to-close duration or live elapsed time in seconds."""
        if self.duration_seconds is not None:
            return max(0.0, self.duration_seconds)
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        if self.status == "active":
            current = now if now is not None else time.time()
            return max(0.0, current - self.start_time)
        return 0.0


def _ordered_activity(
    skills: list[SkillNode],
    tools: list[ToolEvent],
    *,
    scope_path: str | None = None,
) -> list[SkillNode | ToolEvent]:
    opening_ids = {skill.opening_tool_call_id for skill in skills}
    recorded_tools: list[ToolEvent] = []
    for event in tools:
        if skills and scope_path is not None and event.skill_path != scope_path:
            continue
        if event.tool_call_id is not None and event.tool_call_id in opening_ids:
            continue
        record_tool_event(recorded_tools, event)
    return sorted(
        [*skills, *recorded_tools],
        key=lambda activity: (
            activity.start_time
            if isinstance(activity, SkillNode)
            else activity.timestamp
        ),
    )


class ProviderContextUsage(BaseModel):
    """One model request's context, with authoritative totals when available."""

    prompt_tokens: int
    completion_tokens: int | None = None
    return_call_id: str | None = None
    streamed_content: str = ""
    streamed_reasoning: str = ""
    is_final: bool = False
    closure_tokens_ready: bool = False

    @property
    def is_estimated(self) -> bool:
        return self.completion_tokens is None

    @property
    def authoritative_total(self) -> int | None:
        if self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def total_tokens(self, count_tokens: Callable[[str], int]) -> int:
        """Use provider totals when available, otherwise count this request's deltas."""
        total = self.authoritative_total
        if total is not None:
            return total
        return (
            self.prompt_tokens
            + count_tokens(self.streamed_reasoning)
            + count_tokens(self.streamed_content)
        )

    def append_content(self, token: str) -> None:
        self.streamed_content += token

    def append_reasoning(self, token: str) -> None:
        self.streamed_reasoning += token

    def with_stream_from(self, previous: ProviderContextUsage) -> ProviderContextUsage:
        return self.model_copy(
            update={
                "streamed_content": previous.streamed_content,
                "streamed_reasoning": previous.streamed_reasoning,
            },
            deep=True,
        )

    def with_provider_totals(
        self, prompt: int, completion: int | None
    ) -> ProviderContextUsage:
        return self.model_copy(
            update={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "is_final": True,
            }
        )


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


class SteeringInsertion(BaseModel):
    """The execution boundary where ExMekh-Lib confirmed a queued message."""

    message_id: str
    after_tool_call_id: str | None
    tool_name: str | None
    skill_path: str


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

    def arguments_for_tool(self, event: ToolEvent) -> dict[str, Any]:
        """Resolve saved invocations, including legacy nested skill transcripts."""
        if event.arguments:
            return event.arguments
        arguments = extract_tool_arguments(self.raw_history, event.tool_call_id)
        if arguments:
            return arguments
        pending = list(self.skill_history_tree)
        for entry in pending:
            arguments = extract_tool_arguments(entry.get("history"), event.tool_call_id)
            if arguments:
                return arguments
            pending.extend(entry.get("children") or [])
        return {}

    def find_steering_message(
        self,
        content: str,
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
        return _ordered_activity(self.skills, self.tool_events, scope_path="/")

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
            raise ValueError(f"Sub-agent conversation {self.id} is already active")
        self.status = "active"
        self.messages.append(ChatMessage(role="user", content=user_message))
        self.messages.append(ChatMessage(role="assistant", is_streaming=True))

    def append_token(self, token: str) -> None:
        message = self._active_assistant()
        message.close_reasoning()
        message.content += token

    def append_reasoning(self, token: str, skill_path: str, depth: int) -> None:
        message = self._active_assistant()
        message.reasoning_content += token
        if (
            message.reasoning_blocks
            and message.reasoning_blocks[-1].end_time is None
            and message.reasoning_blocks[-1].skill_path == skill_path
            and message.reasoning_blocks[-1].depth == depth
        ):
            message.reasoning_blocks[-1].content += token
        else:
            message.close_reasoning()
            message.reasoning_blocks.append(
                ReasoningBlock(
                    skill_path=skill_path,
                    depth=depth,
                    generation_index=len(message.reasoning_blocks) + 1,
                    content=token,
                )
            )

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
        message.error = "Sub-agent call failed"
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
            raise ValueError(f"Sub-agent conversation {self.id} has no active call")
        return self.messages[-1]


class ProjectSettings(BaseModel):
    """Shared ExMekh-Lib options and the approval default for new project chats."""

    name_hds: str | None = None
    capabilities_path: str | None = None
    coding_agent_persistent_history: bool | None = None
    tool_approval_mode: str = "Manual"

    def migrate_legacy_capabilities_path(self) -> bool:
        """Use the installed library defaults for paths saved before its rename."""
        if self.capabilities_path is None:
            return False
        path = self.capabilities_path.replace("\\", "/").rstrip("/")
        if not path.endswith("/site-packages/qs/capabilities"):
            return False
        self.capabilities_path = None
        return True


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    work_root: str = Field(frozen=True)
    created_at: float = Field(default_factory=time.time)
    settings: ProjectSettings = Field(default_factory=ProjectSettings)

    @property
    def name(self) -> str:
        return Path(self.work_root).name or self.work_root


class ChatSettings(ProjectSettings):
    """Execution snapshot with conversation-local approvals and evaluator history."""

    work_root: str | None = None
    approved_commands_history: list[str] = Field(default_factory=list)
    tool_approval_raw_history: list[dict[str, Any]] = Field(default_factory=list)

    def with_project(self, project: Project) -> ChatSettings:
        """Resolve shared options without changing this chat's approval state."""
        return self.model_copy(
            deep=True,
            update={
                "name_hds": project.settings.name_hds,
                "capabilities_path": project.settings.capabilities_path,
                "coding_agent_persistent_history": project.settings.coding_agent_persistent_history,
                "work_root": project.work_root,
            },
        )

    def set_tool_approval_mode(self, mode: str) -> None:
        self.tool_approval_mode = normalize_tool_approval_mode(mode)

    def clear_command_approvals(self) -> None:
        """Discard evaluator history and cached decisions from an obsolete branch."""
        self.approved_commands_history = []
        self.tool_approval_raw_history = []

    def record_command_approval(
        self, command: str, raw_history: list[dict[str, str]]
    ) -> None:
        """Keep command summaries and evaluator context together without duplicates."""
        if command not in self.approved_commands_history:
            self.approved_commands_history.append(command)
        self.tool_approval_raw_history = [dict(message) for message in raw_history]


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Conversation"
    project_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: list[ChatMessage] = Field(default_factory=list)
    sub_agent_conversations: list[SubAgentConversation] = Field(default_factory=list)
    queued_messages: list[QueuedMessage] = Field(default_factory=list)
    model: str = "Qwen3.8-27B-UD-Q4_K_S"
    agent_mode: Literal["deep_agent", "quick_chat"] = "deep_agent"
    system_prompt: str = ""
    settings: ChatSettings = Field(default_factory=ChatSettings)

    def export_data(self, *, include_history: bool) -> dict[str, Any]:
        """Export conversation content without execution settings or pending input."""
        data = {
            "id": self.id,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "messages": [
                message.export_data(include_history=include_history)
                for message in self.messages
            ],
        }
        if include_history:
            data["sub_agent_conversations"] = [
                child.model_dump(mode="json") for child in self.sub_agent_conversations
            ]
        return data

    @property
    def is_empty(self) -> bool:
        return not (
            self.messages or self.queued_messages or self.sub_agent_conversations
        )

    @property
    def has_queued_messages(self) -> bool:
        return bool(self.queued_messages)

    def message_history(self) -> list[ChatMessage]:
        """Return the conversation without sharing its mutable message list."""
        return list(self.messages)

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def find_user_message(self, message_id: str) -> ChatMessage | None:
        for message in self.messages:
            found = message.find_user_message(message_id)
            if found is not None:
                return found
        return None

    def revised_user_message(
        self, message_id: str, replacement: ChatMessage
    ) -> ChatSession | None:
        """Build an independent conversation branch without modifying this session."""
        if replacement.role != "user":
            return None
        revised = self.model_copy(deep=True)
        boundary = revised._revision_boundary(message_id)
        if boundary is None:
            return None
        prefix, original, cutoff = boundary
        revised.messages = prefix + [
            replacement.model_copy(
                deep=True,
                update={
                    "id": original.id,
                    "timestamp": original.timestamp,
                },
            )
        ]
        revised.sub_agent_conversations = [
            child
            for child in revised.sub_agent_conversations
            if child.discard_calls_from(cutoff)
        ]
        revised.queued_messages = []
        revised.settings.clear_command_approvals()
        return revised

    def _revision_boundary(
        self, message_id: str
    ) -> tuple[list[ChatMessage], ChatMessage, float] | None:
        for index, message in enumerate(self.messages):
            if message.role == "user" and message.id == message_id:
                return self.messages[:index], message, message.timestamp
            earlier_guidance = []
            for guidance in message.steering_messages:
                if guidance.id == message_id:
                    # Guidance bubbles precede their owning response. Restart
                    # that response at its initiating input, including child calls.
                    cutoff = next(
                        (
                            item.timestamp
                            for item in reversed(self.messages[:index])
                            if item.role == "user"
                        ),
                        message.timestamp,
                    )
                    return (
                        self.messages[:index] + earlier_guidance,
                        guidance.to_chat_message(),
                        cutoff,
                    )
                earlier_guidance.append(guidance.to_chat_message())
        return None

    def apply_revision(self, revised: ChatSession) -> None:
        """Publish a persisted branch while retaining this managed instance."""
        if revised.id != self.id:
            raise ValueError("Revision belongs to another conversation")
        self.messages = revised.messages
        self.sub_agent_conversations = revised.sub_agent_conversations
        self.queued_messages = revised.queued_messages
        self.settings = revised.settings
        self.updated_at = revised.updated_at

    def find_sub_agent_conversation(
        self, conversation_id: str | None
    ) -> SubAgentConversation | None:
        return next(
            (
                conversation
                for conversation in self.sub_agent_conversations
                if conversation.id == conversation_id
            ),
            None,
        )

    def start_sub_agent_conversation(
        self,
        conversation_id: str,
        path: str,
        persistent_history: bool,
        user_message: str,
    ) -> SubAgentConversation:
        conversation = self.find_sub_agent_conversation(conversation_id)
        if conversation is None:
            order = 1 + sum(item.path == path for item in self.sub_agent_conversations)
            conversation = SubAgentConversation(
                id=conversation_id,
                path=path,
                order=order,
                persistent_history=persistent_history,
            )
            self.sub_agent_conversations.append(conversation)
        elif (
            conversation.path != path
            or conversation.persistent_history != persistent_history
        ):
            raise ValueError(
                f"Sub-agent conversation {conversation_id} changed identity"
            )
        conversation.start_call(user_message)
        return conversation

    def cleanup_active_sub_agents(self) -> None:
        """Close any unclosed sub-agent conversations after interruption or error."""
        for conversation in self.sub_agent_conversations:
            conversation.interrupt_call()

    def add_queued_message(
        self,
        text: str,
        display_content: str | None = None,
        *,
        image_urls: list[str] | None = None,
        image_paths: list[str] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> QueuedMessage:
        msg = QueuedMessage(
            text=text,
            display_content=display_content,
            image_urls=image_urls or [],
            image_paths=image_paths or [],
            attachment_paths=attachment_paths,
        )
        self.queued_messages.append(msg)
        return msg

    def pop_queued_message(self) -> QueuedMessage | None:
        if self.queued_messages:
            return self.queued_messages.pop(0)
        return None

    def find_queued_message(self, message_id: str) -> QueuedMessage | None:
        return next(
            (message for message in self.queued_messages if message.id == message_id),
            None,
        )

    def remove_queued_message(self, message_id: str) -> QueuedMessage | None:
        for i, m in enumerate(self.queued_messages):
            if m.id == message_id:
                return self.queued_messages.pop(i)
        return None

    def reorder_queued_message(
        self, source_id: str, target_id: str, position: str = "before"
    ) -> bool:
        if source_id == target_id or position not in ("before", "after"):
            return False
        src_idx = next(
            (i for i, m in enumerate(self.queued_messages) if m.id == source_id),
            None,
        )
        if src_idx is None:
            return False
        item = self.queued_messages.pop(src_idx)
        target_idx = next(
            (i for i, m in enumerate(self.queued_messages) if m.id == target_id),
            None,
        )
        if target_idx is None:
            self.queued_messages.insert(src_idx, item)
            return False
        insert_idx = target_idx if position == "before" else target_idx + 1
        self.queued_messages.insert(insert_idx, item)
        return True


class ChatTitle(BaseModel):
    """Structured output schema for generating a concise chat session title."""

    title: str = Field(
        description="A concise, clear, and descriptive title for the conversation in the exact same language as the user's message (maximum 4-6 words, no quotes)."
    )
