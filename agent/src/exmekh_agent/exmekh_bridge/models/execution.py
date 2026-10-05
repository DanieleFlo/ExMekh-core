"""Invocation lifecycle, timing, execution ordering and provider token usage."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

TerminalStatus = Literal["completed", "error"]


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
        return ordered_execution_activity(self.sub_skills, self.tool_events)

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


def ordered_execution_activity(
    skills: list[SkillNode],
    tools: list[ToolEvent],
    *,
    scope_path: str | None = None,
) -> list[SkillNode | ToolEvent]:
    """Expose tools and skill openings once in chronological order."""
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
    completion_estimate: int | None = None
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
        if self.is_final and self.completion_estimate is not None:
            return (
                self.prompt_tokens
                + self.completion_estimate
                + count_tokens(self.streamed_reasoning)
            )
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
                "streamed_reasoning": self.streamed_reasoning
                or previous.streamed_reasoning,
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
