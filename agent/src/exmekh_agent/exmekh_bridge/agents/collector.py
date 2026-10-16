"""Streaming event accumulator for agent conversation turns.

Encapsulates stateful accumulation of stream tokens, tool execution events,
hierarchical skill nodes, and multi-path reasoning block segmentation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from exmekh_lib.core.agents.deep_agent import DeepAgentCompactionProgressEvent

from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.history.compaction import ModelContextCheckpoint
from exmekh_agent.exmekh_bridge.history.skill_history import skill_opening_call_id
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    MessageRole,
    ProviderContextUsage,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
    extract_turn_raw_history,
    record_tool_event,
)


def _find_skill_by_path(
    nodes: list[SkillNode], path: str, opening_tool_call_id: str | None = None
) -> SkillNode | None:
    """Find the most recent invocation through each skill's public interface."""
    for item in reversed(nodes):
        found = item.find_invocation(path, opening_tool_call_id)
        if found is not None:
            return found
    return None


class TurnStreamCollector:
    """Manages accumulation of streaming tokens, multi-path reasoning blocks,
    tool events, and skill nodes during an agent turn.

    Ensures reasoning is cleanly partitioned into distinct ReasoningBlock objects
    whenever execution crosses skill boundaries, changes depth, or executes tools.
    """

    def __init__(self) -> None:
        self.steering = TurnSteering()
        self.tokens: list[str] = []
        self.reasoning_tokens: list[str] = []
        self.reasoning_blocks: list[ReasoningBlock] = []
        self.current_reasoning_block: ReasoningBlock | None = None
        self.generation_counter: int = 0
        self.tools: list[ToolEvent] = []
        self.skills: list[SkillNode] = []
        self.raw_history: list[dict[str, Any]] = []
        self.model_context: ModelContextCheckpoint | None = None
        self.skill_history_tree: list[dict[str, Any]] = []
        self.context_usage: ProviderContextUsage | None = None
        self._pending_skill_before: dict[str, int] = {}
        self._compactions: dict[str, DeepAgentCompactionProgressEvent] = {}

    def set_compaction_progress(self, event: DeepAgentCompactionProgressEvent) -> None:
        operation_id = event.progress.operation_id
        if event.progress.finished:
            self._compactions.pop(operation_id, None)
        else:
            self._compactions[operation_id] = event

    def compaction_snapshot(self) -> list[DeepAgentCompactionProgressEvent]:
        return list(self._compactions.values())

    def _close_current_reasoning_block(self) -> None:
        """Finalize timing on the active reasoning block if one is open."""
        if self.current_reasoning_block is not None:
            self.current_reasoning_block.finalize_timing()
            self.current_reasoning_block = None

    def append_token(self, token: str) -> None:
        """Append a content token to the stream accumulator."""
        self._close_current_reasoning_block()
        self.tokens.append(token)
        if self.context_usage:
            self.context_usage.append_content(token)

    def append_reasoning(
        self, token: str, skill_path: str = "/", depth: int = 0
    ) -> ReasoningBlock:
        """Append a reasoning token.

        If the skill path, depth, or generation boundary has changed,
        creates a new distinct ReasoningBlock with an incremented generation index.
        """
        self.reasoning_tokens.append(token)
        if self.context_usage:
            self.context_usage.append_reasoning(token)
        if (
            self.current_reasoning_block is None
            or self.current_reasoning_block.skill_path != skill_path
            or self.current_reasoning_block.depth != depth
        ):
            self._close_current_reasoning_block()
            self.generation_counter += 1
            self.current_reasoning_block = ReasoningBlock(
                skill_path=skill_path,
                depth=depth,
                generation_index=self.generation_counter,
                content=token,
            )
            self.reasoning_blocks.append(self.current_reasoning_block)
        else:
            self.current_reasoning_block.content += token

        return self.current_reasoning_block

    def append_tool(self, ev: ToolEvent) -> None:
        """Record a tool execution event and demarcate a generation step."""
        self._close_current_reasoning_block()
        record_tool_event(self.tools, ev)

    def turn_raw_history(self) -> list[dict[str, Any]]:
        return extract_turn_raw_history(
            self.raw_history, start=self.steering.history_start
        )

    @property
    def can_steer(self) -> bool:
        """Whether the current execution accepts guidance."""
        return self.steering.available

    def steer(self, message: QueuedMessage) -> bool:
        """Submit guidance through the turn's owned steering state."""
        return self.steering.send(message)

    def append_skill(self, node: SkillNode, action: str = "opened") -> None:
        """Record or update a skill node, preserving opening start_time and computing duration."""
        self._close_current_reasoning_block()
        node = node.model_copy(deep=True)
        node.finalize_timing(action if action in ("completed", "error") else None)
        if self.context_usage and action == "completed":
            node.await_closure_tokens()

        existing = _find_skill_by_path(
            self.skills, node.path, node.opening_tool_call_id
        )
        if existing is not None:
            existing.merge_snapshot(node)
            return

        if node.depth == 1:
            self.skills.append(node)

    def set_raw_history(
        self,
        raw_history: list[dict[str, Any]],
        skill_history_tree: list[dict[str, Any]],
    ) -> None:
        """Set the raw message history and hierarchical skill execution tree."""
        self.raw_history = deepcopy(raw_history)
        self.skill_history_tree = deepcopy(skill_history_tree)
        self._sync_skill_context_tokens()

    def set_provider_context(self, context: ProviderContextUsage) -> None:
        """Reset streamed deltas at each request and freeze exact closure totals."""
        previous = self.context_usage
        if context.is_final and previous:
            context = context.with_stream_from(previous)
        elif (
            context.return_call_id
            and previous
            and previous.authoritative_total is not None
        ):
            self._pending_skill_before[context.return_call_id] = (
                previous.authoritative_total
            )
        self.context_usage = context.model_copy(deep=True)
        if (
            context.is_final or context.closure_tokens_ready
        ) and context.return_call_id:
            before = self._pending_skill_before.get(context.return_call_id)
            if context.is_final:
                self._pending_skill_before.pop(context.return_call_id, None)
            for skill in self._all_skills():
                if (
                    skill.opening_tool_call_id == context.return_call_id
                    and before is not None
                ):
                    skill.record_closure_tokens(before, context.prompt_tokens)
            self._sync_skill_context_tokens()

    def set_model_context(self, context: ModelContextCheckpoint) -> None:
        """Retain an owned checkpoint before publishing the matching raw snapshot."""
        self.model_context = context.model_copy(deep=True)

    def _all_skills(self) -> list[SkillNode]:
        return [node for root in self.skills for node in root.iter_nodes()]

    def get_provider_context(self) -> ProviderContextUsage | None:
        """Return an independent view of the current request's context."""
        return self.context_usage.model_copy(deep=True) if self.context_usage else None

    def skill_snapshots(self) -> list[SkillNode]:
        return [skill.model_copy(deep=True) for skill in self.skills]

    def _sync_skill_context_tokens(self) -> None:
        """Keep persisted and raw-history badges tied to the same skill invocation."""
        nodes = {
            node.opening_tool_call_id: node
            for node in self._all_skills()
            if node.opening_tool_call_id
        }
        if not nodes:
            return
        entries = list(self.skill_history_tree)
        for entry in entries:
            entries.extend(entry.get("children") or [])
            call_id = skill_opening_call_id(entry)
            node = nodes.get(call_id)
            if node:
                entry["tokens_pending"] = node.tokens_pending
                if node.has_closure_tokens:
                    entry["tokens_before_close"] = node.tokens_before_close
                    entry["tokens_after_close"] = node.tokens_after_close

    def get_final_content(self, fallback: str = "") -> str:
        """Return accumulated stream tokens as string, or fallback if empty."""
        return "".join(self.tokens) or fallback

    def get_reasoning_content(self) -> str:
        """Return all reasoning tokens joined as a single string."""
        return "".join(self.reasoning_tokens)

    def snapshot_message(self) -> ChatMessage:
        """Return the current turn without finalizing active events."""
        return self._create_message(is_streaming=True).model_copy(deep=True)

    def _create_message(
        self,
        *,
        role: MessageRole = "assistant",
        fallback_content: str = "",
        error: str | None = None,
        is_streaming: bool = False,
    ) -> ChatMessage:
        """Assemble owned state; callers copy it before exposing it to observers."""
        message = ChatMessage(
            role=role,
            content=self.get_final_content(fallback_content),
            reasoning_content=self.get_reasoning_content(),
            reasoning_blocks=self.reasoning_blocks,
            tool_events=self.tools,
            skills=self.skills,
            raw_history=self.raw_history,
            model_context=self.model_context,
            skill_history_tree=self.skill_history_tree,
            context_usage=self.context_usage,
            compaction_progress=self.compaction_snapshot() if is_streaming else [],
            is_streaming=is_streaming,
            error=error,
        )
        self.steering.apply_to(message)
        if self.model_context is not None:
            message.raw_history_start = self.model_context.turn_start
        return message

    def skill_snapshot(self, path: str) -> SkillNode | None:
        """Return an independent snapshot of a collected skill by path."""
        skill = _find_skill_by_path(self.skills, path)
        return skill.model_copy(deep=True) if skill is not None else None

    def build_message(
        self,
        role: MessageRole = "assistant",
        fallback_content: str = "",
        error: str | None = None,
    ) -> ChatMessage:
        """Construct a final ChatMessage containing all accumulated state."""
        self._close_current_reasoning_block()
        message = self._create_message(
            role=role,
            fallback_content=fallback_content,
            error=error,
        )
        message.finalize_execution("completed" if error is None else "error")
        return message.model_copy(deep=True)
