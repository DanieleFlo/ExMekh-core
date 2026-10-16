"""Keep inspectable execution history independent of the model's working context."""

from __future__ import annotations

import time
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from exmekh_agent.exmekh_bridge.history.tool_calls import parse_tool_calls


class RecordedCompaction(BaseModel):
    """One applied operation, independent of the number of replaced blocks."""

    id: str
    replacement_call_ids: list[str]
    after_tool_call_id: str | None = None
    following_user_count: int = 0
    skill_path: str = "/"
    depth: int = 0
    timestamp: float | None = Field(default_factory=time.time)


class ModelContextCheckpoint(BaseModel):
    history: list[dict[str, Any]] = Field(default_factory=list)
    replacements: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    turn_start: int | None = None
    compactions: list[RecordedCompaction] = Field(default_factory=list)

    def expand(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Restore originals before their compaction marker, retaining both for inspection."""
        expanded: list[dict[str, Any]] = []

        def visit(scope: list[dict[str, Any]], active: frozenset[str]) -> None:
            for message in scope:
                calls = parse_tool_calls(message.get("tool_calls")) or []
                if message.get("role") == "assistant" and len(calls) == 1:
                    call_id = calls[0].get_id()
                    if call_id in self.replacements and call_id not in active:
                        visit(self.replacements[call_id], active | {call_id})
                expanded.append(deepcopy(message))

        visit(messages, frozenset())
        return expanded

    def expand_tree(self, tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = deepcopy(tree)
        pending = list(expanded)
        for skill in pending:
            if isinstance(skill.get("history"), list):
                skill["history"] = self.expand(skill["history"])
            pending.extend(skill.get("children") or [])
        return expanded


def latest_model_context(messages: Any) -> ModelContextCheckpoint | None:
    if not isinstance(messages, (list, tuple)):
        return None
    for message in reversed(messages):
        context = getattr(message, "model_context", None)
        if context is not None:
            return context.model_copy(deep=True)
    return None


def replay_model_context(messages: Sequence[Any]) -> list[Any] | None:
    """Replace the archived prefix, preserving every message after its checkpoint."""
    for index in range(len(messages) - 1, -1, -1):
        context = getattr(messages[index], "model_context", None)
        if context is not None:
            return deepcopy([*context.history, *messages[index + 1 :]])
    return None


class HistoryArchive:
    """Own replacement records, visible history and the current turn boundary."""

    def __init__(self, checkpoint: ModelContextCheckpoint | None = None) -> None:
        self._checkpoint = (
            checkpoint.model_copy(deep=True) if checkpoint else ModelContextCheckpoint()
        )
        self._prior_user_count = 0

    def begin_turn(self, history: list[dict[str, Any]] | None = None) -> None:
        self._checkpoint.compactions = []
        if history is not None:
            self._checkpoint.history = deepcopy(history)
        self._prior_user_count = sum(
            message.get("role") == "user" for message in self._checkpoint.history
        )
        self._checkpoint.turn_start = None

    def record_compaction(
        self,
        replacements: dict[str, list[dict[str, Any]]],
        *,
        history: list[dict[str, Any]] | None = None,
        skill_path: str = "/",
        depth: int = 0,
    ) -> None:
        self._checkpoint.replacements.update(deepcopy(replacements))
        if not replacements:
            return
        operation_id = next(iter(replacements))
        if any(event.id == operation_id for event in self._checkpoint.compactions):
            return
        anchor = next(reversed(replacements))
        following_users = 0
        for item in reversed(history or []):
            if item.get("role") in {"tool", "function"} and item.get("tool_call_id"):
                anchor = item["tool_call_id"]
                break
            following_users += item.get("role") == "user"
        self._checkpoint.compactions.append(
            RecordedCompaction(
                id=operation_id,
                replacement_call_ids=list(replacements),
                after_tool_call_id=anchor,
                following_user_count=following_users,
                skill_path=skill_path,
                depth=depth,
            )
        )

    def update(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._checkpoint.history = deepcopy(history)
        expanded = self.expand(history)
        user_indices = [
            index
            for index, message in enumerate(expanded)
            if message.get("role") == "user"
        ]
        if len(user_indices) > self._prior_user_count:
            self._checkpoint.turn_start = user_indices[self._prior_user_count] + 1
        return expanded

    def checkpoint(self) -> ModelContextCheckpoint | None:
        if not self._checkpoint.replacements:
            return None
        return self._checkpoint.model_copy(deep=True)

    def expand(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._checkpoint.expand(history)

    def expand_tree(self, tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._checkpoint.expand_tree(tree)
