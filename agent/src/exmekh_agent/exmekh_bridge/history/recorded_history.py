"""Interpret recorded invocation transcripts without exposing their storage layout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from exmekh_agent.config.formatting import DEFAULTS as FORMAT_DEFAULTS
from exmekh_agent.exmekh_bridge.history.tool_calls import (
    extract_tool_arguments,
    restore_tool_call_ids,
)


@dataclass(frozen=True)
class RecordedHistory:
    messages: list[dict[str, Any]]
    skills: list[dict[str, Any]]

    def system_prompt_message(self) -> dict[str, Any] | None:
        """Return the first nonempty root prompt without sharing retained data."""
        for message in self.messages:
            role = str(message.get("role", "")).lower()
            if (
                role in {"system", "developer"}
                and str(message.get("content", "")).strip()
            ):
                owned = deepcopy(message)
                owned["role"] = role
                return owned
        return None

    def scopes(self) -> Iterator[list[dict[str, Any]]]:
        """Visit the root transcript and all retained skill transcripts."""
        yield self.messages
        pending = list(self.skills)
        for skill in pending:
            yield skill.get("history") or []
            pending.extend(skill.get("children") or [])

    def arguments_for_call(self, call_id: str | None) -> dict[str, Any]:
        for scope in self.scopes():
            arguments = extract_tool_arguments(scope, call_id)
            if arguments:
                return arguments
        return {}

    def output_for_call(self, call_id: str | None) -> str | None:
        if call_id is None:
            return None
        for scope in self.scopes():
            # Saved legacy transcripts may omit call IDs. Correlate an owned copy
            # so reading an output never rewrites the retained transcript.
            correlated = deepcopy(scope)
            restore_tool_call_ids(correlated)
            for message in correlated:
                if str(message.get("role", "")).lower() not in {"tool", "function"}:
                    continue
                if message.get("tool_call_id") != call_id:
                    continue
                content = message.get("content")
                if content is not None:
                    return (
                        content
                        if isinstance(content, str)
                        else json.dumps(
                            content,
                            ensure_ascii=False,
                            indent=FORMAT_DEFAULTS["indent_spaces"],
                        )
                    )
        return None

    def restored_snapshot(
        self, image_content: dict[str, list[dict[str, Any]]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Restore image payloads and omitted invocation IDs on an owned copy."""
        restored = deepcopy(self)
        for scope in restored.scopes():
            restore_tool_call_ids(scope)
            self._restore_scope_user_content(scope, image_content)
        return restored.messages, restored.skills

    def restore_user_content(
        self, image_content: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Restore serialized user payloads across all scopes of an owned transcript."""
        for scope in self.scopes():
            self._restore_scope_user_content(scope, image_content)

    @staticmethod
    def _restore_scope_user_content(
        scope: list[dict[str, Any]], image_content: dict[str, list[dict[str, Any]]]
    ) -> None:
        for message in scope:
            content = message.get("content")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and content in image_content
            ):
                message["content"] = deepcopy(image_content[content])


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
