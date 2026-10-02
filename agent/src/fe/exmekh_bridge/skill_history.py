"""Interpret recorded ExMekh-Lib skill invocations consistently across views and restores."""

from __future__ import annotations

from typing import Any

from exmekh_lib.core.models.assistant_message import ToolCall

from fe.exmekh_bridge.tool_calls import parse_tool_calls


def inner_skill_history_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skip the parent's opening batch headers, retaining early user guidance."""
    for index in range(1, len(history)):
        if history[index].get("role") in ("user", "assistant"):
            return history[index:]
    return []


def skill_opening_call_id(skill: dict[str, Any]) -> str | None:
    """Read an invocation ID, including legacy snapshots with only opening calls."""
    if skill.get("opening_tool_call_id"):
        return str(skill["opening_tool_call_id"])
    history = skill.get("history") or []
    if history:
        calls = parse_tool_calls(
            history[0].get("tool_calls"), generate_missing_ids=False
        )
        for call in calls or []:
            if call.get_name() == skill.get("skill"):
                return call.get_id()
    return None


def take_skill_for_call(
    skills: list[dict[str, Any]], call: ToolCall
) -> dict[str, Any] | None:
    """Consume one matching invocation; names are a fallback only without an ID."""
    for index, skill in enumerate(skills):
        if skill_opening_call_id(skill) == call.get_id():
            return skills.pop(index)
    for index, skill in enumerate(skills):
        if not skill_opening_call_id(skill) and skill.get("skill") == call.get_name():
            return skills.pop(index)
    return None
