"""Normalize recorded QS tool calls at the integration boundary."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from qs.core.models.assistant_message import FunctionCall, ToolCall


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def parse_tool_calls(
    value: Any, *, generate_missing_ids: bool = True
) -> list[ToolCall] | None:
    """Accept QS calls, recorded dictionaries, and JSON-encoded lists."""
    value = _decode_json(value)
    if not isinstance(value, list):
        return None
    calls = []
    for raw_call in value:
        if isinstance(raw_call, ToolCall):
            calls.append(raw_call)
            continue
        if not isinstance(raw_call, dict):
            continue
        call_id = raw_call.get("id")
        if not call_id and not generate_missing_ids:
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            function = raw_call
        arguments = _decode_json(function.get("arguments", {}))
        calls.append(
            ToolCall(
                id=str(call_id or f"call_{len(calls) + 1}"),
                function=FunctionCall(
                    name=str(function.get("name") or "tool"),
                    arguments=arguments if isinstance(arguments, dict) else {},
                ),
            )
        )
    return calls or None


def extract_tool_arguments(history: Any, call_id: str | None) -> dict[str, Any]:
    """Resolve the exact invocation through the public QS ToolCall methods."""
    if not call_id or not isinstance(history, list):
        return {}
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        for call in (
            parse_tool_calls(message.get("tool_calls"), generate_missing_ids=False)
            or []
        ):
            if call.get_id() == call_id:
                return call.get_arguments()
    return {}


def tool_result_for_call(
    results: Sequence[dict[str, Any]], call_id: str, position: int
) -> dict[str, Any] | None:
    """Match by invocation ID, using position only for legacy results without IDs."""
    for result in results:
        if result.get("tool_call_id") == call_id:
            return result
    if position < len(results) and not results[position].get("tool_call_id"):
        return results[position]
    return None
