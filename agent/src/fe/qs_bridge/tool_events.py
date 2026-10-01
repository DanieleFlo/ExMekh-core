"""Translate framework execution events consistently for every conversation."""

from __future__ import annotations

import time

from qs.core.agents.deep_agent import DeepAgentExecutionEvent

from fe.qs_bridge.models import ToolEvent
from fe.qs_bridge.tool_calls import extract_tool_arguments


def tool_event_from_execution(
    event: DeepAgentExecutionEvent, *, now: float | None = None
) -> ToolEvent:
    timestamp = time.time() if now is None else now
    running = event.is_tool_running
    return ToolEvent(
        tool_call_id=event.tool_call_id or None,
        event_type=str(event.type),
        status="running" if running else "error" if event.failed else "completed",
        tool_name=event.tool_name,
        arguments=extract_tool_arguments(event.history, event.tool_call_id),
        depth=event.depth,
        skill_path=event.skill_path,
        timestamp=timestamp,
        end_time=None if running else timestamp,
        duration_seconds=None,
    )
