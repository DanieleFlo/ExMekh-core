"""Streaming event handlers for quick chat and deep-agent turns."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from exmekh_lib.core.agents.deep_agent import (
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.interfaces.event_handler import (
    DeepAgentEventHandler,
    ProviderEventHandler,
)

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEventRouter
from exmekh_agent.exmekh_bridge.models import SkillNode, TerminalStatus, ToolEvent
from exmekh_agent.exmekh_bridge.tools.tool_events import tool_event_from_execution

logger = logging.getLogger(__name__)


def scoped_reasoning_callback(
    callback: Callable[..., None],
) -> Callable[[str, str, int], None]:
    """Adapt legacy one-argument observers before calling them, never by retrying.

    A TypeError raised inside an observer is a real failure and must propagate.
    Callables without an inspectable signature follow the scoped callback API.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback
    try:
        signature.bind("", "/", 0)
    except TypeError:
        signature.bind("")
        return lambda content, _path, _depth: callback(content)
    return callback


def publish_raw_history(
    callback: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None],
    history: list[dict[str, Any]],
    tree: list[dict[str, Any]],
) -> None:
    """An optional history observer must not interrupt a response or tool execution."""
    try:
        callback(history, tree)
    except Exception as error:  # noqa: BLE001 - observers are best-effort
        logger.warning(
            app_text("errors_agents.event_handlers.unable_to_publish_raw_history_s"),
            error,
        )


class QuickChatEventHandler(ProviderEventHandler):
    """Forward quick chat events and provide a final token when streaming is empty."""

    def __init__(
        self,
        on_token: Callable[[str], None],
        on_reasoning: Callable[..., None],
    ) -> None:
        self._on_token = on_token
        self._on_reasoning = scoped_reasoning_callback(on_reasoning)
        self._received_tokens = False

    def handle(self, event: ProviderStreamEvent) -> None:
        if event.type == ProviderStreamEventType.RESPONSE and event.content:
            self._received_tokens = True
            self._on_token(event.content)
        elif event.type == ProviderStreamEventType.REASONING and event.content:
            self._on_reasoning(event.content, "/", 0)

    def finish(self, final_text: str) -> str:
        if not self._received_tokens:
            self._on_token(final_text)
        return final_text


class DeepAgentTurnEventHandler(DeepAgentEventHandler):
    """Track root agent streaming, tools, and skill progress for one turn."""

    def __init__(
        self,
        event_router: SubAgentEventRouter,
        on_token: Callable[[str], None],
        on_reasoning: Callable[..., None],
        on_tool_event: Callable[[ToolEvent], None],
        on_skill_event: Callable[[SkillNode, str], None] | None,
        on_raw_history: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
        | None,
        steering: TurnSteering | None = None,
        on_steering_added: Callable[[], None] | None = None,
    ) -> None:
        self.event_router = event_router
        self.on_token = on_token
        self.on_reasoning = scoped_reasoning_callback(on_reasoning)
        self.on_tool_event = on_tool_event
        self.on_skill_event = on_skill_event
        self.on_raw_history = on_raw_history
        self.steering = steering
        self.on_steering_added = on_steering_added
        self._preceding_tool: tuple[str, str] | None = None
        self.skill_stack: list[SkillNode] = []
        self.active_tools: dict[tuple[str | None, str, str, int], ToolEvent] = {}
        self.pending_skill_open_times: dict[str, float] = {}
        self.pending_skill_call_ids: dict[str, str] = {}

    def _finalize_skill(
        self, node: SkillNode, status: TerminalStatus = "completed"
    ) -> None:
        node.finalize_timing(status)
        node.is_collapsed = True
        if self.on_skill_event:
            self.on_skill_event(node, node.status)
            self._notify_skill_updated()

    def _current_skill_for(self, skill_path: str, depth: int) -> SkillNode | None:
        while self.skill_stack and not (
            skill_path == self.skill_stack[-1].path
            or skill_path.startswith(f"{self.skill_stack[-1].path}/")
        ):
            self._finalize_skill(self.skill_stack.pop())

        if not skill_path or skill_path == "/" or depth <= 0:
            return None
        if self.skill_stack and self.skill_stack[-1].path == skill_path:
            return self.skill_stack[-1]

        skill_name = skill_path.rstrip("/").split("/")[-1]
        opened_at = self.pending_skill_open_times.pop(skill_name, time.time())
        node = SkillNode(
            name=skill_name,
            path=skill_path,
            depth=depth,
            status="active",
            start_time=opened_at,
            is_collapsed=False,
            opening_tool_call_id=self.pending_skill_call_ids.pop(skill_name, None),
        )
        if self.skill_stack and depth > self.skill_stack[-1].depth:
            self.skill_stack[-1].attach_descendant(node)
        self.skill_stack.append(node)

        if self.on_skill_event:
            self.on_skill_event(node, "opened")
            if len(self.skill_stack) > 1:
                self.on_skill_event(self.skill_stack[0], "updated")
        return node

    def handle(self, event: DeepAgentEvent) -> None:
        if self.event_router.route(event):
            return
        if not isinstance(event, (DeepAgentStreamEvent, DeepAgentExecutionEvent)):
            return
        current_skill = self._current_skill_for(event.skill_path, event.depth)

        if isinstance(event, DeepAgentStreamEvent):
            self._handle_stream_event(event)
        else:
            self._handle_execution_event(event, current_skill)

    def _handle_stream_event(self, event: DeepAgentStreamEvent) -> None:
        provider_event = event.provider_event
        if provider_event.type == ProviderStreamEventType.STEERING_ADDED:
            self._record_steering_insertion(event)
            return

        # A new inference separates its answer from the preceding tool batch.
        self.begin_request()
        if (
            provider_event.type == ProviderStreamEventType.REASONING
            and provider_event.content
        ):
            self.on_reasoning(provider_event.content, event.skill_path, event.depth)
        elif (
            provider_event.type == ProviderStreamEventType.RESPONSE
            and provider_event.content
        ):
            self.on_token(provider_event.content)

    def _record_steering_insertion(self, event: DeepAgentStreamEvent) -> None:
        if self.steering is None:
            return
        call_id, tool_name = self._preceding_tool or (None, None)
        confirmed = self.steering.record_added(
            event.provider_event.message,
            after_tool_call_id=call_id,
            tool_name=tool_name,
            skill_path=event.skill_path,
        )
        if confirmed and self.on_steering_added:
            self.on_steering_added()

    def begin_request(self) -> None:
        """Clear the tool boundary even when the provider emits no stream tokens."""
        self._preceding_tool = None

    def _handle_execution_event(
        self, event: DeepAgentExecutionEvent, current_skill: SkillNode | None
    ) -> None:
        if self.on_raw_history and event.history:
            publish_raw_history(
                self.on_raw_history, list(event.history), list(event.history_tree)
            )

        event_type = str(event.type)
        call_id = event.tool_call_id
        now = time.time()
        tool_event = tool_event_from_execution(event, now=now)
        if event_type == "tool_start":
            self.pending_skill_open_times[event.tool_name] = now
            self.pending_skill_call_ids[event.tool_name] = call_id
            if call_id:
                recorded = self.active_tools.get(tool_event.invocation_key)
                if recorded is None:
                    self.active_tools[tool_event.invocation_key] = tool_event
                else:
                    recorded.merge_update(tool_event)
                    tool_event = recorded
            self._publish_tool_event(tool_event, current_skill)
        elif event_type == "tool_end":
            self._preceding_tool = (call_id, event.tool_name)
            started = (
                self.active_tools.pop(tool_event.invocation_key, None)
                if call_id
                else None
            )
            if started is not None:
                started.merge_update(tool_event)
                self._publish_tool_event(started, current_skill)
            else:
                self._publish_tool_event(tool_event, current_skill)
        else:
            self._publish_tool_event(tool_event, current_skill)

    def _publish_tool_event(
        self, event: ToolEvent, current_skill: SkillNode | None
    ) -> None:
        self.on_tool_event(event)
        if current_skill:
            current_skill.add_tool_event(event)
            self._notify_skill_updated()

    def _notify_skill_updated(self) -> None:
        if self.on_skill_event and self.skill_stack:
            self.on_skill_event(self.skill_stack[0], "updated")

    def finish(self, status: TerminalStatus = "completed") -> None:
        while self.skill_stack:
            self._finalize_skill(self.skill_stack.pop(), status)
        for event in self.active_tools.values():
            event.finalize_timing(status)
            self.on_tool_event(event)
        self.active_tools.clear()
