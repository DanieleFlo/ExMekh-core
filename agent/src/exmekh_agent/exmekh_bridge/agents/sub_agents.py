"""Observe child node calls independently of the root conversation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import uuid4

from exmekh_lib.core.agents.deep_agent import (
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentHistoryCompactionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEventType
from exmekh_lib.interfaces.message import Message

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.compaction import (
    HistoryArchive,
    ModelContextCheckpoint,
    latest_model_context,
)
from exmekh_agent.exmekh_bridge.history.converter import (
    exmekh_message_to_raw_dict,
    restore_node_history,
)
from exmekh_agent.exmekh_bridge.models import SubAgentConversation, ToolEvent
from exmekh_agent.exmekh_bridge.tools.tool_events import tool_event_from_execution
from exmekh_agent.exmekh_bridge.tools.tool_feedback import ToolFeedback


@dataclass(frozen=True, slots=True)
class SubAgentEvent:
    action: Literal[
        "started", "token", "reasoning", "tool", "completed", "error", "history"
    ]
    id: str
    path: str
    persistent_history: bool
    user_message: str = ""
    response: str = ""
    raw_history: list[dict[str, Any]] = field(default_factory=list)
    token: str = ""
    skill_path: str = "/"
    depth: int = 0
    tool_event: ToolEvent | None = None
    model_context: ModelContextCheckpoint | None = None


class SubAgentEventRouter:
    """Route a shared DeepAgent stream to the child currently running."""

    def __init__(
        self,
        on_event: Callable[[SubAgentEvent], None] | None,
        *,
        feedback: ToolFeedback | None = None,
    ) -> None:
        self._on_event = on_event
        self._feedback = feedback
        self._active_nodes: list[ObservedNode] = []

    def enter(self, node: ObservedNode) -> None:
        self._active_nodes.append(node)

    def leave(self, node: ObservedNode) -> None:
        if not self._active_nodes or self._active_nodes[-1] is not node:
            raise RuntimeError(
                app_text(
                    "errors_agents.sub_agents.sub_agent_call_stack_is_out_of_order"
                )
            )
        self._active_nodes.pop()

    def publish(self, event: SubAgentEvent) -> None:
        if self._feedback is not None and event.tool_event is not None:
            self._feedback.publish(
                event.tool_event,
                lambda tool: self._notify(replace(event, tool_event=tool)),
            )
        else:
            self._notify(event)

    def _notify(self, event: SubAgentEvent) -> None:
        if self._on_event:
            self._on_event(event)

    def route(self, event: DeepAgentEvent) -> bool:
        if not self._active_nodes:
            return False
        self._active_nodes[-1].forward_stream_event(event)
        return True


class ObservedNode(Node):
    """Emit a stable identity for each persistent child, or one per fresh call."""

    def __init__(
        self,
        *,
        path: str,
        event_router: SubAgentEventRouter,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = path
        self._event_router = event_router
        self._conversation_id = str(uuid4())
        self._active_call_id: str | None = None
        self._archive = HistoryArchive()

    def restore_conversation(
        self,
        previous: SubAgentConversation,
        prior_exmekh: list[Message[Any]],
    ) -> None:
        if not self.persistent_history:
            return
        self._conversation_id = previous.id
        self._archive = HistoryArchive(latest_model_context(previous.message_history()))
        self._archive.begin_turn([exmekh_message_to_raw_dict(m) for m in prior_exmekh])
        if prior_exmekh:
            restore_node_history(
                self, previous.message_history(), prior_exmekh=prior_exmekh
            )

    def forward_stream_event(self, event: DeepAgentEvent) -> None:
        if self._active_call_id is None:
            return
        shared = {
            "id": self._active_call_id,
            "path": self.path,
            "persistent_history": self.persistent_history,
        }
        if isinstance(event, DeepAgentHistoryCompactionEvent):
            self._archive.record_compaction(
                event.replaced_history,
                history=event.history,
                skill_path=event.skill_path,
                depth=event.depth,
            )
            self._event_router.publish(
                SubAgentEvent(
                    action="history",
                    raw_history=self._archive.update(event.history),
                    model_context=self._archive.checkpoint(),
                    **shared,
                )
            )
            return
        if isinstance(event, DeepAgentStreamEvent):
            provider_event = event.provider_event
            if provider_event.type == ProviderStreamEventType.RESPONSE:
                child_event = SubAgentEvent(
                    action="token", token=provider_event.content or "", **shared
                )
            elif provider_event.type == ProviderStreamEventType.REASONING:
                child_event = SubAgentEvent(
                    action="reasoning",
                    token=provider_event.content or "",
                    skill_path=event.skill_path,
                    depth=event.depth,
                    **shared,
                )
            else:
                return
        elif isinstance(event, DeepAgentExecutionEvent):
            child_event = SubAgentEvent(
                action="tool",
                tool_event=tool_event_from_execution(event),
                **shared,
            )
        else:
            return
        self._event_router.publish(child_event)

    def run(self, user_message: str) -> Any:
        if not self.persistent_history:
            self._archive = HistoryArchive()
        self._archive.begin_turn()
        conversation_id = (
            self._conversation_id if self.persistent_history else str(uuid4())
        )
        self._active_call_id = conversation_id
        self._event_router.enter(self)
        try:
            self._event_router.publish(
                SubAgentEvent(
                    action="started",
                    id=conversation_id,
                    path=self.path,
                    persistent_history=self.persistent_history,
                    user_message=user_message,
                )
            )
            response = super().run(user_message)
            self._event_router.publish(
                SubAgentEvent(
                    action="completed",
                    id=conversation_id,
                    path=self.path,
                    persistent_history=self.persistent_history,
                    response=str(response.get_message() or ""),
                    raw_history=self._archive.update(list(response.history)),
                    model_context=self._archive.checkpoint(),
                )
            )
            return response
        except BaseException:
            self._event_router.publish(
                SubAgentEvent(
                    action="error",
                    id=conversation_id,
                    path=self.path,
                    persistent_history=self.persistent_history,
                )
            )
            raise
        finally:
            self._event_router.leave(self)
            self._active_call_id = None
