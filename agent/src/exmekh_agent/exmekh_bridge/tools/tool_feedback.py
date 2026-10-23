"""Invocation-scoped UI feedback, independent of model attachment delivery."""

from __future__ import annotations

import base64
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from exmekh_lib.core.agents.node import Node
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
)
from exmekh_lib.utils.tool_errors import ToolInputError

from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.agents.cancellation import raise_if_cancelled
from exmekh_agent.exmekh_bridge.models import ToolEvent

logger = logging.getLogger(__name__)


ApprovalState = Literal["pending", "approved", "denied"]
ImageState = Literal["running", "completed", "error"]
ToolCallback = Callable[[ToolEvent], None]


@dataclass
class _InvocationFeedback:
    approval_state: ApprovalState | None = None
    image_state: ImageState | None = None
    image_urls: list[str] | None = None
    images_in_context: bool | None = None

    def apply_to(self, event: ToolEvent) -> ToolEvent:
        updated = event.model_copy(deep=True)
        updated.approval_state = self.approval_state
        updated.image_state = self.image_state
        updated.images_in_context = self.images_in_context
        if self.image_urls is not None:
            updated.image_urls = list(self.image_urls)
        return updated


class ToolFeedback:
    """Serialize invocation updates without calling observers under the state lock."""

    def __init__(self) -> None:
        self._lock = threading.Condition(threading.RLock())
        self._calls: dict[str, tuple[ToolEvent, ToolCallback]] = {}
        self._updates: dict[str, _InvocationFeedback] = {}
        self._notifications: deque[tuple[ToolEvent, ToolCallback]] = deque()
        self._notifying = False

    def publish(self, event: ToolEvent, callback: ToolCallback) -> None:
        with self._lock:
            if event.tool_call_id:
                state = self._updates.setdefault(
                    event.tool_call_id,
                    _InvocationFeedback(
                        approval_state=event.approval_state,
                        image_state=event.image_state,
                        images_in_context=event.images_in_context,
                    ),
                )
                # Observe the launch before its worker starts, including background calls.
                if (
                    event.tool_name == AGENT_DEFAULTS["image_tool"]
                    and event.is_running
                    and state.image_state is None
                ):
                    state.image_state = "running"
                if event.execution_status == "error":
                    if state.image_state == "running":
                        state.image_state = "error"
                    if state.approval_state == "pending":
                        state.approval_state = None
                event = state.apply_to(event)
                self._calls[event.tool_call_id] = (event, callback)
            self._notifications.append((event.model_copy(deep=True), callback))
            self._lock.notify_all()
        self._notify_observers()

    def update(
        self,
        call_id: str,
        *,
        approval_state: ApprovalState | None = None,
        image_state: ImageState | None = None,
        image_urls: list[str] | None = None,
        images_in_context: bool | None = None,
    ) -> None:
        with self._lock:
            state = self._updates.setdefault(call_id, _InvocationFeedback())
            if approval_state is not None:
                state.approval_state = approval_state
            if image_state is not None:
                state.image_state = image_state
            if images_in_context is not None:
                state.images_in_context = images_in_context
            if image_urls is not None:
                state.image_urls = list(image_urls)
            if call_id in self._calls:
                event, callback = self._calls[call_id]
                event = state.apply_to(event)
                self._calls[call_id] = (event, callback)
                self._notifications.append((event.model_copy(deep=True), callback))
            self._lock.notify_all()
        self._notify_observers()

    def _notify_observers(self) -> None:
        with self._lock:
            if self._notifying:
                return
            self._notifying = True
        try:
            while True:
                with self._lock:
                    if not self._notifications:
                        self._notifying = False
                        self._lock.notify_all()
                        return
                    event, callback = self._notifications.popleft()
                callback(event)
        except BaseException:
            with self._lock:
                self._notifying = False
            raise

    def wait_for_images(
        self, cancellation: threading.Event | None, nodes: Iterable[Node] = ()
    ) -> None:
        # A background launch may outlive the LLM turn. Keep its collector alive;
        # failures skip AfterTool, so also observe the public execution status.
        nodes = tuple(nodes)
        while True:
            with self._lock:
                pending = [
                    call_id
                    for call_id, update in self._updates.items()
                    if update.image_state == "running"
                ]
                if not pending and not self._notifications and not self._notifying:
                    return
            if cancellation is not None:
                raise_if_cancelled(cancellation)
            for call_id in pending:
                for node in nodes:
                    try:
                        state = node.control_tool_execution(call_id, wait_seconds=0.1)
                    except ToolInputError as error:
                        if error.code != "unknown_tool_call_id":
                            raise
                        continue
                    if state["status"] in {"failed", "cancelled"}:
                        self.update(call_id, image_state="error")
                    break
            with self._lock:
                self._lock.wait(timeout=0.1)

    def approval(self, call_id: str, state: ApprovalState) -> None:
        self.update(call_id, approval_state=state)


class ImageArguments(BeforeTool):
    def __init__(self, include_in_context: bool, feedback: ToolFeedback) -> None:
        self.feedback = feedback
        self.include_in_context = include_in_context

    def handle(self, context: ToolContext) -> ToolArguments:
        self.feedback.update(
            context.call_id,
            image_state="running",
            images_in_context=self.include_in_context,
        )
        return ToolArguments(
            {**context.arguments, "include_image_in_result": self.include_in_context}
        )


class ImagePreviews(AfterTool):
    def __init__(self, feedback: ToolFeedback) -> None:
        self.feedback = feedback

    def handle(self, context: ToolContext, result: Any) -> Any:
        urls = []
        if isinstance(result, dict):
            for item in result.get("images", []):
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                path = (context.work_root / item["path"]).resolve()
                if path.is_relative_to(context.work_root.resolve()):
                    try:
                        urls.append(
                            "data:image/png;base64,"
                            + base64.b64encode(path.read_bytes()).decode("ascii")
                        )
                    except OSError as error:
                        # A deleted original must not turn a successful generation
                        # into a failed tool call and trigger another generation.
                        logger.warning(
                            app_text("errors_tools.tool_feedback.preview_unavailable"),
                            error,
                        )
        self.feedback.update(
            context.call_id,
            image_urls=urls,
            image_state="completed" if urls else "error",
        )
        return result
