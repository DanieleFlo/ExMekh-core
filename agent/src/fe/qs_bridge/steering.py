"""Thread-safe access to steering for one running QS turn."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from threading import Lock

from qs.core.models.user_message import UserMessage, UserMessageContent

from fe.qs_bridge.image_input import serialize_user_content, user_content
from fe.qs_bridge.models import ChatMessage, QueuedMessage, SteeringInsertion


class TurnSteering:
    """Accept guidance for a single run and retain it after the run closes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._handler: Callable[[UserMessageContent], None] | None = None
        self._history_start: int | None = None
        self._messages: list[QueuedMessage] = []
        self._insertions: list[SteeringInsertion] = []
        self._closed = False
        self._image_content: dict[str, object] = {}

    def bind(
        self, handler: Callable[[UserMessageContent], None], *, history_start: int
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._handler = handler
            self._history_start = history_start

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._handler = None

    @property
    def available(self) -> bool:
        with self._lock:
            return self._handler is not None

    @property
    def history_start(self) -> int | None:
        with self._lock:
            return self._history_start

    def send(self, message: QueuedMessage) -> bool:
        """Queue guidance in QS; confirmation arrives separately through its events."""
        with self._lock:
            if self._handler is None:
                return False
            try:
                self._handler(user_content(message.text, message.image_urls))
            except RuntimeError:
                # QS may close the run between the UI check and the click.
                return False
            self._messages.append(message.model_copy(deep=True))
            return True

    def record_added(
        self,
        message: UserMessage,
        *,
        after_tool_call_id: str | None,
        tool_name: str | None,
        skill_path: str,
    ) -> bool:
        """Confirm the next accepted message in FIFO order, including repeated text."""
        with self._lock:
            index = len(self._insertions)
            if index >= len(self._messages):
                return False
            queued = self._messages[index]
            expected = UserMessage(user_content(queued.text, queued.image_urls))
            if message.message_dump() != expected.message_dump():
                return False
            if queued.image_urls:
                self._image_content[message.message_dump()["content"]] = (
                    serialize_user_content(message)
                )
            self._insertions.append(
                SteeringInsertion(
                    message_id=queued.id,
                    after_tool_call_id=after_tool_call_id,
                    tool_name=tool_name,
                    skill_path=skill_path,
                )
            )
            return True

    def apply_to(self, message: ChatMessage) -> None:
        with self._lock:
            message.steering_insertions = [
                item.model_copy(deep=True) for item in self._insertions
            ]
            if self._messages:
                message.steering_messages = [
                    item.model_copy(deep=True) for item in self._messages
                ]
                message.raw_history_start = self._history_start
            histories = [message.raw_history]
            pending = list(message.skill_history_tree)
            for entry in pending:
                histories.append(entry.get("history", []))
                pending.extend(entry.get("children", []))
            for history in histories:
                for item in history:
                    content = item.get("content")
                    if (
                        item.get("role") == "user"
                        and isinstance(content, str)
                        and content in self._image_content
                    ):
                        item["content"] = deepcopy(self._image_content[content])
