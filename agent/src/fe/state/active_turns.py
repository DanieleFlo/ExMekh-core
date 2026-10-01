"""In-memory state for model turns that outlive a browser page."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.models import ChatMessage, QueuedMessage


@dataclass(frozen=True)
class TurnSnapshot:
    started_at: float
    revision: int
    message: ChatMessage | None


@dataclass
class _ActiveTurn:
    task: asyncio.Task[None]
    collector: TurnStreamCollector
    started_at: float
    revision: int = 0


class ActiveTurnRegistry:
    """Track running turns and pass queued work to a replacement page."""

    def __init__(self) -> None:
        self._turns: dict[str, _ActiveTurn] = {}
        self._queue_handoffs: set[str] = set()

    def start(
        self,
        session_id: str,
        task: asyncio.Task[None],
        collector: TurnStreamCollector,
        *,
        started_at: float,
    ) -> None:
        self._queue_handoffs.discard(session_id)
        self._turns[session_id] = _ActiveTurn(task, collector, started_at)

    def is_running(self, session_id: str) -> bool:
        return session_id in self._turns

    def is_task(self, session_id: str, task: asyncio.Task[None]) -> bool:
        turn = self._turns.get(session_id)
        return turn is not None and turn.task is task

    def snapshot(self, session_id: str, *, after_revision: int) -> TurnSnapshot | None:
        turn = self._turns.get(session_id)
        if turn is None:
            return None
        message = (
            turn.collector.snapshot_message()
            if turn.revision != after_revision
            else None
        )
        return TurnSnapshot(turn.started_at, turn.revision, message)

    def mark_changed(self, session_id: str, collector: TurnStreamCollector) -> None:
        turn = self._turns.get(session_id)
        if turn is not None and turn.collector is collector:
            turn.revision += 1

    def finish(
        self,
        session_id: str,
        task: asyncio.Task[None],
        *,
        handoff_queue: bool = False,
    ) -> None:
        if not self.is_task(session_id, task):
            return
        del self._turns[session_id]
        if handoff_queue:
            self._queue_handoffs.add(session_id)

    def stop(self, session_id: str) -> bool:
        turn = self._turns.get(session_id)
        if turn is None or turn.task.done():
            return False
        turn.task.cancel()
        return True

    def can_steer(self, session_id: str) -> bool:
        turn = self._turns.get(session_id)
        return bool(
            turn is not None
            and not turn.task.done()
            and not turn.task.cancelling()
            and turn.collector.steering.available
        )

    def steer(self, session_id: str, message: QueuedMessage) -> bool:
        if not self.can_steer(session_id):
            return False
        turn = self._turns[session_id]
        if not turn.collector.steering.send(message):
            return False
        turn.revision += 1
        return True

    def cancel_and_forget(self, session_id: str) -> None:
        turn = self._turns.pop(session_id, None)
        self._queue_handoffs.discard(session_id)
        if turn is not None and not turn.task.done():
            turn.task.cancel()

    def cancel_all(self) -> None:
        """Cancel all tracked turns and clear pending queue handoffs."""
        for session_id in list(self._turns):
            self.cancel_and_forget(session_id)
        self._queue_handoffs.clear()

    def claim_queue_handoff(self, session_id: str) -> bool:
        if session_id not in self._queue_handoffs:
            return False
        self._queue_handoffs.remove(session_id)
        return True


active_turns = ActiveTurnRegistry()
