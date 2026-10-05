"""Own snapshot-based token counting, request coalescing and UI delivery."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from nicegui.client import Client

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import ChatSession, SubAgentConversation
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    ContextUsageSummary,
    compute_session_context_usage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextUsageRequest:
    session: ChatSession
    sub_agent: SubAgentConversation | None
    model_name: str
    generation: int

    @property
    def target(self) -> tuple[str, str | None, str, int]:
        return (
            self.session.id,
            self.sub_agent.id if self.sub_agent else None,
            self.model_name,
            self.generation,
        )

    def calculate(self) -> ContextUsageSummary:
        return compute_session_context_usage(
            session=self.session,
            sub_agent=self.sub_agent,
            model_override=self.model_name,
        )


class ContextUsageUpdater:
    """Count owned snapshots in a worker and publish progress for the same target."""

    def __init__(
        self, client: Client, on_result: Callable[[str, ContextUsageSummary], None]
    ) -> None:
        self._client = client
        self._on_result = on_result
        self._pending: ContextUsageRequest | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_scheduled = 0.0
        self._closed = False
        client.on_delete(self.close)

    def request(
        self,
        session: ChatSession,
        *,
        model_name: str,
        collector: TurnStreamCollector | None = None,
        sub_agent: SubAgentConversation | None = None,
    ) -> None:
        if self._closed:
            return
        snapshot = session.model_copy(deep=True)
        if collector is not None and sub_agent is None:
            snapshot.add_message(collector.snapshot_message())
        self._pending = ContextUsageRequest(
            snapshot,
            sub_agent.model_copy(deep=True) if sub_agent else None,
            model_name,
            id(collector) if collector else 0,
        )
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def schedule(
        self,
        loop: asyncio.AbstractEventLoop,
        update: Callable[[], None],
        *,
        force: bool = False,
        accepts_events: Callable[[], bool] | None = None,
    ) -> None:
        """Throttle stream callbacks using elapsed time, including a forced final count."""
        if self._closed:
            return
        now = time.monotonic()
        if (
            not force
            and now - self._last_scheduled
            < EXECUTION_DEFAULTS["context_update_interval_seconds"]
        ):
            return
        self._last_scheduled = now

        def deliver() -> None:
            if not self._closed and (accepts_events is None or accepts_events()):
                update()

        loop.call_soon_threadsafe(deliver)

    async def wait_idle(self) -> None:
        """Wait for outstanding counts without exposing worker tasks to callers."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    def close(self) -> None:
        self._closed = True
        self._pending = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while self._pending is not None:
            request = self._pending
            try:
                result = await loop.run_in_executor(None, request.calculate)
            except Exception:
                logger.exception(
                    app_text("chat.context_updates.unable_to_count_context_tokens")
                )
                if self._pending is request:
                    self._pending = None
                continue
            pending = self._pending
            if pending is None:
                return
            if request.target != pending.target:
                continue
            with self._client:
                self._on_result(request.session.id, result)
            if self._pending is request:
                self._pending = None
