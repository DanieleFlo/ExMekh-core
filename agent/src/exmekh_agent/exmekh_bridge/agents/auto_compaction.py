"""Application threshold policy; node runtimes own safe execution and replacement."""

from __future__ import annotations

from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.request_budget import RequestBudget

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.providers.provider_context import (
    ContextTrackingProvider,
)


class AutoCompactionPolicy:
    def __init__(
        self, tracker: ContextTrackingProvider, *, model: str, base_url: str
    ) -> None:
        self._tracker = tracker
        self._model = model
        self._base_url = base_url
        self._compacted_messages: tuple[dict[str, str], ...] | None = None

    def request_budget(self) -> RequestBudget | None:
        capacity = config.get_server_context_length(self._model, self._base_url)
        return (
            RequestBudget(capacity, self._tracker.count_request_tokens)
            if capacity is not None
            else None
        )

    def _over_threshold(self, request: ProviderRequest) -> bool:
        capacity = config.get_server_context_length(self._model, self._base_url)
        if capacity is None:
            return False
        usage = self._tracker.estimate_context(request)
        used_tokens = max(
            usage.prompt_tokens, self._tracker.last_context_total(request)
        )
        return used_tokens * 100 >= capacity * config.auto_compaction_threshold_percent

    def before_inference(self, request: ProviderRequest) -> bool:
        if not self._over_threshold(request):
            self._compacted_messages = None
            return False
        return self._snapshot(request) != self._compacted_messages

    def after_compaction(self, request: ProviderRequest) -> None:
        # Compare content so equivalent restored message objects do not rearm the
        # policy, while edits to existing messages still trigger compaction.
        self._compacted_messages = (
            self._snapshot(request) if self._over_threshold(request) else None
        )

    @staticmethod
    def _snapshot(request: ProviderRequest) -> tuple[dict[str, str], ...]:
        return tuple(message.message_dump() for message in request.get_messages())
