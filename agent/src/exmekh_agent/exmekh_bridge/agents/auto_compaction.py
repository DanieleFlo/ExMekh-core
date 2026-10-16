"""Application threshold policy; node runtimes own safe execution and replacement."""

from __future__ import annotations

from exmekh_lib.core.models.provider_request import ProviderRequest

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
        self._armed = True

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
            self._armed = True
            return False
        if not self._armed:
            return False
        # Rearm only after usage drops below the threshold. Failed or ineffective
        # compaction must not spin on the same oversized protected context.
        self._armed = False
        return True

    def after_compaction(self, request: ProviderRequest) -> None:
        self._armed = not self._over_threshold(request)
