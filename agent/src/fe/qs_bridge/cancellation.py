"""Propagate turn cancellation across the UI, tool workers, and provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Event

from pydantic import BaseModel
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.event_handler import ProviderEventHandler
from qs.interfaces.provider import Provider


def raise_if_cancelled(cancellation: Event) -> None:
    if cancellation.is_set():
        raise asyncio.CancelledError


class CancellationEventHandler(ProviderEventHandler):
    """Stop consuming a cancelled stream before forwarding more events."""

    def __init__(
        self, cancellation: Event, forward: Callable[[ProviderStreamEvent], None]
    ) -> None:
        self._cancellation = cancellation
        self._forward = forward

    def handle(self, event: ProviderStreamEvent) -> None:
        raise_if_cancelled(self._cancellation)
        self._forward(event)


class CancellableProvider(Provider):
    """Prevent a cancelled agent from starting or continuing inference."""

    def __init__(self, provider: Provider, cancellation: Event) -> None:
        super().__init__(provider.get_tokenizer())
        self._provider = provider
        self._cancellation = cancellation

    def call(self, request: ProviderRequest) -> ProviderResponse:
        raise_if_cancelled(self._cancellation)
        guarded_request = ProviderRequest(
            messages=request.get_messages(),
            extra=request.get_extra(),
            tool_choice=request.get_tool_choice(),
            tools=request.get_tools(),
            stream=request.get_streaming(),
            on_event=CancellationEventHandler(
                self._cancellation, request.execute_on_stream_event
            ),
            reasoning_effort_override=request.get_reasoning_effort_override(),
        )
        response = self._provider.call(guarded_request)
        raise_if_cancelled(self._cancellation)
        return response

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        raise_if_cancelled(self._cancellation)
        guarded_request = StructuredProviderRequest(
            messages=request.get_messages(),
            response_model=request.get_response_model(),
            strict=request.get_strict(),
            extra=request.get_extra(),
            tool_choice=request.get_tool_choice(),
            tools=request.get_tools(),
            stream=request.get_streaming(),
            on_event=CancellationEventHandler(
                self._cancellation, request.execute_on_stream_event
            ),
        )
        response = self._provider.call_structured(guarded_request)
        raise_if_cancelled(self._cancellation)
        return response
