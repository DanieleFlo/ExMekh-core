"""Observe model request boundaries and preserve provider token totals."""

from __future__ import annotations

import json
from collections.abc import Callable

from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.interfaces.provider import Provider
from pydantic import BaseModel

from exmekh_agent.exmekh_bridge.context_usage import (
    count_message_dict_tokens,
    count_text_tokens,
)
from exmekh_agent.exmekh_bridge.ds4_context import Ds4ContextEstimator
from exmekh_agent.exmekh_bridge.models import ProviderContextUsage


class ContextTrackingProvider(Provider):
    """Wrap the typed provider without changing its requests or event handlers."""

    def __init__(
        self,
        provider: Provider,
        *,
        on_context: Callable[[ProviderContextUsage], None] | None,
        context_id: str | None,
        ds4_qwen: bool,
        thinking: bool,
        api_type: str = "responses",
        on_request: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(provider.get_tokenizer())
        self._provider = provider
        self._on_context = on_context
        self._on_request = on_request
        self._context_id = context_id
        self._tool_mapper = (
            OpenAIResponseToolMapper
            if api_type == "responses"
            else OpenAIChatCompletionToolMapper
        )
        self._ds4_context = Ds4ContextEstimator(thinking=thinking) if ds4_qwen else None

    def set_request_observer(self, observer: Callable[[], None] | None) -> None:
        """Observe owned requests without exposing the provider's callback storage."""
        self._on_request = observer

    def call(self, request: ProviderRequest) -> ProviderResponse:
        extra = request.get_extra() or {}
        agentic = extra.get("agentic") or {}
        if agentic and agentic.get("agent_context_id") != self._context_id:
            return self._provider.call(request)

        if self._on_request:
            self._on_request()
        if self._on_context is None:
            return self._provider.call(request)

        context = self._estimate_request(request, agentic)
        self._on_context(context)
        response = self._provider.call(request)
        if response.usage and response.usage.prompt_tokens is not None:
            if self._ds4_context:
                self._ds4_context.record_response(
                    response,
                    agentic.get("allowed_skills", []),
                    context.return_call_id,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            self._on_context(
                context.with_provider_totals(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
            )
        return response

    def _estimate_request(
        self, request: ProviderRequest, agentic: dict
    ) -> ProviderContextUsage:
        messages = [message.message_dump() for message in request.get_messages()]
        tools = request.get_tools()
        schemas = (
            self._tool_mapper.map_tools(tools)
            if tools and request.get_tool_choice() != "none"
            else []
        )
        if self._ds4_context:
            return self._ds4_context.estimate_request(messages, schemas, agentic)
        prompt_tokens = sum(count_message_dict_tokens(message) for message in messages)
        prompt_tokens += count_text_tokens(json.dumps(schemas, ensure_ascii=False))
        return ProviderContextUsage(
            prompt_tokens=prompt_tokens, return_call_id=agentic.get("skill_call_id")
        )

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        return self._provider.call_structured(request)
