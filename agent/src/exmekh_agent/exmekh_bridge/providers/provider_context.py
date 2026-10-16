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

from exmekh_agent.exmekh_bridge.models import ProviderContextUsage
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    count_message_dict_tokens,
    count_text_tokens,
)
from exmekh_agent.exmekh_bridge.providers.ds4_context import Ds4ContextEstimator


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
        agentic_only: bool = False,
    ) -> None:
        super().__init__(provider.get_tokenizer())
        self._provider = provider
        self._on_context = on_context
        self._on_request = on_request
        self._context_id = context_id
        self._agentic_only = agentic_only
        self._tool_mapper = (
            OpenAIResponseToolMapper
            if api_type == "responses"
            else OpenAIChatCompletionToolMapper
        )
        self._ds4_context = Ds4ContextEstimator(thinking=thinking) if ds4_qwen else None
        self._scope_estimators: dict[str, Ds4ContextEstimator] = {}
        self._last_totals: dict[str, int] = {}

    def invalidate_context(self, context_id: str | None) -> None:
        self._last_totals.pop(context_id or "", None)
        self._scope_estimators.pop(context_id or "", None)

    def restore_context_usage(
        self, context_id: str, usage: ProviderContextUsage
    ) -> None:
        if usage.authoritative_total is not None:
            self._last_totals[context_id] = usage.authoritative_total

    def last_context_total(self, request: ProviderRequest) -> int:
        agentic = (request.get_extra() or {}).get("agentic") or {}
        return self._last_totals.get(agentic.get("agent_context_id") or "", 0)

    def _remember_totals(self, request, response) -> None:
        agentic = (request.get_extra() or {}).get("agentic") or {}
        usage = response.usage
        if agentic and usage is not None and usage.prompt_tokens is not None:
            self._last_totals[agentic.get("agent_context_id") or ""] = (
                usage.prompt_tokens + (usage.completion_tokens or 0)
            )

    def set_request_observer(self, observer: Callable[[], None] | None) -> None:
        """Observe owned requests without exposing the provider's callback storage."""
        self._on_request = observer

    def estimate_context(self, request: ProviderRequest) -> ProviderContextUsage:
        """Estimate a node's next prompt without notifying UI or admitting inference."""
        return self._estimate_request(
            request, (request.get_extra() or {}).get("agentic") or {}
        )

    def _observes(self, request: ProviderRequest) -> bool:
        agentic = (request.get_extra() or {}).get("agentic") or {}
        return not agentic or agentic.get("agent_context_id") == self._context_id

    def _begin_request(self, request: ProviderRequest) -> ProviderContextUsage | None:
        agentic = (request.get_extra() or {}).get("agentic") or {}
        # Stateless thinking summaries have no agentic scope and are not chat context.
        if self._agentic_only and not agentic:
            return None
        observes = self._observes(request)
        if observes and self._on_request:
            self._on_request()
        if self._ds4_context is None and (not observes or self._on_context is None):
            return None
        context = self._estimate_request(request, agentic)
        if observes and self._on_context:
            self._on_context(context)
        return context

    def _record_response(
        self,
        context: ProviderContextUsage | None,
        request: ProviderRequest,
        response: ProviderResponse | StructuredProviderResponse,
    ) -> None:
        if context is None:
            return
        usage = response.usage
        has_prompt_usage = usage is not None and usage.prompt_tokens is not None
        prompt_tokens = (
            usage.prompt_tokens if has_prompt_usage else context.prompt_tokens
        )
        completion_tokens = usage.completion_tokens if has_prompt_usage else None
        agentic = (request.get_extra() or {}).get("agentic") or {}
        estimator = self._estimator(agentic)
        if estimator and has_prompt_usage:
            estimator.record_response(
                response,
                agentic.get("allowed_skills", []),
                context.return_call_id,
                prompt_tokens,
                completion_tokens,
            )
        final_context = context.with_provider_totals(prompt_tokens, completion_tokens)
        if completion_tokens is None:
            # Recount the complete response, including tool arguments not streamed by
            # the library. Provider totals, when present, remain authoritative.
            final_context.completion_estimate = count_message_dict_tokens(
                response.message_dump()
            )
            if response.thinking and response.thinking.thoughts:
                final_context.streamed_reasoning = response.thinking.thoughts
        if self._observes(request) and self._on_context:
            self._on_context(final_context)

    def call(self, request: ProviderRequest) -> ProviderResponse:
        context = self._begin_request(request)
        response = self._provider.call(request)
        self._remember_totals(request, response)
        self._record_response(context, request, response)
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
        estimator = self._estimator(agentic)
        if estimator:
            return estimator.estimate_request(messages, schemas, agentic)
        prompt_tokens = sum(count_message_dict_tokens(message) for message in messages)
        prompt_tokens += count_text_tokens(json.dumps(schemas, ensure_ascii=False))
        return ProviderContextUsage(
            prompt_tokens=prompt_tokens, return_call_id=agentic.get("skill_call_id")
        )

    def _estimator(self, agentic: dict) -> Ds4ContextEstimator | None:
        if self._ds4_context is None:
            return None
        context_id = agentic.get("agent_context_id") or ""
        if not agentic or context_id == self._context_id:
            return self._ds4_context
        if context_id not in self._scope_estimators:
            self._scope_estimators[context_id] = Ds4ContextEstimator(
                thinking=self._ds4_context.thinking
            )
        return self._scope_estimators[context_id]

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        context = self._begin_request(request)
        response = self._provider.call_structured(request)
        self._record_response(context, request, response)
        return response
