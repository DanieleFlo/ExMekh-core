"""Estimate literal DS4 prompts and retain sampled skill checkpoint totals."""

from __future__ import annotations

from typing import Any

from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)

from exmekh_agent.exmekh_bridge.models import ProviderContextUsage
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    count_text_tokens,
    find_skill_opening_index,
)
from exmekh_agent.exmekh_bridge.providers.ds4_prompt import render_ds4_qwen_prompt


class Ds4ContextEstimator:
    """Own sampled skill-opening totals; prompt estimation never changes them."""

    def __init__(self, *, thinking: bool) -> None:
        self._thinking = thinking
        self._skill_checkpoints: dict[str, tuple[int, bool]] = {}

    @property
    def thinking(self) -> bool:
        return self._thinking

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        agentic: dict[str, Any],
        *,
        response_schema: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> ProviderContextUsage:
        thinking = self._thinking if thinking is None else thinking
        prompt_tokens = count_text_tokens(
            render_ds4_qwen_prompt(
                messages, schemas, thinking=thinking, response_schema=response_schema
            )
        )
        return_call_id = agentic.get("skill_call_id")
        checkpoint = self._skill_checkpoints.get(return_call_id)
        opening_index = find_skill_opening_index(messages, "", return_call_id)
        closure_tokens_ready = (
            checkpoint is not None
            and checkpoint[1] == thinking
            and response_schema is None
            and opening_index >= 0
        )
        if closure_tokens_ready:
            # DS4 restores the sampled opening, including reasoning, then
            # evaluates a tool-result tail beginning with the terminal token.
            tail = "<|im_end|>\n" + render_ds4_qwen_prompt(
                messages[opening_index + 1 :], [], thinking=thinking
            )
            prompt_tokens = checkpoint[0] + count_text_tokens(tail)
        return ProviderContextUsage(
            prompt_tokens=prompt_tokens,
            return_call_id=return_call_id,
            closure_tokens_ready=closure_tokens_ready,
        )

    def record_response(
        self,
        response: ProviderResponse | StructuredProviderResponse,
        allowed_skills: list[str],
        return_call_id: str | None,
        prompt_tokens: int,
        completion_tokens: int | None,
        *,
        thinking: bool | None = None,
    ) -> None:
        if completion_tokens is None:
            return
        self._skill_checkpoints.pop(return_call_id, None)
        for call in response.get_tool_calls() or []:
            if call.get_name() in allowed_skills:
                self._skill_checkpoints[call.get_id()] = (
                    prompt_tokens + completion_tokens + 1,
                    self._thinking if thinking is None else thinking,
                )
