"""Estimate literal DS4 prompts and retain sampled skill checkpoint totals."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _SkillCheckpoint:
    sampled_tokens: int
    thinking: bool
    request_prefix: str

    def matches(self, prompt: str, *, thinking: bool) -> bool:
        return self.thinking == thinking and prompt.startswith(self.request_prefix)


class Ds4ContextEstimator:
    """Own sampled skill-opening totals; prompt estimation never changes them."""

    def __init__(self, *, thinking: bool) -> None:
        self._thinking = thinking
        self._skill_checkpoints: dict[str, _SkillCheckpoint] = {}

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
        # Scoring can reuse a thinking conversation while disabling only its
        # new generation. Retain that history's template in the prompt budget.
        history_thinking = (
            self._thinking if response_schema is not None and not thinking else thinking
        )
        prompt = render_ds4_qwen_prompt(
            messages,
            schemas,
            thinking=history_thinking,
            generation_thinking=thinking,
            response_schema=response_schema,
        )
        prompt_tokens = count_text_tokens(prompt)
        return_call_id = agentic.get("skill_call_id")
        checkpoint = self._skill_checkpoints.get(return_call_id)
        opening_index = find_skill_opening_index(messages, "", return_call_id)
        closure_tokens_ready = (
            checkpoint is not None
            and checkpoint.matches(prompt, thinking=thinking)
            and response_schema is None
            and opening_index >= 0
        )
        if closure_tokens_ready:
            # DS4 restores the sampled opening, including reasoning, then
            # evaluates a tool-result tail beginning with the terminal token.
            tail = "<|im_end|>\n" + render_ds4_qwen_prompt(
                messages[opening_index + 1 :], [], thinking=thinking
            )
            prompt_tokens = checkpoint.sampled_tokens + count_text_tokens(tail)
        return ProviderContextUsage(
            prompt_tokens=prompt_tokens,
            return_call_id=return_call_id,
            closure_tokens_ready=closure_tokens_ready,
        )

    def record_response(
        self,
        response: ProviderResponse | StructuredProviderResponse,
        agentic: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        thinking: bool | None = None,
    ) -> None:
        # A completed return consumes its checkpoint even without usage totals.
        self._skill_checkpoints.pop(agentic.get("skill_call_id"), None)
        usage = response.usage
        if (
            usage is None
            or usage.prompt_tokens is None
            or usage.completion_tokens is None
        ):
            return
        thinking = self._thinking if thinking is None else thinking
        for call in response.get_tool_calls() or []:
            if call.get_name() in agentic.get("allowed_skills", []):
                self._skill_checkpoints[call.get_id()] = _SkillCheckpoint(
                    sampled_tokens=usage.prompt_tokens + usage.completion_tokens + 1,
                    thinking=thinking,
                    request_prefix=render_ds4_qwen_prompt(
                        messages, schemas, thinking=thinking
                    ),
                )
