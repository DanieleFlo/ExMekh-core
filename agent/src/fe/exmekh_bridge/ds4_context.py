"""Track the prompt prefixes and skill checkpoints retained by DS4."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from exmekh_lib.core.models.provider_response import ProviderResponse

from fe.exmekh_bridge.context_usage import count_text_tokens, find_skill_opening_index
from fe.exmekh_bridge.ds4_prompt import apply_ds4_scope, render_ds4_qwen_prompt
from fe.exmekh_bridge.models import ProviderContextUsage


class Ds4ContextEstimator:
    """Own DS4's cached scope prefixes and sampled skill-opening token totals."""

    def __init__(self, *, thinking: bool) -> None:
        self._thinking = thinking
        self._raw_messages: list[dict[str, Any]] = []
        self._rendered_messages: list[dict[str, Any]] = []
        self._skill_checkpoints: dict[str, int] = {}

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        agentic: dict[str, Any],
    ) -> ProviderContextUsage:
        rendered = self._restore_scope_prefix(messages)
        if agentic:
            apply_ds4_scope(rendered, agentic)
        prompt_tokens = count_text_tokens(
            render_ds4_qwen_prompt(rendered, schemas, thinking=self._thinking)
        )
        return_call_id = agentic.get("skill_call_id")
        checkpoint = self._skill_checkpoints.get(return_call_id)
        opening_index = find_skill_opening_index(rendered, "", return_call_id)
        closure_tokens_ready = checkpoint is not None and opening_index >= 0
        if closure_tokens_ready:
            # DS4 restores the sampled opening, including reasoning, then
            # evaluates a tool-result tail beginning with the terminal token.
            tail = "<|im_end|>\n" + render_ds4_qwen_prompt(
                rendered[opening_index + 1 :], [], thinking=self._thinking
            )
            prompt_tokens = checkpoint + count_text_tokens(tail)
        self._raw_messages = deepcopy(messages)
        self._rendered_messages = rendered
        return ProviderContextUsage(
            prompt_tokens=prompt_tokens,
            return_call_id=return_call_id,
            closure_tokens_ready=closure_tokens_ready,
        )

    def record_response(
        self,
        response: ProviderResponse,
        allowed_skills: list[str],
        return_call_id: str | None,
        prompt_tokens: int,
        completion_tokens: int | None,
    ) -> None:
        if completion_tokens is None:
            return
        self._skill_checkpoints.pop(return_call_id, None)
        for call in response.get_tool_calls() or []:
            if call.get_name() in allowed_skills:
                self._skill_checkpoints[call.get_id()] = (
                    prompt_tokens + completion_tokens + 1
                )

    def _restore_scope_prefix(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Reuse injected scope text only along the unchanged cached prefix."""
        rendered = deepcopy(messages)
        for index, (previous, current) in enumerate(zip(self._raw_messages, messages)):
            if previous != current:
                break
            rendered[index] = deepcopy(self._rendered_messages[index])
        return rendered
