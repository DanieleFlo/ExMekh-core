"""Defaults and policy for execution."""

from dataclasses import dataclass
from typing import Any

from exmekh_agent.config.catalogs import message, settings

DEFAULTS = settings("execution")


@dataclass(frozen=True, slots=True)
class TitleGenerationSettings:
    """Own title sampling, normalization and deterministic fallback policy."""

    fallback_max_characters: int
    temperature: float
    max_output_tokens: int
    reasoning_effort: str
    trim_characters: str

    def fallback(self, prompt: str) -> str:
        clean = prompt.strip().replace("\n", " ")
        shortened = clean[: self.fallback_max_characters]
        if len(clean) > self.fallback_max_characters:
            shortened += message("execution.agent_runner.title_ellipsis")
        return shortened

    def normalize(self, title: str) -> str:
        return title.strip().strip(self.trim_characters)

    def request_extra(self, api_type: str) -> dict[str, Any]:
        sampling: dict[str, Any] = {"temperature": self.temperature}
        if api_type == "responses":
            sampling.update(
                max_output_tokens=self.max_output_tokens,
                reasoning={"effort": self.reasoning_effort},
            )
        else:
            sampling.update(
                max_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
            )
        return sampling


TITLE_GENERATION = TitleGenerationSettings(**DEFAULTS["chat_title"])
