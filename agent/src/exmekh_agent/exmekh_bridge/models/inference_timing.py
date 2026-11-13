"""Live timing for one inference phase."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

InferencePhase = Literal["prefill", "decode", "reasoning", "tool"]


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    phase: InferencePhase
    tokens_per_second: float | None

    @property
    def preparing_tool(self) -> bool:
        return self.phase == "tool"


TimingObserver = Callable[[InferenceTiming | None], None]
