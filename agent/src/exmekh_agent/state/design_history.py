"""Undo snapshots for local agent-board edits, separate from persisted chat state."""

from __future__ import annotations

from dataclasses import dataclass

from exmekh_agent.state.agent_design import AgentDesign, DesignNode


@dataclass(frozen=True)
class DesignSnapshot:
    design: AgentDesign
    catalog: list[DesignNode]
    selected_id: str

    @classmethod
    def capture(
        cls, design: AgentDesign, catalog: list[DesignNode], selected_id: str
    ) -> DesignSnapshot:
        return cls(
            design.model_copy(deep=True),
            [node.model_copy(deep=True) for node in catalog],
            selected_id,
        )


class DesignHistory:
    def __init__(self) -> None:
        self._undo: list[DesignSnapshot] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def clear(self) -> None:
        self._undo.clear()

    def record(
        self, before: DesignSnapshot, design: AgentDesign, catalog: list[DesignNode]
    ) -> None:
        """Ignore selection and no-op changes; keep every successful board edit."""
        if before.design != design or before.catalog != catalog:
            self._undo.append(before)

    def undo(self) -> DesignSnapshot | None:
        return self._undo.pop() if self._undo else None
