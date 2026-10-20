"""Selection of board nodes, independent of rendering and undo storage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from exmekh_agent.state.agent_design import AgentDesign


@dataclass(frozen=True)
class DesignSelection:
    node_ids: tuple[str, ...] = ()

    @property
    def primary_id(self) -> str:
        return next(iter(self.node_ids), "")

    @classmethod
    def for_nodes(cls, design: AgentDesign, node_ids: Iterable[str]) -> DesignSelection:
        """Validate the whole selection and use the board's stable node order."""
        requested = {design.node(node_id).id for node_id in node_ids}
        return cls(
            tuple(node_id for node_id in design.node_ids() if node_id in requested)
        )

    def retain_existing(self, design: AgentDesign) -> DesignSelection:
        """Keep surviving selections when a refresh or edit removes a card."""
        remaining = tuple(
            node_id for node_id in self.node_ids if design.has_node(node_id)
        )
        if self.node_ids and not remaining:
            remaining = (design.root_id,)
        return type(self)(remaining)
