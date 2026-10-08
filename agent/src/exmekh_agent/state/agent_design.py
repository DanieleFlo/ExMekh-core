"""Serializable agent boards and the operations that preserve their invariants."""

from __future__ import annotations

import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from exmekh_agent.config.agents import AGENT_PROFILES as AGENT_PROFILES
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.catalogs import presentation_settings

BOARD_DEFAULTS = presentation_settings("agent_board")

AgentProfile = Literal["generic", "programmer", "creative", "custom"]


class DesignNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal["hds", "skill", "tool"]
    name: str
    source: str = ""
    content: str = ""
    x: float = BOARD_DEFAULTS["initial_node_x"]
    y: float = BOARD_DEFAULTS["initial_node_y"]
    persistent_history: bool = True


class DesignLink(BaseModel):
    parent: str
    child: str


class AgentDesign(BaseModel):
    nodes: list[DesignNode] = Field(default_factory=list)
    links: list[DesignLink] = Field(default_factory=list)
    root_id: str = ""
    capability_paths: list[str] = Field(default_factory=list)
    skill_connections_explicit: bool = False

    def node(self, node_id: str) -> DesignNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError(
            app_text("errors_state.agent_design.unknown_agent_node", value_1=node_id)
        )

    def has_node(self, node_id: str) -> bool:
        return any(node.id == node_id for node in self.nodes)

    def connected_nodes(self) -> list[tuple[DesignNode, DesignNode]]:
        """Expose endpoints for presentation without leaking link resolution."""
        return [(self.node(link.parent), self.node(link.child)) for link in self.links]

    def children(self, node_id: str, *, kind: str | None = None) -> list[DesignNode]:
        children = [
            self.node(link.child) for link in self.links if link.parent == node_id
        ]
        return [node for node in children if kind is None or node.kind == kind]

    def capability(self, kind: str, name: str) -> DesignNode | None:
        return next(
            (node for node in self.nodes if node.kind == kind and node.name == name),
            None,
        )

    def connections(self, node_id: str) -> list[DesignLink]:
        return [link for link in self.links if node_id in {link.parent, link.child}]

    def connection_targets(self, parent_id: str) -> list[DesignNode]:
        """Offer only new connections accepted by the same graph invariants as connect."""
        parent = self.node(parent_id)
        if parent.kind == "tool":
            return []
        connected = {node.id for node in self.children(parent_id)}
        targets = []
        for node in self.nodes:
            if node.id == parent_id or node.id in connected:
                continue
            candidate = self.model_copy(deep=True)
            try:
                candidate.connect(parent_id, node.id)
            except ValueError:
                continue
            targets.append(node)
        return targets

    def consolidate_capabilities(self) -> None:
        """Migrate repeated legacy cards to shared nodes without losing connections."""
        unique: list[DesignNode] = []
        capabilities: dict[tuple[str, str], DesignNode] = {}
        aliases: dict[str, str] = {}
        for node in self.nodes:
            key = (node.kind, node.name)
            previous = capabilities.get(key)
            if node.kind == "hds" or previous is None:
                unique.append(node)
                capabilities[key] = node
            else:
                if node.source != previous.source or node.content != previous.content:
                    raise ValueError(
                        app_text(
                            "errors_state.agent_design.conflicting_definitions_for_capability",
                            value_1=node.name,
                        )
                    )
                aliases[node.id] = previous.id
        connections: list[DesignLink] = []
        for link in self.links:
            rewritten = DesignLink(
                parent=aliases.get(link.parent, link.parent),
                child=aliases.get(link.child, link.child),
            )
            if rewritten not in connections:
                connections.append(rewritten)
        previous_nodes, previous_links = self.nodes, self.links
        self.nodes, self.links = unique, connections
        try:
            self.validate_graph()
        except ValueError:
            self.nodes, self.links = previous_nodes, previous_links
            raise

    def add_node(self, node: DesignNode, parent_id: str | None = None) -> None:
        """Add a card atomically, including its optional ownership connection."""
        self.nodes.append(node)
        try:
            if parent_id is not None:
                self.connect(parent_id, node.id)
            else:
                self.validate_graph()
        except ValueError:
            self.nodes.pop()
            raise

    def move_node(self, node_id: str, x: float, y: float) -> None:
        node = self.node(node_id)
        node.x = max(0, x)
        node.y = max(0, y)

    def replace_node(self, node_id: str, replacement: DesignNode) -> None:
        """Keep card identity and connections when editing a capability copy."""
        current = self.node(node_id)
        index = self.nodes.index(current)
        self.nodes[index] = replacement.model_copy(update={"id": node_id}, deep=True)
        try:
            self.validate_graph()
        except ValueError:
            self.nodes[index] = current
            raise

    def disconnect(self, link: DesignLink) -> None:
        self.links.remove(link)

    def remove_dependencies(
        self, parent_id: str, capabilities: set[tuple[str, str]]
    ) -> None:
        """Remove obsolete dependencies, retaining cards still owned elsewhere."""
        removed: set[str] = set()
        for link in list(self.connections(parent_id)):
            child = self.node(link.child)
            if link.parent == parent_id and (child.kind, child.name) in capabilities:
                self.disconnect(link)
                removed.add(child.id)
        self._remove_orphaned_nodes(removed)

    def _remove_orphaned_nodes(self, candidates: set[str]) -> None:
        while candidates:
            node_id = candidates.pop()
            if any(link.child == node_id for link in self.links):
                continue
            candidates.update(child.id for child in self.children(node_id))
            self.remove_node(node_id)

    def remove_node(self, node_id: str) -> None:
        if node_id == self.root_id:
            raise ValueError(
                app_text(
                    "errors_state.agent_design.choose_another_root_before_removing_this_hds"
                )
            )
        node = self.node(node_id)
        self.nodes.remove(node)
        self.links = [
            link for link in self.links if node_id not in {link.parent, link.child}
        ]

    def set_root(self, node_id: str) -> None:
        if self.node(node_id).kind != "hds":
            raise ValueError(
                app_text("errors_state.agent_design.choose_an_hds_as_the_root_agent")
            )
        previous_root, previous_links = self.root_id, self.links
        self.root_id = node_id
        self.links = [link for link in self.links if link.child != node_id]
        try:
            self.validate_graph()
        except ValueError:
            self.root_id, self.links = previous_root, previous_links
            raise

    def set_persistent_history(self, node_id: str, enabled: bool) -> None:
        node = self.node(node_id)
        if node.kind != "hds":
            raise ValueError(
                app_text(
                    "errors_state.agent_design.only_hds_nodes_have_persistent_history"
                )
            )
        node.persistent_history = enabled

    def connect(self, parent: str, child: str) -> None:
        link = DesignLink(parent=parent, child=child)
        if link in self.links:
            return
        self.links.append(link)
        try:
            self.validate_graph()
        except ValueError:
            self.links.remove(link)
            raise

    def validate_graph(self) -> None:
        ids = [node.id for node in self.nodes]
        names = [(node.kind, node.name) for node in self.nodes]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError(
                app_text(
                    "errors_state.agent_design.agent_nodes_must_have_unique_ids_and_capability_names_per_kind"
                )
            )
        if self.root_id not in ids or self.node(self.root_id).kind != "hds":
            raise ValueError(
                app_text("errors_state.agent_design.choose_an_hds_as_the_root_agent")
            )
        for node in self.nodes:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", node.id):
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.use_safe_node_ids_without_path_separators"
                    )
                )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", node.name):
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.use_letters_numbers_underscores_or_hyphens_for_names"
                    )
                )
        for link in self.links:
            if (
                link.parent not in ids
                or link.child not in ids
                or link.parent == link.child
            ):
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.a_connection_must_join_two_different_existing_nodes"
                    )
                )
            parent = self.node(link.parent)
            child = self.node(link.child)
            if parent.kind == "tool":
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.start_connections_from_an_hds_or_skill"
                    )
                )
            if parent.kind == "skill" and child.kind == "hds":
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.skills_can_only_connect_to_sub_skills_and_tools"
                    )
                )
            if link.child == self.root_id:
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.the_root_hds_cannot_be_a_sub_agent"
                    )
                )
        for node in self.nodes:
            if (
                node.kind == "hds"
                and sum(link.child == node.id for link in self.links) > 1
            ):
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.each_sub_agent_must_have_a_single_parent"
                    )
                )

        def visit(node_id: str, ancestors: set[str]) -> None:
            if node_id in ancestors:
                raise ValueError(
                    app_text(
                        "errors_state.agent_design.agent_and_skill_connections_cannot_contain_cycles"
                    )
                )
            for link in self.links:
                if link.parent == node_id:
                    visit(link.child, ancestors | {node_id})

        for node in self.nodes:
            if node.kind != "tool":
                visit(node.id, set())

    def validate_connected(self) -> None:
        self.validate_graph()
        attached = {self.root_id}
        while True:
            expanded = attached | {
                link.child for link in self.links if link.parent in attached
            }
            if expanded == attached:
                break
            attached = expanded
        if any(node.id not in attached for node in self.nodes):
            raise ValueError(
                app_text(
                    "errors_state.agent_design.connect_every_board_node_to_the_root_or_remove_unused_nodes"
                )
            )
