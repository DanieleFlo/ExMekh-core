"""Serializable agent boards and the operations that preserve their invariants."""

from __future__ import annotations

import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

AGENT_PROFILES = {
    "generic": "Generic",
    "programmer": "Programmer",
    "creative": "Creative",
    "custom": "Custom",
}
AgentProfile = Literal["generic", "programmer", "creative", "custom"]


class DesignNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal["hds", "skill", "tool"]
    name: str
    source: str = ""
    content: str = ""
    x: float = 40
    y: float = 40
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
        raise ValueError(f"Unknown agent node: {node_id}")

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
                        f"Conflicting definitions for capability: {node.name}"
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

    def remove_node(self, node_id: str) -> None:
        if node_id == self.root_id:
            raise ValueError("Choose another root before removing this HDS")
        node = self.node(node_id)
        self.nodes.remove(node)
        self.links = [
            link for link in self.links if node_id not in {link.parent, link.child}
        ]

    def set_root(self, node_id: str) -> None:
        if self.node(node_id).kind != "hds":
            raise ValueError("Choose an HDS as the root agent")
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
            raise ValueError("Only HDS nodes have persistent history")
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
                "Agent nodes must have unique IDs and capability names per kind"
            )
        if self.root_id not in ids or self.node(self.root_id).kind != "hds":
            raise ValueError("Choose an HDS as the root agent")
        for node in self.nodes:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", node.id):
                raise ValueError("Use safe node IDs without path separators")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", node.name):
                raise ValueError(
                    "Use letters, numbers, underscores or hyphens for names"
                )
        for link in self.links:
            if (
                link.parent not in ids
                or link.child not in ids
                or link.parent == link.child
            ):
                raise ValueError("A connection must join two different existing nodes")
            parent = self.node(link.parent)
            child = self.node(link.child)
            if parent.kind == "tool":
                raise ValueError("Start connections from an HDS or skill")
            if parent.kind == "skill" and child.kind == "hds":
                raise ValueError("Skills can only connect to sub-skills and tools")
            if link.child == self.root_id:
                raise ValueError("The root HDS cannot be a sub-agent")
        for node in self.nodes:
            if (
                node.kind == "hds"
                and sum(link.child == node.id for link in self.links) > 1
            ):
                raise ValueError("Each sub-agent must have a single parent")

        def visit(node_id: str, ancestors: set[str]) -> None:
            if node_id in ancestors:
                raise ValueError("Agent and skill connections cannot contain cycles")
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
                "Connect every board node to the root, or remove unused nodes"
            )
