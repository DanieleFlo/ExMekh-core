"""Build configured Node hierarchies using the library's public API."""

from __future__ import annotations

from pathlib import Path

from exmekh_lib.core.agents.node import DefaultCapabilities, Node, ToolHooks

from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.paths import BUNDLED_CAPABILITIES_DIR
from exmekh_agent.exmekh_bridge.agents.sub_agents import (
    ObservedNode,
    SubAgentEventRouter,
)
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.state.agent_design import AgentDesign


def build_quick_agent(
    tool_hooks: dict[str, ToolHooks], agent_context_id: str | None = None
) -> Node:
    """Keep QuickChat's capability selection at the node construction boundary."""
    included = AGENT_DEFAULTS["quick_chat_tools"]
    return Node(
        capabilities_paths=[],
        name_hds=AGENT_DEFAULTS["quick_chat_hds"],
        agent_context_id=agent_context_id,
        default_capabilities=DefaultCapabilities(
            harness=False,
            include=included,
            tool_hooks={
                name: hook for name, hook in tool_hooks.items() if name in included
            },
        ),
    )


def build_generic_agent(
    capabilities_paths: list[Path],
    router: SubAgentEventRouter,
    tool_hooks: dict[str, ToolHooks],
    *,
    root_name: str,
    coding_persistent_history: bool,
    agent_context_id: str | None = None,
) -> tuple[Node, list[ObservedNode]]:
    """Build the bundled hierarchy through the same boundary as custom designs."""
    defaults = CapabilityRepository().default_capabilities_for(
        root_name, capabilities_paths, tool_hooks
    )
    child_name = AGENT_DEFAULTS["coding_capability"]
    children = [
        ObservedNode(
            capabilities_paths=[BUNDLED_CAPABILITIES_DIR],
            default_capabilities=defaults,
            name_hds=child_name,
            path=f"{root_name}/{child_name}",
            event_router=router,
            persistent_history=coding_persistent_history,
        )
    ]
    root = Node(
        capabilities_paths=capabilities_paths,
        children=children,
        name_hds=root_name,
        agent_context_id=agent_context_id,
        default_capabilities=defaults,
    )
    return root, children


def build_designed_agent(
    design: AgentDesign,
    router: SubAgentEventRouter,
    tool_hooks: dict[str, ToolHooks],
    agent_context_id: str | None = None,
) -> tuple[Node, list[ObservedNode]]:
    compiled = CapabilityRepository().compile(design)
    observed: list[ObservedNode] = []

    def build(node_id: str, parent_path: str = "") -> Node:
        item = design.node(node_id)
        included = compiled.default_names_for(item.id)
        defaults = DefaultCapabilities(
            harness=False,
            include=included,
            tool_hooks={
                name: hook for name, hook in tool_hooks.items() if name in included
            },
        )
        path = f"{parent_path}/{item.name}" if parent_path else item.name
        children = [
            build(child.id, path) for child in design.children(item.id, kind="hds")
        ]
        options = dict(
            capabilities_paths=[compiled.capabilities_for(item.id)],
            name_hds=item.name,
            children=children,
            persistent_history=item.persistent_history,
            default_capabilities=defaults,
        )
        if node_id == design.root_id:
            return Node(**options, agent_context_id=agent_context_id)
        child = ObservedNode(**options, path=path, event_router=router)
        observed.append(child)
        return child

    return build(design.root_id), observed


def validate_agent_design(design: AgentDesign) -> None:
    """Validate graph connectivity and actual capability loading through public APIs."""
    root, children = build_designed_agent(design, SubAgentEventRouter(None), {})
    for node in [root, *children]:
        node.load_capabilities(work_root=Path.cwd())
