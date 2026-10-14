"""Typed input for one agent turn, independent of execution and UI state."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.exmekh_bridge.agents.steering import TurnSteering
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEvent
from exmekh_agent.exmekh_bridge.models import (
    ProviderContextUsage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.tools.tool_approval import CommandApprovedCallback
from exmekh_agent.state.agent_design import AgentDesign


@dataclass(frozen=True, slots=True)
class AgentTurnRequest:
    """Keep execution inputs together across the serialized worker boundary."""

    user_prompt: str
    model: str
    base_url: str
    api_key: str
    api_type: str
    reasoning_effort_str: str
    agent_mode: str
    capabilities_path_str: str
    name_hds: str
    on_token: Callable[[str], None]
    on_reasoning: Callable[..., None]
    on_tool_event: Callable[[ToolEvent], None]
    on_skill_event: Callable[[SkillNode, str], None] | None = None
    on_raw_history: (
        Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None
    ) = None
    history: Any | None = None
    agent_context_id: str | None = None
    on_sub_agent_event: Callable[[SubAgentEvent], None] | None = None
    sub_agent_history: list[SubAgentConversation] | None = None
    work_root: str | None = None
    coding_agent_persistent_history: bool | None = None
    on_context: Callable[[ProviderContextUsage], None] | None = None
    on_request_manual_approval: Callable[[str], bool] | None = None
    on_command_approved: CommandApprovedCallback | None = None
    tool_approval_mode: str | None = None
    get_tool_approval_mode: Callable[[], str] | None = None
    approved_commands_history: list[str] | None = None
    tool_approval_raw_history: list[dict[str, Any]] | None = None
    cancellation_event: threading.Event | None = None
    steering: TurnSteering | None = None
    on_steering_added: Callable[[], None] | None = None
    image_urls: list[str] | None = None
    agent_profile: str = AGENT_DEFAULTS["default_profile"]
    agent_design: AgentDesign | None = None
    capabilities_paths: list[str] | None = None
