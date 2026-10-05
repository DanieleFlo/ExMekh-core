"""Shared project configuration and conversation-local execution overrides."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from exmekh_agent.config.approvals import DEFAULT_TOOL_APPROVAL_MODE
from exmekh_agent.exmekh_bridge.tools.tool_approval import normalize_tool_approval_mode
from exmekh_agent.state.agent_design import AgentDesign, AgentProfile


class ProjectSettings(BaseModel):
    """Shared ExMekh-Lib options and the approval default for new project chats."""

    agent_profile: AgentProfile = "generic"
    agent_design: AgentDesign | None = None
    name_hds: str | None = None
    capabilities_path: str | None = None
    coding_agent_persistent_history: bool | None = None
    tool_approval_mode: str = DEFAULT_TOOL_APPROVAL_MODE

    def with_agent(self, profile: AgentProfile, design: AgentDesign | None) -> Self:
        """Return independent settings for the selected agent configuration."""
        return type(self).model_validate(
            {
                **self.model_dump(),
                "agent_profile": profile,
                "agent_design": design.model_dump() if design else None,
            }
        )

    def migrate_legacy_capabilities_path(self) -> bool:
        """Use the installed library defaults for paths saved before its rename."""
        if self.capabilities_path is None:
            return False
        path = self.capabilities_path.replace("\\", "/").rstrip("/")
        if not path.endswith("/site-packages/qs/capabilities"):
            return False
        self.capabilities_path = None
        return True


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    work_root: str = Field(frozen=True)
    created_at: float = Field(default_factory=time.time)
    settings: ProjectSettings = Field(default_factory=ProjectSettings)

    @property
    def name(self) -> str:
        return Path(self.work_root).name or self.work_root


class ChatSettings(ProjectSettings):
    """Execution snapshot with conversation-local approvals and evaluator history."""

    agent_override: bool = False
    work_root: str | None = None
    approved_commands_history: list[str] = Field(default_factory=list)
    tool_approval_raw_history: list[dict[str, Any]] = Field(default_factory=list)

    def with_project(self, project: Project) -> ChatSettings:
        """Resolve shared options without changing this chat's approval state."""
        agent_settings = self if self.agent_override else project.settings
        resolved = self.with_agent(
            agent_settings.agent_profile, agent_settings.agent_design
        )
        return resolved.model_copy(
            update={
                "name_hds": project.settings.name_hds,
                "capabilities_path": project.settings.capabilities_path,
                "coding_agent_persistent_history": project.settings.coding_agent_persistent_history,
                "work_root": project.work_root,
            }
        )

    def with_agent_override(
        self, profile: AgentProfile, design: AgentDesign | None
    ) -> ChatSettings:
        settings = self.with_agent(profile, design)
        settings.agent_override = True
        return settings

    def set_tool_approval_mode(self, mode: str) -> None:
        self.tool_approval_mode = normalize_tool_approval_mode(mode)

    def clear_command_approvals(self) -> None:
        """Discard evaluator history and cached decisions from an obsolete branch."""
        self.approved_commands_history = []
        self.tool_approval_raw_history = []

    def record_command_approval(
        self, command: str, raw_history: list[dict[str, str]]
    ) -> None:
        """Keep command summaries and evaluator context together without duplicates."""
        if command not in self.approved_commands_history:
            self.approved_commands_history.append(command)
        self.tool_approval_raw_history = [dict(message) for message in raw_history]
