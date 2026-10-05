"""Profile picker shared by project defaults and chat overrides."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from nicegui import ui

from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.agent_design import AGENT_PROFILES, AgentDesign, AgentProfile
from exmekh_agent.ui.agents.agent_designer import AgentDesigner
from exmekh_agent.ui.resources import ui_text

Settings = TypeVar("Settings", bound=ProjectSettings)


class AgentSelector:
    def __init__(self, paths: Callable[[], list[str]]) -> None:
        self._paths = paths
        self._profile: AgentProfile = "generic"
        self._design: AgentDesign | None = None
        self._loading = False
        self._designer = AgentDesigner()
        self._select: ui.select | None = None

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    def design_snapshot(self) -> AgentDesign | None:
        return self._design.model_copy(deep=True) if self._design else None

    def apply_to(self, settings: Settings) -> Settings:
        return settings.with_agent(self.profile, self.design_snapshot())

    def build(self) -> None:
        self._designer.build()
        self._select = (
            ui.select(
                AGENT_PROFILES,
                value=self._profile,
                label=ui_text("agents.agent_selector.agent_type"),
                on_change=self._changed,
            )
            .props("outlined dense dark")
            .classes("w-full")
        )
        ui.button(
            ui_text("agents.agent_selector.customize_agent"),
            icon="account_tree",
            on_click=self.customize,
        ).props("flat no-caps").classes("w-full")

    def set_value(self, profile: AgentProfile, design: AgentDesign | None) -> None:
        self._profile = profile
        self._design = design.model_copy(deep=True) if design else None
        self._restore_selection()

    def _restore_selection(self) -> None:
        if self._select is None:
            return
        self._loading = True
        try:
            self._select.value = self.profile
        finally:
            self._loading = False

    def _changed(self, event) -> Awaitable[None] | None:
        if self._loading:
            return
        if event.value == "custom":
            return self.customize()
        else:
            self._profile = event.value
            self._design = None

    async def customize(self) -> None:
        await self._designer.open(
            self.design_snapshot(),
            self._saved,
            self._paths(),
            self.profile,
            on_cancelled=self._restore_selection,
        )

    def _saved(self, design: AgentDesign) -> None:
        self.set_value("custom", design)
