"""Profile picker shared by project defaults and chat overrides."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TypeVar

from nicegui import ui

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.agent_design import AGENT_PROFILES, AgentDesign, AgentProfile
from exmekh_agent.ui.agents.agent_designer import AgentDesigner
from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/agent_selector")

Settings = TypeVar(app_text("agents.agent_selector.settings"), bound=ProjectSettings)


class AgentSelector:
    def __init__(self, paths: Callable[[], list[str]]) -> None:
        self._paths = paths
        self._profile: AgentProfile = "generic"
        self._design: AgentDesign | None = None
        self._designer = AgentDesigner()
        self._buttons: dict[AgentProfile, ui.button] = {}

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    def design_snapshot(self) -> AgentDesign | None:
        return self._design.model_copy(deep=True) if self._design else None

    def apply_to(self, settings: Settings) -> Settings:
        return settings.with_agent(self.profile, self.design_snapshot())

    def build(self) -> None:
        self._designer.build()
        ui.label(ui_text("agents.agent_selector.agent_type")).classes(
            UI_PRESENTATION["build.classes.heading"]
        )
        with (
            ui.row()
            .classes(UI_PRESENTATION["build.classes.profile_group"])
            .props(
                f'role=group aria-label="{ui_text("agents.agent_selector.agent_type")}"'
            )
        ):
            for profile, label in AGENT_PROFILES.items():
                presentation = UI_PRESENTATION["profiles"][profile]
                self._buttons[profile] = (
                    ui.button(
                        icon=presentation["icon"],
                        on_click=partial(self._choose_profile, profile),
                    )
                    .props(UI_PRESENTATION["build.props.profile_option"])
                    .props(f'aria-label="{label}"')
                    .classes(UI_PRESENTATION["build.classes.profile_option"])
                    .tooltip(ui_text(presentation["hint"]))
                )
        self._restore_selection()

    def set_value(self, profile: AgentProfile, design: AgentDesign | None) -> None:
        self._profile = profile
        self._design = design.model_copy(deep=True) if design else None
        self._restore_selection()

    def _restore_selection(self) -> None:
        for profile, button in self._buttons.items():
            selected = profile == self.profile
            button.props(f"aria-pressed={str(selected).lower()}")
            if selected:
                button.classes(add=UI_PRESENTATION["build.classes.profile_selected"])
            else:
                button.classes(remove=UI_PRESENTATION["build.classes.profile_selected"])

    async def _choose_profile(self, profile: AgentProfile) -> None:
        if profile == "custom":
            await self.customize()
        else:
            self._profile = profile
            self._design = None
            self._restore_selection()

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
