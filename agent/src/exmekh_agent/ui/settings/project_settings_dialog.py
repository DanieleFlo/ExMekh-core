from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    TOOL_APPROVAL_MODES,
    normalize_tool_approval_mode,
)
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.agents.agent_selector import AgentSelector
from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/project_settings_dialog")


class ProjectSettingsDialogComponent:
    """Edit shared options, or prepare options before creating a project."""

    def __init__(self, on_saved: Callable[[], None] | None = None) -> None:
        self.on_saved = on_saved
        self._saving = False
        self._card: ui.card | None = None
        self.dialog: ui.dialog | None = None
        self.project_id: str | None = None
        self.draft_saved: Callable[[ProjectSettings], None] | None = None
        self.input_hds: ui.input | None = None
        self.input_capabilities: ui.input | None = None
        self.switch_coding_history: ui.switch | None = None
        self.approval_select: ui.select | None = None
        self.root_label: ui.label | None = None
        self.agent_selector = AgentSelector(
            lambda: [self.input_capabilities.value or config.capabilities_path]
        )

    def build(self) -> ui.dialog:
        with (
            ui.dialog() as dialog,
            ui.card().classes(
                UI_PRESENTATION[
                    "build.classes.bg_zinc_900_text_zinc_100_p_4_w_min_90vw_440p"
                ]
            ) as self._card,
        ):
            self.dialog = dialog
            with ui.row().classes(
                UI_PRESENTATION["build.classes.items_center_justify_between_w_full"]
            ):
                ui.label(
                    ui_text("settings.project_settings_dialog.project_settings")
                ).classes(UI_PRESENTATION["build.classes.font_bold_text_sm"])
                ui.button(
                    icon=UI_PRESENTATION["build.icon.close"], on_click=self.close
                ).props(UI_PRESENTATION["build.props.flat_round_dense"])
            self.root_label = ui.label().classes(
                UI_PRESENTATION["build.classes.text_xs_text_zinc_400_break_all"]
            )
            self.input_hds = (
                ui.input(
                    ui_text("settings.project_settings_dialog.system_hds_prompt_name")
                )
                .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                .classes(UI_PRESENTATION["build.classes.w_full"])
            )
            self.input_capabilities = (
                ui.input(ui_text("settings.project_settings_dialog.capabilities_path"))
                .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                .classes(UI_PRESENTATION["build.classes.w_full"])
            )
            self.agent_selector.build()
            self.switch_coding_history = ui.switch(
                ui_text(
                    "settings.project_settings_dialog.coding_agent_preserve_history"
                )
            )
            self.approval_select = (
                ui.select(
                    TOOL_APPROVAL_MODES,
                    label=ui_text(
                        "settings.project_settings_dialog.default_bash_approval"
                    ),
                )
                .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                .classes(UI_PRESENTATION["build.classes.w_full"])
            )
            ui.label(
                ui_text(
                    "settings.project_settings_dialog.the_approval_default_applies_to_new_chats"
                )
            ).classes(UI_PRESENTATION["build.classes.text_xs_text_zinc_400"])
            ui.button(
                ui_text("settings.project_settings_dialog.save_project_settings"),
                on_click=self._save_settings,
            ).props(UI_PRESENTATION["build.props.no_caps_icon_check"]).classes(
                UI_PRESENTATION["build.classes.w_full"]
            )
        return dialog

    def open(self, project_id: str) -> None:
        project = session_manager.get_project(project_id)
        if project is None:
            return
        self.project_id = project_id
        self.draft_saved = None
        self._show(project.settings, project.work_root)

    def open_draft(
        self,
        settings: ProjectSettings,
        work_root: str,
        on_saved: Callable[[ProjectSettings], None],
    ) -> None:
        self.project_id = None
        self.draft_saved = on_saved
        self._show(settings, work_root)

    def _show(self, settings: ProjectSettings, work_root: str) -> None:
        if self.dialog is None:
            return
        self.root_label.text = work_root
        self.input_hds.value = settings.name_hds or config.name_hds
        self.input_capabilities.value = (
            settings.capabilities_path or config.capabilities_path
        )
        self.switch_coding_history.value = (
            config.coding_agent_persistent_history
            if settings.coding_agent_persistent_history is None
            else settings.coding_agent_persistent_history
        )
        self.approval_select.value = normalize_tool_approval_mode(
            settings.tool_approval_mode
        )
        self.agent_selector.set_value(settings.agent_profile, settings.agent_design)
        self.dialog.open()

    def close(self) -> None:
        if self.dialog:
            self.dialog.close()

    async def _save_settings(self) -> None:
        if self.dialog is None or self._saving:
            return
        client = ui.context.client
        settings = ProjectSettings(
            name_hds=self.input_hds.value,
            capabilities_path=self.input_capabilities.value,
            coding_agent_persistent_history=self.switch_coding_history.value,
            tool_approval_mode=self.approval_select.value,
        )
        settings = self.agent_selector.apply_to(settings)
        self._saving = True
        self.dialog.props(UI_PRESENTATION["save_settings.props.persistent"])
        self._card.props(UI_PRESENTATION["save_settings.props.inert"])
        try:
            if self.draft_saved:
                self.draft_saved(settings)
            elif (
                self.project_id is None
                or not await session_manager.save_project_settings_async(
                    self.project_id, settings
                )
            ):
                return
            if not client.is_deleted:
                with client:
                    if self.on_saved:
                        self.on_saved()
                    ui.notify(
                        ui_text(
                            "settings.project_settings_dialog.project_settings_saved"
                        ),
                        type="positive",
                    )
                    self.close()
        except OSError as error:
            if not client.is_deleted:
                with client:
                    ui.notify(
                        ui_text(
                            "settings.project_settings_dialog.unable_to_save_project_settings",
                            error=error,
                        ),
                        type="negative",
                    )
        finally:
            self._saving = False
            if not client.is_deleted:
                with client:
                    self.dialog.props(remove="persistent")
                    self._card.props(remove="inert")
