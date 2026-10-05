from __future__ import annotations

import asyncio
import tkinter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import filedialog

from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.session_manager import CODE_WORK_ROOT, session_manager
from exmekh_agent.ui.agents.agent_selector import AgentSelector
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.settings.project_settings_dialog import (
    ProjectSettingsDialogComponent,
)

UI_PRESENTATION = presentation_config("controls/new_project_dialog")


def pick_folder_dialog(initial_dir: str | Path | None = None) -> str | None:
    """Open the native picker and always release its Tk window."""
    initial_path = Path(initial_dir).expanduser() if initial_dir else CODE_WORK_ROOT
    if not initial_path.is_dir():
        initial_path = initial_path.parent
    initial_directory = str(initial_path.resolve()) if initial_path.is_dir() else None
    root = tkinter.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=initial_directory,
            title=app_text("navigation.new_project_dialog.choose_project_folder"),
            mustexist=True,
        )
        return selected or None
    finally:
        root.destroy()


class NewProjectDialogComponent:
    def __init__(
        self,
        on_created: Callable[[str], None],
        settings_dialog: ProjectSettingsDialogComponent,
    ) -> None:
        self.on_created = on_created
        self.settings_dialog = settings_dialog
        self.settings = ProjectSettings()
        self.agent_selector = AgentSelector(
            lambda: [self.settings.capabilities_path or config.capabilities_path]
        )
        self.dialog: ui.dialog | None = None
        self.folder_input: ui.input | None = None
        self.create_button: ui.button | None = None

    def build(self) -> None:
        with (
            ui.dialog() as self.dialog,
            ui.card().classes(
                UI_PRESENTATION[
                    "build.classes.bg_zinc_900_text_zinc_100_p_4_w_min_90vw_480p"
                ]
            ),
        ):
            ui.label(ui_text("navigation.new_project_dialog.new_project")).classes(
                UI_PRESENTATION["build.classes.font_bold_text_base"]
            )
            self.folder_input = (
                ui.input(
                    ui_text("navigation.new_project_dialog.project_folder"),
                    value=str(CODE_WORK_ROOT),
                )
                .props(UI_PRESENTATION["build.props.outlined_dense_dark"])
                .classes(UI_PRESENTATION["build.classes.w_full"])
            )
            ui.button(
                ui_text("navigation.new_project_dialog.browse_folders"),
                icon=UI_PRESENTATION["build.icon.folder_open"],
                on_click=self.browse,
            ).props(UI_PRESENTATION["build.props.no_caps"])
            ui.label(
                ui_text(
                    "navigation.new_project_dialog.the_project_uses_the_folder_name_its"
                )
            ).classes(UI_PRESENTATION["build.classes.text_xs_text_zinc_400"])
            self.agent_selector.build()
            ui.button(
                ui_text("navigation.new_project_dialog.project_settings"),
                icon=UI_PRESENTATION["build.icon.settings"],
                on_click=self.open_settings,
            ).props(UI_PRESENTATION["build.props.flat_no_caps"])
            with ui.row().classes(UI_PRESENTATION["build.classes.w_full_justify_end"]):
                ui.button(
                    ui_text("navigation.new_project_dialog.cancel"),
                    on_click=self.dialog.close,
                ).props(UI_PRESENTATION["build.props.flat_no_caps"])
                self.create_button = ui.button(
                    ui_text("navigation.new_project_dialog.create_project"),
                    on_click=self.create,
                ).props(UI_PRESENTATION["build.props.no_caps"])

    def open(self) -> None:
        self.settings = ProjectSettings()
        self.agent_selector.set_value("generic", None)
        self.folder_input.value = str(CODE_WORK_ROOT)
        self.dialog.open()

    def open_settings(self) -> None:
        self.settings = self.agent_selector.apply_to(self.settings)
        self.settings_dialog.open_draft(
            self.settings, self.folder_input.value, self._set_settings
        )

    def _set_settings(self, settings: ProjectSettings) -> None:
        self.settings = settings
        self.agent_selector.set_value(settings.agent_profile, settings.agent_design)

    async def browse(self) -> None:
        client = ui.context.client
        try:
            selected = await asyncio.get_running_loop().run_in_executor(
                None, pick_folder_dialog, self.folder_input.value
            )
        except (OSError, tkinter.TclError) as error:
            if not client.is_deleted:
                with client:
                    ui.notify(
                        ui_text(
                            "navigation.new_project_dialog.unable_to_open_folder_picker",
                            error=error,
                        ),
                        type="negative",
                    )
            return
        if selected and not client.is_deleted:
            with client:
                self.folder_input.value = str(selected)

    async def create(self) -> None:
        client = ui.context.client
        self.settings = self.agent_selector.apply_to(self.settings)
        self.create_button.props(UI_PRESENTATION["create.props.loading"])
        try:
            project = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    session_manager.create_project,
                    self.folder_input.value,
                    self.settings.model_copy(deep=True),
                ),
            )
        except (OSError, ValueError) as error:
            if not client.is_deleted:
                with client:
                    ui.notify(
                        ui_text(
                            "navigation.new_project_dialog.unable_to_create_project",
                            error=error,
                        ),
                        type="negative",
                    )
        else:
            if not client.is_deleted:
                with client:
                    self.dialog.close()
                    self.on_created(project.id)
        finally:
            if not client.is_deleted:
                with client:
                    self.create_button.props(remove="loading")
