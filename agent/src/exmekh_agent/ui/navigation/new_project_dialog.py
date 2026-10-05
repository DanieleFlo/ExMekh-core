from __future__ import annotations

import asyncio
import tkinter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import filedialog

from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.session_manager import CODE_WORK_ROOT, session_manager
from exmekh_agent.ui.agents.agent_selector import AgentSelector
from exmekh_agent.ui.resources import ui_text
from exmekh_agent.ui.settings.project_settings_dialog import (
    ProjectSettingsDialogComponent,
)


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
            title="Choose project folder",
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
            ui.card().classes("bg-zinc-900 text-zinc-100 p-4 w-[min(90vw,480px)]"),
        ):
            ui.label(ui_text("navigation.new_project_dialog.new_project")).classes(
                "font-bold text-base"
            )
            self.folder_input = (
                ui.input("Project folder", value=str(CODE_WORK_ROOT))
                .props("outlined dense dark")
                .classes("w-full")
            )
            ui.button(
                ui_text("navigation.new_project_dialog.browse_folders"),
                icon="folder_open",
                on_click=self.browse,
            ).props("no-caps")
            ui.label(
                ui_text(
                    "navigation.new_project_dialog.the_project_uses_the_folder_name_its"
                )
            ).classes("text-xs text-zinc-400")
            self.agent_selector.build()
            ui.button(
                ui_text("navigation.new_project_dialog.project_settings"),
                icon="settings",
                on_click=self.open_settings,
            ).props("flat no-caps")
            with ui.row().classes("w-full justify-end"):
                ui.button(
                    ui_text("navigation.new_project_dialog.cancel"),
                    on_click=self.dialog.close,
                ).props("flat no-caps")
                self.create_button = ui.button(
                    ui_text("navigation.new_project_dialog.create_project"),
                    on_click=self.create,
                ).props("no-caps")

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
        self.create_button.props("loading")
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
