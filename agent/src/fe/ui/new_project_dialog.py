from __future__ import annotations

import asyncio
import tkinter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import filedialog

from nicegui import ui

from fe.exmekh_bridge.models import ProjectSettings
from fe.state.session_manager import CODE_WORK_ROOT, session_manager
from fe.ui.project_settings_dialog import ProjectSettingsDialogComponent


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
        self.dialog: ui.dialog | None = None
        self.folder_input: ui.input | None = None
        self.create_button: ui.button | None = None

    def build(self) -> None:
        with (
            ui.dialog() as self.dialog,
            ui.card().classes("bg-zinc-900 text-zinc-100 p-4 w-[min(90vw,480px)]"),
        ):
            ui.label("New project").classes("font-bold text-base")
            self.folder_input = (
                ui.input("Project folder", value=str(CODE_WORK_ROOT))
                .props("outlined dense dark")
                .classes("w-full")
            )
            ui.button("Browse folders", icon="folder_open", on_click=self.browse).props(
                "no-caps"
            )
            ui.label(
                "The project uses the folder name. Its folder cannot be changed later."
            ).classes("text-xs text-zinc-400")
            ui.button(
                "Project settings", icon="settings", on_click=self.open_settings
            ).props("flat no-caps")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=self.dialog.close).props("flat no-caps")
                self.create_button = ui.button(
                    "Create project", on_click=self.create
                ).props("no-caps")

    def open(self) -> None:
        self.settings = ProjectSettings()
        self.folder_input.value = str(CODE_WORK_ROOT)
        self.dialog.open()

    def open_settings(self) -> None:
        self.settings_dialog.open_draft(
            self.settings, self.folder_input.value, self._set_settings
        )

    def _set_settings(self, settings: ProjectSettings) -> None:
        self.settings = settings

    async def browse(self) -> None:
        client = ui.context.client
        try:
            selected = await asyncio.get_running_loop().run_in_executor(
                None, pick_folder_dialog, self.folder_input.value
            )
        except (OSError, tkinter.TclError) as error:
            if not client.is_deleted:
                with client:
                    ui.notify(f"Unable to open folder picker: {error}", type="negative")
            return
        if selected and not client.is_deleted:
            with client:
                self.folder_input.value = str(selected)

    async def create(self) -> None:
        client = ui.context.client
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
                    ui.notify(f"Unable to create project: {error}", type="negative")
        else:
            if not client.is_deleted:
                with client:
                    self.dialog.close()
                    self.on_created(project.id)
        finally:
            if not client.is_deleted:
                with client:
                    self.create_button.props(remove="loading")
