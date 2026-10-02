from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.exmekh_bridge.tool_approval import (
    TOOL_APPROVAL_MODES,
    normalize_tool_approval_mode,
)
from exmekh_agent.state.session_manager import session_manager


class ProjectSettingsDialogComponent:
    """Edit shared options, or prepare options before creating a project."""

    def __init__(self, on_saved: Callable[[], None] | None = None) -> None:
        self.on_saved = on_saved
        self.dialog: ui.dialog | None = None
        self.project_id: str | None = None
        self.draft_saved: Callable[[ProjectSettings], None] | None = None
        self.input_hds: ui.input | None = None
        self.input_capabilities: ui.input | None = None
        self.switch_coding_history: ui.switch | None = None
        self.approval_select: ui.select | None = None
        self.root_label: ui.label | None = None

    def build(self) -> ui.dialog:
        with (
            ui.dialog() as dialog,
            ui.card().classes("bg-zinc-900 text-zinc-100 p-4 w-[min(90vw,440px)]"),
        ):
            self.dialog = dialog
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Project settings").classes("font-bold text-sm")
                ui.button(icon="close", on_click=self.close).props("flat round dense")
            self.root_label = ui.label().classes("text-xs text-zinc-400 break-all")
            self.input_hds = (
                ui.input("System HDS Prompt Name")
                .props("outlined dense dark")
                .classes("w-full")
            )
            self.input_capabilities = (
                ui.input("Capabilities Path")
                .props("outlined dense dark")
                .classes("w-full")
            )
            self.switch_coding_history = ui.switch("Coding agent: preserve history")
            self.approval_select = (
                ui.select(TOOL_APPROVAL_MODES, label="Default bash approval")
                .props("outlined dense dark")
                .classes("w-full")
            )
            ui.label(
                "The approval default applies to new chats. Each chat can change its own approval mode."
            ).classes("text-xs text-zinc-400")
            ui.button("Save project settings", on_click=self._save_settings).props(
                "no-caps icon=check"
            ).classes("w-full")
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
        self.dialog.open()

    def close(self) -> None:
        if self.dialog:
            self.dialog.close()

    def _save_settings(self) -> None:
        if self.dialog is None:
            return
        settings = ProjectSettings(
            name_hds=self.input_hds.value,
            capabilities_path=self.input_capabilities.value,
            coding_agent_persistent_history=self.switch_coding_history.value,
            tool_approval_mode=self.approval_select.value,
        )
        try:
            if self.draft_saved:
                self.draft_saved(settings)
            elif self.project_id is None or not session_manager.save_project_settings(
                self.project_id, settings
            ):
                return
        except OSError as error:
            ui.notify(f"Unable to save project settings: {error}", type="negative")
            return
        if self.on_saved:
            self.on_saved()
        ui.notify("Project settings saved", type="positive")
        self.close()
