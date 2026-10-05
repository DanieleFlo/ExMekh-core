"""Own additional folder references and keep the default capabilities immutable."""

from __future__ import annotations

import asyncio
import tkinter
from functools import partial

from nicegui import ui

from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.shared.folder_picker import pick_folder_dialog
from exmekh_agent.utils.paths import normalize_paths

PRESENTATION = presentation_config("controls/capability_paths_editor")


def browse_capability_folder(initial_dir: str) -> str | None:
    selected = pick_folder_dialog(initial_dir)
    return CapabilityRepository.folder_reference(selected) if selected else None


class CapabilityPathsEditor:
    def __init__(self, default_path: str) -> None:
        self.default_path = normalize_paths([default_path])[0]
        self._paths: list[str] = []
        self._rows: ui.column | None = None
        self._browse_button: ui.button | None = None
        self._browsing = False

    def all_paths(self) -> list[str]:
        return [self.default_path, *self._paths]

    def additional_paths(self) -> list[str]:
        return list(self._paths)

    def set_paths(self, paths: list[str]) -> None:
        self._paths = [
            path for path in normalize_paths(paths) if path != self.default_path
        ]
        if self._rows is not None:
            self._render_paths()

    def build(self) -> None:
        with ui.expansion(
            ui_text("settings.capability_paths.title"),
            icon=PRESENTATION["section_icon"],
        ).classes(PRESENTATION["section"]):
            ui.input(
                ui_text("settings.capability_paths.default"), value=self.default_path
            ).props(PRESENTATION["readonly"]).classes(PRESENTATION["input"]).tooltip(
                ui_text("settings.capability_paths.default_hint")
            )
            self._rows = ui.column().classes(PRESENTATION["rows"])
            self._render_paths()
            self._browse_button = (
                ui.button(
                    ui_text("settings.capability_paths.add"),
                    icon=PRESENTATION["browse_icon"],
                    on_click=self.browse,
                )
                .props(PRESENTATION["browse_props"])
                .classes(PRESENTATION["browse_class"])
            )

    def _render_paths(self) -> None:
        self._rows.clear()
        with self._rows:
            for path in self._paths:
                with ui.row().classes(PRESENTATION["row"]):
                    ui.input(
                        ui_text("settings.capability_paths.additional"), value=path
                    ).props(PRESENTATION["readonly"]).classes(
                        PRESENTATION["additional_input"]
                    ).tooltip(path)
                    ui.button(
                        icon=PRESENTATION["remove_icon"],
                        on_click=partial(self._remove, path),
                    ).props(PRESENTATION["remove_props"]).tooltip(
                        ui_text("settings.capability_paths.remove")
                    ).props(
                        f'aria-label="{ui_text("settings.capability_paths.remove")}"'
                    )

    def _remove(self, path: str) -> None:
        self._paths.remove(path)
        self._render_paths()

    async def browse(self) -> None:
        if self._browsing:
            return
        client = ui.context.client
        self._browsing = True
        self._browse_button.disable()
        self._browse_button.props(PRESENTATION["loading"])
        try:
            selected = await asyncio.get_running_loop().run_in_executor(
                None,
                browse_capability_folder,
                self._paths[-1] if self._paths else self.default_path,
            )
            if selected and not client.is_deleted:
                with client:
                    self.set_paths([*self._paths, selected])
        except (OSError, ValueError, tkinter.TclError) as error:
            if not client.is_deleted:
                with client:
                    ui.notify(
                        ui_text("settings.capability_paths.browse_failed", error=error),
                        type="negative",
                    )
        finally:
            self._browsing = False
            if not client.is_deleted:
                with client:
                    self._browse_button.enable()
                    self._browse_button.props(remove="loading")
