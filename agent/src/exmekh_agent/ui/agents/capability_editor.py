"""Edit capability prose and source code without owning the agent board."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from uuid import uuid4

from nicegui import ui

from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.state.agent_design import AgentDesign, DesignNode
from exmekh_agent.ui.resources import ui_text


class CapabilityEditor:
    def __init__(
        self,
        capabilities: CapabilityRepository,
        on_saved: Callable[[str, DesignNode], None],
    ) -> None:
        self._capabilities = capabilities
        self._on_saved = on_saved
        self._saving = False

    def open(self, design: AgentDesign, node_id: str) -> None:
        self._design = design.model_copy(deep=True)
        self._node = self._design.node(node_id)
        self._client = ui.context.client
        with (
            ui.dialog() as self._dialog,
            ui.card().classes(
                "w-[760px] max-w-[96vw] max-h-[92vh] overflow-y-auto bg-zinc-900"
            ),
        ):
            ui.label(ui_text("agents.capability_editor.edit_capability_copy")).classes(
                "text-lg"
            )
            self._build_fields()
            if self._node.kind != "tool":
                self._render_dependencies()
            ui.label(
                ui_text(
                    "agents.capability_editor.the_original_remains_available_in_capabilities_the"
                )
            ).classes("text-xs text-zinc-400")
            with ui.row().classes("justify-end w-full"):
                self._cancel_button = ui.button(
                    ui_text("agents.capability_editor.cancel"),
                    on_click=self._dialog.close,
                ).props("flat")
                self._save_button = ui.button(
                    ui_text("agents.capability_editor.save_copy"), on_click=self._save
                )
        self._dialog.open()

    def _build_fields(self) -> None:
        values = (
            self._capabilities.editable_values(self._node)
            if self._node.kind != "tool"
            else None
        )
        self._name = (
            ui.input(
                "New capability name",
                value=f"{self._node.name}-custom-{uuid4().hex[:4]}",
            )
            .props("outlined dense")
            .classes("w-full")
        )
        self._description = (
            ui.textarea("Description", value=values.description)
            .props("outlined autogrow")
            .classes("w-full")
            if values
            else None
        )
        self._instructions = (
            ui.textarea(
                "Instructions" if values else "Source code",
                value=values.instructions if values else self._node.content,
            )
            .props("outlined autogrow")
            .classes("w-full font-mono text-xs")
        )

    def _render_dependencies(self) -> None:
        for kind, label in (("skill", "Sub-skills"), ("tool", "Sub-tools")):
            dependencies = (
                ", ".join(
                    item.name
                    for item in self._design.children(self._node.id, kind=kind)
                )
                or "None"
            )
            ui.label(f"{label}: {dependencies}").classes(
                "text-xs text-zinc-300 break-all"
            )
        ui.label(
            ui_text(
                "agents.capability_editor.dependencies_are_controlled_by_board_connections_change"
            )
        ).classes("text-xs text-sky-300")

    def _set_saving(self, saving: bool) -> None:
        self._saving = saving
        for control in [
            self._name,
            self._instructions,
            self._description,
            self._save_button,
            self._cancel_button,
        ]:
            if control is not None:
                control.set_enabled(not saving)
        if saving:
            self._dialog.props("persistent")
        else:
            self._dialog.props(remove="persistent")

    async def _save(self) -> None:
        if self._saving:
            return
        self._set_saving(True)
        try:
            operation = (
                partial(
                    self._capabilities.copy_capability,
                    self._node,
                    self._name.value.strip(),
                    self._instructions.value,
                )
                if self._node.kind == "tool"
                else partial(
                    self._capabilities.copy_for_design,
                    self._design,
                    self._node.id,
                    self._name.value.strip(),
                    self._description.value,
                    self._instructions.value,
                )
            )
            replacement = await asyncio.get_running_loop().run_in_executor(
                None, operation
            )
            if not self._client.is_deleted:
                with self._client:
                    self._on_saved(self._node.id, replacement)
                    self._dialog.close()
        except (OSError, ValueError, SyntaxError) as error:
            if not self._client.is_deleted:
                with self._client:
                    ui.notify(
                        ui_text(
                            "agents.capability_editor.unable_to_save_capability",
                            error=error,
                        ),
                        type="negative",
                    )
        finally:
            self._saving = False
            if not self._client.is_deleted:
                with self._client:
                    self._set_saving(False)
