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
from exmekh_agent.ui.resources import presentation_config, ui_text

UI_PRESENTATION = presentation_config("controls/capability_editor")


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
            self._client,
            ui.dialog() as self._dialog,
            ui.card().classes(
                UI_PRESENTATION[
                    "open.classes.w_760px_max_w_96vw_max_h_92vh_overflow_y_auto"
                ]
            ),
        ):
            ui.label(ui_text("agents.capability_editor.edit_capability_copy")).classes(
                UI_PRESENTATION["open.classes.text_lg"]
            )
            self._build_fields()
            if self._node.kind != "tool":
                self._render_dependencies()
            ui.label(
                ui_text(
                    "agents.capability_editor.the_original_remains_available_in_capabilities_the"
                )
            ).classes(UI_PRESENTATION["open.classes.text_xs_text_zinc_400"])
            with ui.row().classes(UI_PRESENTATION["open.classes.justify_end_w_full"]):
                self._cancel_button = ui.button(
                    ui_text("agents.capability_editor.cancel"),
                    on_click=self._dialog.close,
                ).props(UI_PRESENTATION["open.props.flat"])
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
                ui_text("agents.capability_editor.new_capability_name"),
                value=f"{self._node.name}-custom-{uuid4().hex[:4]}",
            )
            .props(UI_PRESENTATION["build_fields.props.outlined_dense"])
            .classes(UI_PRESENTATION["build_fields.classes.w_full"])
        )
        self._description = (
            ui.textarea(
                ui_text("agents.capability_editor.description"),
                value=values.description,
            )
            .props(UI_PRESENTATION["build_fields.props.outlined_autogrow"])
            .classes(UI_PRESENTATION["build_fields.classes.w_full"])
            if values
            else None
        )
        self._instructions = (
            ui.textarea(
                ui_text("agents.capability_editor.instructions")
                if values
                else ui_text("agents.capability_editor.source_code"),
                value=values.instructions if values else self._node.content,
            )
            .props(UI_PRESENTATION["build_fields.props.outlined_autogrow"])
            .classes(UI_PRESENTATION["build_fields.classes.w_full_font_mono_text_xs"])
        )

    def _render_dependencies(self) -> None:
        for kind, label in (
            ("skill", ui_text("agents.capability_editor.sub_skills")),
            ("tool", ui_text("agents.capability_editor.sub_tools")),
        ):
            dependencies = ", ".join(
                item.name for item in self._design.children(self._node.id, kind=kind)
            ) or ui_text("agents.capability_editor.no_dependencies")
            ui.label(
                ui_text(
                    "agents.capability_editor.value",
                    value_1=label,
                    value_2=dependencies,
                )
            ).classes(
                UI_PRESENTATION[
                    "render_dependencies.classes.text_xs_text_zinc_300_break_all"
                ]
            )
        ui.label(
            ui_text(
                "agents.capability_editor.dependencies_are_controlled_by_board_connections_change"
            )
        ).classes(UI_PRESENTATION["render_dependencies.classes.text_xs_text_sky_300"])

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
            self._dialog.props(UI_PRESENTATION["set_saving.props.persistent"])
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
