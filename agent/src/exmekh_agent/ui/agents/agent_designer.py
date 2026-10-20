"""Visual editor for agent hierarchies and copied capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from pathlib import Path
from uuid import uuid4

from nicegui import ui

from exmekh_agent.exmekh_bridge.capabilities.agent_design import validate_agent_design
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.state.agent_design import AgentDesign, AgentProfile, DesignNode
from exmekh_agent.state.design_history import DesignHistory, DesignSnapshot
from exmekh_agent.state.design_selection import DesignSelection
from exmekh_agent.ui.agents.capability_editor import CapabilityEditor
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text

BOARD_DEFAULTS = presentation_config("agent_board")

UI_PRESENTATION = presentation_config("controls/agent_designer")


class AgentDesigner:
    def __init__(self) -> None:
        self._capabilities = CapabilityRepository()
        self._dialog: ui.dialog | None = None
        self._design = AgentDesign()
        self._catalog: list[DesignNode] = []
        self._selection = DesignSelection()
        self._on_cancelled: Callable[[], None] | None = None
        self._on_saved: Callable[[AgentDesign], None] | None = None
        self._board: ui.html | None = None
        self._details: ui.column | None = None
        self._scope = uuid4().hex
        self._saving = False
        self._history = DesignHistory()
        self._undo_button: ui.button | None = None
        self._shell: ui.card | None = None
        self._save_button: ui.button | None = None
        self._refresh_timer: ui.timer | None = None
        self._copied_sources: set[str] = set()
        self._generation = 0
        self._refreshing = False

    def build(self) -> None:
        script = (
            Path(__file__).parents[1] / "resources" / "js" / "agent_board.js"
        ).read_text(encoding="utf-8")
        ui.add_head_html(f"<script>{script}</script>")
        ui.on("agent_board_action", self._handle_action)
        with (
            ui.dialog()
            .props(UI_PRESENTATION["build.props.maximized"])
            .classes(
                UI_PRESENTATION["build.classes.agent_designer_dialog"]
            ) as self._dialog
        ):
            with ui.card().classes(
                UI_PRESENTATION[
                    "build.classes.agent_designer_shell_w_full_h_full_bg_zinc_95"
                ]
            ) as self._shell:
                with ui.row().classes(
                    UI_PRESENTATION["build.classes.w_full_items_center_justify_between"]
                ):
                    ui.label(ui_text("agents.agent_designer.customize_agent")).classes(
                        UI_PRESENTATION["build.classes.text_lg_font_semibold"]
                    )
                    with ui.row().classes(UI_PRESENTATION["build.classes.gap_2"]):
                        self._undo_button = (
                            ui.button(
                                ui_text("agents.agent_designer.undo"),
                                icon=UI_PRESENTATION["build.icon.undo"],
                                on_click=self._undo,
                            )
                            .props(UI_PRESENTATION["build.props.flat"])
                            .tooltip(ui_text("agents.agent_designer.undo_shortcut"))
                        )
                        self._undo_button.disable()
                        ui.button(
                            ui_text("agents.agent_designer.arrange"),
                            icon="auto_fix_high",
                            on_click=lambda: self._change_design(self._design.arrange),
                        ).props(UI_PRESENTATION["build.props.flat"])
                        ui.button(
                            ui_text("agents.agent_designer.new_hds"),
                            icon=UI_PRESENTATION["build.icon.add"],
                            on_click=self._new_hds,
                        ).props(UI_PRESENTATION["build.props.flat"])
                        ui.button(
                            ui_text("agents.agent_designer.cancel"),
                            on_click=self._dialog.close,
                        ).props(UI_PRESENTATION["build.props.flat"])
                        self._save_button = ui.button(
                            ui_text("agents.agent_designer.save_agent"),
                            on_click=self._save,
                        ).props(UI_PRESENTATION["build.props.unelevated"])
                ui.label(
                    ui_text("agents.agent_designer.select_an_hds_or_skill_to_connect")
                ).classes(UI_PRESENTATION["build.classes.text_xs_text_zinc_400"])
                with ui.row().classes(
                    UI_PRESENTATION[
                        "build.classes.agent_designer_layout_w_full_flex_1_min_h_0_f"
                    ]
                ):
                    self._board = ui.html("").classes(
                        UI_PRESENTATION[
                            "build.classes.agent_board_html_flex_1_min_w_0_h_full"
                        ]
                    )
                    self._details = ui.column().classes(
                        UI_PRESENTATION["build.classes.agent_node_details_gap_3"]
                    )

        self._dialog.on("hide", self._cancelled)
        self._refresh_timer = ui.timer(3, self._refresh_capabilities, active=False)

    def _cancelled(self) -> None:
        self._generation += 1
        self._refresh_timer.deactivate()
        if self._on_cancelled:
            self._on_cancelled()
            self._on_cancelled = None

    async def open(
        self,
        design: AgentDesign | None,
        on_saved: Callable[[AgentDesign], None],
        paths: list[str] | None = None,
        profile: AgentProfile = "generic",
        *,
        on_cancelled: Callable[[], None] | None = None,
    ) -> None:
        client = ui.context.client
        self._generation += 1
        generation = self._generation
        self._refresh_timer.deactivate()
        source_design = (
            design.model_copy(deep=True, update={"capability_paths": list(paths)})
            if design is not None and paths is not None
            else design
        )
        try:
            catalog = await asyncio.get_running_loop().run_in_executor(
                None, partial(self._capabilities.catalog, paths)
            )
            updated_design = (
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    partial(self._capabilities.expand_skill_connections, source_design),
                )
                if source_design
                else await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._capabilities.default_design, profile, paths)
                )
            )
            updated_design = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(self._capabilities.refresh_design, updated_design, catalog),
            )
        except (OSError, ValueError, SyntaxError) as error:
            if client.is_deleted or generation != self._generation:
                return
            with client:
                ui.notify(
                    ui_text(
                        "agents.agent_designer.unable_to_open_agent_capabilities",
                        error=error,
                    ),
                    type="negative",
                )
                if on_cancelled:
                    on_cancelled()
            return
        if client.is_deleted or generation != self._generation:
            return
        self._catalog = catalog
        self._design = updated_design
        self._on_saved = on_saved
        self._on_cancelled = on_cancelled
        self._select_nodes([self._design.root_id])
        self._history.clear()
        self._copied_sources.clear()
        with client:
            self._render()
            self._dialog.open()
            self._refresh_timer.activate()

    async def _refresh_capabilities(self) -> None:
        if self._saving or self._refreshing or not self._dialog.value:
            return
        client = self._dialog.client
        generation = self._generation
        before = self._design.model_copy(deep=True)
        catalog = [item.model_copy(deep=True) for item in self._catalog]
        copies = set(self._copied_sources)
        self._refreshing = True
        try:
            refresh = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(self._capabilities.refresh_from_disk, before, catalog, copies),
            )
        except (OSError, ValueError, SyntaxError, TypeError):
            # Retry on the next timer tick if a file is currently being edited.
            return
        finally:
            self._refreshing = False
        if (
            refresh is None
            or client.is_deleted
            or not self._dialog.value
            or generation != self._generation
            or self._saving
            or self._design != before
            or self._catalog != catalog
        ):
            return
        with client:
            self._catalog = refresh.catalog
            self._design = refresh.design
            self._history.clear()
            self._render()

    def _render(self) -> None:
        self._selection = self._selection.retain_existing(self._design)
        if self._undo_button:
            self._undo_button.set_enabled(self._history.can_undo)
        self._render_board()
        self._details.clear()
        with self._details:
            for warning in self._capabilities.dependency_warnings(self._design):
                ui.label(warning).classes(UI_PRESENTATION["dependency_warning.classes"])
            if len(self._selection.node_ids) > 1:
                ui.label(
                    ui_text(
                        "agents.agent_designer.selected_count",
                        count=len(self._selection.node_ids),
                    )
                )
                ui.button(
                    ui_text("agents.agent_designer.disconnect_selected"),
                    icon="link_off",
                    on_click=self._disconnect_selected,
                ).props(UI_PRESENTATION["collective.props"])
                removable = set(self._selection.node_ids) - {self._design.root_id}
                button = ui.button(
                    ui_text("agents.agent_designer.remove_selected"),
                    icon="delete_outline",
                    on_click=self._remove_selected,
                ).props(UI_PRESENTATION["remove_selected.props"])
                button.set_enabled(bool(removable))
            elif self._selection.primary_id:
                self._render_details(self._design.node(self._selection.primary_id))

    def _change_design(self, change: Callable[[], None]) -> None:
        before = DesignSnapshot.capture(self._design, self._catalog, self._selection)
        previous_warnings = set(self._capabilities.dependency_warnings(self._design))
        change()
        self._history.record(before, self._design, self._catalog)
        if (
            before.design != self._design
            or before.catalog != self._catalog
            or before.selection != self._selection
        ):
            self._render()
            for warning in self._capabilities.dependency_warnings(self._design):
                if warning not in previous_warnings:
                    ui.notify(warning, type="warning")

    def _undo(self) -> None:
        if self._saving:
            return
        snapshot = self._history.undo()
        if snapshot is None:
            return
        self._design = snapshot.design
        self._catalog = snapshot.catalog
        self._selection = snapshot.selection
        self._render()

    def _render_board(self) -> None:
        layout = presentation_config("agent_board")
        nodes = self._design.nodes
        width = max(
            layout["minimum_width"],
            max((node.x + layout["horizontal_padding"] for node in nodes), default=0),
        )
        height = max(
            layout["minimum_height"],
            max((node.y + layout["vertical_padding"] for node in nodes), default=0),
        )
        self._board.content = render_html(
            "agents/board.html",
            scope=self._scope,
            catalog=self._catalog,
            nodes=nodes,
            connections=self._design.connected_nodes(),
            selected_id=self._selection.primary_id,
            selected_ids=self._selection.node_ids,
            root_id=self._design.root_id,
            width=width,
            height=height,
            directions=[("left", "←"), ("right", "→"), ("up", "↑"), ("down", "↓")],
            **{
                key: layout[key]
                for key in (
                    "colors",
                    "node_width",
                    "node_height",
                    "curve_offset",
                    "pan_step",
                )
            },
        )

    def _render_details(self, node: DesignNode) -> None:
        ui.label(node.name).classes(
            UI_PRESENTATION["render_details.classes.text_sm_font_semibold_break_all"]
        )
        ui.label(
            ui_text(
                "agents.agent_designer.select_a_card_to_edit_its_properties",
                node_kind_upper=node.kind.upper(),
            )
        ).classes(UI_PRESENTATION["render_details.classes.text_xs_text_zinc_400"])
        if node.kind == "hds":
            ui.switch(
                ui_text("agents.agent_designer.preserve_sub_agent_history"),
                value=node.persistent_history,
                on_change=lambda event: self._change_design(
                    partial(self._design.set_persistent_history, node.id, event.value)
                ),
            ).props(UI_PRESENTATION["render_details.props.dense"])
            if node.id != self._design.root_id:
                ui.button(
                    ui_text("agents.agent_designer.use_as_root"),
                    on_click=lambda: self._set_root(node.id),
                ).props(UI_PRESENTATION["render_details.props.flat_no_caps"])
        if node.kind in {"hds", "skill"}:
            targets = {
                item.id: f"{item.kind.upper()} · {item.name}"
                for item in self._design.connection_targets(node.id)
            }
            target = (
                ui.select(targets, label=ui_text("agents.agent_designer.connect_to"))
                .props(UI_PRESENTATION["render_details.props.outlined_dense"])
                .classes(UI_PRESENTATION["render_details.classes.w_full"])
            )
            connect_button = ui.button(
                ui_text("agents.agent_designer.connect"),
                icon=UI_PRESENTATION["render_details.icon.add_link"],
                on_click=lambda: self._connect(node.id, target.value),
            ).props(UI_PRESENTATION["render_details.props.flat_no_caps"])
            if not targets:
                target.disable()
                connect_button.disable()
                ui.label(
                    ui_text("agents.agent_designer.no_connection_targets")
                ).classes(
                    UI_PRESENTATION["render_details.classes.text_xs_text_zinc_400"]
                )
        if node.kind == "skill":
            ui.label(
                ui_text(
                    "agents.agent_designer.edit_description_and_instructions_separately_sub_skills"
                )
            ).classes(UI_PRESENTATION["render_details.classes.text_xs_text_zinc_400"])
            if node.name == "deep-thinking":
                ui.label(
                    ui_text(
                        "agents.agent_designer.deep_thinking_keeps_provider_thinking_in_direct"
                    )
                ).classes(
                    UI_PRESENTATION["render_details.classes.text_xs_text_sky_300"]
                )
        if node.source or node.kind == "hds":
            ui.button(
                ui_text("agents.agent_designer.edit_a_copy"),
                icon=UI_PRESENTATION["render_details.icon.edit"],
                on_click=lambda: self._edit_copy(node),
            ).props(UI_PRESENTATION["render_details.props.flat_no_caps"])
        for link in self._design.connections(node.id):
            label = f"{self._design.node(link.parent).name} → {self._design.node(link.child).name}"
            ui.button(
                label,
                icon=UI_PRESENTATION["render_details.icon.link_off"],
                on_click=partial(self._disconnect, link),
            ).props(UI_PRESENTATION["render_details.props.connection_button"]).classes(
                UI_PRESENTATION["render_details.classes.connection_button"]
            ).tooltip(label)
        if node.id != self._design.root_id:
            ui.button(
                ui_text("agents.agent_designer.remove_from_board"),
                icon=UI_PRESENTATION["render_details.icon.delete_outline"],
                on_click=lambda: self._remove_node(node.id),
            ).props(UI_PRESENTATION["render_details.props.flat_no_caps_color_negative"])

    def _handle_action(self, event) -> None:
        data = event.args
        if (
            not isinstance(data, dict)
            or self._saving
            or data.get("scope") != self._scope
            or not self._dialog.value
        ):
            return
        action = data.get("action")
        if action == "undo":
            self._undo()
            return

        def apply_action() -> None:
            if action == "add":
                self._add_capability(data)
            elif action == "move":
                self._design.move_node(data["id"], float(data["x"]), float(data["y"]))
            elif action == "move_many":
                self._move_nodes(data["positions"], data["ids"])
            elif action == "select":
                self._select_nodes(
                    data.get("ids", [data["id"]] if "id" in data else [])
                )

        try:
            if action == "select":
                apply_action()
                self._render()
            else:
                self._change_design(apply_action)
        except (ValueError, TypeError, StopIteration, KeyError) as error:
            ui.notify(str(error), type="warning")

    def _add_capability(self, data: dict) -> None:
        original = next(node for node in self._catalog if node.id == data["id"])
        default_y = (
            max((node.y for node in self._design.nodes), default=0)
            + BOARD_DEFAULTS["palette_vertical_gap"]
        )
        node = original.model_copy(
            deep=True,
            update={
                "x": float(data.get("x", BOARD_DEFAULTS["palette_node_x"])),
                "y": float(data.get("y", default_y)),
            },
        )
        parent = self._design.node(self._selection.primary_id or self._design.root_id)
        parent_id = (
            parent.id if parent.kind in {"hds", "skill"} else self._design.root_id
        )
        self._design = self._capabilities.add_to_design(
            self._design, node, parent_id, self._catalog
        )
        self._select_nodes([self._design.capability(node.kind, node.name).id])

    def _move_nodes(self, positions: list[dict], ids: list[str]) -> None:
        selection = DesignSelection.for_nodes(self._design, ids)
        self._design.move_nodes(
            {item["id"]: (float(item["x"]), float(item["y"])) for item in positions}
        )
        self._selection = selection

    def _select_nodes(self, ids: list[str]) -> None:
        self._selection = DesignSelection.for_nodes(self._design, ids)

    def _remove_selected(self) -> None:
        def remove() -> None:
            self._design.remove_nodes(self._selection.node_ids)
            self._select_nodes([self._design.root_id])

        self._change_design(remove)

    def _disconnect_selected(self) -> None:
        self._change_design(
            partial(self._design.disconnect_nodes, self._selection.node_ids)
        )

    def _connect(self, parent: str, child: str | None) -> None:
        if not child:
            return
        try:
            self._change_design(partial(self._design.connect, parent, child))
        except ValueError as error:
            ui.notify(str(error), type="warning")

    def _disconnect(self, link) -> None:
        self._change_design(partial(self._design.disconnect, link))

    def _set_root(self, node_id: str) -> None:
        self._change_design(partial(self._design.set_root, node_id))

    def _remove_node(self, node_id: str) -> None:
        def remove() -> None:
            self._design.remove_node(node_id)
            self._select_nodes([self._design.root_id])

        self._change_design(remove)

    def _new_hds(self) -> None:
        with (
            ui.dialog() as dialog,
            ui.card().classes(
                UI_PRESENTATION["new_hds.classes.w_560px_max_w_94vw_bg_zinc_900"]
            ),
        ):
            ui.label(ui_text("agents.agent_designer.new_sub_agent_hds")).classes(
                UI_PRESENTATION["new_hds.classes.text_lg"]
            )
            name = (
                ui.input(
                    ui_text("agents.agent_designer.hds_name"),
                    value=BOARD_DEFAULTS["worker_name_prefix"]
                    + uuid4().hex[: BOARD_DEFAULTS["worker_name_id_length"]],
                )
                .props(UI_PRESENTATION["new_hds.props.outlined"])
                .classes(UI_PRESENTATION["new_hds.classes.w_full"])
            )
            instructions = (
                ui.textarea(
                    ui_text("agents.agent_designer.instructions"),
                    value=ui_text(
                        "agents.agent_designer.you_are_a_specialist_assistant_complete_the_delegated_task_and_r"
                    ),
                )
                .props(UI_PRESENTATION["new_hds.props.outlined_autogrow"])
                .classes(UI_PRESENTATION["new_hds.classes.w_full"])
            )

            def add() -> None:
                node = DesignNode(
                    kind="hds",
                    name=name.value.strip(),
                    content=instructions.value,
                    x=BOARD_DEFAULTS["new_hds_x"],
                    y=BOARD_DEFAULTS["new_hds_y"],
                )
                try:

                    def add_node() -> None:
                        self._design.add_node(node, self._design.root_id)
                        self._select_nodes([node.id])

                    self._change_design(add_node)
                except ValueError as error:
                    ui.notify(str(error), type="warning")
                    return
                dialog.close()

            ui.button(ui_text("agents.agent_designer.add_sub_agent"), on_click=add)
        dialog.open()

    def _edit_copy(self, node: DesignNode) -> None:
        CapabilityEditor(self._capabilities, self._apply_capability_copy).open(
            self._design, node.id
        )

    def _apply_capability_copy(self, node_id: str, replacement: DesignNode) -> None:
        self._copied_sources.add(replacement.source)

        def replace() -> None:
            self._design.replace_node(node_id, replacement)
            self._catalog.append(
                replacement.model_copy(update={"id": uuid4().hex}, deep=True)
            )

        self._change_design(replace)

    def _set_saving(self, saving: bool) -> None:
        self._saving = saving
        if saving:
            self._dialog.props(UI_PRESENTATION["set_saving.props.persistent"])
            self._shell.props(UI_PRESENTATION["set_saving.props.inert"])
            self._save_button.props(UI_PRESENTATION["set_saving.props.loading"])
        else:
            self._dialog.props(remove="persistent")
            self._shell.props(remove="inert")
            self._save_button.props(remove="loading")

    async def _save(self) -> None:
        if self._saving:
            return
        client = self._dialog.client
        design = self._design.model_copy(deep=True)
        self._set_saving(True)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, validate_agent_design, design
            )
            if not client.is_deleted:
                with client:
                    self._on_saved(design)
                    self._on_cancelled = None
                    self._dialog.close()
        except (OSError, ValueError, TypeError, SyntaxError) as error:
            if not client.is_deleted:
                with client:
                    ui.notify(str(error), type="warning")
        finally:
            self._saving = False
            if not client.is_deleted:
                with client:
                    self._set_saving(False)
