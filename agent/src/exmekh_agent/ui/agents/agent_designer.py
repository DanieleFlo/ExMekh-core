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
from exmekh_agent.ui.agents.capability_editor import CapabilityEditor
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text


class AgentDesigner:
    def __init__(self) -> None:
        self._capabilities = CapabilityRepository()
        self._dialog: ui.dialog | None = None
        self._design = AgentDesign()
        self._catalog: list[DesignNode] = []
        self._selected_id = ""
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

    def build(self) -> None:
        script = (
            Path(__file__).parents[1] / "resources" / "js" / "agent_board.js"
        ).read_text(encoding="utf-8")
        ui.add_head_html(f"<script>{script}</script>")
        ui.on("agent_board_action", self._handle_action)
        with (
            ui.dialog()
            .props("maximized")
            .classes("agent-designer-dialog") as self._dialog
        ):
            with ui.card().classes(
                "agent-designer-shell w-full h-full bg-zinc-950 gap-3"
            ) as self._shell:
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(ui_text("agents.agent_designer.customize_agent")).classes(
                        "text-lg font-semibold"
                    )
                    with ui.row().classes("gap-2"):
                        self._undo_button = (
                            ui.button(
                                ui_text("agents.agent_designer.undo"),
                                icon="undo",
                                on_click=self._undo,
                            )
                            .props("flat")
                            .tooltip(ui_text("agents.agent_designer.undo_shortcut"))
                        )
                        self._undo_button.disable()
                        ui.button(
                            ui_text("agents.agent_designer.new_hds"),
                            icon="add",
                            on_click=self._new_hds,
                        ).props("flat")
                        ui.button(
                            ui_text("agents.agent_designer.cancel"),
                            on_click=self._dialog.close,
                        ).props("flat")
                        self._save_button = ui.button(
                            ui_text("agents.agent_designer.save_agent"),
                            on_click=self._save,
                        ).props("unelevated")
                ui.label(
                    ui_text("agents.agent_designer.select_an_hds_or_skill_to_connect")
                ).classes("text-xs text-zinc-400")
                with ui.row().classes(
                    "agent-designer-layout w-full flex-1 min-h-0 flex-nowrap"
                ):
                    self._board = ui.html("").classes(
                        "agent-board-html flex-1 min-w-0 h-full"
                    )
                    self._details = ui.column().classes("agent-node-details gap-3")

        self._dialog.on("hide", self._cancelled)

    def _cancelled(self) -> None:
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
        try:
            self._catalog = await asyncio.get_running_loop().run_in_executor(
                None, partial(self._capabilities.catalog, paths)
            )
            self._design = (
                await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._capabilities.expand_skill_connections, design)
                )
                if design
                else await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._capabilities.default_design, profile, paths)
                )
            )
        except (OSError, ValueError, SyntaxError) as error:
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
        self._on_saved = on_saved
        self._on_cancelled = on_cancelled
        self._selected_id = self._design.root_id
        self._history.clear()
        with client:
            self._render()
            self._dialog.open()

    def _render(self) -> None:
        if self._undo_button:
            self._undo_button.set_enabled(self._history.can_undo)
        self._render_board()
        self._details.clear()
        if self._selected_id:
            with self._details:
                self._render_details(self._design.node(self._selected_id))

    def _change_design(self, change: Callable[[], None]) -> None:
        before = DesignSnapshot.capture(self._design, self._catalog, self._selected_id)
        change()
        self._history.record(before, self._design, self._catalog)
        if (
            before.design != self._design
            or before.catalog != self._catalog
            or before.selected_id != self._selected_id
        ):
            self._render()

    def _undo(self) -> None:
        if self._saving:
            return
        snapshot = self._history.undo()
        if snapshot is None:
            return
        self._design = snapshot.design
        self._catalog = snapshot.catalog
        self._selected_id = snapshot.selected_id
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
            connections=[
                (self._design.node(link.parent), self._design.node(link.child))
                for link in self._design.links
            ],
            selected_id=self._selected_id,
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
        ui.label(node.name).classes("text-sm font-semibold break-all")
        ui.label(
            ui_text(
                "agents.agent_designer.select_a_card_to_edit_its_properties",
                node_kind_upper=node.kind.upper(),
            )
        ).classes("text-xs text-zinc-400")
        if node.kind == "hds":
            ui.switch(
                ui_text("agents.agent_designer.preserve_sub_agent_history"),
                value=node.persistent_history,
                on_change=lambda event: self._change_design(
                    partial(self._design.set_persistent_history, node.id, event.value)
                ),
            ).props("dense")
            if node.id != self._design.root_id:
                ui.button(
                    ui_text("agents.agent_designer.use_as_root"),
                    on_click=lambda: self._set_root(node.id),
                ).props("flat no-caps")
        if node.kind in {"hds", "skill"}:
            targets = {
                item.id: f"{item.kind.upper()} · {item.name}"
                for item in self._design.connection_targets(node.id)
            }
            target = (
                ui.select(targets, label=ui_text("agents.agent_designer.connect_to"))
                .props("outlined dense")
                .classes("w-full")
            )
            connect_button = ui.button(
                ui_text("agents.agent_designer.connect"),
                icon="add_link",
                on_click=lambda: self._connect(node.id, target.value),
            ).props("flat no-caps")
            if not targets:
                target.disable()
                connect_button.disable()
                ui.label(
                    ui_text("agents.agent_designer.no_connection_targets")
                ).classes("text-xs text-zinc-400")
        if node.kind == "skill":
            ui.label(
                ui_text(
                    "agents.agent_designer.edit_description_and_instructions_separately_sub_skills"
                )
            ).classes("text-xs text-zinc-400")
            if node.name == "deep-thinking":
                ui.label(
                    ui_text(
                        "agents.agent_designer.deep_thinking_keeps_provider_thinking_in_direct"
                    )
                ).classes("text-xs text-sky-300")
        if node.source or node.kind == "hds":
            ui.button(
                ui_text("agents.agent_designer.edit_a_copy"),
                icon="edit",
                on_click=lambda: self._edit_copy(node),
            ).props("flat no-caps")
        for link in self._design.connections(node.id):
            label = f"{self._design.node(link.parent).name} → {self._design.node(link.child).name}"
            ui.button(
                label, icon="link_off", on_click=partial(self._disconnect, link)
            ).props("flat no-caps").classes("w-full text-xs")
        if node.id != self._design.root_id:
            ui.button(
                ui_text("agents.agent_designer.remove_from_board"),
                icon="delete_outline",
                on_click=lambda: self._remove_node(node.id),
            ).props("flat no-caps color=negative")

    def _handle_action(self, event) -> None:
        data = event.args
        if self._saving or data.get("scope") != self._scope or not self._dialog.value:
            return
        if data["action"] == "undo":
            self._undo()
            return

        def apply_action() -> None:
            if data["action"] == "add":
                original = next(node for node in self._catalog if node.id == data["id"])
                node = original.model_copy(
                    update={
                        "x": max(0, float(data.get("x", 50))),
                        "y": max(
                            0,
                            float(
                                data.get(
                                    "y",
                                    max(
                                        (item.y for item in self._design.nodes),
                                        default=0,
                                    )
                                    + 120,
                                )
                            ),
                        ),
                    },
                    deep=True,
                )
                parent = self._design.node(self._selected_id)
                parent_id = (
                    parent.id
                    if parent.kind in {"hds", "skill"}
                    else self._design.root_id
                )
                self._design = self._capabilities.add_to_design(
                    self._design, node, parent_id, self._catalog
                )
                self._selected_id = self._design.capability(node.kind, node.name).id
            elif data["action"] == "move":
                self._design.move_node(data["id"], float(data["x"]), float(data["y"]))
            elif data["action"] == "select":
                self._selected_id = self._design.node(data["id"]).id

        try:
            if data["action"] == "select":
                apply_action()
                self._render()
            else:
                self._change_design(apply_action)
        except (ValueError, StopIteration, KeyError) as error:
            ui.notify(str(error), type="warning")

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
            self._selected_id = self._design.root_id

        self._change_design(remove)

    def _new_hds(self) -> None:
        with (
            ui.dialog() as dialog,
            ui.card().classes("w-[560px] max-w-[94vw] bg-zinc-900"),
        ):
            ui.label(ui_text("agents.agent_designer.new_sub_agent_hds")).classes(
                "text-lg"
            )
            name = (
                ui.input("HDS name", value=f"worker-{uuid4().hex[:4]}")
                .props("outlined")
                .classes("w-full")
            )
            instructions = (
                ui.textarea(
                    "Instructions",
                    value="You are a specialist assistant. Complete the delegated task and respond in the user's language.",
                )
                .props("outlined autogrow")
                .classes("w-full")
            )

            def add() -> None:
                node = DesignNode(
                    kind="hds",
                    name=name.value.strip(),
                    content=instructions.value,
                    x=50,
                    y=650,
                )
                try:

                    def add_node() -> None:
                        self._design.add_node(node, self._design.root_id)
                        self._selected_id = node.id

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
        def replace() -> None:
            self._design.replace_node(node_id, replacement)
            self._catalog.append(
                replacement.model_copy(update={"id": uuid4().hex}, deep=True)
            )

        self._change_design(replace)

    def _set_saving(self, saving: bool) -> None:
        self._saving = saving
        if saving:
            self._dialog.props("persistent")
            self._shell.props("inert")
            self._save_button.props("loading")
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
