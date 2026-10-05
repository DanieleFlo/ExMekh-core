from __future__ import annotations

import json
from typing import Any

from markupsafe import Markup
from nicegui import ui

from exmekh_agent.exmekh_bridge.models import ChatMessage, ChatSession
from exmekh_agent.ui.graph.graph_data import (
    NODE_COLORS,
    GraphNode,
    SessionGraphData,
    build_graph_data_from_session,
)
from exmekh_agent.ui.graph.graph_svg import render_star_svg_html
from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.resources import render_html, ui_text


class GraphViewComponent:
    """Interactive Constellation Graph component for the 'Graph' tab.

    Renders an independent graph per ChatSession:
    - Main history is always horizontal with fixed distance between consecutive nodes.
    - Horizontal canvas size grows dynamically with node count and scrolls horizontally when needed.
    - System Prompt and User are separated into distinct nodes using the real System Prompt.
    - While a response is being generated, displays a loading animation and builds the graph once the response finishes.
    - Sub-agent histories are parallel dual lines intersecting their parent history at the points of interaction (constellation layout).
    - Opened skills stay permanently visible and step downward following the counterclockwise logic, with identical offset for all nodes in the same skill_path.
    - Clicking a node opens a horizontal menu at the bottom with the events grouped in that node.
    """

    def __init__(self) -> None:
        self.container: ui.column | None = None
        self.status_label: ui.label | None = None
        self.stage_container: ui.element | None = None
        self.bottom_menu_container: ui.column | None = None
        self.current_session: ChatSession | None = None
        self.current_graph_data: SessionGraphData | None = None
        self.selected_node_by_session: dict[str, str | None] = {}
        self.highlighted_sub_agent_id: str | None = None
        self.is_generating: bool = False

    def build(self) -> ui.column:
        with ui.column().classes(
            "w-full h-full flex-1 min-h-0 flex-grow px-4 py-3 gap-3 overflow-y-auto graph-view-container"
        ) as col:
            self.container = col
            self._render_toolbar()
            self.stage_container = ui.element("div").classes(
                "graph-stage relative w-full h-[340px] min-h-[290px] shrink-0 "
                "bg-[#0b0b0e] border border-zinc-800/90 rounded-xl overflow-x-auto overflow-y-auto shadow-inner"
            )
            self.bottom_menu_container = ui.column().classes(
                "graph-bottom-menu w-full shrink-0 gap-2"
            )
        return col

    def _render_toolbar(self) -> None:
        with ui.row().classes(
            "w-full items-center justify-between pb-2 border-b border-zinc-800/80 flex-wrap gap-2"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("account_tree", size="sm").classes("text-sky-400")
                ui.label(ui_text("graph.graph_view.graph_view")).classes(
                    "text-sm font-semibold tracking-tight text-zinc-100"
                )
                self.status_label = ui.label(
                    ui_text("graph.graph_view.0_histories_0_nodes")
                ).classes(
                    "font-mono text-xs text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800"
                )

            with ui.row().classes("items-center gap-3 flex-wrap text-[11px] font-mono"):
                legend_items = [
                    ("System Prompt", NODE_COLORS["system"]),
                    ("User", NODE_COLORS["user"]),
                    ("Tool (Call+Exec+Resp)", NODE_COLORS["tool_bundle"]),
                    ("Skill (Offset ↺ downward)", NODE_COLORS["skill_bundle"]),
                    (
                        "Sub-Agent Interaction (Constellation)",
                        NODE_COLORS["sub_agent_bundle"],
                    ),
                    ("Assistant", NODE_COLORS["assistant"]),
                ]
                for lbl, color in legend_items:
                    with ui.row().classes("items-center gap-1.5"):
                        ui.html(
                            render_html("shared/color_dot.html", size=9, color=color)
                        )
                        ui.label(lbl).classes("text-zinc-400")

    def get_selected_node_id(self, session_id: str) -> str | None:
        return self.selected_node_by_session.get(session_id)

    def select_node(self, node_id: str | None) -> None:
        if not self.current_session or self.is_generating:
            return
        self.selected_node_by_session[self.current_session.id] = node_id
        self._render_stage_and_menu()

    def render_session(
        self,
        session: ChatSession,
        *,
        selected_sub_agent_id: str | None = None,
        live_message: ChatMessage | None = None,
        is_generating: bool = False,
    ) -> None:
        if not self.container or not self.stage_container:
            return

        self.current_session = session
        self.highlighted_sub_agent_id = selected_sub_agent_id
        effective_generating = is_generating or (
            live_message is not None and live_message.is_streaming
        )
        self.is_generating = effective_generating

        if effective_generating:
            self.current_graph_data = None
            if self.status_label:
                self.status_label.text = ui_text(
                    "graph.graph_view.chat_generating_response",
                    session_title_22=session.title[:22],
                )
            self._render_loading_animation(session)
            return

        self.current_graph_data = build_graph_data_from_session(session)

        selected_id = self.selected_node_by_session.get(session.id)
        if selected_id and self.current_graph_data.find_node(selected_id) is None:
            self.selected_node_by_session[session.id] = None

        if self.status_label:
            h_count = len(self.current_graph_data.histories)
            n_count = self.current_graph_data.total_nodes
            self.status_label.text = ui_text(
                "graph.graph_view.chat_histories_nodes",
                session_title_22=session.title[:22],
                h_count=h_count,
                n_count=n_count,
            )

        self._render_stage_and_menu()

    def _render_loading_animation(self, session: ChatSession) -> None:
        """Render a constellation loading animation while the assistant response is being generated."""
        if not self.stage_container or not self.bottom_menu_container:
            return

        self.stage_container.clear()
        with self.stage_container:
            loading_svg = render_html("graph/loading.svg")
            ui.html(loading_svg).classes(
                "w-full h-full absolute inset-0 pointer-events-none"
            )
            with ui.column().classes(
                "graph-loading-state absolute inset-0 items-center justify-center text-center gap-3 px-6 bg-zinc-950/65 backdrop-blur-[1px]"
            ):
                ui.spinner("rings", size="xl", color="sky-4")
                ui.label(ui_text("graph.graph_view.generating_response")).classes(
                    "text-sm font-mono font-semibold text-sky-300 tracking-tight"
                )
                ui.label(
                    ui_text(
                        "graph.graph_view.the_constellation_graph_is_automatically_generated_once"
                    )
                ).classes("text-xs font-mono text-zinc-400 max-w-lg")

        self.bottom_menu_container.clear()
        with self.bottom_menu_container:
            with ui.row().classes(
                "graph-bottom-menu-loading w-full items-center justify-between "
                "px-3.5 py-2.5 rounded-lg bg-[#111114] border border-zinc-800/80 text-xs font-mono text-zinc-400"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.spinner("dots", size="xs", color="sky-4")
                    ui.label(
                        ui_text(
                            "graph.graph_view.waiting_for_response_completion_for_to_build",
                            session_title_28=session.title[:28],
                        )
                    )

    def _render_stage_and_menu(self) -> None:
        if (
            not self.stage_container
            or not self.bottom_menu_container
            or not self.current_graph_data
            or not self.current_session
        ):
            return

        graph_data = self.current_graph_data
        selected_id = self.selected_node_by_session.get(self.current_session.id)

        self.stage_container.clear()
        with self.stage_container:
            canvas = (
                ui.element("div")
                .classes("graph-canvas relative shrink-0 mx-auto")
                .style(
                    f"width:{int(graph_data.width)}px;height:{int(graph_data.height)}px;min-height:100%;"
                )
            )
            canvas.props["data-canvas-width"] = str(int(graph_data.width))
            canvas.props["data-canvas-height"] = str(int(graph_data.height))
            with canvas:
                svg_markup = render_star_svg_html(
                    graph_data,
                    selected_node_id=selected_id,
                    highlighted_history_id=self.highlighted_sub_agent_id,
                )
                ui.html(svg_markup).classes(
                    "w-full h-full absolute inset-0 pointer-events-none"
                )

                if graph_data.total_nodes == 0:
                    with ui.column().classes(
                        "absolute inset-0 items-center justify-center text-center pointer-events-none gap-1"
                    ):
                        ui.label(
                            ui_text(
                                "graph.graph_view.main_horizontal_line_ready_send_a_message"
                            )
                        ).classes(
                            "text-xs font-mono text-zinc-400 bg-zinc-950/85 px-3 py-1.5 rounded-lg border border-zinc-800"
                        )

                for history in graph_data.histories:
                    for node in history.nodes:
                        is_selected = node.id == selected_id
                        offset_cls = (
                            "graph-node-skill-offset"
                            if node.depth > 0
                            else "graph-node-main-line"
                        )
                        sel_cls = "graph-node-selected" if is_selected else ""

                        node_btn = (
                            ui.element("button")
                            .classes(
                                f"graph-node graph-node-{node.kind} {offset_cls} {sel_cls}"
                            )
                            .style(
                                f"left:{node.x:.2f}px;top:{node.y:.2f}px;background-color:{node.color};color:{node.color};"
                            )
                            .on("click", lambda _, nid=node.id: self.select_node(nid))
                        )
                        node_btn.props["type"] = "button"
                        node_btn.props["data-node-id"] = str(node.id)
                        node_btn.props["data-history-id"] = str(node.history_id)
                        node_btn.props["data-depth"] = str(node.depth)
                        node_btn.props["data-skill-path"] = str(node.skill_path)
                        node_btn.props["data-kind"] = str(node.kind)
                        node_btn.props["aria-label"] = str(node.label)
                        node_btn.props["title"] = (
                            f"{node.label} ({len(node.events)} events)"
                        )
                        with node_btn:
                            ui.label(str(node.step_index + 1)).classes(
                                "text-[10px] font-mono font-bold text-zinc-950 pointer-events-none select-none"
                            )
                            caption_pos = "top-6" if node.depth > 0 else "-top-5"
                            ui.label(node.short_label).classes(
                                f"absolute {caption_pos} left-1/2 -translate-x-1/2 whitespace-nowrap "
                                "text-[10px] font-mono text-zinc-300 bg-zinc-950/85 px-1.5 py-0.2 "
                                "rounded border border-zinc-800/80 pointer-events-none select-none"
                            )

        self._render_bottom_horizontal_menu(graph_data.find_node(selected_id))

    def _render_bottom_horizontal_menu(self, selected_node: GraphNode | None) -> None:
        if not self.bottom_menu_container:
            return

        self.bottom_menu_container.clear()
        with self.bottom_menu_container:
            if selected_node is None:
                with ui.row().classes(
                    "graph-bottom-menu-placeholder w-full items-center justify-between "
                    "px-3.5 py-2.5 rounded-lg bg-[#111114] border border-zinc-800/80 text-xs font-mono text-zinc-400"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("touch_app", size="xs").classes("text-sky-400")
                        ui.label(
                            ui_text("graph.graph_view.click_a_node_on_the_graph_to")
                        )
                return

            with ui.column().classes(
                "graph-bottom-menu-open w-full bg-[#0e0e12] border border-zinc-700/80 rounded-xl p-3 gap-2.5 shadow-xl"
            ):
                with ui.row().classes(
                    "w-full items-center justify-between pb-2 border-b border-zinc-800/80 flex-wrap gap-2"
                ):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.html(
                            render_html(
                                "shared/color_dot.html",
                                size=12,
                                color=selected_node.color,
                            )
                        )
                        ui.label(
                            ui_text(
                                "graph.graph_view.node",
                                selected_node_step_index_1=selected_node.step_index + 1,
                                selected_node_label=selected_node.label,
                            )
                        ).classes("text-xs font-mono font-semibold text-zinc-100")
                        ui.label(selected_node.history_label).classes(
                            "text-[11px] font-mono text-sky-300 bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800/50"
                        )
                        if selected_node.depth > 0:
                            ui.label(
                                ui_text(
                                    "graph.graph_view.skill_counterclockwise_offset_downward_step",
                                    selected_node_depth=selected_node.depth,
                                    selected_node_skill_path=selected_node.skill_path,
                                )
                            ).classes(
                                "text-[11px] font-mono text-purple-300 bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800/50"
                            )
                        ui.label(
                            ui_text(
                                "graph.graph_view.events_in_node",
                                len_selected_node_events=len(selected_node.events),
                            )
                        ).classes(
                            "text-[11px] font-mono text-amber-300 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/50"
                        )

                    with ui.row().classes("items-center gap-1.5"):
                        all_events_json = json.dumps(
                            selected_node.events, indent=2, ensure_ascii=False
                        )
                        ui.button(
                            ui_text("graph.graph_view.copy_node_events"),
                            icon="content_copy",
                            on_click=lambda _, t=all_events_json: self._copy_json(
                                t, "Node events copied"
                            ),
                        ).props("flat dense no-caps size=xs text-color=grey-3").classes(
                            "text-[11px] px-2.5 py-0.5 bg-zinc-900 hover:bg-zinc-800 rounded border border-zinc-700"
                        )
                        ui.button(
                            icon="close",
                            on_click=lambda: self.select_node(None),
                        ).props(
                            "flat round dense size=xs text-color=grey-4 aria-label='Close node menu'"
                        )

                with ui.row().classes(
                    "graph-horizontal-events-row w-full flex-row flex-nowrap overflow-x-auto gap-3 pb-1 items-stretch"
                ):
                    for idx, ev in enumerate(selected_node.events):
                        self._render_horizontal_event_card(ev, index=idx + 1)

    def _render_horizontal_event_card(self, ev: dict[str, Any], index: int) -> None:
        role = str(ev.get("role", "event")).lower()
        stage_label = str(
            ev.get("stage_label") or ev.get("event_stage") or role.upper()
        )

        if role in ("system", "developer"):
            border_cls = "border-pink-800/60"
            badge_cls = "bg-pink-950/80 text-pink-300 border-pink-700/60"
            icon_name = "terminal"
        elif ev.get("event_stage") == "steering_message":
            border_cls = "border-orange-700/50"
            badge_cls = "bg-orange-950/80 text-orange-300 border-orange-700/60"
            icon_name = "alt_route"
        elif role == "user":
            border_cls = "border-sky-800/50"
            badge_cls = "bg-sky-950/80 text-sky-300 border-sky-700/60"
            icon_name = "person"
        elif role == "tool":
            border_cls = "border-amber-700/50"
            badge_cls = "bg-amber-950/80 text-amber-300 border-amber-700/60"
            icon_name = "output"
        elif role == "execution":
            border_cls = "border-purple-700/50"
            badge_cls = "bg-purple-950/80 text-purple-300 border-purple-700/60"
            icon_name = "play_circle"
        else:
            border_cls = "border-emerald-800/50"
            badge_cls = "bg-emerald-950/80 text-emerald-300 border-emerald-700/60"
            icon_name = "smart_toy"

        with ui.column().classes(
            f"graph-event-card min-w-[330px] flex-1 shrink-0 bg-[#111114] border {border_cls} rounded-lg p-3 gap-2"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-1.5 flex-wrap"):
                    ui.icon(icon_name, size="xs").classes("text-zinc-400")
                    with ui.row().classes(
                        f"items-center px-1.5 py-0.5 rounded text-[10px] font-mono border {badge_cls}"
                    ):
                        ui.label(stage_label)
                        if ev.get("pending"):
                            ui.label(ui_text("graph.graph_view.pending")).classes(
                                "text-orange-300"
                            )
                    ui.label(ui_text("graph.graph_view.event", index=index)).classes(
                        "font-mono text-[10px] text-zinc-500"
                    )
                ev_json = json.dumps(ev, indent=2, ensure_ascii=False)
                ui.button(
                    icon="content_copy",
                    on_click=lambda _, text=ev_json: self._copy_json(
                        text, "Event copied"
                    ),
                ).props("flat round dense size=xs").classes(
                    "text-zinc-600 hover:text-zinc-300"
                )

            highlighted_html = format_json_to_highlighted_html(ev)
            ui.html(
                render_html(
                    "shared/raw_json.html",
                    content=Markup(highlighted_html),
                    extra_classes="max-h-[220px] overflow-y-auto",
                )
            ).classes("w-full")

    @staticmethod
    def _copy_json(text: str, notification: str) -> None:
        ui.clipboard.write(text)
        ui.notify(notification, type="positive", position="top", timeout=1500)
