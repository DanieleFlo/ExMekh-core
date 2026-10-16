from __future__ import annotations

import json
from typing import Any

from markupsafe import Markup
from nicegui import ui

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.exmekh_bridge.models import ChatMessage, ChatSession
from exmekh_agent.ui.graph.graph_data import (
    NODE_COLORS,
    GraphNode,
    SessionGraphData,
    build_graph_data_from_session,
)
from exmekh_agent.ui.graph.graph_svg import render_star_svg_html
from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text

UI_PRESENTATION = presentation_config("controls/graph_view")

DRAWING_DEFAULTS = presentation_config("graph_view_drawing")


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
            UI_PRESENTATION[
                "build.classes.w_full_h_full_flex_1_min_h_0_flex_grow_px_4_p"
            ]
        ) as col:
            self.container = col
            self._render_toolbar()
            self.stage_container = ui.element("div").classes(
                UI_PRESENTATION[
                    "build.classes.graph_stage_relative_w_full_h_340px_min_h_290"
                ]
            )
            self.bottom_menu_container = ui.column().classes(
                UI_PRESENTATION["build.classes.graph_bottom_menu_w_full_shrink_0_gap_2"]
            )
        return col

    def _render_toolbar(self) -> None:
        with ui.row().classes(
            UI_PRESENTATION[
                "render_toolbar.classes.w_full_items_center_justify_between_pb_2_bord"
            ]
        ):
            with ui.row().classes(
                UI_PRESENTATION["render_toolbar.classes.items_center_gap_2"]
            ):
                ui.icon(
                    UI_PRESENTATION["render_toolbar.icon.account_tree"],
                    size=UI_PRESENTATION["render_toolbar.size.sm"],
                ).classes(UI_PRESENTATION["render_toolbar.classes.text_sky_400"])
                ui.label(ui_text("graph.graph_view.graph_view")).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.text_sm_font_semibold_tracking_tight_text_zin"
                    ]
                )
                self.status_label = ui.label(
                    ui_text("graph.graph_view.0_histories_0_nodes")
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.font_mono_text_xs_text_zinc_400_bg_zinc_900_8"
                    ]
                )

            with ui.row().classes(
                UI_PRESENTATION[
                    "render_toolbar.classes.items_center_gap_3_flex_wrap_text_11px_font_m"
                ]
            ):
                legend_items = [
                    (app_text("graph.graph_data.system_prompt"), NODE_COLORS["system"]),
                    (app_text("graph.graph_data.user_31"), NODE_COLORS["user"]),
                    (
                        app_text("graph.graph_view.tool_call_exec_resp"),
                        NODE_COLORS["tool_bundle"],
                    ),
                    (
                        app_text("graph.graph_view.skill_offset_downward"),
                        NODE_COLORS["skill_bundle"],
                    ),
                    (
                        app_text(
                            "graph.graph_view.sub_agent_interaction_constellation"
                        ),
                        NODE_COLORS["sub_agent_bundle"],
                    ),
                    (app_text("graph.graph_view.assistant"), NODE_COLORS["assistant"]),
                    (
                        app_text("graph.graph_data.compaction_short"),
                        NODE_COLORS["compaction"],
                    ),
                ]
                for lbl, color in legend_items:
                    with ui.row().classes(
                        UI_PRESENTATION["render_toolbar.classes.items_center_gap_1_5"]
                    ):
                        ui.html(
                            render_html(
                                "shared/color_dot.html",
                                size=DRAWING_DEFAULTS["toolbar_text_size"],
                                color=color,
                            )
                        )
                        ui.label(lbl).classes(
                            UI_PRESENTATION["render_toolbar.classes.text_zinc_400"]
                        )

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
                    session_title_22=session.title[
                        : DRAWING_DEFAULTS["selected_label_length"]
                    ],
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
                session_title_22=session.title[
                    : DRAWING_DEFAULTS["selected_label_length"]
                ],
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
                UI_PRESENTATION[
                    "render_loading_animation.classes.w_full_h_full_absolute_inset_0_pointer_events"
                ]
            )
            with ui.column().classes(
                UI_PRESENTATION[
                    "render_loading_animation.classes.graph_loading_state_absolute_inset_0_items_ce"
                ]
            ):
                ui.spinner(
                    UI_PRESENTATION["render_loading_animation.spinner.rings"],
                    size=UI_PRESENTATION["render_loading_animation.size.xl"],
                    color=UI_PRESENTATION["render_loading_animation.color.sky_4"],
                )
                ui.label(ui_text("graph.graph_view.generating_response")).classes(
                    UI_PRESENTATION[
                        "render_loading_animation.classes.text_sm_font_mono_font_semibold_text_sky_300_"
                    ]
                )
                ui.label(
                    ui_text(
                        "graph.graph_view.the_constellation_graph_is_automatically_generated_once"
                    )
                ).classes(
                    UI_PRESENTATION[
                        "render_loading_animation.classes.text_xs_font_mono_text_zinc_400_max_w_lg"
                    ]
                )

        self.bottom_menu_container.clear()
        with self.bottom_menu_container:
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_loading_animation.classes.graph_bottom_menu_loading_w_full_items_center"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_loading_animation.classes.items_center_gap_2"
                    ]
                ):
                    ui.spinner(
                        UI_PRESENTATION["render_loading_animation.spinner.dots"],
                        size=UI_PRESENTATION["render_loading_animation.size.xs"],
                        color=UI_PRESENTATION["render_loading_animation.color.sky_4"],
                    )
                    ui.label(
                        ui_text(
                            "graph.graph_view.waiting_for_response_completion_for_to_build",
                            session_title_28=session.title[
                                : DRAWING_DEFAULTS["loading_circle_radius"]
                            ],
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
                .classes(
                    UI_PRESENTATION[
                        "render_stage_and_menu.classes.graph_canvas_relative_shrink_0_mx_auto"
                    ]
                )
                .style(
                    UI_PRESENTATION["canvas.style"].format(
                        state_1=int(graph_data.width), state_2=int(graph_data.height)
                    )
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
                    UI_PRESENTATION[
                        "render_stage_and_menu.classes.w_full_h_full_absolute_inset_0_pointer_events"
                    ]
                )

                if graph_data.total_nodes == 0:
                    with ui.column().classes(
                        UI_PRESENTATION[
                            "render_stage_and_menu.classes.absolute_inset_0_items_center_justify_center_"
                        ]
                    ):
                        ui.label(
                            ui_text(
                                "graph.graph_view.main_horizontal_line_ready_send_a_message"
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_stage_and_menu.classes.text_xs_font_mono_text_zinc_400_bg_zinc_950_8"
                            ]
                        )

                for history in graph_data.histories:
                    for node in history.nodes:
                        is_selected = node.id == selected_id
                        offset_cls = (
                            UI_PRESENTATION["node.skill.offset"]
                            if node.depth > 0
                            else UI_PRESENTATION["node.main.offset"]
                        )
                        sel_cls = (
                            UI_PRESENTATION["node.selected"] if is_selected else ""
                        )

                        node_btn = (
                            ui.element("button")
                            .classes(
                                UI_PRESENTATION["node.classes"].format(
                                    kind=node.kind,
                                    offset_cls=offset_cls,
                                    sel_cls=sel_cls,
                                )
                            )
                            .style(
                                UI_PRESENTATION["node.style"].format(
                                    x=node.x, y=node.y, color=node.color
                                )
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
                                UI_PRESENTATION[
                                    "render_stage_and_menu.classes.text_10px_font_mono_font_bold_text_zinc_950_p"
                                ]
                            )
                            caption_pos = (
                                UI_PRESENTATION["node.skill.caption"]
                                if node.depth > 0
                                else "-top-5"
                            )
                            ui.label(node.short_label).classes(
                                UI_PRESENTATION["node.caption"].format(
                                    caption_pos=caption_pos
                                )
                            )

        self._render_bottom_horizontal_menu(graph_data.find_node(selected_id))

    def _render_bottom_horizontal_menu(self, selected_node: GraphNode | None) -> None:
        if not self.bottom_menu_container:
            return

        self.bottom_menu_container.clear()
        with self.bottom_menu_container:
            if selected_node is None:
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_bottom_horizontal_menu.classes.graph_bottom_menu_placeholder_w_full_items_ce"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_bottom_horizontal_menu.classes.items_center_gap_2"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.icon.touch_app"
                            ],
                            size=UI_PRESENTATION[
                                "render_bottom_horizontal_menu.size.xs"
                            ],
                        ).classes(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.classes.text_sky_400"
                            ]
                        )
                        ui.label(
                            ui_text("graph.graph_view.click_a_node_on_the_graph_to")
                        )
                return

            with ui.column().classes(
                UI_PRESENTATION[
                    "render_bottom_horizontal_menu.classes.graph_bottom_menu_open_w_full_bg_0e0e12_borde"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_bottom_horizontal_menu.classes.w_full_items_center_justify_between_pb_2_bord"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_bottom_horizontal_menu.classes.items_center_gap_2_flex_wrap"
                        ]
                    ):
                        ui.html(
                            render_html(
                                "shared/color_dot.html",
                                size=DRAWING_DEFAULTS["history_label_length"],
                                color=selected_node.color,
                            )
                        )
                        ui.label(
                            ui_text(
                                "graph.graph_view.node",
                                selected_node_step_index_1=selected_node.step_index + 1,
                                selected_node_label=selected_node.label,
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.classes.text_xs_font_mono_font_semibold_text_zinc_100"
                            ]
                        )
                        ui.label(selected_node.history_label).classes(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.classes.text_11px_font_mono_text_sky_300_bg_sky_950_6"
                            ]
                        )
                        if selected_node.depth > 0:
                            ui.label(
                                ui_text(
                                    "graph.graph_view.skill_counterclockwise_offset_downward_step",
                                    selected_node_depth=selected_node.depth,
                                    selected_node_skill_path=selected_node.skill_path,
                                )
                            ).classes(
                                UI_PRESENTATION[
                                    "render_bottom_horizontal_menu.classes.text_11px_font_mono_text_purple_300_bg_purple"
                                ]
                            )
                        ui.label(
                            ui_text(
                                "graph.graph_view.events_in_node",
                                len_selected_node_events=len(selected_node.events),
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.classes.text_11px_font_mono_text_amber_300_bg_amber_9"
                            ]
                        )

                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_bottom_horizontal_menu.classes.items_center_gap_1_5"
                        ]
                    ):
                        all_events_json = json.dumps(
                            selected_node.events, indent=2, ensure_ascii=False
                        )
                        ui.button(
                            ui_text("graph.graph_view.copy_node_events"),
                            icon=UI_PRESENTATION[
                                "render_bottom_horizontal_menu.icon.content_copy"
                            ],
                            on_click=lambda _, t=all_events_json: self._copy_json(
                                t, app_text("graph.graph_view.node_events_copied")
                            ),
                        ).props(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.props.flat_dense_no_caps_size_xs_text_color_grey_3"
                            ]
                        ).classes(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.classes.text_11px_px_2_5_py_0_5_bg_zinc_900_hover_bg_"
                            ]
                        )
                        ui.button(
                            icon=UI_PRESENTATION[
                                "render_bottom_horizontal_menu.icon.close"
                            ],
                            on_click=lambda: self.select_node(None),
                        ).props(
                            UI_PRESENTATION[
                                "render_bottom_horizontal_menu.props.flat_round_dense_size_xs_text_color_grey_4_ar"
                            ]
                        )

                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_bottom_horizontal_menu.classes.graph_horizontal_events_row_w_full_flex_row_f"
                    ]
                ):
                    for idx, ev in enumerate(selected_node.events):
                        self._render_horizontal_event_card(ev, index=idx + 1)

    def _render_horizontal_event_card(self, ev: dict[str, Any], index: int) -> None:
        role = str(ev.get("role", "event")).lower()
        stage_label = str(
            ev.get("stage_label") or ev.get("event_stage") or role.upper()
        )

        if role in ("system", "developer"):
            border_cls = UI_PRESENTATION["event.system.border"]
            badge_cls = UI_PRESENTATION["event.system.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.terminal"]
        elif ev.get("event_stage") == "history_compaction":
            border_cls = UI_PRESENTATION["event.compaction.border"]
            badge_cls = UI_PRESENTATION["event.compaction.badge"]
            icon_name = UI_PRESENTATION["event.compaction.icon"]
        elif ev.get("event_stage") == "steering_message":
            border_cls = UI_PRESENTATION["event.steering.border"]
            badge_cls = UI_PRESENTATION["event.steering.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.alt_route"]
        elif role == "user":
            border_cls = UI_PRESENTATION["event.user.border"]
            badge_cls = UI_PRESENTATION["event.user.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.person"]
        elif role == "tool":
            border_cls = UI_PRESENTATION["event.tool.border"]
            badge_cls = UI_PRESENTATION["event.tool.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.output"]
        elif role == "execution":
            border_cls = UI_PRESENTATION["event.execution.border"]
            badge_cls = UI_PRESENTATION["event.execution.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.play_circle"]
        else:
            border_cls = UI_PRESENTATION["event.assistant.border"]
            badge_cls = UI_PRESENTATION["event.assistant.badge"]
            icon_name = UI_PRESENTATION["render_horizontal_event_card.icon.smart_toy"]

        with ui.column().classes(
            UI_PRESENTATION["event.card"].format(border_cls=border_cls)
        ):
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_horizontal_event_card.classes.w_full_items_center_justify_between"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_horizontal_event_card.classes.items_center_gap_1_5_flex_wrap"
                    ]
                ):
                    ui.icon(
                        icon_name,
                        size=UI_PRESENTATION["render_horizontal_event_card.size.xs"],
                    ).classes(
                        UI_PRESENTATION[
                            "render_horizontal_event_card.classes.text_zinc_400"
                        ]
                    )
                    with ui.row().classes(
                        UI_PRESENTATION["event.badge"].format(badge_cls=badge_cls)
                    ):
                        ui.label(stage_label)
                        if ev.get("pending"):
                            ui.label(ui_text("graph.graph_view.pending")).classes(
                                UI_PRESENTATION[
                                    "render_horizontal_event_card.classes.text_orange_300"
                                ]
                            )
                    ui.label(ui_text("graph.graph_view.event", index=index)).classes(
                        UI_PRESENTATION[
                            "render_horizontal_event_card.classes.font_mono_text_10px_text_zinc_500"
                        ]
                    )
                ev_json = json.dumps(ev, indent=2, ensure_ascii=False)
                ui.button(
                    icon=UI_PRESENTATION[
                        "render_horizontal_event_card.icon.content_copy"
                    ],
                    on_click=lambda _, text=ev_json: self._copy_json(
                        text, app_text("graph.graph_view.event_copied")
                    ),
                ).props(
                    UI_PRESENTATION[
                        "render_horizontal_event_card.props.flat_round_dense_size_xs"
                    ]
                ).classes(
                    UI_PRESENTATION[
                        "render_horizontal_event_card.classes.text_zinc_600_hover_text_zinc_300"
                    ]
                )

            highlighted_html = format_json_to_highlighted_html(ev)
            ui.html(
                render_html(
                    "shared/raw_json.html",
                    content=Markup(highlighted_html),
                    extra_classes="max-h-[220px] overflow-y-auto",
                )
            ).classes(UI_PRESENTATION["render_horizontal_event_card.classes.w_full"])

    @staticmethod
    def _copy_json(text: str, notification: str) -> None:
        ui.clipboard.write(text)
        ui.notify(
            notification,
            type="positive",
            position="top",
            timeout=EXECUTION_DEFAULTS["graph_notification_timeout_ms"],
        )
