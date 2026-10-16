from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

from nicegui import ui

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.exmekh_bridge.history.image_input import IMAGE_SUFFIXES
from exmekh_agent.exmekh_bridge.history.message_edit import retained_attachment_paths
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    extract_attached_files_from_content,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ExecutionTimeline,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SteeringInsertion,
    ToolEvent,
    record_tool_event,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    compute_skill_node_closure_tokens,
    format_skill_closure_tokens,
    populate_skill_closure_tokens,
)
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.ui.chat.chat_export import ExportFormat, export_chat
from exmekh_agent.ui.chat.message_editor import EditMessageCallback, MessageEditor
from exmekh_agent.ui.chat.message_text import message_display_text
from exmekh_agent.ui.graph.graph_view import GraphViewComponent
from exmekh_agent.ui.history.raw_history_view import RawHistoryComponent
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text
from exmekh_agent.ui.shared.disclosure import Disclosure, DisclosureState
from exmekh_agent.ui.shared.file_icons import get_file_icon
from exmekh_agent.ui.shared.image_preview import render_image_thumbnail
from exmekh_agent.ui.shared.markdown_parser import parse_markdown_to_html
from exmekh_agent.ui.shared.output_format import format_output_html
from exmekh_agent.utils.formatting import format_duration

UI_PRESENTATION = presentation_config("controls/chat_view")

CHAT_SCROLL_JS_PATH: Path = (
    Path(__file__).parents[1] / "resources" / "js" / "chat_scroll.js"
)


def get_chat_scroll_js() -> str:
    """Read and return the raw JavaScript rules from chat_scroll.js."""
    if CHAT_SCROLL_JS_PATH.exists():
        return CHAT_SCROLL_JS_PATH.read_text(encoding="utf-8")
    return ""


def load_chat_scroll_js() -> None:
    """Inject the smart chat scroll JS script into the current NiceGUI page."""
    js_content = get_chat_scroll_js()
    if js_content:
        ui.add_head_html(f"<script>\n{js_content}\n</script>")


def _upsert_streaming_skill(
    skills: list[SkillNode], incoming: SkillNode, action: str
) -> None:
    """Insert or update a SkillNode inside the streaming skills list, preserving start_time."""
    incoming.finalize_timing(action if action in ("completed", "error") else None)

    for root in reversed(skills):
        existing = root.find_invocation(incoming.path, incoming.opening_tool_call_id)
        if existing is not None:
            existing.merge_snapshot(incoming)
            return

    if incoming.depth <= 1 or not skills:
        skills.append(incoming)
        return

    for existing in skills:
        if existing.attach_descendant(incoming):
            return
    skills.append(incoming)


class ChatViewComponent:
    def __init__(
        self,
        on_suggested_prompt: Callable[[str], None],
        on_edit_message: EditMessageCallback | None = None,
        on_agent_settings: Callable[[], None] | None = None,
        sessions: SessionManager | None = None,
    ) -> None:
        self._sessions = sessions
        self.on_suggested_prompt = on_suggested_prompt
        self.on_edit_message = on_edit_message
        self.on_agent_settings = on_agent_settings
        self.agent_settings_button: ui.button | None = None
        self.chat_container: ui.column | None = None
        self.child_chat_container: ui.column | None = None
        self.other_chat_container: ui.column | None = None
        self.stream_session_id: str | None = None
        self.current_stream_card: ui.row | None = None
        self.current_steering_container: ui.column | None = None
        self.streaming_steering_ids: set[str] = set()
        self.streaming_steering_insertions: list[SteeringInsertion] = []
        self.streaming_steering_spinners: dict[str, ui.spinner] = {}
        self._rendering_timeline: ExecutionTimeline | None = None
        self.current_html: ui.html | None = None
        self.current_response_text: str = ""
        self.stream_started_at: float | None = None
        self.stream_duration_label: ui.label | None = None
        self.stream_timer: ui.timer | None = None
        self.current_thinking_label: ui.html | None = None
        self.current_live_reasoning_timer_label: ui.label | None = None
        self.current_loading_placeholder: ui.row | None = None
        self.current_reasoning_container: ui.column | None = None
        self.current_live_reasoning_container: ui.column | None = None
        self.current_live_reasoning_block: ReasoningBlock | None = None
        self.current_skills_container: ui.column | None = None
        self.streaming_reasoning_blocks: list[ReasoningBlock] = []
        self.streaming_skills: list[SkillNode] = []
        self.streaming_root_tools: list[ToolEvent] = []
        self.live_skill_timer_labels: list[tuple[SkillNode, ui.label]] = []
        self.live_tool_timer_labels: list[tuple[ToolEvent, ui.label]] = []
        self.current_master_reasoning_expansion: ui.expansion | None = None
        self.active_tab: str = "chat"
        self.selected_sub_agent_id: str | None = None
        self.path_label: ui.label | None = None
        self.raw_history_view = RawHistoryComponent()
        self.child_raw_history_view = RawHistoryComponent()
        self.other_raw_history_view = RawHistoryComponent()
        self.raw_history_container: ui.column | None = None
        self.root_raw_element: ui.column | None = None
        self.child_raw_element: ui.column | None = None
        self.other_raw_element: ui.column | None = None
        self.graph_view = GraphViewComponent()
        self.graph_container: ui.column | None = None
        self.tabs_ref: ui.tabs | None = None
        self.tab_chat: ui.tab | None = None
        self.tab_raw: ui.tab | None = None
        self.tab_graph: ui.tab | None = None
        self.scroll_reset_seq: int = 0
        self.scroll_to_bottom_btn: ui.button | None = None
        self.poi_track: ui.element | None = None
        self.include_subskill_history: ui.switch | None = None
        self._rendering_message: ChatMessage | None = None
        self._streaming_history: ChatMessage | None = None
        self._disclosures = DisclosureState()

    @property
    def sessions(self) -> SessionManager:
        return self._sessions if self._sessions is not None else session_manager

    def build(self) -> ui.column:
        load_chat_scroll_js()
        with ui.column().classes(
            UI_PRESENTATION[
                "build.classes.w_full_h_full_flex_1_min_h_0_flex_grow_overfl"
            ]
        ) as outer_container:
            # 1. Top Sub-Tabs Bar directly under header
            with ui.row().classes(
                UI_PRESENTATION[
                    "build.classes.chat_subtabs_toolbar_w_full_border_b_border_z"
                ]
            ):
                with (
                    ui.tabs(value="chat", on_change=self._handle_tab_changed)
                    .props(
                        UI_PRESENTATION[
                            "build.props.dense_no_caps_inline_label_indicator_color_sk"
                        ]
                    )
                    .classes(UI_PRESENTATION["build.classes.sub_tabs_bar"]) as tabs
                ):
                    self.tabs_ref = tabs
                    self.tab_chat = ui.tab(
                        "chat",
                        label=ui_text("chat.chat_view.chat"),
                        icon=UI_PRESENTATION["build.icon.chat"],
                    )
                    self.tab_raw = ui.tab(
                        "raw_history",
                        label=ui_text("chat.chat_view.raw_history"),
                        icon=UI_PRESENTATION["build.icon.data_object"],
                    )
                    self.tab_graph = ui.tab(
                        "graph",
                        label=ui_text("chat.chat_view.graph"),
                        icon=UI_PRESENTATION["build.icon.account_tree"],
                    )
                with ui.row().classes(
                    UI_PRESENTATION[
                        "build.classes.chat_tab_tools_items_center_gap_2_min_w_0_fle"
                    ]
                ):
                    if self.on_agent_settings:
                        self.agent_settings_button = (
                            ui.button(
                                ui_text("chat.chat_view.agent"),
                                icon=UI_PRESENTATION["build.icon.account_tree"],
                                on_click=self.on_agent_settings,
                            )
                            .props(
                                UI_PRESENTATION[
                                    "build.props.flat_dense_no_caps_aria_label_chat_agent_sett"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "build.classes.chat_agent_settings_text_xs"
                                ]
                            )
                        )
                    self.path_label = ui.label("").classes(
                        UI_PRESENTATION[
                            "build.classes.text_10px_font_mono_text_sky_300_truncate"
                        ]
                    )
                    with (
                        ui.button(icon=UI_PRESENTATION["build.icon.more_vert"])
                        .props(
                            UI_PRESENTATION[
                                "build.props.flat_round_dense_size_sm_aria_label_chat_menu"
                            ]
                        )
                        .classes(
                            UI_PRESENTATION[
                                "build.classes.chat_export_menu_btn_text_zinc_400"
                            ]
                        )
                    ):
                        ui.tooltip(ui_text("chat.chat_view.copy_entire_chat"))
                        with ui.menu().classes(
                            UI_PRESENTATION[
                                "build.classes.bg_zinc_900_border_border_zinc_700"
                            ]
                        ):
                            ui.label(
                                ui_text("chat.chat_view.copy_entire_chat")
                            ).classes(
                                UI_PRESENTATION[
                                    "build.classes.text_xs_text_zinc_400_px_4_pt_2"
                                ]
                            )
                            self.include_subskill_history = (
                                ui.switch(
                                    ui_text("chat.chat_view.include_subskill_history"),
                                    value=False,
                                )
                                .classes(UI_PRESENTATION["build.classes.text_xs_px_3"])
                                .props(UI_PRESENTATION["build.props.dense"])
                            )
                            ui.separator()
                            ui.menu_item(
                                app_text("chat.chat_view.copy_as_json"),
                                on_click=lambda: self._copy_chat("json"),
                            )
                            ui.menu_item(
                                app_text("chat.chat_view.copy_as_markdown"),
                                on_click=lambda: self._copy_chat("markdown"),
                            )

            # 2. Tab 1 Container: Chat
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.chat_scroll_container_w_full_flex_1_min_h_0_f"
                ]
            ) as chat_box:
                chat_box.props["data-scroll-reset-seq"] = "0"
                chat_box.props["data-scroll-pinned"] = "true"
                self.chat_container = chat_box
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.chat_scroll_container_w_full_flex_1_min_h_0_f"
                ]
            ) as child_chat_box:
                child_chat_box.props["data-scroll-reset-seq"] = "0"
                child_chat_box.props["data-scroll-pinned"] = "true"
                self.child_chat_container = child_chat_box
                child_chat_box.set_visibility(False)
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.chat_scroll_container_w_full_flex_1_min_h_0_f"
                ]
            ) as other_chat_box:
                other_chat_box.props["data-scroll-reset-seq"] = "0"
                other_chat_box.props["data-scroll-pinned"] = "true"
                self.other_chat_container = other_chat_box
                other_chat_box.set_visibility(False)

            # 3. Tab 2 Container: Raw History
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.chat_scroll_container_w_full_max_w_5xl_mx_aut"
                ]
            ) as raw_box:
                raw_box.props["data-scroll-reset-seq"] = "0"
                raw_box.props["data-scroll-pinned"] = "true"
                self.raw_history_container = raw_box
                self.root_raw_element = self.raw_history_view.build()
                self.child_raw_element = self.child_raw_history_view.build()
                self.other_raw_element = self.other_raw_history_view.build()
                self.child_raw_element.set_visibility(False)
                self.other_raw_element.set_visibility(False)
                raw_box.set_visibility(False)

            # 4. Tab 3 Container: Graph (Interactive Star Graph per Chat)
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.w_full_max_w_6xl_mx_auto_flex_1_min_h_0_flex_"
                ]
            ) as graph_box:
                self.graph_container = graph_box
                self.graph_view.build()
                graph_box.set_visibility(False)

            # Floating "Scroll to bottom" indicator button (shown when detached from bottom)
            self.scroll_to_bottom_btn = (
                ui.button(
                    icon=UI_PRESENTATION["build.icon.arrow_downward"],
                    on_click=lambda: self.reset_scroll_to_bottom(),
                )
                .props(
                    UI_PRESENTATION[
                        "build.props.id_exmekh_scroll_to_bottom_btn_round_dense_co"
                    ]
                )
                .classes(UI_PRESENTATION["build.classes.scroll_to_bottom_btn"])
            )

            # Floating points of interest rail near right scrollbar
            self.poi_track = (
                ui.element("div")
                .props(UI_PRESENTATION["build.props.id_exmekh_chat_poi_track"])
                .classes(UI_PRESENTATION["build.classes.chat_poi_track"])
            )

            self.render_current_session()
        return outer_container

    async def _copy_chat(self, export_format: ExportFormat) -> None:
        client = ui.context.client
        session = self.sessions.snapshot_current_session()
        include_history = bool(
            self.include_subskill_history and self.include_subskill_history.value
        )
        text = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: export_chat(
                session, export_format, include_history=include_history
            ),
        )
        with client:
            ui.clipboard.write(text)
            ui.notify(
                ui_text(
                    "chat.chat_view.chat_copied_as",
                    export_format_upper=export_format.upper(),
                ),
                type="positive",
            )

    def _render_copy_button(self, text: str, *, user: bool = False) -> None:
        ui.button(
            icon=UI_PRESENTATION["render_copy_button.icon.content_copy"],
            on_click=lambda: ui.clipboard.write(text),
        ).props(
            UI_PRESENTATION[
                "render_copy_button.props.flat_round_dense_size_xs_aria_label_copy_mess"
            ]
        ).classes(
            UI_PRESENTATION["render_copy_button.state_classes"]
            + (UI_PRESENTATION["render_copy_button.state_classes_2"] if user else "")
        ).tooltip(ui_text("chat.chat_view.copy_message"))

    def _handle_tab_changed(self, event: Any) -> None:
        tab_val = getattr(event, "value", event)
        self.switch_tab(str(tab_val))

    def switch_tab(self, tab: str) -> None:
        self.active_tab = tab
        self._set_chat_visibility()
        if self.raw_history_container:
            self.raw_history_container.set_visibility(tab == "raw_history")
            if tab == "raw_history":
                self._render_selected_raw_history()
        if self.graph_container:
            self.graph_container.set_visibility(tab == "graph")
            if tab == "graph":
                self._render_selected_graph()
        if self.tabs_ref and self.tabs_ref.value != tab:
            self.tabs_ref.value = tab

    def _render_selected_graph(self) -> None:
        if not self.graph_view:
            return
        session = self.sessions.get_current_session()
        is_generating = self.is_streaming_session(session.id) or any(
            sub.status == "active" for sub in session.sub_agent_conversations
        )
        self.graph_view.render_session(
            session,
            selected_sub_agent_id=self.selected_sub_agent_id,
            is_generating=is_generating,
        )

    def render_current_session(self) -> None:
        session = self.sessions.get_current_session()
        if self.agent_settings_button:
            self.agent_settings_button.set_visibility(
                session.agent_mode == "deep_agent"
            )
        child = session.find_sub_agent_conversation(self.selected_sub_agent_id)
        populate_skill_closure_tokens(session, sub_agent=child)
        if self.path_label:
            self.path_label.text = f"{child.path} #{child.order}" if child else ""

        if self.raw_history_container:
            self._render_selected_raw_history()
        if self.graph_container:
            self._render_selected_graph()

        if not self.chat_container:
            return
        scope = self._display_scope(session)
        target = {
            "root": self.chat_container,
            "child": self.child_chat_container,
            "other": self.other_chat_container,
        }[scope]
        self._set_chat_visibility()
        if target is None:
            return
        if target is self.chat_container and self.current_stream_card:
            return

        target.clear()

        with target:
            messages = child.messages if child else session.messages
            if not messages:
                self._render_welcome_hero()
                return

            for msg in messages:
                self._render_message(msg)

    def _set_chat_visibility(self) -> None:
        scope = self._display_scope(self.sessions.get_current_session())
        if self.chat_container:
            self.chat_container.set_visibility(
                self.active_tab == "chat" and scope == "root"
            )
        if self.child_chat_container:
            self.child_chat_container.set_visibility(
                self.active_tab == "chat" and scope == "child"
            )
        if self.other_chat_container:
            self.other_chat_container.set_visibility(
                self.active_tab == "chat" and scope == "other"
            )

    def _display_scope(self, session: ChatSession) -> Literal["root", "child", "other"]:
        if self.selected_sub_agent_id is not None:
            return "child"
        if self.current_stream_card and self.stream_session_id != session.id:
            return "other"
        return "root"

    def _raw_display_session(self) -> ChatSession:
        session = self.sessions.get_current_session()
        child = session.find_sub_agent_conversation(self.selected_sub_agent_id)
        if child is None:
            return session
        return ChatSession(
            title=child.path,
            messages=[ChatMessage(role="assistant", raw_history=child.raw_history)]
            if child.raw_history
            else [],
        )

    def _render_selected_raw_history(self) -> None:
        scope = self._display_scope(self.sessions.get_current_session())
        if self.root_raw_element:
            self.root_raw_element.set_visibility(scope == "root")
        if self.child_raw_element:
            self.child_raw_element.set_visibility(scope == "child")
        if self.other_raw_element:
            self.other_raw_element.set_visibility(scope == "other")
        if scope == "child":
            self.child_raw_history_view.render_session(self._raw_display_session())
        elif scope == "other":
            self.other_raw_history_view.render_session(
                self.sessions.get_current_session()
            )
        elif not self.raw_history_view.is_streaming:
            self.raw_history_view.render_session(self.sessions.get_current_session())

    def is_streaming_session(self, session_id: str) -> bool:
        return (
            self.stream_session_id == session_id
            and self.current_stream_card is not None
        )

    def has_active_stream(self) -> bool:
        return self.current_stream_card is not None

    def finalize_session_stream(self, session_id: str) -> None:
        """Close a session's live UI and refresh it when that session is visible."""
        visible = self.sessions.is_current_session(session_id)
        if self.is_streaming_session(session_id) or (
            visible and self.has_active_stream()
        ):
            self.finalize_stream()
        if visible:
            self.render_current_session()

    def is_showing_sub_agent(self, conversation_id: str) -> bool:
        return self.selected_sub_agent_id == conversation_id

    def reset_scroll_to_bottom(self, target: ui.column | None = None) -> None:
        """Pin the chat scroll container back to the bottom and scroll down."""
        self.scroll_reset_seq += 1
        seq_str = str(self.scroll_reset_seq)
        containers = (
            [target]
            if target is not None
            else [
                self.chat_container,
                self.child_chat_container,
                self.other_chat_container,
                self.raw_history_container,
            ]
        )
        for container in containers:
            if container is not None:
                container.props["data-scroll-reset-seq"] = seq_str
                container.props["data-scroll-pinned"] = "true"
                container.update()

    def show_sub_agent(self, conversation_id: str) -> None:
        self.selected_sub_agent_id = conversation_id
        self.render_current_session()
        self.switch_tab(self.active_tab)
        self.reset_scroll_to_bottom(self.child_chat_container)

    def show_main(self) -> None:
        self.selected_sub_agent_id = None
        self.render_current_session()
        self.switch_tab(self.active_tab)
        self.reset_scroll_to_bottom(self.chat_container)

    def _render_welcome_hero(self) -> None:
        with ui.column().classes(
            UI_PRESENTATION[
                "render_welcome_hero.classes.items_center_justify_center_w_full_py_16_gap_"
            ]
        ):
            ui.icon(
                UI_PRESENTATION["render_welcome_hero.icon.terminal"],
                size=UI_PRESENTATION["render_welcome_hero.size.md"],
            ).classes(UI_PRESENTATION["render_welcome_hero.classes.text_zinc_400_mb_1"])
            ui.label(ui_text("chat.chat_view.exmekh_agent")).classes(
                UI_PRESENTATION[
                    "render_welcome_hero.classes.text_2xl_font_bold_tracking_tight_text_zinc_1"
                ]
            )
            ui.label(
                ui_text(
                    "chat.chat_view.hierarchical_agentic_framework_with_multi_depth_execution"
                )
            ).classes(
                UI_PRESENTATION[
                    "render_welcome_hero.classes.text_sm_text_zinc_400_max_w_md"
                ]
            )

            with ui.row().classes(
                UI_PRESENTATION[
                    "render_welcome_hero.classes.gap_2_mt_4_flex_wrap_justify_center"
                ]
            ):
                suggestions = [
                    (
                        app_text("chat.chat_view.run_mock_tool"),
                        app_text(
                            "chat.chat_view.use_the_mock_tool_to_verify_tool_execution"
                        ),
                    ),
                    (
                        app_text("chat.chat_view.demonstrate_nested_skills"),
                        app_text(
                            "chat.chat_view.use_the_mock_skill_and_invoke_sub_tools_within_it_to_demonstrate_nesti"
                        ),
                    ),
                    (
                        app_text("chat.chat_view.reasoning_test"),
                        app_text(
                            "chat.chat_view.solve_step_by_step_how_many_r_s_are_in_the_word_strawberry"
                        ),
                    ),
                ]
                for label, prompt in suggestions:
                    ui.button(
                        label, on_click=lambda _, p=prompt: self.on_suggested_prompt(p)
                    ).props(
                        UI_PRESENTATION[
                            "render_welcome_hero.props.flat_rounded_lg_no_caps_dense"
                        ]
                    ).classes(
                        UI_PRESENTATION[
                            "render_welcome_hero.classes.text_zinc_400_hover_text_white_text_xs_px_3_p"
                        ]
                    )

    async def _show_attached_file_modal(self, file_path: str) -> None:
        filename = Path(file_path).name
        session = self.sessions.get_current_session()
        storage = self.sessions.temp_storage(session)
        client = ui.context.client
        is_binary = False
        icon_name, color_class, _ = get_file_icon(filename)
        try:
            full_path = storage.resolve_attachment(file_path, allow_legacy=True)
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, full_path.read_bytes)
            if b"\x00" in data:
                is_binary = True
                content_text = app_text(
                    "chat.chat_view.binary_file_value_1_bytes_text_preview_unavailable",
                    value_1=len(data),
                )
            else:
                try:
                    content_text = data.decode("utf-8")
                except UnicodeDecodeError:
                    is_binary = True
                    content_text = app_text(
                        "chat.chat_view.file_with_non_utf_8_encoding_value_1_bytes_text_preview_unavailable",
                        value_1=len(data),
                    )
        except ValueError:
            content_text = app_text("chat.chat_view.invalid_attachment_path")
        except FileNotFoundError:
            content_text = app_text("chat.chat_view.attached_file_not_found")
        except OSError:
            content_text = app_text("chat.chat_view.unable_to_read_the_file_from_disk")

        with client:
            with (
                ui.dialog() as dlg,
                ui.card().classes(
                    UI_PRESENTATION[
                        "show_attached_file_modal.classes.bg_zinc_900_border_border_zinc_800_text_zinc_"
                    ]
                ),
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "show_attached_file_modal.classes.w_full_items_center_justify_between_border_b_"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "show_attached_file_modal.classes.items_center_gap_1_5_min_w_0_flex_1"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION[
                                "show_attached_file_modal.icon.attach_file"
                            ],
                            size=UI_PRESENTATION["show_attached_file_modal.size.18px"],
                        ).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.text_zinc_400_shrink_0"
                            ]
                        )
                        ui.icon(
                            icon_name,
                            size=UI_PRESENTATION["show_attached_file_modal.size.18px"],
                        ).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes_template"
                            ].format(color_class=color_class)
                        )
                        dlg_lbl = ui.label(filename).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.font_mono_text_sm_font_semibold_text_zinc_200"
                            ]
                        )
                        dlg_lbl.tooltip(filename)
                    ui.button(
                        icon=UI_PRESENTATION["show_attached_file_modal.icon.close"],
                        on_click=dlg.close,
                    ).props(
                        UI_PRESENTATION[
                            "show_attached_file_modal.props.dense_flat_round_size_sm_text_color_zinc_400_"
                        ]
                    )
                if is_binary:
                    with ui.column().classes(
                        UI_PRESENTATION[
                            "show_attached_file_modal.classes.w_full_items_center_justify_center_p_6_gap_2_"
                        ]
                    ):
                        ui.icon(
                            icon_name,
                            size=UI_PRESENTATION["show_attached_file_modal.size.36px"],
                        ).classes(color_class)
                        ui.label(content_text).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.text_xs_font_mono_text_zinc_300"
                            ]
                        )
                        ui.label(
                            ui_text(
                                "chat.chat_view.path_formatted", file_path=file_path
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.text_10px_font_mono_text_zinc_500"
                            ]
                        )
                        ui.label(
                            ui_text("chat.chat_view.the_agent_can_access_the_file_on")
                        ).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.text_xs_text_zinc_400_italic"
                            ]
                        )
                else:
                    with ui.column().classes(
                        UI_PRESENTATION[
                            "show_attached_file_modal.classes.w_full_max_h_96_overflow_y_auto_bg_zinc_950_p"
                        ]
                    ):
                        ui.code(content_text).classes(
                            UI_PRESENTATION[
                                "show_attached_file_modal.classes.w_full_text_xs_font_mono"
                            ]
                        )
            dlg.open()

    def _open_message_editor(self, session_id: str, message: ChatMessage) -> None:
        session = self.sessions.get_session(session_id)
        if session is not None and self.on_edit_message:
            MessageEditor(self.on_edit_message).open(
                session_id, message, self.sessions.temp_storage(session)
            )

    def _render_message(self, msg: ChatMessage) -> None:
        for guidance in msg.steering_messages:
            self._render_message_body(
                guidance.to_chat_message(),
                steering=True,
            )
        self._render_message_body(msg)

    def _render_message_body(
        self,
        msg: ChatMessage,
        *,
        steering: bool = False,
        pending_steering: bool = False,
    ) -> None:
        if msg.role == "user":
            attached_files = [
                path
                for path in dict.fromkeys(
                    [
                        *retained_attachment_paths(msg),
                        *extract_attached_files_from_content(msg.content),
                    ]
                )
                if path not in msg.image_paths
                and not (
                    msg.image_urls
                    and not msg.image_paths
                    and Path(path).suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
                )
            ]
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_message_body.classes.w_full_justify_end_items_start_chat_message_r"
                ]
            ):
                bubble_class = (
                    UI_PRESENTATION["render_message_body.state_classes"]
                    if steering
                    else ""
                )
                with ui.column().classes(
                    UI_PRESENTATION["render_message_body.classes_template"].format(
                        bubble_class=bubble_class
                    )
                ):
                    if steering:
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.steering_message_label_items_center_gap_1_5"
                            ]
                        ):
                            ui.icon(
                                UI_PRESENTATION["render_message_body.icon.alt_route"],
                                size=UI_PRESENTATION["render_message_body.size.13px"],
                            )
                            ui.label(ui_text("chat.chat_view.steering"))
                            if pending_steering:
                                spinner = ui.spinner(
                                    size=UI_PRESENTATION[
                                        "render_message_body.size.11px"
                                    ],
                                    color=UI_PRESENTATION[
                                        "render_message_body.color.grey_5"
                                    ],
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_message_body.classes.steering_loading_spinner"
                                    ]
                                )
                                spinner.tooltip(
                                    ui_text(
                                        "chat.chat_view.pending_waiting_to_be_inserted_into_history"
                                    )
                                )
                                self.streaming_steering_spinners[msg.id] = spinner
                    display_text = message_display_text(msg)

                    if display_text:
                        ui.markdown(display_text).classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.text_sm_text_zinc_100"
                            ]
                        )

                    if msg.image_urls:
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.exmekh_message_images_gap_2_flex_wrap_mt_2"
                            ]
                        ):
                            for index, url in enumerate(msg.image_urls):
                                render_image_thumbnail(
                                    url,
                                    app_text(
                                        "chat.chat_view.attached_image_value_1",
                                        value_1=index + 1,
                                    ),
                                )

                    if attached_files:
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.exmekh_message_attachments_items_center_gap_1"
                            ]
                        ):
                            for file_path in attached_files:
                                filename = Path(file_path).name
                                icon_name, color_class, _ = get_file_icon(filename)
                                with (
                                    ui.row()
                                    .classes(
                                        UI_PRESENTATION[
                                            "render_message_body.classes.exmekh_message_attachment_chip_items_center_g"
                                        ]
                                    )
                                    .on(
                                        "click",
                                        lambda _, p=file_path: (
                                            self._show_attached_file_modal(p)
                                        ),
                                    )
                                ):
                                    ui.icon(
                                        UI_PRESENTATION[
                                            "render_message_body.icon.attach_file"
                                        ],
                                        size=UI_PRESENTATION[
                                            "render_message_body.size.13px"
                                        ],
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_message_body.classes.text_zinc_400_shrink_0"
                                        ]
                                    )
                                    ui.icon(
                                        icon_name,
                                        size=UI_PRESENTATION[
                                            "render_message_body.size.13px"
                                        ],
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_message_body.classes_template_2"
                                        ].format(color_class=color_class)
                                    )
                                    ui.label(filename).classes(
                                        UI_PRESENTATION[
                                            "render_message_body.classes.font_mono_text_xs_text_zinc_300_truncate_max_"
                                        ]
                                    )
                                    ui.tooltip(
                                        ui_text(
                                            "chat.chat_view.attached_file_click_to_preview",
                                            filename=filename,
                                        )
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_message_body.classes.text_xs_font_mono"
                                        ]
                                    )

                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_message_body.classes.user_message_footer_w_full_justify_end_items_"
                        ]
                    ):
                        ui.label(
                            datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                        ).classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.message_time_text_10px"
                            ]
                        )
                        self._render_copy_button(display_text, user=True)
                        if self.on_edit_message and self.selected_sub_agent_id is None:
                            session_id = self.sessions.get_current_session().id
                            ui.button(
                                icon=UI_PRESENTATION["render_message_body.icon.edit"],
                                on_click=lambda: self._open_message_editor(
                                    session_id, msg
                                ),
                            ).props(
                                UI_PRESENTATION[
                                    "render_message_body.props.flat_round_dense_size_xs_aria_label_edit_mess"
                                ]
                            ).classes(
                                UI_PRESENTATION[
                                    "render_message_body.classes.text_zinc_600_hover_text_zinc_300_response_co"
                                ]
                            ).tooltip(ui_text("chat.chat_view.edit_message"))

        else:
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_message_body.classes.w_full_justify_start_items_start_chat_message"
                ]
            ):
                with ui.column().classes(
                    UI_PRESENTATION[
                        "render_message_body.classes.msg_assistant_w_full_gap_2_5"
                    ]
                ):
                    if msg.error == "cancelled":
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.generation_stopped_items_center_gap_1_5_text_"
                            ]
                        ):
                            ui.icon(
                                UI_PRESENTATION["render_message_body.icon.stop_circle"],
                                size=UI_PRESENTATION["render_message_body.size.16px"],
                            )
                            ui.label(
                                ui_text("chat.chat_view.generation_stopped_by_user")
                            )
                    # Thinking Blocks (Reasoning separated by generation and path)
                    reasoning_blocks = msg.reasoning_blocks
                    if not reasoning_blocks and msg.reasoning_content:
                        reasoning_blocks = [
                            ReasoningBlock(
                                skill_path="/",
                                depth=0,
                                generation_index=1,
                                content=msg.reasoning_content,
                            )
                        ]

                    if reasoning_blocks:
                        with ui.column().classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.w_full_gap_1_reasoning_blocks_container"
                            ]
                        ):
                            self._render_reasoning_group(reasoning_blocks)

                    # Hierarchical Skills & Nested Blocks
                    with ui.column().classes(
                        UI_PRESENTATION["render_message_body.classes.w_full_gap_2"]
                    ):
                        self._render_turn_activity(
                            msg,
                        )

                    # Response Parsed HTML (with LaTeX MathML support)
                    if msg.content:
                        ui.html(parse_markdown_to_html(msg.content)).classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.text_sm_text_zinc_200_leading_relaxed_parsed_"
                            ]
                        )

                    # Footer actions
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_message_body.classes.w_full_justify_end_items_center_gap_2_pt_1"
                        ]
                    ):
                        if msg.duration_seconds is not None:
                            ui.label(format_duration(msg.duration_seconds)).classes(
                                UI_PRESENTATION[
                                    "render_message_body.classes.message_time_text_10px_font_mono"
                                ]
                            )
                        ui.label(
                            datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                        ).classes(
                            UI_PRESENTATION[
                                "render_message_body.classes.message_time_text_10px"
                            ]
                        )
                        self._render_copy_button(msg.content)

    def _render_reasoning_group(self, blocks: list[ReasoningBlock]) -> ui.expansion:
        """Renders multiple reasoning blocks compacted into a single expandable row,
        which expands to reveal the individual sub-blocks, each individually expandable.
        """
        count = len(blocks)
        total_duration = sum(b.elapsed_seconds() for b in blocks)
        expansion_key = f"reasoning:{blocks[0].id}"
        with ui.expansion(
            "",
            value=self._disclosures.is_expanded(expansion_key),
            on_value_change=lambda event: self._disclosures.set_expanded(
                expansion_key, event.value
            ),
        ).classes(
            UI_PRESENTATION[
                "render_reasoning_group.classes.w_full_reasoning_master_expansion_reasoning_e"
            ]
        ) as master_exp:
            with master_exp.add_slot("header"):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_reasoning_group.classes.reasoning_master_header_items_center_justify_"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_reasoning_group.classes.items_center_gap_2"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION["render_reasoning_group.icon.psychology"],
                            size=UI_PRESENTATION["render_reasoning_group.size.xs"],
                        ).classes(
                            UI_PRESENTATION[
                                "render_reasoning_group.classes.text_zinc_400"
                            ]
                        )
                        ui.label(ui_text("chat.chat_view.reasoning_process")).classes(
                            UI_PRESENTATION[
                                "render_reasoning_group.classes.font_mono_text_zinc_300_text_xs_font_medium"
                            ]
                        )
                        ui.label(
                            ui_text(
                                "chat.chat_view.value",
                                value_1=count,
                                value_2=app_text("chat.chat_view.block")
                                if count == 1
                                else app_text("chat.chat_view.blocks"),
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_reasoning_group.classes.font_mono_text_11px_text_zinc_400_reasoning_c"
                            ]
                        )
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_reasoning_group.classes.items_center_gap_2_font_mono_text_11px_text_z"
                        ]
                    ):
                        if count > 1:
                            unique_paths = list(
                                dict.fromkeys(b.skill_path for b in blocks)
                            )
                            if len(unique_paths) > 1:
                                paths_str = ", ".join(
                                    p
                                    if p != "/"
                                    else ui_text("chat.chat_view.steering_root")
                                    for p in unique_paths
                                )
                                ui.label(paths_str).classes(
                                    UI_PRESENTATION[
                                        "render_reasoning_group.classes.text_zinc_500_text_10px"
                                    ]
                                )
                        ui.label(format_duration(total_duration)).classes(
                            UI_PRESENTATION[
                                "render_reasoning_group.classes.message_time_text_10px_font_mono"
                            ]
                        )

            with (
                master_exp,
                ui.column().classes(
                    UI_PRESENTATION[
                        "render_reasoning_group.classes.w_full_pl_2_5_py_1_5_gap_1_5_border_l_border_"
                    ]
                ),
            ):
                for block in blocks:
                    self._render_reasoning_sub_block(block)

            return master_exp

    def _render_reasoning_sub_block(self, block: ReasoningBlock) -> ui.expansion:
        """Renders an individual reasoning sub-block with its path/gen badges and timer,
        expandable with a click to display its full thinking content.
        """
        block_duration = block.elapsed_seconds()
        expansion_key = f"reasoning-block:{block.id}"
        with ui.expansion(
            "",
            value=self._disclosures.is_expanded(expansion_key),
            on_value_change=lambda event: self._disclosures.set_expanded(
                expansion_key, event.value
            ),
        ).classes(
            UI_PRESENTATION[
                "render_reasoning_sub_block.classes.w_full_reasoning_sub_block_text_xs"
            ]
        ) as sub_exp:
            with sub_exp.add_slot("header"):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_reasoning_sub_block.classes.reasoning_header_reasoning_sub_header_items_c"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_reasoning_sub_block.classes.items_center_gap_2"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.icon.subdirectory_arrow_right"
                            ],
                            size=UI_PRESENTATION["render_reasoning_sub_block.size.xs"],
                        ).classes(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.classes.text_zinc_500"
                            ]
                        )
                        ui.label(
                            ui_text(
                                "chat.chat_view.step_formatted",
                                block_generation_index=block.generation_index,
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.classes.font_mono_text_zinc_400_text_xs_font_medium"
                            ]
                        )
                        ui.label(block.skill_path).classes(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.classes.font_mono_text_11px_text_zinc_400_reasoning_p"
                            ]
                        )
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_reasoning_sub_block.classes.items_center_gap_2_font_mono_text_11px_text_z"
                        ]
                    ):
                        if block.depth > 0:
                            ui.label(
                                ui_text("chat.chat_view.depth", depth=block.depth)
                            ).classes(
                                UI_PRESENTATION[
                                    "render_reasoning_sub_block.classes.text_zinc_500"
                                ]
                            )
                        ui.label(
                            ui_text(
                                "chat.chat_view.gen",
                                block_generation_index=block.generation_index,
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.classes.text_zinc_500_reasoning_gen_badge"
                            ]
                        )
                        ui.label(format_duration(block_duration)).classes(
                            UI_PRESENTATION[
                                "render_reasoning_sub_block.classes.message_time_text_10px_font_mono"
                            ]
                        )

            with sub_exp:
                ui.html(parse_markdown_to_html(block.content)).classes(
                    UI_PRESENTATION[
                        "render_reasoning_sub_block.classes.thinking_box_whitespace_pre_wrap_w_full_font_"
                    ]
                )

            return sub_exp

    def _render_skill_node(self, node: SkillNode) -> None:
        depth_level = min(max(node.depth, 1), 3)
        depth_class = f"skill-depth-{depth_level}"

        if node.status == "active":
            # Active state: seamless container with subtle guideline and live skill timer
            with ui.column().classes(
                UI_PRESENTATION["render_skill_node.classes_template"].format(
                    depth_class=depth_class
                )
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_skill_node.classes.items_center_justify_between_w_full"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION["render_skill_node.classes.items_center_gap_2"]
                    ):
                        ui.icon(
                            UI_PRESENTATION["render_skill_node.icon.hub"],
                            size=UI_PRESENTATION["render_skill_node.size.xs"],
                        ).classes(
                            UI_PRESENTATION["render_skill_node.classes.text_zinc_400"]
                        )
                        ui.label(
                            ui_text("chat.chat_view.active_skill", node_path=node.path)
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.font_mono_font_medium_text_zinc_200_text_xs"
                            ]
                        )
                    with ui.row().classes(
                        UI_PRESENTATION["render_skill_node.classes.items_center_gap_2"]
                    ):
                        ui.label(
                            ui_text("chat.chat_view.depth", depth=node.depth)
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_10px_font_mono_text_zinc_500"
                            ]
                        )
                        skill_timer_lbl = ui.label(
                            format_duration(node.elapsed_seconds())
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.message_time_text_10px_font_mono"
                            ]
                        )
                        self.live_skill_timer_labels.append((node, skill_timer_lbl))
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.skill_active_badge_text_10px_font_mono_items_"
                            ]
                        ):
                            ui.spinner(
                                UI_PRESENTATION["render_skill_node.spinner.dots"],
                                size=UI_PRESENTATION["render_skill_node.size.xs"],
                                color=UI_PRESENTATION["render_skill_node.color.grey_5"],
                            )
                            ui.label(ui_text("chat.chat_view.running"))

                # Render reasoning blocks inside this skill if present
                if node.reasoning_blocks:
                    with ui.column().classes(
                        UI_PRESENTATION["render_skill_node.classes.w_full_gap_1_pl_2"]
                    ):
                        self._render_reasoning_group(node.reasoning_blocks)

                with ui.column().classes(
                    UI_PRESENTATION["render_skill_node.classes.w_full_gap_2_pl_2"]
                ):
                    self._render_skill_activity(node)

        else:
            expansion_key = f"skill:{node.id}"
            expanded = self._disclosures.is_expanded(expansion_key)
            output_key = f"skill-output:{node.id}"
            # Completed state: seamless accordion header with open-to-close skill timer
            with ui.column().classes(
                UI_PRESENTATION["render_skill_node.classes_template_2"].format(
                    depth_class=depth_class
                )
            ):
                body_ref: ui.column | None = None
                chevron_ref: ui.icon | None = None

                # Compacted clickable bar
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_skill_node.classes.skill_compact_header_items_center_justify_bet"
                    ]
                ) as header:
                    with ui.row().classes(
                        UI_PRESENTATION["render_skill_node.classes.items_center_gap_2"]
                    ):
                        ui.icon(
                            "error" if node.status == "error" else "check_circle",
                            size=UI_PRESENTATION["render_skill_node.size.xs"],
                        ).classes(
                            UI_PRESENTATION["render_skill_node.state_classes"]
                            if node.status == "error"
                            else UI_PRESENTATION["render_skill_node.state_classes_2"]
                        )
                        ui.label(
                            ui_text("chat.chat_view.skill", node_name=node.name)
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.font_mono_font_medium_text_zinc_300_text_xs"
                            ]
                        )
                        ui.label(node.path).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_11px_text_zinc_500_font_mono"
                            ]
                        )

                    with ui.row().classes(
                        UI_PRESENTATION["render_skill_node.classes.items_center_gap_2"]
                    ):
                        if not node.tokens_pending:
                            before_tok, after_tok = compute_skill_node_closure_tokens(
                                node
                            )
                            token_badge_text = format_skill_closure_tokens(
                                before_tok, after_tok
                            )
                            with (
                                ui.row()
                                .classes(
                                    UI_PRESENTATION[
                                        "render_skill_node.classes.skill_closure_tokens_items_center_gap_1_px_1_"
                                    ]
                                )
                                .props(
                                    f'title="{ui_text("shared.context_compaction.tooltip", before=before_tok, after=after_tok)}"'
                                )
                            ):
                                ui.icon(
                                    UI_PRESENTATION["render_skill_node.icon.compress"],
                                    size=UI_PRESENTATION["render_skill_node.size.12px"],
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_skill_node.classes.text_emerald_400"
                                    ]
                                )
                                ui.label(token_badge_text).classes(
                                    UI_PRESENTATION[
                                        "render_skill_node.classes.skill_closure_tokens_label"
                                    ]
                                )
                        ui.label(
                            ui_text(
                                "chat.chat_view.actions",
                                node_action_count=node.action_count,
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_11px_font_mono_text_zinc_500"
                            ]
                        )
                        ui.label(
                            ui_text("chat.chat_view.depth", depth=node.depth)
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_11px_font_mono_text_zinc_500"
                            ]
                        )
                        ui.label(format_duration(node.elapsed_seconds())).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.message_time_text_10px_font_mono"
                            ]
                        )
                        chevron_ref = ui.icon(
                            "expand_less" if expanded else "expand_more",
                            size=UI_PRESENTATION["render_skill_node.size.xs"],
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_zinc_500_transition_transform_duration_2"
                            ]
                        )

                # Collapsible body containing all actions and nested sub-skills
                with ui.column().classes(
                    UI_PRESENTATION["render_skill_node.classes.w_full_py_2_pl_3_gap_2"]
                ) as body:
                    body_ref = body

                    if node.reasoning_blocks:
                        ui.label(
                            ui_text("chat.chat_view.reasoning_within_skill")
                        ).classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.text_11px_text_zinc_500_font_mono"
                            ]
                        )
                        self._render_reasoning_group(node.reasoning_blocks)

                    self._render_skill_activity(node)
                    with (
                        ui.expansion(
                            ui_text("chat.chat_view.skill_output"),
                            icon=UI_PRESENTATION["render_skill_node.icon.output"],
                            value=self._disclosures.is_expanded(output_key),
                            on_value_change=lambda event: (
                                self._disclosures.set_expanded(output_key, event.value)
                            ),
                        )
                        .props(UI_PRESENTATION["render_skill_node.props.dense"])
                        .classes(
                            UI_PRESENTATION[
                                "render_skill_node.classes.skill_output_expansion_w_full_text_xs"
                            ]
                        )
                    ):
                        self._render_output_panel(node.opening_tool_call_id, skill=True)

                Disclosure(
                    header,
                    body_ref,
                    label=app_text("chat.chat_view.show_skill_name", name=node.name),
                    expanded=expanded,
                    on_change=partial(self._disclosures.set_expanded, expansion_key),
                    chevron=chevron_ref,
                )

    def _render_skill_activity(self, node: SkillNode) -> None:
        activities = (
            self._rendering_timeline.skill_activity(node)
            if self._rendering_timeline is not None
            else node.ordered_activity()
        )
        self._render_scope_activity(activities, node=node)

    def _render_scope_activity(
        self,
        activities: list[SkillNode | ToolEvent | SteeringInsertion],
        *,
        node: SkillNode | None = None,
    ) -> None:
        for activity in activities:
            if isinstance(activity, SteeringInsertion):
                self._render_steering_insertion(activity)
                continue
            is_skill = isinstance(activity, SkillNode)
            if node is not None and node.status != "active":
                ui.label(
                    app_text("chat.chat_view.nested_sub_skills")
                    if is_skill
                    else app_text("chat.chat_view.actions_executed_in_skill")
                ).classes(
                    UI_PRESENTATION[
                        "render_skill_activity.classes.text_11px_text_zinc_500_font_mono"
                    ]
                )
            if is_skill:
                self._render_skill_node(activity)
            else:
                self._render_tool_event(activity)

    def _render_output_panel(
        self, call_id: str | None, *, skill: bool = False
    ) -> ui.column:
        message = self._rendering_message
        output = message.output_for_call(call_id) if message else None
        with ui.column().classes(
            "invocation-output-panel "
            + ("skill-output-panel" if skill else "tool-output-panel")
        ) as panel:
            ui.html(format_output_html(output)).classes(
                UI_PRESENTATION["render_output_panel.classes.w_full"]
            )
            if output is not None:
                ui.button(
                    icon=UI_PRESENTATION["render_output_panel.icon.content_copy"],
                    on_click=lambda: ui.clipboard.write(output),
                ).props(
                    UI_PRESENTATION[
                        "render_output_panel.props.flat_round_dense_size_xs_aria_label_copy_outp"
                    ]
                ).classes(
                    UI_PRESENTATION["render_output_panel.classes.output_copy_button"]
                )
        return panel

    def _render_tool_event(self, ev: ToolEvent) -> None:
        is_running = ev.is_running
        with ui.row().classes(
            UI_PRESENTATION[
                "render_tool_event.classes.tool_event_card_w_full_items_center_justify_b"
            ]
        ) as tool_row:
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_tool_event.classes.tool_event_main_items_center_gap_2"
                ]
            ):
                if is_running:
                    ui.spinner(
                        UI_PRESENTATION["render_tool_event.spinner.dots"],
                        size=UI_PRESENTATION["render_tool_event.size.xs"],
                        color=UI_PRESENTATION["render_tool_event.color.grey_5"],
                    )
                else:
                    ui.icon(
                        UI_PRESENTATION["render_tool_event.icon.terminal"],
                        size=UI_PRESENTATION["render_tool_event.size.xs"],
                    ).classes(
                        UI_PRESENTATION["render_tool_event.classes.text_zinc_500"]
                    )
                ui.label(
                    ui_text("chat.chat_view.tool", ev_tool_name=ev.tool_name)
                ).classes(
                    UI_PRESENTATION[
                        "render_tool_event.classes.tool_event_name_font_mono_font_medium_text_zi"
                    ]
                )
                arguments = (
                    self._rendering_message.arguments_for_tool(ev)
                    if self._rendering_message is not None
                    else ev.arguments
                )
                if arguments:
                    argument_preview = []
                    for name, value in arguments.items():
                        text = (
                            value
                            if isinstance(value, str)
                            else json.dumps(value, ensure_ascii=False)
                        )
                        argument_preview.append((name, " ".join(text.split())))
                    ui.html(
                        render_html(
                            "shared/tool_arguments.html", arguments=argument_preview
                        )
                    ).classes(
                        UI_PRESENTATION[
                            "render_tool_event.classes.tool_arguments_preview_font_mono_text_xs"
                        ]
                    ).tooltip(json.dumps(arguments, ensure_ascii=False, indent=2))
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_tool_event.classes.tool_event_meta_items_center_gap_2_text_11px_"
                ]
            ):
                if not is_running:
                    ui.icon(
                        "error" if ev.execution_status == "error" else "check_circle",
                        size=UI_PRESENTATION["render_tool_event.size.xs"],
                    ).classes(
                        UI_PRESENTATION["render_tool_event.state_classes"]
                        if ev.execution_status == "error"
                        else UI_PRESENTATION["render_tool_event.state_classes_2"]
                    ).tooltip(
                        app_text("chat.chat_view.failed")
                        if ev.execution_status == "error"
                        else app_text("chat.chat_view.completed")
                    )
                ui.label(ui_text("chat.chat_view.depth", depth=ev.depth))
                ui.label(ui_text("chat.chat_view.path", ev_skill_path=ev.skill_path))
                tool_timer_lbl = ui.label(
                    format_duration(ev.elapsed_seconds())
                ).classes(
                    UI_PRESENTATION[
                        "render_tool_event.classes.message_time_text_10px_font_mono"
                    ]
                )
                if is_running:
                    self.live_tool_timer_labels.append((ev, tool_timer_lbl))
        if not is_running:
            panel = self._render_output_panel(ev.tool_call_id)
            tool_row.classes(
                UI_PRESENTATION["render_tool_event.classes.tool_output_toggle"]
            )

            expansion_key = f"tool:{ev.id}"
            Disclosure(
                tool_row,
                panel,
                label=app_text(
                    "chat.chat_view.show_output_for_tool_name", tool_name=ev.tool_name
                ),
                expanded=self._disclosures.is_expanded(expansion_key),
                on_change=partial(self._disclosures.set_expanded, expansion_key),
            )

    def _render_turn_activity(self, message: ChatMessage) -> None:
        """Render the model's timeline identically during streaming and after reload."""
        self._rendering_timeline = message.execution_timeline()
        self._rendering_message = message
        try:
            self._render_scope_activity(self._rendering_timeline.root_activity())
        finally:
            self._rendering_timeline = None
            self._rendering_message = None

    def _render_steering_insertion(self, insertion: SteeringInsertion) -> None:
        location = (
            ui_text("chat.chat_view.steering_after_tool", tool_name=insertion.tool_name)
            if insertion.tool_name
            else ui_text("chat.chat_view.steering_before_response")
        )
        skill = (
            insertion.skill_path
            if insertion.skill_path != "/"
            else ui_text("chat.chat_view.steering_root")
        )
        with ui.row().classes(
            UI_PRESENTATION[
                "render_steering_insertion.classes.steering_tool_marker_items_center_gap_1_5"
            ]
        ):
            ui.icon(
                UI_PRESENTATION["render_steering_insertion.icon.alt_route"],
                size=UI_PRESENTATION["render_steering_insertion.size.13px"],
            )
            ui.label(
                ui_text(
                    "chat.chat_view.steering_inserted_skill",
                    location=location,
                    skill=skill,
                )
            )

    # Streaming support methods
    def start_assistant_stream(self, *, started_at: float | None = None) -> None:
        if not self.chat_container:
            return

        self._streaming_history = None
        self.stream_session_id = self.sessions.get_current_session().id
        self.stream_started_at = (
            started_at if started_at is not None else time.monotonic()
        )

        self.streaming_reasoning_blocks.clear()
        self.streaming_skills.clear()
        self.streaming_root_tools.clear()
        self.streaming_steering_insertions.clear()
        self.streaming_steering_spinners.clear()
        self.live_skill_timer_labels.clear()
        self.live_tool_timer_labels.clear()
        self.current_live_reasoning_container = None
        self.current_live_reasoning_block = None
        self.current_live_reasoning_timer_label = None
        self.current_master_reasoning_expansion = None

        with self.chat_container:
            self.current_steering_container = ui.column().classes(
                UI_PRESENTATION["start_assistant_stream.classes.w_full_gap_3"]
            )
            self.streaming_steering_ids.clear()
            with ui.row().classes(
                UI_PRESENTATION[
                    "start_assistant_stream.classes.w_full_justify_start_items_start_chat_message"
                ]
            ) as row:
                self.current_stream_card = row
                with ui.column().classes(
                    UI_PRESENTATION[
                        "start_assistant_stream.classes.msg_assistant_w_full_gap_2_5"
                    ]
                ):
                    # Initial loading indicator while waiting for LLM generation
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "start_assistant_stream.classes.items_center_gap_2_text_xs_text_zinc_500_font"
                        ]
                    ) as loader:
                        self.current_loading_placeholder = loader
                        ui.spinner(
                            UI_PRESENTATION["start_assistant_stream.spinner.dots"],
                            size=UI_PRESENTATION["start_assistant_stream.size.sm"],
                            color=UI_PRESENTATION[
                                "start_assistant_stream.color.grey_5"
                            ],
                        )
                        ui.label(ui_text("chat.chat_view.waiting_for_response"))

                    # Container for separated reasoning blocks
                    self.current_reasoning_container = ui.column().classes(
                        UI_PRESENTATION[
                            "start_assistant_stream.classes.w_full_gap_1_reasoning_blocks_container"
                        ]
                    )

                    # Dynamic skills container
                    self.current_skills_container = ui.column().classes(
                        UI_PRESENTATION["start_assistant_stream.classes.w_full_gap_2"]
                    )

                    # Parsed HTML response (with LaTeX MathML support)
                    self.current_response_text = ""
                    self.current_html = ui.html("").classes(
                        UI_PRESENTATION[
                            "start_assistant_stream.classes.text_sm_text_zinc_200_leading_relaxed_parsed_"
                        ]
                    )
                    self.stream_duration_label = ui.label(format_duration(0)).classes(
                        UI_PRESENTATION[
                            "start_assistant_stream.classes.message_time_text_10px_font_mono_self_end"
                        ]
                    )
                    self.stream_timer = ui.timer(
                        EXECUTION_DEFAULTS["stream_duration_interval_seconds"],
                        self._update_stream_duration,
                    )

        if self.raw_history_view:
            session = self.sessions.get_current_session()
            self.raw_history_view.start_live_turn(session)
        if self.graph_view:
            self._render_selected_graph()
        self.reset_scroll_to_bottom(self.chat_container)

    def restore_stream_snapshot(self, message: ChatMessage) -> None:
        """Show the latest live turn in a page opened after generation began."""
        if not self.current_stream_card:
            return
        previous = self._streaming_history
        execution_changed = previous is None or not message.has_same_execution_details(
            previous
        )
        self._streaming_history = message
        self.streaming_steering_insertions = message.confirmed_steering_insertions()
        self._update_steering_spinners()
        self.show_steering_messages(message.steering_messages)
        self.current_response_text = message.content
        if self.current_html:
            self.current_html.content = parse_markdown_to_html(message.content)
        if (
            message.content
            or message.reasoning_blocks
            or message.skills
            or message.tool_events
        ) and self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)

        self.streaming_reasoning_blocks = message.reasoning_blocks
        active_block = next(
            (
                block
                for block in reversed(message.reasoning_blocks)
                if block.end_time is None
            ),
            None,
        )
        self._render_streaming_reasoning(active_block=active_block)
        self.streaming_skills = message.skills
        self.streaming_root_tools = [
            event for event in message.tool_events if event.skill_path == "/"
        ]
        if execution_changed:
            self._refresh_streaming_skills_and_tools()

    def show_steering_messages(self, messages: list[QueuedMessage]) -> None:
        if self.current_steering_container is None:
            return
        with self.current_steering_container:
            for message in messages:
                if message.id in self.streaming_steering_ids:
                    continue
                is_inserted = any(
                    ins.message_id == message.id
                    for ins in self.streaming_steering_insertions
                )
                self._render_message_body(
                    message.to_chat_message(),
                    steering=True,
                    pending_steering=not is_inserted,
                )
                self.streaming_steering_ids.add(message.id)

    def update_stream_history(self, message: ChatMessage) -> None:
        """Publish newly recorded output even when no tool or skill changes state."""
        previous = self._streaming_history
        self._streaming_history = message
        if self.raw_history_view:
            self.raw_history_view.update_context_snapshot(self.sessions.get_current_session(), message)
        if previous is None or not message.has_same_execution_details(previous):
            self._refresh_streaming_skills_and_tools()

    def show_steering_insertions(self, message: ChatMessage) -> None:
        insertions = message.confirmed_steering_insertions()
        if insertions != self.streaming_steering_insertions:
            self.streaming_steering_insertions = insertions
            self._refresh_streaming_skills_and_tools()
        self._update_steering_spinners()

    def _update_steering_spinners(self) -> None:
        for insertion in self.streaming_steering_insertions:
            spinner = self.streaming_steering_spinners.pop(insertion.message_id, None)
            if spinner is not None and not spinner.is_deleted:
                spinner.set_visibility(False)

    def _update_stream_duration(self) -> None:
        if self.stream_duration_label and self.stream_started_at is not None:
            elapsed = time.monotonic() - self.stream_started_at
            self.stream_duration_label.text = format_duration(elapsed)
        now = time.time()
        if (
            self.current_live_reasoning_block is not None
            and self.current_live_reasoning_timer_label is not None
        ):
            self.current_live_reasoning_timer_label.text = format_duration(
                self.current_live_reasoning_block.elapsed_seconds(now)
            )
        for skill_node, skill_lbl in list(self.live_skill_timer_labels):
            skill_lbl.text = format_duration(skill_node.elapsed_seconds(now))
        for tool_ev, tool_lbl in list(self.live_tool_timer_labels):
            tool_lbl.text = format_duration(tool_ev.elapsed_seconds(now))

    def _close_live_reasoning_block(self) -> None:
        """Finalize end_time and duration_seconds on the currently active live thinking block."""
        if self.current_live_reasoning_block is not None:
            self.current_live_reasoning_block.finalize_timing()

    def stream_reasoning(
        self,
        text: str,
        skill_path: str = "/",
        depth: int = 0,
        generation_index: int = 1,
    ) -> None:
        if self.raw_history_view:
            self.raw_history_view.stream_reasoning(
                text,
                skill_path=skill_path,
                depth=depth,
                generation_index=generation_index,
            )

        if self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)

        if not self.current_reasoning_container:
            return

        block = next(
            (
                item
                for item in self.streaming_reasoning_blocks
                if item.generation_index == generation_index
                and item.skill_path == skill_path
                and item.depth == depth
            ),
            None,
        )
        if block is None:
            block = ReasoningBlock(
                skill_path=skill_path, depth=depth, generation_index=generation_index
            )
            self.streaming_reasoning_blocks.append(block)
        if block is not self.current_live_reasoning_block:
            self._close_live_reasoning_block()
            self._render_streaming_reasoning(active_block=block)

        block.content += text
        if self.current_thinking_label:
            self.current_thinking_label.content = parse_markdown_to_html(block.content)

    def _render_streaming_reasoning(self, active_block: ReasoningBlock | None) -> None:
        """Show completed steps above the one currently receiving tokens."""
        if not self.current_reasoning_container:
            return
        self.current_reasoning_container.clear()
        self.current_master_reasoning_expansion = None
        self.current_live_reasoning_container = None
        self.current_live_reasoning_block = active_block
        self.current_thinking_label = None
        self.current_live_reasoning_timer_label = None
        completed = [
            block
            for block in self.streaming_reasoning_blocks
            if block is not active_block
        ]
        with self.current_reasoning_container:
            if completed:
                self.current_master_reasoning_expansion = self._render_reasoning_group(
                    completed
                )
            if active_block:
                with ui.column().classes(
                    UI_PRESENTATION[
                        "render_streaming_reasoning.classes.w_full_gap_2_live_reasoning"
                    ]
                ) as live:
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_streaming_reasoning.classes.items_center_justify_between_w_full_text_xs_t"
                        ]
                    ):
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_streaming_reasoning.classes.items_center_gap_2"
                            ]
                        ):
                            ui.spinner(
                                UI_PRESENTATION[
                                    "render_streaming_reasoning.spinner.dots"
                                ],
                                size=UI_PRESENTATION[
                                    "render_streaming_reasoning.size.xs"
                                ],
                                color=UI_PRESENTATION[
                                    "render_streaming_reasoning.color.grey_5"
                                ],
                            )
                            ui.label(ui_text("chat.chat_view.reasoning_in_progress"))
                        self.current_live_reasoning_timer_label = ui.label(
                            format_duration(active_block.elapsed_seconds())
                        ).classes(
                            UI_PRESENTATION[
                                "render_streaming_reasoning.classes.message_time_text_10px_font_mono"
                            ]
                        )
                    ui.label(
                        ui_text(
                            "chat.chat_view.step",
                            active_block_generation_index=active_block.generation_index,
                            active_block_skill_path=active_block.skill_path,
                        )
                    ).classes(
                        UI_PRESENTATION[
                            "render_streaming_reasoning.classes.text_11px_text_zinc_500_font_mono"
                        ]
                    )
                    self.current_thinking_label = ui.html(
                        parse_markdown_to_html(active_block.content)
                    ).classes(
                        UI_PRESENTATION[
                            "render_streaming_reasoning.classes.thinking_box_whitespace_pre_wrap_w_full_font_"
                        ]
                    )
                self.current_live_reasoning_container = live

    def _compact_streaming_reasoning(self) -> None:
        """Move completed live thinking into the compact history row."""
        if self.current_live_reasoning_block is None:
            return
        self._close_live_reasoning_block()
        self._render_streaming_reasoning(active_block=None)

    def stream_token(self, token: str) -> None:
        if self.raw_history_view:
            self.raw_history_view.stream_token(token)

        if self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)
        # Compact reasoning into a single line once response text starts streaming
        self._compact_streaming_reasoning()
        self.current_response_text += token
        if self.current_html:
            self.current_html.content = parse_markdown_to_html(
                self.current_response_text
            )

    def _refresh_streaming_skills_and_tools(self) -> None:
        """Re-render the live skills and root tools container with updated timers."""
        if (
            self.active_tab == "graph"
            and self.graph_view
            and not self.graph_view.is_generating
        ):
            self._render_selected_graph()
        if not self.current_skills_container:
            return
        self.live_skill_timer_labels.clear()
        self.live_tool_timer_labels.clear()
        snapshot = active_turns.snapshot(
            self.stream_session_id or "", after_revision=-1
        )
        recorded = self._streaming_history or (snapshot.message if snapshot else None)
        with self.current_skills_container:
            self.current_skills_container.clear()
            self._render_turn_activity(
                ChatMessage(
                    role="assistant",
                    raw_history=recorded.raw_history if recorded else [],
                    skill_history_tree=recorded.skill_history_tree if recorded else [],
                    skills=self.streaming_skills,
                    tool_events=self.streaming_root_tools,
                    steering_insertions=self.streaming_steering_insertions,
                ),
            )

    def handle_live_skill_event(self, node: SkillNode, action: str) -> None:
        self._compact_streaming_reasoning()
        if self.raw_history_view:
            self.raw_history_view.handle_live_skill_event(node, action)

        if self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)
        if not self.current_skills_container:
            return

        _upsert_streaming_skill(self.streaming_skills, node, action)
        self._refresh_streaming_skills_and_tools()

    def add_tool_event(self, ev: ToolEvent) -> None:
        self._compact_streaming_reasoning()
        if self.raw_history_view:
            self.raw_history_view.add_tool_event(ev)

        if self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)
        if self.current_skills_container and ev.skill_path == "/":
            record_tool_event(self.streaming_root_tools, ev)
            self._refresh_streaming_skills_and_tools()

    def finalize_stream(self) -> None:
        self._streaming_history = None
        if self.stream_timer:
            self.stream_timer.cancel()
            self.stream_timer = None
        self.stream_started_at = None
        self.stream_duration_label = None
        if self.raw_history_view:
            self.raw_history_view.finalize_live_turn()

        if self.current_loading_placeholder:
            self.current_loading_placeholder.set_visibility(False)
        self._compact_streaming_reasoning()

        self.current_stream_card = None
        self.current_steering_container = None
        self.streaming_steering_ids.clear()
        self.streaming_steering_insertions.clear()
        for spinner in self.streaming_steering_spinners.values():
            if not spinner.is_deleted:
                spinner.set_visibility(False)
        self.streaming_steering_spinners.clear()
        self.stream_session_id = None
        self.current_html = None
        self.current_response_text = ""
        self.current_thinking_label = None
        self.current_live_reasoning_timer_label = None
        self.current_loading_placeholder = None
        self.current_reasoning_container = None
        self.current_live_reasoning_container = None
        self.current_live_reasoning_block = None
        self.current_skills_container = None
        self.current_master_reasoning_expansion = None
        self.streaming_reasoning_blocks.clear()
        self.streaming_skills.clear()
        self.streaming_root_tools.clear()
        self.live_skill_timer_labels.clear()
        self.live_tool_timer_labels.clear()
