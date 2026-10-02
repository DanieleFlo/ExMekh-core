from __future__ import annotations

import asyncio
import html
import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from nicegui import ui

from fe.qs_bridge.context_usage import (
    compute_skill_node_closure_tokens,
    format_skill_closure_tokens,
    populate_skill_closure_tokens,
)
from fe.qs_bridge.image_input import IMAGE_SUFFIXES
from fe.qs_bridge.message_edit import retained_attachment_paths
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SteeringInsertion,
    ToolEvent,
    format_duration,
    record_tool_event,
)
from fe.qs_bridge.pasted_input import (
    extract_attached_files_from_content,
)
from fe.state.session_manager import session_manager
from fe.ui.chat_export import ExportFormat, export_chat
from fe.ui.file_icons import get_file_icon
from fe.ui.graph_view import GraphViewComponent
from fe.ui.image_preview import render_image_thumbnail
from fe.ui.markdown_parser import parse_markdown_to_html
from fe.ui.message_text import message_display_text
from fe.ui.raw_history_view import RawHistoryComponent

CHAT_SCROLL_JS_PATH: Path = Path(__file__).parent / "js" / "chat_scroll.js"


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
        on_edit_message: Callable[[str, str, str], Awaitable[bool]] | None = None,
    ) -> None:
        self.on_suggested_prompt = on_suggested_prompt
        self.on_edit_message = on_edit_message
        self.chat_container: ui.column | None = None
        self.child_chat_container: ui.column | None = None
        self.other_chat_container: ui.column | None = None
        self.stream_session_id: str | None = None
        self.current_stream_card: ui.row | None = None
        self.current_steering_container: ui.column | None = None
        self.streaming_steering_ids: set[str] = set()
        self.streaming_steering_insertions: list[SteeringInsertion] = []
        self.streaming_steering_spinners: dict[str, ui.spinner] = {}
        self._rendering_steering_insertions: list[SteeringInsertion] = []
        self._rendered_insertion_ids: set[str] = set()
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

    def build(self) -> ui.column:
        load_chat_scroll_js()
        with ui.column().classes(
            "w-full h-full flex-1 min-h-0 flex-grow overflow-hidden gap-0 relative"
        ) as outer_container:
            # 1. Top Sub-Tabs Bar directly under header
            with ui.row().classes(
                "w-full border-b border-zinc-800/80 bg-zinc-950/60 px-4 items-center justify-between shrink-0"
            ):
                with (
                    ui.tabs(value="chat", on_change=self._handle_tab_changed)
                    .props(
                        "dense no-caps inline-label indicator-color=sky-4 active-color=white text-color=grey-5"
                    )
                    .classes("sub-tabs-bar") as tabs
                ):
                    self.tabs_ref = tabs
                    self.tab_chat = ui.tab("chat", label="Chat", icon="chat")
                    self.tab_raw = ui.tab(
                        "raw_history", label="Raw History", icon="data_object"
                    )
                    self.tab_graph = ui.tab("graph", label="Graph", icon="account_tree")
                with ui.row().classes("items-center gap-2 min-w-0 flex-1 justify-end"):
                    self.path_label = ui.label("").classes(
                        "text-[10px] font-mono text-sky-300 truncate"
                    )
                    with (
                        ui.button(icon="more_vert")
                        .props("flat round dense size=sm aria-label='Chat menu'")
                        .classes("chat-export-menu-btn text-zinc-400")
                    ):
                        ui.tooltip("Copy entire chat")
                        with ui.menu().classes("bg-zinc-900 border border-zinc-700"):
                            ui.label("Copy entire chat").classes(
                                "text-xs text-zinc-400 px-4 pt-2"
                            )
                            self.include_subskill_history = (
                                ui.switch("Include subskill history", value=False)
                                .classes("text-xs px-3")
                                .props("dense")
                            )
                            ui.separator()
                            ui.menu_item(
                                "Copy as JSON", on_click=lambda: self._copy_chat("json")
                            )
                            ui.menu_item(
                                "Copy as Markdown",
                                on_click=lambda: self._copy_chat("markdown"),
                            )

            # 2. Tab 1 Container: Chat
            with ui.column().classes(
                "chat-scroll-container w-full flex-1 min-h-0 flex-grow px-4 py-6 gap-6 overflow-y-auto"
            ) as chat_box:
                chat_box.props["data-scroll-reset-seq"] = "0"
                chat_box.props["data-scroll-pinned"] = "true"
                self.chat_container = chat_box
            with ui.column().classes(
                "chat-scroll-container w-full flex-1 min-h-0 flex-grow px-4 py-6 gap-6 overflow-y-auto"
            ) as child_chat_box:
                child_chat_box.props["data-scroll-reset-seq"] = "0"
                child_chat_box.props["data-scroll-pinned"] = "true"
                self.child_chat_container = child_chat_box
                child_chat_box.set_visibility(False)
            with ui.column().classes(
                "chat-scroll-container w-full flex-1 min-h-0 flex-grow px-4 py-6 gap-6 overflow-y-auto"
            ) as other_chat_box:
                other_chat_box.props["data-scroll-reset-seq"] = "0"
                other_chat_box.props["data-scroll-pinned"] = "true"
                self.other_chat_container = other_chat_box
                other_chat_box.set_visibility(False)

            # 3. Tab 2 Container: Raw History
            with ui.column().classes(
                "chat-scroll-container w-full max-w-5xl mx-auto flex-1 min-h-0 flex-grow px-4 py-4 overflow-y-auto"
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
                "w-full max-w-6xl mx-auto flex-1 min-h-0 flex-grow px-4 py-3 overflow-y-auto"
            ) as graph_box:
                self.graph_container = graph_box
                self.graph_view.build()
                graph_box.set_visibility(False)

            # Floating "Scroll to bottom" indicator button (shown when detached from bottom)
            self.scroll_to_bottom_btn = (
                ui.button(
                    icon="arrow_downward",
                    on_click=lambda: self.reset_scroll_to_bottom(),
                )
                .props(
                    "id=qs-scroll-to-bottom-btn round dense color=dark text-color=white size=sm aria-label='Torna in fondo'"
                )
                .classes("scroll-to-bottom-btn")
            )

            # Floating points of interest rail near right scrollbar
            self.poi_track = (
                ui.element("div")
                .props("id=qs-chat-poi-track")
                .classes("chat-poi-track")
            )

            self.render_current_session()
        return outer_container

    async def _copy_chat(self, export_format: ExportFormat) -> None:
        client = ui.context.client
        session = session_manager.snapshot_current_session()
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
            ui.notify(f"Chat copied as {export_format.upper()}", type="positive")

    def _render_copy_button(self, text: str, *, user: bool = False) -> None:
        ui.button(
            icon="content_copy",
            on_click=lambda: ui.clipboard.write(text),
        ).props("flat round dense size=xs aria-label='Copy message'").classes(
            "text-zinc-600 hover:text-zinc-300 response-copy-btn "
            + ("user-copy-btn" if user else "")
        ).tooltip("Copy message")

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
        session = session_manager.get_current_session()
        is_generating = self.is_streaming_session(session.id) or any(
            sub.status == "active" for sub in session.sub_agent_conversations
        )
        self.graph_view.render_session(
            session,
            selected_sub_agent_id=self.selected_sub_agent_id,
            is_generating=is_generating,
        )

    def render_current_session(self) -> None:
        session = session_manager.get_current_session()
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
        scope = self._display_scope(session_manager.get_current_session())
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
        session = session_manager.get_current_session()
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
        scope = self._display_scope(session_manager.get_current_session())
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
                session_manager.get_current_session()
            )
        elif not self.raw_history_view.is_streaming:
            self.raw_history_view.render_session(session_manager.get_current_session())

    def is_streaming_session(self, session_id: str) -> bool:
        return (
            self.stream_session_id == session_id
            and self.current_stream_card is not None
        )

    def has_active_stream(self) -> bool:
        return self.current_stream_card is not None

    def finalize_session_stream(self, session_id: str) -> None:
        """Close a session's live UI and refresh it when that session is visible."""
        visible = session_manager.is_current_session(session_id)
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
            "items-center justify-center w-full py-16 gap-3 text-center chat-message-row"
        ):
            ui.icon("terminal", size="md").classes("text-zinc-400 mb-1")
            ui.label("QS Studio").classes(
                "text-2xl font-bold tracking-tight text-zinc-100"
            )
            ui.label(
                "Hierarchical agentic framework with multi-depth execution, tool calling, and real-time streaming."
            ).classes("text-sm text-zinc-400 max-w-md")

            with ui.row().classes("gap-2 mt-4 flex-wrap justify-center"):
                suggestions = [
                    (
                        "🔧 Run mock-tool",
                        "Use the mock-tool to verify tool execution.",
                    ),
                    (
                        "🌿 Demonstrate Nested Skills",
                        "Use the mock-skill and invoke sub-tools within it to demonstrate nesting.",
                    ),
                    (
                        "⚡ Reasoning Test",
                        "Solve step by step: How many 'r's are in the word 'strawberry'?",
                    ),
                ]
                for label, prompt in suggestions:
                    ui.button(
                        label, on_click=lambda _, p=prompt: self.on_suggested_prompt(p)
                    ).props("flat rounded-lg no-caps dense").classes(
                        "text-zinc-400 hover:text-white text-xs px-3 py-1.5 bg-zinc-900/60 hover:bg-zinc-800"
                    )

    async def _show_attached_file_modal(self, file_path: str) -> None:
        filename = Path(file_path).name
        session = session_manager.get_current_session()
        storage = session_manager.temp_storage(session)
        client = ui.context.client
        is_binary = False
        icon_name, color_class, _ = get_file_icon(filename)
        try:
            full_path = storage.resolve_attachment(file_path, allow_legacy=True)
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, full_path.read_bytes)
            if b"\x00" in data:
                is_binary = True
                content_text = (
                    f"Binary file ({len(data):,} bytes). Text preview unavailable."
                )
            else:
                try:
                    content_text = data.decode("utf-8")
                except UnicodeDecodeError:
                    is_binary = True
                    content_text = (
                        f"File with non-UTF-8 encoding ({len(data):,} bytes). "
                        "Text preview unavailable."
                    )
        except ValueError:
            content_text = "Invalid attachment path."
        except FileNotFoundError:
            content_text = "Attached file not found."
        except OSError:
            content_text = "Unable to read the file from disk."

        with client:
            with (
                ui.dialog() as dlg,
                ui.card().classes(
                    "bg-zinc-900 border border-zinc-800 text-zinc-100 max-w-2xl w-full p-4 gap-2"
                ),
            ):
                with ui.row().classes(
                    "w-full items-center justify-between border-b border-zinc-800 pb-2"
                ):
                    with ui.row().classes("items-center gap-1.5 min-w-0 flex-1"):
                        ui.icon("attach_file", size="18px").classes(
                            "text-zinc-400 shrink-0"
                        )
                        ui.icon(icon_name, size="18px").classes(
                            f"{color_class} shrink-0"
                        )
                        dlg_lbl = ui.label(filename).classes(
                            "font-mono text-sm font-semibold text-zinc-200 truncate max-w-[420px]"
                        )
                        dlg_lbl.tooltip(filename)
                    ui.button(icon="close", on_click=dlg.close).props(
                        "dense flat round size=sm text-color=zinc-400 shrink-0"
                    )
                if is_binary:
                    with ui.column().classes(
                        "w-full items-center justify-center p-6 gap-2 bg-zinc-950/80 rounded border border-zinc-800 text-center"
                    ):
                        ui.icon(icon_name, size="36px").classes(color_class)
                        ui.label(content_text).classes(
                            "text-xs font-mono text-zinc-300"
                        )
                        ui.label(f"Path: {file_path}").classes(
                            "text-[10px] font-mono text-zinc-500"
                        )
                        ui.label(
                            "The agent can access the file on disk and will determine whether it can process it."
                        ).classes("text-xs text-zinc-400 italic")
                else:
                    with ui.column().classes(
                        "w-full max-h-96 overflow-y-auto bg-zinc-950 p-3 rounded border border-zinc-800/80"
                    ):
                        ui.code(content_text).classes("w-full text-xs font-mono")
            dlg.open()

    def _open_message_editor(self, session_id: str, message: ChatMessage) -> None:
        has_attachments = bool(retained_attachment_paths(message) or message.image_urls)
        with ui.dialog().classes("message-edit-dialog") as dialog:
            with ui.card().classes("w-[640px] max-w-[94vw] bg-zinc-900 gap-4"):
                ui.label("Edit message").classes("text-lg text-zinc-100")
                editor = (
                    ui.textarea(label="Message", value=message_display_text(message))
                    .props("outlined autogrow autofocus")
                    .classes("w-full")
                )
                ui.label(
                    "Saving restarts the response and removes the conversation after this message."
                ).classes("text-xs text-zinc-400")
                if has_attachments:
                    ui.label("Attachments will be kept.").classes(
                        "text-xs text-zinc-400"
                    )

                async def save() -> None:
                    save_button.disable()
                    editor.disable()
                    try:
                        if self.on_edit_message and await self.on_edit_message(
                            session_id, message.id, editor.value or ""
                        ):
                            dialog.close()
                    finally:
                        save_button.enable()
                        editor.enable()

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    save_button = (
                        ui.button("Save & regenerate", on_click=save)
                        .props("unelevated")
                        .bind_enabled_from(
                            editor,
                            "value",
                            backward=lambda value: bool(
                                str(value or "").strip() or has_attachments
                            ),
                        )
                    )
        dialog.open()

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
                "w-full justify-end items-start chat-message-row chat-message-user"
            ):
                bubble_class = "msg-user-steering" if steering else ""
                with ui.column().classes(
                    f"msg-user {bubble_class} max-w-2xl px-4 py-2.5"
                ):
                    if steering:
                        with ui.row().classes(
                            "steering-message-label items-center gap-1.5"
                        ):
                            ui.icon("alt_route", size="13px")
                            ui.label("Steering")
                            if pending_steering:
                                spinner = ui.spinner(
                                    size="11px", color="grey-5"
                                ).classes("steering-loading-spinner")
                                spinner.tooltip(
                                    "Pending: waiting to be inserted into history"
                                )
                                self.streaming_steering_spinners[msg.id] = spinner
                    display_text = message_display_text(msg)

                    if display_text:
                        ui.markdown(display_text).classes("text-sm text-zinc-100")

                    if msg.image_urls:
                        with ui.row().classes("qs-message-images gap-2 flex-wrap mt-2"):
                            for index, url in enumerate(msg.image_urls):
                                render_image_thumbnail(
                                    url, f"Attached image {index + 1}"
                                )

                    if attached_files:
                        with ui.row().classes(
                            "qs-message-attachments items-center gap-1.5 mt-1.5 pt-1.5 border-t border-zinc-700/50 flex-wrap"
                        ):
                            for file_path in attached_files:
                                filename = Path(file_path).name
                                icon_name, color_class, _ = get_file_icon(filename)
                                with (
                                    ui.row()
                                    .classes(
                                        "qs-message-attachment-chip items-center gap-1.5 px-2 py-0.5 rounded "
                                        "bg-zinc-800/80 border border-zinc-700/60 max-w-[240px] "
                                        "shadow-sm cursor-pointer hover:bg-zinc-700/80 transition-colors select-none"
                                    )
                                    .on(
                                        "click",
                                        lambda _, p=file_path: (
                                            self._show_attached_file_modal(p)
                                        ),
                                    )
                                ):
                                    ui.icon("attach_file", size="13px").classes(
                                        "text-zinc-400 shrink-0"
                                    )
                                    ui.icon(icon_name, size="13px").classes(
                                        f"{color_class} shrink-0"
                                    )
                                    ui.label(filename).classes(
                                        "font-mono text-xs text-zinc-300 truncate max-w-[150px]"
                                    )
                                    ui.tooltip(
                                        f"Attached file: {filename}\n(Click to preview)"
                                    ).classes("text-xs font-mono")

                    with ui.row().classes(
                        "user-message-footer w-full justify-end items-center gap-2"
                    ):
                        ui.label(
                            datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                        ).classes("message-time text-[10px]")
                        self._render_copy_button(display_text, user=True)
                        if self.on_edit_message and self.selected_sub_agent_id is None:
                            session_id = session_manager.get_current_session().id
                            ui.button(
                                icon="edit",
                                on_click=lambda: self._open_message_editor(
                                    session_id, msg
                                ),
                            ).props(
                                'flat round dense size=xs aria-label="Edit message"'
                            ).classes(
                                "text-zinc-600 hover:text-zinc-300 response-copy-btn user-edit-btn"
                            ).tooltip("Edit message")

        else:
            with ui.row().classes(
                "w-full justify-start items-start chat-message-row chat-message-assistant"
            ):
                with ui.column().classes("msg-assistant w-full gap-2.5"):
                    if msg.error == "cancelled":
                        with ui.row().classes(
                            "generation-stopped items-center gap-1.5 text-xs text-amber-400"
                        ):
                            ui.icon("stop_circle", size="16px")
                            ui.label("Generation stopped by user")
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
                            "w-full gap-1 reasoning-blocks-container"
                        ):
                            self._render_reasoning_group(reasoning_blocks)

                    # Hierarchical Skills & Nested Blocks
                    with ui.column().classes("w-full gap-2"):
                        self._render_turn_activity(
                            msg,
                        )

                    # Response Parsed HTML (with LaTeX MathML support)
                    if msg.content:
                        ui.html(parse_markdown_to_html(msg.content)).classes(
                            "text-sm text-zinc-200 leading-relaxed parsed-response-content w-full"
                        )

                    # Footer actions
                    with ui.row().classes("w-full justify-end items-center gap-2 pt-1"):
                        if msg.duration_seconds is not None:
                            ui.label(format_duration(msg.duration_seconds)).classes(
                                "message-time text-[10px] font-mono"
                            )
                        ui.label(
                            datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                        ).classes("message-time text-[10px]")
                        self._render_copy_button(msg.content)

    def _render_reasoning_group(self, blocks: list[ReasoningBlock]) -> ui.expansion:
        """Renders multiple reasoning blocks compacted into a single expandable row,
        which expands to reveal the individual sub-blocks, each individually expandable.
        """
        count = len(blocks)
        total_duration = sum(b.elapsed_seconds() for b in blocks)
        with ui.expansion("").classes(
            "w-full reasoning-master-expansion reasoning-expansion text-xs"
        ) as master_exp:
            master_exp.value = False  # Initially compacted into a single line
            with master_exp.add_slot("header"):
                with ui.row().classes(
                    "reasoning-master-header items-center justify-between w-full py-1 cursor-pointer"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("psychology", size="xs").classes("text-zinc-400")
                        ui.label("Reasoning Process").classes(
                            "font-mono text-zinc-300 text-xs font-medium"
                        )
                        ui.label(
                            f"{count} {'block' if count == 1 else 'blocks'}"
                        ).classes(
                            "font-mono text-[11px] text-zinc-400 reasoning-count-badge px-1.5 py-0.5 rounded"
                        )
                    with ui.row().classes(
                        "items-center gap-2 font-mono text-[11px] text-zinc-500"
                    ):
                        if count > 1:
                            unique_paths = list(
                                dict.fromkeys(b.skill_path for b in blocks)
                            )
                            if len(unique_paths) > 1:
                                paths_str = ", ".join(
                                    p if p != "/" else "root" for p in unique_paths
                                )
                                ui.label(paths_str).classes("text-zinc-500 text-[10px]")
                        ui.label(format_duration(total_duration)).classes(
                            "message-time text-[10px] font-mono"
                        )

            with (
                master_exp,
                ui.column().classes(
                    "w-full pl-2.5 py-1.5 gap-1.5 border-l border-zinc-800/80 ml-2"
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
        with ui.expansion("").classes("w-full reasoning-sub-block text-xs") as sub_exp:
            sub_exp.value = (
                False  # Initially collapsed so each sub-block can be expanded in turn
            )
            with sub_exp.add_slot("header"):
                with ui.row().classes(
                    "reasoning-header reasoning-sub-header items-center justify-between w-full py-0.5 cursor-pointer"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("subdirectory_arrow_right", size="xs").classes(
                            "text-zinc-500"
                        )
                        ui.label(f"Step #{block.generation_index}").classes(
                            "font-mono text-zinc-400 text-xs font-medium"
                        )
                        ui.label(block.skill_path).classes(
                            "font-mono text-[11px] text-zinc-400 reasoning-path-badge bg-zinc-800/50 px-1.5 py-0.5 rounded"
                        )
                    with ui.row().classes(
                        "items-center gap-2 font-mono text-[11px] text-zinc-500"
                    ):
                        if block.depth > 0:
                            ui.label(f"depth: {block.depth}").classes("text-zinc-500")
                        ui.label(f"gen #{block.generation_index}").classes(
                            "text-zinc-500 reasoning-gen-badge"
                        )
                        ui.label(format_duration(block_duration)).classes(
                            "message-time text-[10px] font-mono"
                        )

            with sub_exp:
                ui.html(parse_markdown_to_html(block.content)).classes(
                    "thinking-box whitespace-pre-wrap w-full font-mono text-xs"
                )

            return sub_exp

    def _render_skill_node(self, node: SkillNode) -> None:
        depth_level = min(max(node.depth, 1), 3)
        depth_class = f"skill-depth-{depth_level}"

        if node.status == "active":
            # Active state: seamless container with subtle guideline and live skill timer
            with ui.column().classes(
                f"skill-container skill-container-active {depth_class} py-1 gap-2"
            ):
                with ui.row().classes("items-center justify-between w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("hub", size="xs").classes("text-zinc-400")
                        ui.label(f"Active skill: {node.path}").classes(
                            "font-mono font-medium text-zinc-200 text-xs"
                        )
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"Depth {node.depth}").classes(
                            "text-[10px] font-mono text-zinc-500"
                        )
                        skill_timer_lbl = ui.label(
                            format_duration(node.elapsed_seconds())
                        ).classes("message-time text-[10px] font-mono")
                        self.live_skill_timer_labels.append((node, skill_timer_lbl))
                        with ui.row().classes(
                            "skill-active-badge text-[10px] font-mono items-center gap-1.5 text-zinc-400"
                        ):
                            ui.spinner("dots", size="xs", color="grey-5")
                            ui.label("RUNNING")

                # Render reasoning blocks inside this skill if present
                if node.reasoning_blocks:
                    with ui.column().classes("w-full gap-1 pl-2"):
                        self._render_reasoning_group(node.reasoning_blocks)

                with ui.column().classes("w-full gap-2 pl-2"):
                    self._render_skill_activity(node)

        else:
            # Completed state: seamless accordion header with open-to-close skill timer
            with ui.column().classes(f"skill-container {depth_class} gap-0"):
                body_ref: ui.column | None = None
                chevron_ref: ui.icon | None = None

                # Compacted clickable bar
                with ui.row().classes(
                    "skill-compact-header items-center justify-between w-full"
                ) as header:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(
                            "error" if node.status == "error" else "check_circle",
                            size="xs",
                        ).classes(
                            "text-rose-400"
                            if node.status == "error"
                            else "text-emerald-500"
                        )
                        ui.label(f"Skill: {node.name}").classes(
                            "font-mono font-medium text-zinc-300 text-xs"
                        )
                        ui.label(node.path).classes(
                            "text-[11px] text-zinc-500 font-mono"
                        )

                    with ui.row().classes("items-center gap-2"):
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
                                    "skill-closure-tokens items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/40 text-[10px] font-mono text-emerald-300"
                                )
                                .props(
                                    f'title="Context tokens before closing: {before_tok:,} • after closing: {after_tok:,}"'
                                )
                            ):
                                ui.icon("compress", size="12px").classes(
                                    "text-emerald-400"
                                )
                                ui.label(token_badge_text).classes(
                                    "skill-closure-tokens-label"
                                )
                        ui.label(f"{node.action_count} actions").classes(
                            "text-[11px] font-mono text-zinc-500"
                        )
                        ui.label(f"Depth {node.depth}").classes(
                            "text-[11px] font-mono text-zinc-500"
                        )
                        ui.label(format_duration(node.elapsed_seconds())).classes(
                            "message-time text-[10px] font-mono"
                        )
                        chevron_ref = ui.icon(
                            "expand_more" if node.is_collapsed else "expand_less",
                            size="xs",
                        ).classes("text-zinc-500 transition-transform duration-200")

                # Collapsible body containing all actions and nested sub-skills
                with ui.column().classes("w-full py-2 pl-3 gap-2") as body:
                    body_ref = body
                    body.set_visibility(not node.is_collapsed)

                    if node.reasoning_blocks:
                        ui.label("Reasoning within skill:").classes(
                            "text-[11px] text-zinc-500 font-mono"
                        )
                        self._render_reasoning_group(node.reasoning_blocks)

                    self._render_skill_activity(node)

                def toggle_collapse(n=node, b=body_ref, c=chevron_ref):
                    n.is_collapsed = not n.is_collapsed
                    if b:
                        b.set_visibility(not n.is_collapsed)
                    if c:
                        c.name = "expand_more" if n.is_collapsed else "expand_less"

                header.on("click", toggle_collapse)

        self._render_insertions_after(node.opening_tool_call_id)

    def _render_skill_activity(self, node: SkillNode) -> None:
        for activity in node.ordered_activity():
            is_skill = isinstance(activity, SkillNode)
            if node.status != "active":
                ui.label(
                    "Nested sub-skills:" if is_skill else "Actions executed in skill:"
                ).classes("text-[11px] text-zinc-500 font-mono")
            if is_skill:
                self._render_skill_node(activity)
            else:
                self._render_tool_event(activity)

    def _render_tool_event(self, ev: ToolEvent) -> None:
        is_running = ev.is_running
        with ui.row().classes(
            "tool-event-card w-full items-center justify-between py-1"
        ):
            with ui.row().classes("tool-event-main items-center gap-2"):
                if is_running:
                    ui.spinner("dots", size="xs", color="grey-5")
                else:
                    ui.icon("terminal", size="xs").classes("text-zinc-500")
                ui.label(f"Tool: {ev.tool_name}").classes(
                    "tool-event-name font-mono font-medium text-zinc-300 text-xs"
                )
                arguments = (
                    self._rendering_message.arguments_for_tool(ev)
                    if self._rendering_message is not None
                    else ev.arguments
                )
                if arguments:
                    fragments = []
                    for name, value in arguments.items():
                        text = (
                            value
                            if isinstance(value, str)
                            else json.dumps(value, ensure_ascii=False)
                        )
                        fragments.append(
                            f'<span class="tool-argument-name">{html.escape(name)}</span>'
                            f'<span class="tool-argument-value">{html.escape(" ".join(text.split()))}</span>'
                        )
                    ui.html(
                        '<span class="tool-argument-separator">—</span> '
                        + " · ".join(fragments)
                    ).classes("tool-arguments-preview font-mono text-xs").tooltip(
                        json.dumps(arguments, ensure_ascii=False, indent=2)
                    )
            with ui.row().classes(
                "tool-event-meta items-center gap-2 text-[11px] text-zinc-500 font-mono"
            ):
                if not is_running:
                    ui.icon(
                        "error" if ev.execution_status == "error" else "check_circle",
                        size="xs",
                    ).classes(
                        "tool-event-status text-red-400"
                        if ev.execution_status == "error"
                        else "tool-event-status text-emerald-400"
                    ).tooltip(
                        "Failed" if ev.execution_status == "error" else "Completed"
                    )
                ui.label(f"depth: {ev.depth}")
                ui.label(f"path: {ev.skill_path}")
                tool_timer_lbl = ui.label(
                    format_duration(ev.elapsed_seconds())
                ).classes("message-time text-[10px] font-mono")
                if is_running:
                    self.live_tool_timer_labels.append((ev, tool_timer_lbl))
        self._render_insertions_after(ev.tool_call_id)

    def _render_turn_activity(
        self,
        message: ChatMessage,
    ) -> None:
        """Render live and saved invocations once, in their opening order."""
        self._rendering_steering_insertions = message.confirmed_steering_insertions()
        self._rendering_message = message
        self._rendered_insertion_ids.clear()
        try:
            for activity in message.ordered_activity():
                if isinstance(activity, SkillNode):
                    self._render_skill_node(activity)
                else:
                    self._render_tool_event(activity)
            self._render_unmatched_steering_insertions()
        finally:
            self._rendering_steering_insertions = []
            self._rendering_message = None

    def _render_insertions_after(self, call_id: str | None) -> None:
        if call_id is None:
            return
        for insertion in self._rendering_steering_insertions:
            if (
                insertion.after_tool_call_id == call_id
                and insertion.message_id not in self._rendered_insertion_ids
            ):
                self._render_steering_insertion(insertion)

    def _render_unmatched_steering_insertions(self) -> None:
        for insertion in self._rendering_steering_insertions:
            if insertion.message_id not in self._rendered_insertion_ids:
                self._render_steering_insertion(insertion)

    def _render_steering_insertion(self, insertion: SteeringInsertion) -> None:
        self._rendered_insertion_ids.add(insertion.message_id)
        location = (
            f"after {insertion.tool_name}"
            if insertion.tool_name
            else "before the response"
        )
        skill = insertion.skill_path if insertion.skill_path != "/" else "root"
        with ui.row().classes("steering-tool-marker items-center gap-1.5"):
            ui.icon("alt_route", size="13px")
            ui.label(f"Steering inserted {location} · skill {skill}")

    # Streaming support methods
    def start_assistant_stream(self, *, started_at: float | None = None) -> None:
        if not self.chat_container:
            return

        self.stream_session_id = session_manager.get_current_session().id
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
            self.current_steering_container = ui.column().classes("w-full gap-3")
            self.streaming_steering_ids.clear()
            with ui.row().classes(
                "w-full justify-start items-start chat-message-row chat-message-assistant chat-message-streaming"
            ) as row:
                self.current_stream_card = row
                with ui.column().classes("msg-assistant w-full gap-2.5"):
                    # Initial loading indicator while waiting for LLM generation
                    with ui.row().classes(
                        "items-center gap-2 text-xs text-zinc-500 font-mono py-1"
                    ) as loader:
                        self.current_loading_placeholder = loader
                        ui.spinner("dots", size="sm", color="grey-5")
                        ui.label("Waiting for response...")

                    # Container for separated reasoning blocks
                    self.current_reasoning_container = ui.column().classes(
                        "w-full gap-1 reasoning-blocks-container"
                    )

                    # Dynamic skills container
                    self.current_skills_container = ui.column().classes("w-full gap-2")

                    # Parsed HTML response (with LaTeX MathML support)
                    self.current_response_text = ""
                    self.current_html = ui.html("").classes(
                        "text-sm text-zinc-200 leading-relaxed parsed-response-content w-full"
                    )
                    self.stream_duration_label = ui.label(format_duration(0)).classes(
                        "message-time text-[10px] font-mono self-end"
                    )
                    self.stream_timer = ui.timer(1, self._update_stream_duration)

        if self.raw_history_view:
            session = session_manager.get_current_session()
            self.raw_history_view.start_live_turn(session)
        if self.graph_view:
            self._render_selected_graph()
        self.reset_scroll_to_bottom(self.chat_container)

    def restore_stream_snapshot(self, message: ChatMessage) -> None:
        """Show the latest live turn in a page opened after generation began."""
        if not self.current_stream_card:
            return
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
            now = time.time()
            if self.current_live_reasoning_block.end_time is None:
                self.current_live_reasoning_block.end_time = now
            if self.current_live_reasoning_block.duration_seconds is None:
                self.current_live_reasoning_block.duration_seconds = max(
                    0.0,
                    self.current_live_reasoning_block.end_time
                    - self.current_live_reasoning_block.timestamp,
                )

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
                with ui.column().classes("w-full gap-2 live-reasoning") as live:
                    with ui.row().classes(
                        "items-center justify-between w-full text-xs text-zinc-300 font-mono"
                    ):
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner("dots", size="xs", color="grey-5")
                            ui.label("Reasoning in progress...")
                        self.current_live_reasoning_timer_label = ui.label(
                            format_duration(active_block.elapsed_seconds())
                        ).classes("message-time text-[10px] font-mono")
                    ui.label(
                        f"Step #{active_block.generation_index} · {active_block.skill_path}"
                    ).classes("text-[11px] text-zinc-500 font-mono")
                    self.current_thinking_label = ui.html(
                        parse_markdown_to_html(active_block.content)
                    ).classes(
                        "thinking-box whitespace-pre-wrap w-full font-mono text-xs"
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
        with self.current_skills_container:
            self.current_skills_container.clear()
            self._render_turn_activity(
                ChatMessage(
                    role="assistant",
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
