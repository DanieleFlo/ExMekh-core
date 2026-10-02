from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from nicegui import ui

from fe.config import config
from fe.qs_bridge.context_usage import (
    format_compact_tokens,
    format_context_percentage,
)
from fe.qs_bridge.models import QueuedMessage
from fe.qs_bridge.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
    create_staged_attachments_from_text,
)
from fe.qs_bridge.tool_approval import (
    DEFAULT_TOOL_APPROVAL_MODE,
    TOOL_APPROVAL_MODES,
    normalize_tool_approval_mode,
)
from fe.state.temp_storage import TempStorage
from fe.ui.file_icons import format_file_size, get_file_icon
from fe.ui.image_preview import render_image_thumbnail

PASTE_JS_HANDLER = (
    "(e) => {"
    " if (e.clipboardData?.files?.length) {"
    "   e.preventDefault(); window.__qsUploadFiles(e.clipboardData.files); return;"
    " }"
    " const text = (e.clipboardData || window.clipboardData)?.getData('text') || '';"
    " if (text.length > 400) {"
    "   e.preventDefault();"
    "   emit({text: text, prevented: true});"
    " } else {"
    "   emit({text: text, prevented: false});"
    " }"
    "}"
)


@dataclass
class _InputDraft:
    text: str = ""
    attachments: list[StagedAttachment] = field(default_factory=list)


class InputBarComponent:
    def __init__(
        self,
        on_send: Callable[[str], None],
        on_stop: Callable[[], None],
        on_edit_queued: Callable[[str], None] | None = None,
        on_delete_queued: Callable[[str], None] | None = None,
        on_reorder_queued: Callable[[str, str, str], None] | None = None,
        on_resume_queue: Callable[[], None] | None = None,
        temp_storage_getter: Callable[[str | None], TempStorage] | None = None,
        on_send_with_attachments: (
            Callable[[str, list[StagedAttachment]], None] | None
        ) = None,
        on_approval_mode_changed: Callable[[str], None] | None = None,
        on_steer_queued: Callable[[str], None] | None = None,
    ) -> None:
        self.on_send = on_send
        self.on_send_with_attachments = on_send_with_attachments
        self.on_stop = on_stop
        self.on_edit_queued = on_edit_queued
        self.on_delete_queued = on_delete_queued
        self.on_reorder_queued = on_reorder_queued
        self.on_resume_queue = on_resume_queue
        self.temp_storage_getter = temp_storage_getter
        self.on_approval_mode_changed = on_approval_mode_changed
        self.on_steer_queued = on_steer_queued

        self.root_container: ui.column | None = None
        self.queue_container: ui.column | None = None
        self.queue_list: ui.column | None = None
        self.queue_count_label: ui.label | None = None
        self.resume_queue_btn: ui.button | None = None
        self.queued_messages: list[QueuedMessage] = []
        self.steering_buttons: list[ui.button] = []
        self.steering_enabled = False

        self.attachments_container: ui.row | None = None
        self.staged_attachments: list[StagedAttachment] = []
        self.drop_overlay: ui.element | None = None
        self.input_container: ui.column | None = None
        self.session_id: str | None = None
        self._drafts: dict[str, _InputDraft] = {}

        self.approval_mode: str = DEFAULT_TOOL_APPROVAL_MODE
        self.approval_select: ui.select | None = None
        self.approval_container: ui.column | None = None
        self.approval_command_label: ui.label | None = None
        self.approval_approve_btn: ui.button | None = None
        self.approval_decline_btn: ui.button | None = None
        self._current_approval_accept_cb: Callable[[], None] | None = None
        self._current_approval_decline_cb: Callable[[], None] | None = None

        self.textarea: ui.textarea | None = None
        self.send_button: ui.button | None = None
        self.send_tooltip: ui.tooltip | None = None
        self.stop_button: ui.button | None = None
        self.model_label: ui.label | None = None
        self.context_percent_label: ui.label | None = None
        self.context_tokens_label: ui.label | None = None
        self.context_tooltip: ui.tooltip | None = None
        self.progress_container: ui.element | None = None
        self.is_busy: bool = False

        self.model_name: str = config.default_model
        self.used_tokens: int = 0
        self.context_length: int = config.get_context_length(config.default_model)

    def build(self) -> ui.column:
        ui.on("qs_files_dropped", self._handle_files_dropped)
        ui.on("qs_files_dropped_error", self._handle_files_dropped_error)

        with ui.column().classes(
            "w-full max-w-4xl mx-auto mb-4 items-center gap-0 relative"
        ) as root:
            self.root_container = root

            # -------------------------------------------------------------
            # Queue Box (peeks out partially above the input box)
            # -------------------------------------------------------------
            with ui.column().classes(
                "qs-queue-box w-[calc(100%-1.25rem)] mx-auto bg-zinc-900/95 backdrop-blur-md "
                "border border-b-0 border-zinc-800 rounded-t-xl overflow-hidden shadow-2xl "
                "transition-all duration-200 z-0 -mb-1 px-3 pt-2 pb-1.5 gap-1.5"
            ) as q_box:
                self.queue_container = q_box

                with ui.row().classes(
                    "w-full items-center justify-between text-xs text-zinc-400 select-none pb-1 border-b border-zinc-800/60"
                ):
                    with ui.row().classes("items-center gap-1.5"):
                        ui.icon("hourglass_top", size="14px").classes("text-amber-400")
                        self.queue_count_label = ui.label(
                            "Queued messages (0)"
                        ).classes("font-semibold text-zinc-300 text-[11px]")
                    with ui.row().classes("items-center gap-2"):
                        self.resume_queue_btn = (
                            ui.button(
                                "Send queue",
                                icon="play_arrow",
                                on_click=self._trigger_resume_queue,
                            )
                            .props("dense flat size=xs text-color=emerald-400 no-caps")
                            .classes(
                                "text-[11px] px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/50 hover:bg-emerald-900/50"
                            )
                        )
                        self.resume_queue_btn.set_visibility(False)
                        ui.label("Automatic sequential sending").classes(
                            "text-[10px] text-zinc-500 font-mono hidden sm:inline"
                        )

                self.queue_list = ui.column().classes(
                    "w-full max-h-40 overflow-y-auto gap-1 pr-0.5"
                )

            self.queue_container.set_visibility(False)

            # -------------------------------------------------------------
            # Attachments Container (peeks out / sits above the input box)
            # -------------------------------------------------------------
            with ui.row().classes(
                "qs-attachments-bar w-[calc(100%-1.25rem)] mx-auto bg-zinc-900/95 backdrop-blur-md "
                "border border-b-0 border-zinc-800 rounded-t-xl px-2.5 py-1.5 gap-2 items-center flex-wrap z-0 -mb-1"
            ) as att_bar:
                self.attachments_container = att_bar

            self.attachments_container.set_visibility(False)

            self._build_approval_prompt()

            # -------------------------------------------------------------
            # Main Input Bar
            # -------------------------------------------------------------
            with ui.column().classes(
                "qs-chat-input-box w-full px-2.5 pt-2 pb-1.5 items-center gap-1 bg-zinc-900/90 backdrop-blur-md rounded-xl border border-zinc-800 z-10 relative overflow-hidden"
            ) as input_container:
                self.input_container = input_container
                with ui.element("div").classes("qs-drop-overlay") as drop_overlay:
                    self.drop_overlay = drop_overlay
                    with ui.element("div").classes(
                        "w-11 h-11 rounded-full bg-sky-500/20 text-sky-400 border border-sky-500/40 flex items-center justify-center qs-drop-bounce-icon"
                    ):
                        ui.icon("cloud_upload", size="26px")
                    ui.label("Drop files here").classes(
                        "text-sm font-semibold text-zinc-100 tracking-wide select-none leading-none"
                    )
                    ui.label("Files will be added to temporary storage").classes(
                        "text-[11px] text-sky-300 font-mono select-none leading-none"
                    )

                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.element("input").props("type=file multiple").classes(
                        "hidden qs-file-picker"
                    ).on(
                        "change",
                        js_handler="(e) => { window.__qsUploadFiles(e.target.files); e.target.value = ''; }",
                    )
                    ui.button(icon="attach_file").props("flat round dense").on(
                        "click",
                        js_handler="() => document.querySelector('.qs-file-picker').click()",
                    ).tooltip("Attach files, images or PDFs")
                    self.textarea = (
                        ui.textarea(
                            placeholder="Send a message to QS Studio... (Enter to send, Shift+Enter for new line)"
                        )
                        .props("autogrow borderless dark rows=1")
                        .classes(
                            "chat-input-textarea flex-grow text-sm px-2 py-1 text-zinc-100"
                        )
                    )

                    # Bind Enter key to send (preventing default newline unless shift is pressed)
                    self.textarea.on("keydown.enter.prevent", self._handle_enter)
                    self.textarea.on(
                        "paste", self._handle_paste, js_handler=PASTE_JS_HANDLER
                    )

                    # Stop Button (visible while generating)
                    self.stop_button = (
                        ui.button(icon="stop", on_click=self.on_stop)
                        .props("round dense color=red-9 text-color=white")
                        .classes("self-center w-9 h-9")
                    )
                    self.stop_button.tooltip("Stop generation")
                    self.stop_button.set_visibility(False)

                    # Send / Queue Button (always accessible)
                    self.send_button = (
                        ui.button(icon="arrow_upward", on_click=self._trigger_send)
                        .props("round dense color=dark text-color=white")
                        .classes(
                            "self-center border border-zinc-700/80 w-9 h-9 bg-zinc-800 hover:bg-zinc-700"
                        )
                    )
                    self.send_tooltip = ui.tooltip("Send message (Enter)")

                # Bottom bar: left shows model name, context percentage, and tokens/context_length;
                # right shows command approval mode selector with clear spacing from context info
                with ui.row().classes(
                    "w-full items-center justify-between px-2 pt-2 pb-0.5 border-t border-zinc-800/60 gap-3"
                ):
                    pct_str = format_context_percentage(
                        self.used_tokens, self.context_length
                    )
                    tokens_str = f"{format_compact_tokens(self.used_tokens)}/{format_compact_tokens(self.context_length)}"
                    with ui.row().classes(
                        "input-bar-context-info items-center gap-1.5 text-[11px] font-mono text-zinc-400 select-none flex-wrap"
                    ):
                        ui.icon("memory", size="14px").classes("text-zinc-500")
                        self.model_label = ui.label(self.model_name).classes(
                            "input-bar-model-name font-medium text-zinc-300 truncate max-w-[200px]"
                        )
                        ui.label("·").classes("text-zinc-600")
                        self.context_percent_label = ui.label(pct_str).classes(
                            "input-bar-context-percent font-semibold text-emerald-400"
                        )
                        ui.label("·").classes("text-zinc-600")
                        self.context_tokens_label = ui.label(tokens_str).classes(
                            "input-bar-context-tokens text-zinc-400"
                        )
                        self.context_tooltip = ui.tooltip(
                            f"Model: {self.model_name} • Context tokens: {self.used_tokens:,} / {self.context_length:,} ({pct_str})"
                        )

                    with ui.row().classes(
                        "input-bar-approval-container items-center gap-1.5 ml-auto flex-shrink-0"
                    ):
                        self.approval_select = (
                            ui.select(
                                options=TOOL_APPROVAL_MODES,
                                value=self.approval_mode,
                                on_change=self._handle_approval_mode_change,
                            )
                            .props(
                                "dense options-dense dark borderless "
                                "popup-content-class=input-bar-approval-menu "
                                'menu-anchor="top right" menu-self="bottom right"'
                            )
                            .classes(
                                "input-bar-approval-select text-[10.5px] font-sans text-zinc-300 bg-zinc-800/80 rounded px-1.5 py-0 border border-zinc-700/60"
                            )
                        )
                        self.approval_select.tooltip(
                            "Command approval mode (Manual, Approve All, Approve via LLM)"
                        )

            # -------------------------------------------------------------
            # Linear Generation Progress Bar (below the input box)
            # -------------------------------------------------------------
            with ui.element("div").classes(
                "input-progress-container w-full"
            ) as prog_container:
                self.progress_container = prog_container
                ui.element("div").classes("input-linear-progress")

            self.progress_container.set_visibility(False)

        return root

    def _build_approval_prompt(self) -> None:
        with ui.column().classes(
            "qs-approval-box w-[calc(100%-1.25rem)] mx-auto bg-zinc-900/95 backdrop-blur-md "
            "border border-b-0 border-amber-500/60 rounded-t-xl px-3 py-2 gap-2 shadow-2xl z-0 -mb-1"
        ) as app_box:
            self.approval_container = app_box

            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes(
                    "items-center gap-1.5 text-xs font-semibold text-amber-400"
                ):
                    ui.icon("terminal", size="16px").classes("text-amber-400")
                    ui.label("Command Approval Request").classes("tracking-wide")
                ui.label("Awaiting Confirmation").classes(
                    "text-[10px] font-mono text-zinc-400 uppercase tracking-wider"
                )

            self.approval_command_label = ui.label("").classes(
                "qs-approval-command text-xs font-mono text-zinc-200 bg-zinc-950/80 px-2.5 py-1.5 rounded border border-zinc-800 break-all select-text max-h-32 overflow-y-auto w-full"
            )

            with ui.row().classes("w-full items-center justify-end gap-2 pt-0.5"):
                self.approval_decline_btn = (
                    ui.button(
                        "Decline",
                        icon="close",
                        on_click=self._handle_approval_decline,
                    )
                    .props("dense size=sm no-caps")
                    .classes(
                        "bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800/60 px-3 py-1 text-xs rounded-lg transition-colors cursor-pointer"
                    )
                )
                self.approval_approve_btn = (
                    ui.button(
                        "Approve",
                        icon="check",
                        on_click=self._handle_approval_accept,
                    )
                    .props("dense size=sm no-caps")
                    .classes(
                        "bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-800/60 px-3 py-1 text-xs rounded-lg font-medium transition-colors cursor-pointer"
                    )
                )

        self.approval_container.set_visibility(False)

    def sync_queue(
        self, queued_messages: list[QueuedMessage], *, can_steer: bool
    ) -> None:
        """Refresh queue controls and rebuild rows only when their contents change."""
        self.set_steering_enabled(can_steer)
        if self.queued_messages != queued_messages:
            self.render_queue(queued_messages)

    def render_queue(self, queued_messages: list[QueuedMessage]) -> None:
        """Render the queue box with items, Steering button, 3-dots menu, and drag handle."""
        self.queued_messages = [
            message.model_copy(deep=True) for message in queued_messages
        ]
        if not self.queue_container or not self.queue_list:
            return

        count = len(self.queued_messages)
        self.steering_buttons.clear()
        if count == 0:
            self.queue_container.set_visibility(False)
            return

        self.queue_container.set_visibility(True)
        if self.queue_count_label:
            self.queue_count_label.text = f"Queued messages ({count})"
        if self.resume_queue_btn:
            self.resume_queue_btn.set_visibility(not self.is_busy)

        self.queue_list.clear()
        with self.queue_list:
            for idx, msg in enumerate(self.queued_messages):
                visible_text = msg.visible_text
                first_line = (
                    visible_text.strip().splitlines()[0] if visible_text.strip() else ""
                )
                with (
                    ui.row()
                    .classes(
                        "qs-queue-item w-full items-center justify-between px-2.5 py-1.5 "
                        "bg-zinc-800/50 hover:bg-zinc-800/80 rounded-lg border border-zinc-700/40 "
                        "gap-2 select-none"
                    )
                    .props(f'data-id="{msg.id}" data-index="{idx}"')
                ):
                    # Left: Drag handle (if count > 1) + Order badge + 1st line of message (single line only)
                    with ui.row().classes("items-center gap-1.5 flex-1 min-w-0"):
                        # Drag handle button: MUST appear ONLY if more than 1 message
                        if count > 1:
                            ui.button(icon="drag_indicator").props(
                                "dense flat round size=xs text-color=zinc-500"
                            ).classes(
                                "qs-queue-drag-handle cursor-grab active:cursor-grabbing hover:text-zinc-200 select-none shrink-0"
                            ).tooltip("Drag to reorder queue")

                        ui.label(f"#{idx + 1}").classes(
                            "text-[10px] font-mono font-bold text-amber-400/90 shrink-0 bg-amber-950/40 px-1 py-0.5 rounded border border-amber-800/40"
                        )
                        msg_label = ui.label(first_line).classes(
                            "text-xs text-zinc-200 truncate flex-1 font-sans cursor-default"
                        )
                        if len(visible_text) > len(first_line) or len(first_line) > 50:
                            msg_label.tooltip(visible_text)

                    # Right: Steering button, 3 dots menu
                    with ui.row().classes("items-center gap-1 shrink-0"):
                        steering_button = (
                            ui.button(
                                "Steering",
                                icon="alt_route",
                                on_click=lambda _, mid=msg.id: self._trigger_steer(mid),
                            )
                            .props("dense flat size=xs text-color=zinc-400 no-caps")
                            .classes(
                                "text-[11px] px-1.5 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700/50"
                            )
                        )
                        steering_button.tooltip(
                            "Send guidance to the active agent turn"
                        )
                        steering_button.set_enabled(self.steering_enabled)
                        self.steering_buttons.append(steering_button)

                        # 3-dots menu button (Edit, Delete, Move up/down)
                        with (
                            ui.button(icon="more_vert")
                            .props("dense flat round size=xs text-color=zinc-400")
                            .classes("hover:text-zinc-200")
                        ):
                            with ui.menu().classes(
                                "bg-zinc-900 border border-zinc-800 text-zinc-200 shadow-xl py-1"
                            ):
                                with ui.menu_item(
                                    on_click=lambda _, mid=msg.id: self._trigger_edit(
                                        mid
                                    ),
                                ).classes(
                                    "qs-queue-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer"
                                ):
                                    ui.icon("edit", size="14px").classes(
                                        "qs-queue-menu-icon flex items-center justify-center self-center text-zinc-400 shrink-0"
                                    )
                                    ui.label("Edit").classes(
                                        "self-center leading-none text-zinc-200"
                                    )

                                with ui.menu_item(
                                    on_click=lambda _, mid=msg.id: self._trigger_delete(
                                        mid
                                    ),
                                ).classes(
                                    "qs-queue-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer text-rose-400"
                                ):
                                    ui.icon("delete", size="14px").classes(
                                        "qs-queue-menu-icon flex items-center justify-center self-center text-rose-400 shrink-0"
                                    )
                                    ui.label("Delete").classes(
                                        "self-center leading-none text-rose-400"
                                    )

                                if count > 1:
                                    if idx > 0:
                                        prev_id = self.queued_messages[idx - 1].id
                                        with ui.menu_item(
                                            on_click=lambda _, mid=msg.id, pid=prev_id: (
                                                self._trigger_reorder(
                                                    mid, pid, "before"
                                                )
                                            ),
                                        ).classes(
                                            "qs-queue-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer"
                                        ):
                                            ui.icon(
                                                "arrow_upward", size="14px"
                                            ).classes(
                                                "qs-queue-menu-icon flex items-center justify-center self-center text-zinc-400 shrink-0"
                                            )
                                            ui.label("Move up").classes(
                                                "self-center leading-none text-zinc-200"
                                            )

                                    if idx < count - 1:
                                        nxt_id = self.queued_messages[idx + 1].id
                                        with ui.menu_item(
                                            on_click=lambda _, mid=msg.id, nid=nxt_id: (
                                                self._trigger_reorder(mid, nid, "after")
                                            ),
                                        ).classes(
                                            "qs-queue-menu-item flex flex-row items-center gap-2.5 px-3 py-1.5 min-h-0 text-xs hover:bg-zinc-800 cursor-pointer"
                                        ):
                                            ui.icon(
                                                "arrow_downward", size="14px"
                                            ).classes(
                                                "qs-queue-menu-icon flex items-center justify-center self-center text-zinc-400 shrink-0"
                                            )
                                            ui.label("Move down").classes(
                                                "self-center leading-none text-zinc-200"
                                            )

    def _trigger_edit(self, message_id: str) -> None:
        if self.on_edit_queued:
            self.on_edit_queued(message_id)

    def _trigger_steer(self, message_id: str) -> None:
        if self.steering_enabled and self.on_steer_queued:
            self.on_steer_queued(message_id)

    def set_steering_enabled(self, enabled: bool) -> None:
        enabled = enabled and self.on_steer_queued is not None
        if self.steering_enabled == enabled:
            return
        self.steering_enabled = enabled
        for button in self.steering_buttons:
            button.set_enabled(enabled)

    def _trigger_delete(self, message_id: str) -> None:
        if self.on_delete_queued:
            self.on_delete_queued(message_id)

    def _trigger_reorder(self, source_id: str, target_id: str, position: str) -> None:
        if self.on_reorder_queued:
            self.on_reorder_queued(source_id, target_id, position)

    def _trigger_resume_queue(self) -> None:
        if self.on_resume_queue:
            self.on_resume_queue()

    def update_context_info(
        self,
        model_name: str,
        used_tokens: int,
        context_length: int,
        estimated: bool = True,
    ) -> None:
        """Update the model name, context percentage, and token ratio in the bottom-left bar."""
        self.model_name = model_name or config.default_model
        self.used_tokens = max(0, int(used_tokens))
        self.context_length = max(1, int(context_length))

        pct_str = format_context_percentage(self.used_tokens, self.context_length)
        tokens_str = f"{format_compact_tokens(self.used_tokens)}/{format_compact_tokens(self.context_length)}"
        pct_val = (self.used_tokens / float(self.context_length)) * 100.0

        if self.model_label:
            self.model_label.text = self.model_name
        if self.context_percent_label:
            self.context_percent_label.text = pct_str
            self.context_percent_label.classes(
                remove="text-emerald-400 text-amber-400 text-rose-400"
            )
            if pct_val >= 85.0:
                self.context_percent_label.classes("text-rose-400")
            elif pct_val >= 60.0:
                self.context_percent_label.classes("text-amber-400")
            else:
                self.context_percent_label.classes("text-emerald-400")
        if self.context_tokens_label:
            self.context_tokens_label.text = tokens_str
        if self.context_tooltip:
            count_label = "Estimated context tokens" if estimated else "Context tokens"
            self.context_tooltip.text = (
                f"Model: {self.model_name} • {count_label}: "
                f"{self.used_tokens:,} / {self.context_length:,} ({pct_str})"
            )

    def _handle_enter(self, e: object) -> None:
        # Check if shift key was held
        is_shift = (
            getattr(e, "args", {}).get("shiftKey", False)
            if hasattr(e, "args")
            else False
        )
        if is_shift:
            if self.textarea and self.textarea.value is not None:
                self.textarea.value += "\n"
        else:
            self._trigger_send()

    def get_temp_storage(self, session_id: str | None = None) -> TempStorage:
        return (
            self.temp_storage_getter(session_id or self.session_id)
            if self.temp_storage_getter
            else TempStorage()
        )

    def set_agent_mode(self, mode: str) -> None:
        if self.approval_select:
            self.approval_select.set_visibility(mode == "deep_agent")

    def render_attachments(self) -> None:
        if not self.attachments_container:
            return

        if not self.staged_attachments:
            self.attachments_container.set_visibility(False)
            self.attachments_container.clear()
            return

        self.attachments_container.set_visibility(True)
        self.attachments_container.clear()
        with self.attachments_container:
            for att in self.staged_attachments:
                self._render_attachment_chip(att)

    def _render_attachment_chip(self, att: StagedAttachment) -> None:
        icon_name, color_class, ext_label = get_file_icon(att.filename)
        with ui.row().classes(
            "qs-attachment-chip items-center gap-1.5 px-2.5 py-1 bg-zinc-800/90 hover:bg-zinc-800 "
            "border border-zinc-700/70 rounded-lg shadow-sm select-none max-w-[320px] transition-all"
        ):
            if att.image_urls:
                with ui.row().classes("gap-1 flex-wrap"):
                    for index, url in enumerate(att.image_urls):
                        label = att.filename
                        if att.pdf_pages:
                            label += f" · page {index + 1}"
                        render_image_thumbnail(url, label)
            else:
                ui.icon(icon_name, size="16px").classes(f"{color_class} shrink-0")
            with ui.column().classes("min-w-0 flex-1 gap-0"):
                ui.label(att.filename or att.preview).classes(
                    "text-xs font-mono text-zinc-200 truncate max-w-[140px] leading-tight"
                )
                if att.is_binary or att.token_count == 0:
                    info_text = f"{format_file_size(att.file_size)} • {ext_label}"
                else:
                    info_text = f"{att.token_count} tok • {ext_label}"
                if att.pdf_pages:
                    info_text = f"Pages 1–{len(att.image_urls)} of {att.pdf_pages}"
                ui.label(info_text).classes(
                    "text-[10px] text-zinc-400 leading-tight font-mono"
                )

            preview_snippet = (
                att.content[:200] + ("..." if len(att.content) > 200 else "")
                if not att.is_binary and att.content
                else ""
            )
            ui.tooltip(
                f"File: {att.filename}\n"
                f"Size: {format_file_size(att.file_size)} • {att.token_count} tok\n"
                f"Path: {att.relative_path or ''}\n\n"
                + (f"Preview:\n{preview_snippet}\n\n" if preview_snippet else "")
                + (
                    "• Insert into the message as plain text\n"
                    if not att.is_binary
                    else ""
                )
                + "• Remove attachment"
            ).classes("whitespace-pre-line text-xs font-mono")

            # Insert text files into the message.
            if not att.is_binary:
                (
                    ui.button(
                        icon="text_fields",
                        on_click=lambda _, a=att: self._insert_attachment_into_message(
                            a
                        ),
                    )
                    .props("dense flat round size=xs text-color=zinc-400")
                    .classes("hover:text-zinc-100 hover:bg-zinc-700/60 shrink-0")
                    .tooltip("Insert into the message as plain text")
                )

            # Remove attachment.
            (
                ui.button(
                    icon="close",
                    on_click=lambda _, a=att: self._delete_attachment(a),
                )
                .props("dense flat round size=xs text-color=zinc-400")
                .classes("hover:text-rose-400 hover:bg-rose-950/40 shrink-0")
                .tooltip("Remove attachment")
            )

    async def stage_file(
        self,
        filename: str,
        data: bytes,
    ) -> StagedAttachment:
        """Stage a file directly into temporary files."""
        storage = self.get_temp_storage()
        client = ui.context.client
        session_id = self.session_id
        att = await asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                create_staged_attachment_from_file,
                filename,
                data,
                storage=storage,
            ),
        )
        if not client.is_deleted:
            with client:
                self._accept_attachments([att], session_id)
        return att

    def set_session_id(self, session_id: str) -> None:
        if self.session_id != session_id and self.session_id is not None:
            self._drafts[self.session_id] = _InputDraft(
                text=self.textarea.value or "" if self.textarea else "",
                attachments=list(self.staged_attachments),
            )
            draft = self._drafts.get(session_id, _InputDraft())
            self.set_text(draft.text)
            self.staged_attachments = list(draft.attachments)
            self.render_attachments()
        self.session_id = session_id
        if self.input_container:
            self.input_container.props(f"data-session-id={session_id}")

    def _accept_attachments(
        self, attachments: list[StagedAttachment], session_id: str | None
    ) -> None:
        """Keep an upload with its original draft when the user changes chats."""
        if self.session_id == session_id:
            self.staged_attachments.extend(attachments)
            self.render_attachments()
            for attachment in attachments:
                if attachment.pdf_pages > len(attachment.image_urls):
                    ui.notify(
                        f"{attachment.filename}: only the first {len(attachment.image_urls)} "
                        f"of {attachment.pdf_pages} pages will be sent",
                        type="warning",
                    )
        elif session_id is not None:
            self._drafts.setdefault(session_id, _InputDraft()).attachments.extend(
                attachments
            )

    async def _handle_files_dropped(self, e: object) -> None:
        args = getattr(e, "args", None)
        session_id = self.session_id
        if isinstance(args, list):
            items = args
        elif isinstance(args, dict):
            session_id = args.get("session_id", self.session_id)
            items = args.get("files", [args])
        else:
            return

        client = ui.context.client
        try:
            storage = self.get_temp_storage(session_id)
        except ValueError as error:
            ui.notify(f"Upload error: {error}", type="negative")
            return
        loop = asyncio.get_running_loop()
        added_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                attachment = await loop.run_in_executor(
                    None,
                    partial(
                        StagedAttachment.from_upload,
                        item,
                        storage=storage,
                    ),
                )
            except (OSError, ValueError, TypeError) as error:
                with client:
                    ui.notify(f"Upload error: {error}", type="negative")
                continue
            if client.is_deleted:
                return
            with client:
                self._accept_attachments([attachment], session_id)
            added_count += 1

        if added_count > 0 and self.session_id == session_id:
            with client:
                ui.notify(
                    f"{added_count} file{'s' if added_count != 1 else ''} added to temporary storage",
                    type="positive",
                )

    def _handle_files_dropped_error(self, e: object) -> None:
        args = getattr(e, "args", {}) if hasattr(e, "args") else {}
        err_msg = (
            args.get("error", "File upload failed")
            if isinstance(args, dict)
            else str(args)
        )
        ui.notify(f"Upload error: {err_msg}", type="negative")

    def _insert_attachment_into_message(self, att: StagedAttachment) -> None:
        self._remove_staged_attachment(att)
        if self.textarea:
            curr = (self.textarea.value or "").strip()
            if curr:
                self.textarea.set_value(f"{curr}\n\n{att.content}")
            else:
                self.textarea.set_value(att.content)
            self.focus()

    def _delete_attachment(self, att: StagedAttachment) -> None:
        self._remove_staged_attachment(att)

    def _remove_staged_attachment(self, att: StagedAttachment) -> None:
        if att in self.staged_attachments:
            self.staged_attachments.remove(att)
        att.delete_file()
        self.render_attachments()

    async def _handle_paste(self, e: object) -> None:
        args = getattr(e, "args", {})
        pasted_text = (
            args.get("text", "")
            if isinstance(args, dict)
            else args
            if isinstance(args, str)
            else ""
        )
        prevented = isinstance(args, dict) and bool(args.get("prevented", False))
        pasted_text = pasted_text or (
            self.textarea.value or "" if self.textarea else ""
        )
        if not pasted_text:
            return
        client = ui.context.client
        session_id = self.session_id
        storage = self.get_temp_storage()
        try:
            attachments = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    create_staged_attachments_from_text, pasted_text, storage=storage
                ),
            )
        except (OSError, ValueError) as error:
            with client:
                ui.notify(f"Unable to prepare pasted input: {error}", type="negative")
            return
        if client.is_deleted:
            return
        with client:
            if attachments:
                if self.session_id == session_id and self.textarea and not prevented:
                    self.textarea.set_value(
                        (self.textarea.value or "").replace(pasted_text, "", 1)
                    )
                self._accept_attachments(attachments, session_id)
            elif prevented:
                if self.session_id == session_id and self.textarea:
                    self.textarea.set_value((self.textarea.value or "") + pasted_text)
                elif session_id is not None:
                    self._drafts.setdefault(
                        session_id, _InputDraft()
                    ).text += pasted_text

    def _trigger_send(self) -> None:
        if not self.textarea:
            return

        text = (self.textarea.value or "").strip()
        attachments = list(self.staged_attachments)
        if not text and not attachments:
            return

        self.textarea.set_value("")
        self.staged_attachments.clear()
        self.render_attachments()
        if self.on_send_with_attachments is None:
            self.on_send(text)
        else:
            self.on_send_with_attachments(text, attachments)

    def _handle_approval_mode_change(self) -> None:
        if self.approval_select and self.approval_select.value:
            self.approval_mode = str(self.approval_select.value)
            if self.on_approval_mode_changed:
                self.on_approval_mode_changed(self.approval_mode)

    def set_approval_mode(self, mode: str) -> None:
        self.approval_mode = normalize_tool_approval_mode(mode)
        if self.approval_select and self.approval_select.value != self.approval_mode:
            self.approval_select.value = self.approval_mode

    def show_approval_prompt(
        self,
        command: str,
        on_accept: Callable[[], None],
        on_decline: Callable[[], None],
    ) -> None:
        """Display the command confirmation prompt above the input bar."""
        self._current_approval_accept_cb = on_accept
        self._current_approval_decline_cb = on_decline
        if self.approval_command_label:
            self.approval_command_label.text = command
        if self.approval_container:
            self.approval_container.set_visibility(True)

    def hide_approval_prompt(self) -> None:
        """Hide the command confirmation prompt."""
        self._current_approval_accept_cb = None
        self._current_approval_decline_cb = None
        if self.approval_container:
            self.approval_container.set_visibility(False)

    def _handle_approval_accept(self) -> None:
        cb = self._current_approval_accept_cb
        self.hide_approval_prompt()
        if cb:
            cb()

    def _handle_approval_decline(self) -> None:
        cb = self._current_approval_decline_cb
        self.hide_approval_prompt()
        if cb:
            cb()

    def set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        if not busy:
            self.hide_approval_prompt()
            self.set_steering_enabled(False)
        if self.progress_container:
            self.progress_container.set_visibility(busy)
        if self.stop_button:
            self.stop_button.set_visibility(busy)
        if self.send_button:
            if busy:
                self.send_button.props("icon=schedule_send")
                if self.send_tooltip:
                    self.send_tooltip.text = "Add to queue (Enter)"
            else:
                self.send_button.props("icon=arrow_upward")
                if self.send_tooltip:
                    self.send_tooltip.text = "Send message (Enter)"
        if self.resume_queue_btn:
            self.resume_queue_btn.set_visibility(
                not busy and bool(self.queued_messages)
            )

    def set_text(self, text: str) -> None:
        if self.textarea:
            self.textarea.value = text

    def focus(self) -> None:
        if self.textarea:
            self.textarea.run_method("focus")
