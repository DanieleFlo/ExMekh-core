from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from nicegui import ui

from exmekh_agent.config import config
from exmekh_agent.config.approvals import (
    DEFAULT_TOOL_APPROVAL_MODE,
    TOOL_APPROVAL_MODES,
    normalize_tool_approval_mode,
)
from exmekh_agent.config.attachments import DEFAULTS as ATTACHMENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.image_input import image_bytes_from_url
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
    create_staged_attachments_from_text,
)
from exmekh_agent.exmekh_bridge.models import QueuedMessage
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text
from exmekh_agent.ui.shared.file_icons import get_file_icon
from exmekh_agent.ui.shared.image_preview import render_image_thumbnail
from exmekh_agent.ui.shared.neural_loader import NeuralLoader
from exmekh_agent.utils.formatting import (
    format_compact_tokens,
    format_context_percentage,
    format_file_size,
)

UI_PRESENTATION = presentation_config("controls/input_bar")

PASTE_JS_HANDLER = render_html(
    "shared/paste_handler.js",
    paste_char_threshold=ATTACHMENT_DEFAULTS["paste_char_threshold"],
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
        self.neural_loader: NeuralLoader | None = None
        self.is_busy: bool = False

        self.model_name: str = config.default_model
        self.used_tokens: int = 0
        self.context_length: int = config.get_context_length(config.default_model)

    def build(self) -> ui.column:
        ui.on("exmekh_files_dropped", self._handle_files_dropped)
        ui.on("exmekh_files_dropped_error", self._handle_files_dropped_error)

        with ui.column().classes(
            UI_PRESENTATION[
                "build.classes.chat_composer_w_full_max_w_4xl_mx_auto_mb_4_i"
            ]
        ) as root:
            self.root_container = root

            # -------------------------------------------------------------
            # Queue Box (peeks out partially above the input box)
            # -------------------------------------------------------------
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.exmekh_queue_box_w_calc_100_1_25rem_mx_auto_b"
                ]
            ) as q_box:
                self.queue_container = q_box

                with ui.row().classes(
                    UI_PRESENTATION[
                        "build.classes.w_full_items_center_justify_between_text_xs_t"
                    ]
                ):
                    with ui.row().classes(
                        UI_PRESENTATION["build.classes.items_center_gap_1_5"]
                    ):
                        ui.icon(
                            UI_PRESENTATION["build.icon.hourglass_top"],
                            size=UI_PRESENTATION["build.size.14px"],
                        ).classes(UI_PRESENTATION["build.classes.text_amber_400"])
                        self.queue_count_label = ui.label(
                            ui_text("chat.input_bar.queued_messages_0")
                        ).classes(
                            UI_PRESENTATION[
                                "build.classes.font_semibold_text_zinc_300_text_11px"
                            ]
                        )
                    with ui.row().classes(
                        UI_PRESENTATION["build.classes.items_center_gap_2"]
                    ):
                        self.resume_queue_btn = (
                            ui.button(
                                ui_text("chat.input_bar.send_queue"),
                                icon=UI_PRESENTATION["build.icon.play_arrow"],
                                on_click=self._trigger_resume_queue,
                            )
                            .props(
                                UI_PRESENTATION[
                                    "build.props.dense_flat_size_xs_text_color_emerald_400_no_"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "build.classes.text_11px_px_2_py_0_5_rounded_bg_emerald_950_"
                                ]
                            )
                        )
                        self.resume_queue_btn.set_visibility(False)
                        ui.label(
                            ui_text("chat.input_bar.automatic_sequential_sending")
                        ).classes(
                            UI_PRESENTATION[
                                "build.classes.text_10px_text_zinc_500_font_mono_hidden_sm_i"
                            ]
                        )

                self.queue_list = ui.column().classes(
                    UI_PRESENTATION[
                        "build.classes.w_full_max_h_40_overflow_y_auto_gap_1_pr_0_5"
                    ]
                )

            self.queue_container.set_visibility(False)

            # -------------------------------------------------------------
            # Attachments Container (peeks out / sits above the input box)
            # -------------------------------------------------------------
            with ui.row().classes(
                UI_PRESENTATION[
                    "build.classes.exmekh_attachments_bar_w_calc_100_1_25rem_mx_"
                ]
            ) as att_bar:
                self.attachments_container = att_bar

            self.attachments_container.set_visibility(False)

            self._build_approval_prompt()

            # -------------------------------------------------------------
            # Main Input Bar
            # -------------------------------------------------------------
            with ui.column().classes(
                UI_PRESENTATION[
                    "build.classes.exmekh_chat_input_box_w_full_px_2_5_pt_2_pb_1"
                ]
            ) as input_container:
                self.input_container = input_container
                with ui.element("div").classes(
                    UI_PRESENTATION["build.classes.exmekh_drop_overlay"]
                ) as drop_overlay:
                    self.drop_overlay = drop_overlay
                    with ui.element("div").classes(
                        UI_PRESENTATION[
                            "build.classes.w_11_h_11_rounded_full_bg_sky_500_20_text_sky"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION["build.icon.cloud_upload"],
                            size=UI_PRESENTATION["build.size.26px"],
                        )
                    ui.label(ui_text("chat.input_bar.drop_files_here")).classes(
                        UI_PRESENTATION[
                            "build.classes.text_sm_font_semibold_text_zinc_100_tracking_"
                        ]
                    )
                    ui.label(
                        ui_text(
                            "chat.input_bar.files_will_be_added_to_temporary_storage"
                        )
                    ).classes(
                        UI_PRESENTATION[
                            "build.classes.text_11px_text_sky_300_font_mono_select_none_"
                        ]
                    )

                with ui.row().classes(
                    UI_PRESENTATION["build.classes.w_full_items_center_gap_2_no_wrap"]
                ):
                    ui.element("input").props(
                        UI_PRESENTATION["build.props.type_file_multiple"]
                    ).classes(
                        UI_PRESENTATION["build.classes.hidden_exmekh_file_picker"]
                    ).on(
                        "change",
                        js_handler="(e) => { window.__exmekhUploadFiles(e.target.files); e.target.value = ''; }",
                    )
                    ui.button(icon=UI_PRESENTATION["build.icon.attach_file"]).props(
                        UI_PRESENTATION["build.props.flat_round_dense"]
                    ).on(
                        "click",
                        js_handler="() => document.querySelector('.exmekh-file-picker').click()",
                    ).tooltip(ui_text("chat.input_bar.attach_files_images_or_pdfs"))
                    self.textarea = (
                        ui.textarea(
                            placeholder=ui_text(
                                "chat.input_bar.send_a_message_to_exmekh_agent_enter"
                            )
                        )
                        .props(
                            UI_PRESENTATION[
                                "build.props.autogrow_borderless_dark_rows_1"
                            ]
                        )
                        .classes(
                            UI_PRESENTATION[
                                "build.classes.chat_input_textarea_flex_grow_text_sm_px_2_py"
                            ]
                        )
                    )

                    # Bind Enter key to send (preventing default newline unless shift is pressed)
                    self.textarea.on("keydown.enter.prevent", self._handle_enter)
                    self.textarea.on(
                        "paste", self._handle_paste, js_handler=PASTE_JS_HANDLER
                    )

                    # Stop Button (visible while generating)
                    self.stop_button = (
                        ui.button(
                            icon=UI_PRESENTATION["build.icon.stop"],
                            on_click=self.on_stop,
                        )
                        .props(
                            UI_PRESENTATION[
                                "build.props.round_dense_color_red_9_text_color_white"
                            ]
                        )
                        .classes(UI_PRESENTATION["build.classes.self_center_w_9_h_9"])
                    )
                    self.stop_button.tooltip(ui_text("chat.input_bar.stop_generation"))
                    self.stop_button.set_visibility(False)

                    # Send / Queue Button (always accessible)
                    self.send_button = (
                        ui.button(
                            icon=UI_PRESENTATION["build.icon.arrow_upward"],
                            on_click=self._trigger_send,
                        )
                        .props(
                            UI_PRESENTATION[
                                "build.props.round_dense_color_dark_text_color_white"
                            ]
                        )
                        .classes(
                            UI_PRESENTATION[
                                "build.classes.self_center_border_border_zinc_700_80_w_9_h_9"
                            ]
                        )
                    )
                    self.send_tooltip = ui.tooltip(
                        ui_text("chat.input_bar.send_message_enter")
                    )

                # Bottom bar: left shows model name, context percentage, and tokens/context_length;
                # right shows command approval mode selector with clear spacing from context info
                with ui.row().classes(
                    UI_PRESENTATION[
                        "build.classes.w_full_items_center_justify_between_px_2_pt_2"
                    ]
                ):
                    pct_str = format_context_percentage(
                        self.used_tokens, self.context_length
                    )
                    tokens_str = f"{format_compact_tokens(self.used_tokens)}/{format_compact_tokens(self.context_length)}"
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "build.classes.input_bar_context_info_items_center_gap_1_5_t"
                        ]
                    ):
                        ui.icon(
                            UI_PRESENTATION["build.icon.memory"],
                            size=UI_PRESENTATION["build.size.14px"],
                        ).classes(UI_PRESENTATION["build.classes.text_zinc_500"])
                        self.model_label = ui.label(self.model_name).classes(
                            UI_PRESENTATION[
                                "build.classes.input_bar_model_name_font_medium_text_zinc_30"
                            ]
                        )
                        ui.label(ui_text("chat.input_bar.value")).classes(
                            UI_PRESENTATION["build.classes.text_zinc_600"]
                        )
                        self.context_percent_label = ui.label(pct_str).classes(
                            UI_PRESENTATION[
                                "build.classes.input_bar_context_percent_font_semibold_text_"
                            ]
                        )
                        ui.label(ui_text("chat.input_bar.value")).classes(
                            UI_PRESENTATION["build.classes.text_zinc_600"]
                        )
                        self.context_tokens_label = ui.label(tokens_str).classes(
                            UI_PRESENTATION[
                                "build.classes.input_bar_context_tokens_text_zinc_400"
                            ]
                        )
                        self.context_tooltip = ui.tooltip(
                            ui_text(
                                "chat.input_bar.model_context_tokens",
                                self_model_name=self.model_name,
                                self_used_tokens=self.used_tokens,
                                self_context_length=self.context_length,
                                pct_str=pct_str,
                            )
                        )

                    with ui.row().classes(
                        UI_PRESENTATION[
                            "build.classes.input_bar_approval_container_items_center_gap"
                        ]
                    ):
                        self.approval_select = (
                            ui.select(
                                options=TOOL_APPROVAL_MODES,
                                value=self.approval_mode,
                                on_change=self._handle_approval_mode_change,
                            )
                            .props(
                                UI_PRESENTATION[
                                    "build.props.dense_options_dense_dark_borderless_popup_con"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "build.classes.input_bar_approval_select_text_10_5px_font_sa"
                                ]
                            )
                        )
                        self.approval_select.tooltip(
                            ui_text(
                                "chat.input_bar.command_approval_mode_manual_approve_all_approve"
                            )
                        )

            # -------------------------------------------------------------
            # Neural generation indicator (below the input box)
            # -------------------------------------------------------------
            with ui.element("div").classes(
                UI_PRESENTATION["build.classes.input_progress_container_w_full"]
            ) as prog_container:
                self.progress_container = prog_container
                self.neural_loader = NeuralLoader()

            self.progress_container.set_visibility(False)

        return root

    def _build_approval_prompt(self) -> None:
        with ui.column().classes(
            UI_PRESENTATION[
                "build_approval_prompt.classes.exmekh_approval_box_w_calc_100_1_25rem_mx_aut"
            ]
        ) as app_box:
            self.approval_container = app_box

            with ui.row().classes(
                UI_PRESENTATION[
                    "build_approval_prompt.classes.w_full_items_center_justify_between"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "build_approval_prompt.classes.items_center_gap_1_5_text_xs_font_semibold_te"
                    ]
                ):
                    ui.icon(
                        UI_PRESENTATION["build_approval_prompt.icon.terminal"],
                        size=UI_PRESENTATION["build_approval_prompt.size.16px"],
                    ).classes(
                        UI_PRESENTATION["build_approval_prompt.classes.text_amber_400"]
                    )
                    ui.label(
                        ui_text("chat.input_bar.command_approval_request")
                    ).classes(
                        UI_PRESENTATION["build_approval_prompt.classes.tracking_wide"]
                    )
                ui.label(ui_text("chat.input_bar.awaiting_confirmation")).classes(
                    UI_PRESENTATION[
                        "build_approval_prompt.classes.text_10px_font_mono_text_zinc_400_uppercase_t"
                    ]
                )

            self.approval_command_label = ui.label("").classes(
                UI_PRESENTATION[
                    "build_approval_prompt.classes.exmekh_approval_command_text_xs_font_mono_tex"
                ]
            )

            with ui.row().classes(
                UI_PRESENTATION[
                    "build_approval_prompt.classes.w_full_items_center_justify_end_gap_2_pt_0_5"
                ]
            ):
                self.approval_decline_btn = (
                    ui.button(
                        ui_text("chat.input_bar.decline"),
                        icon=UI_PRESENTATION["build_approval_prompt.icon.close"],
                        on_click=self._handle_approval_decline,
                    )
                    .props(
                        UI_PRESENTATION[
                            "build_approval_prompt.props.dense_size_sm_no_caps"
                        ]
                    )
                    .classes(
                        UI_PRESENTATION[
                            "build_approval_prompt.classes.bg_rose_950_80_hover_bg_rose_900_text_rose_30"
                        ]
                    )
                )
                self.approval_approve_btn = (
                    ui.button(
                        ui_text("chat.input_bar.approve"),
                        icon=UI_PRESENTATION["build_approval_prompt.icon.check"],
                        on_click=self._handle_approval_accept,
                    )
                    .props(
                        UI_PRESENTATION[
                            "build_approval_prompt.props.dense_size_sm_no_caps"
                        ]
                    )
                    .classes(
                        UI_PRESENTATION[
                            "build_approval_prompt.classes.bg_emerald_950_80_hover_bg_emerald_900_text_e"
                        ]
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
            self.queue_count_label.text = ui_text(
                "chat.input_bar.queued_messages", count=count
            )
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
                        UI_PRESENTATION[
                            "render_queue.classes.exmekh_queue_item_w_full_items_center_justify"
                        ]
                    )
                    .props(f'data-id="{msg.id}" data-index="{idx}"')
                ):
                    # Left: Drag handle (if count > 1) + Order badge + 1st line of message (single line only)
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_queue.classes.items_center_gap_1_5_flex_1_min_w_0"
                        ]
                    ):
                        # Drag handle button: MUST appear ONLY if more than 1 message
                        if count > 1:
                            ui.button(
                                icon=UI_PRESENTATION["render_queue.icon.drag_indicator"]
                            ).props(
                                UI_PRESENTATION[
                                    "render_queue.props.dense_flat_round_size_xs_text_color_zinc_500"
                                ]
                            ).classes(
                                UI_PRESENTATION[
                                    "render_queue.classes.exmekh_queue_drag_handle_cursor_grab_active_c"
                                ]
                            ).tooltip(ui_text("chat.input_bar.drag_to_reorder_queue"))

                        ui.label(
                            ui_text("chat.input_bar.value_2", value_1=idx + 1)
                        ).classes(
                            UI_PRESENTATION[
                                "render_queue.classes.text_10px_font_mono_font_bold_text_amber_400_"
                            ]
                        )
                        msg_label = ui.label(first_line).classes(
                            UI_PRESENTATION[
                                "render_queue.classes.text_xs_text_zinc_200_truncate_flex_1_font_sa"
                            ]
                        )
                        if len(visible_text) > len(first_line) or len(first_line) > 50:
                            msg_label.tooltip(visible_text)

                    # Right: Steering button, 3 dots menu
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_queue.classes.items_center_gap_1_shrink_0"
                        ]
                    ):
                        steering_button = (
                            ui.button(
                                ui_text("chat.input_bar.steering"),
                                icon=UI_PRESENTATION["render_queue.icon.alt_route"],
                                on_click=lambda _, mid=msg.id: self._trigger_steer(mid),
                            )
                            .props(
                                UI_PRESENTATION[
                                    "render_queue.props.dense_flat_size_xs_text_color_zinc_400_no_cap"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "render_queue.classes.text_11px_px_1_5_py_0_5_rounded_bg_zinc_800_h"
                                ]
                            )
                        )
                        steering_button.tooltip(
                            ui_text(
                                "chat.input_bar.send_guidance_to_the_active_agent_turn"
                            )
                        )
                        steering_button.set_enabled(self.steering_enabled)
                        self.steering_buttons.append(steering_button)

                        # 3-dots menu button (Edit, Delete, Move up/down)
                        with (
                            ui.button(
                                icon=UI_PRESENTATION["render_queue.icon.more_vert"]
                            )
                            .props(
                                UI_PRESENTATION[
                                    "render_queue.props.dense_flat_round_size_xs_text_color_zinc_400"
                                ]
                            )
                            .classes(
                                UI_PRESENTATION[
                                    "render_queue.classes.hover_text_zinc_200"
                                ]
                            )
                        ):
                            with ui.menu().classes(
                                UI_PRESENTATION[
                                    "render_queue.classes.bg_zinc_900_border_border_zinc_800_text_zinc_"
                                ]
                            ):
                                with ui.menu_item(
                                    on_click=lambda _, mid=msg.id: self._trigger_edit(
                                        mid
                                    ),
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_queue.classes.exmekh_queue_menu_item_flex_flex_row_items_ce"
                                    ]
                                ):
                                    ui.icon(
                                        UI_PRESENTATION["render_queue.icon.edit"],
                                        size=UI_PRESENTATION["render_queue.size.14px"],
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_queue.classes.exmekh_queue_menu_icon_flex_items_center_just"
                                        ]
                                    )
                                    ui.label(ui_text("chat.input_bar.edit")).classes(
                                        UI_PRESENTATION[
                                            "render_queue.classes.self_center_leading_none_text_zinc_200"
                                        ]
                                    )

                                with ui.menu_item(
                                    on_click=lambda _, mid=msg.id: self._trigger_delete(
                                        mid
                                    ),
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_queue.classes.exmekh_queue_menu_item_flex_flex_row_items_ce_2"
                                    ]
                                ):
                                    ui.icon(
                                        UI_PRESENTATION["render_queue.icon.delete"],
                                        size=UI_PRESENTATION["render_queue.size.14px"],
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_queue.classes.exmekh_queue_menu_icon_flex_items_center_just_2"
                                        ]
                                    )
                                    ui.label(ui_text("chat.input_bar.delete")).classes(
                                        UI_PRESENTATION[
                                            "render_queue.classes.self_center_leading_none_text_rose_400"
                                        ]
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
                                            UI_PRESENTATION[
                                                "render_queue.classes.exmekh_queue_menu_item_flex_flex_row_items_ce"
                                            ]
                                        ):
                                            ui.icon(
                                                UI_PRESENTATION[
                                                    "render_queue.icon.arrow_upward"
                                                ],
                                                size=UI_PRESENTATION[
                                                    "render_queue.size.14px"
                                                ],
                                            ).classes(
                                                UI_PRESENTATION[
                                                    "render_queue.classes.exmekh_queue_menu_icon_flex_items_center_just"
                                                ]
                                            )
                                            ui.label(
                                                ui_text("chat.input_bar.move_up")
                                            ).classes(
                                                UI_PRESENTATION[
                                                    "render_queue.classes.self_center_leading_none_text_zinc_200"
                                                ]
                                            )

                                    if idx < count - 1:
                                        nxt_id = self.queued_messages[idx + 1].id
                                        with ui.menu_item(
                                            on_click=lambda _, mid=msg.id, nid=nxt_id: (
                                                self._trigger_reorder(mid, nid, "after")
                                            ),
                                        ).classes(
                                            UI_PRESENTATION[
                                                "render_queue.classes.exmekh_queue_menu_item_flex_flex_row_items_ce"
                                            ]
                                        ):
                                            ui.icon(
                                                UI_PRESENTATION[
                                                    "render_queue.icon.arrow_downward"
                                                ],
                                                size=UI_PRESENTATION[
                                                    "render_queue.size.14px"
                                                ],
                                            ).classes(
                                                UI_PRESENTATION[
                                                    "render_queue.classes.exmekh_queue_menu_icon_flex_items_center_just"
                                                ]
                                            )
                                            ui.label(
                                                ui_text("chat.input_bar.move_down")
                                            ).classes(
                                                UI_PRESENTATION[
                                                    "render_queue.classes.self_center_leading_none_text_zinc_200"
                                                ]
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
                remove=UI_PRESENTATION["update_context_info.state_classes"]
            )
            if pct_val >= 85.0:
                self.context_percent_label.classes(
                    UI_PRESENTATION["update_context_info.classes.text_rose_400"]
                )
            elif pct_val >= 60.0:
                self.context_percent_label.classes(
                    UI_PRESENTATION["update_context_info.classes.text_amber_400"]
                )
            else:
                self.context_percent_label.classes(
                    UI_PRESENTATION["update_context_info.classes.text_emerald_400"]
                )
        if self.context_tokens_label:
            self.context_tokens_label.text = tokens_str
        if self.context_tooltip:
            count_label = (
                app_text("chat.input_bar.estimated_context_tokens")
                if estimated
                else app_text("chat.input_bar.context_tokens")
            )
            self.context_tooltip.text = ui_text(
                "chat.input_bar.model",
                self_model_name=self.model_name,
                count_label=count_label,
                self_used_tokens=self.used_tokens,
                self_context_length=self.context_length,
                pct_str=pct_str,
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
            UI_PRESENTATION[
                "render_attachment_chip.classes.exmekh_attachment_chip_items_center_gap_1_5_p"
            ]
        ):
            if att.image_urls:
                with ui.row().classes(
                    UI_PRESENTATION["render_attachment_chip.classes.gap_1_flex_wrap"]
                ):
                    for index, url in enumerate(att.image_urls):
                        label = att.filename
                        if att.pdf_pages:
                            label += f" · page {index + 1}"
                        render_image_thumbnail(url, label)
            else:
                ui.icon(
                    icon_name, size=UI_PRESENTATION["render_attachment_chip.size.16px"]
                ).classes(
                    UI_PRESENTATION["render_attachment_chip.classes_template"].format(
                        color_class=color_class
                    )
                )
            with ui.column().classes(
                UI_PRESENTATION["render_attachment_chip.classes.min_w_0_flex_1_gap_0"]
            ):
                ui.label(att.filename or att.preview).classes(
                    UI_PRESENTATION[
                        "render_attachment_chip.classes.text_xs_font_mono_text_zinc_200_truncate_max_"
                    ]
                )
                if att.is_binary or att.token_count == 0:
                    info_text = app_text(
                        "chat.input_bar.attachment_size",
                        size=format_file_size(att.file_size),
                        extension=ext_label,
                    )
                else:
                    info_text = app_text(
                        "chat.input_bar.attachment_tokens",
                        tokens=att.token_count,
                        extension=ext_label,
                    )
                if att.pdf_pages:
                    info_text = app_text(
                        "chat.input_bar.pages_1_value_1_of_pdf_pages",
                        value_1=len(att.image_urls),
                        pdf_pages=att.pdf_pages,
                    )
                ui.label(info_text).classes(
                    UI_PRESENTATION[
                        "render_attachment_chip.classes.text_10px_text_zinc_400_leading_tight_font_mo"
                    ]
                )

            preview_snippet = (
                att.content[: ATTACHMENT_DEFAULTS["tooltip_preview_char_limit"]]
                + (
                    app_text("chat.input_bar.preview_ellipsis")
                    if len(att.content)
                    > ATTACHMENT_DEFAULTS["tooltip_preview_char_limit"]
                    else ""
                )
                if not att.is_binary and att.content
                else ""
            )
            ui.tooltip(
                app_text(
                    "chat.input_bar.attachment_details",
                    filename=att.filename,
                    size=format_file_size(att.file_size),
                    tokens=att.token_count,
                    path=att.relative_path or "",
                    preview=app_text(
                        "chat.input_bar.attachment_preview", snippet=preview_snippet
                    )
                    if preview_snippet
                    else "",
                    insert_hint=app_text("chat.input_bar.attachment_insert_hint")
                    if not att.is_binary
                    else "",
                )
            ).classes(
                UI_PRESENTATION[
                    "render_attachment_chip.classes.whitespace_pre_line_text_xs_font_mono"
                ]
            )

            # Insert text files into the message.
            if not att.is_binary:
                (
                    ui.button(
                        icon=UI_PRESENTATION["render_attachment_chip.icon.text_fields"],
                        on_click=lambda _, a=att: self._insert_attachment_into_message(
                            a
                        ),
                    )
                    .props(
                        UI_PRESENTATION[
                            "render_attachment_chip.props.dense_flat_round_size_xs_text_color_zinc_400"
                        ]
                    )
                    .classes(
                        UI_PRESENTATION[
                            "render_attachment_chip.classes.hover_text_zinc_100_hover_bg_zinc_700_60_shri"
                        ]
                    )
                    .tooltip(
                        ui_text("chat.input_bar.insert_into_the_message_as_plain_text")
                    )
                )

            # Remove attachment.
            (
                ui.button(
                    icon=UI_PRESENTATION["render_attachment_chip.icon.close"],
                    on_click=lambda _, a=att: self._delete_attachment(a),
                )
                .props(
                    UI_PRESENTATION[
                        "render_attachment_chip.props.dense_flat_round_size_xs_text_color_zinc_400"
                    ]
                )
                .classes(
                    UI_PRESENTATION[
                        "render_attachment_chip.classes.hover_text_rose_400_hover_bg_rose_950_40_shri"
                    ]
                )
                .tooltip(ui_text("chat.input_bar.remove_attachment"))
            )

    async def stage_image(self, url: str) -> StagedAttachment:
        """Attach a generated PNG to the current draft without sending it."""
        session_id = self.session_id
        storage = self.get_temp_storage()
        client = ui.context.client
        attachment = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: create_staged_attachment_from_file(
                "generated-image.png", image_bytes_from_url(url), storage=storage
            ),
        )
        if not client.is_deleted:
            with client:
                self._accept_attachments([attachment], session_id)
        return attachment

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
                        ui_text(
                            "chat.input_bar.only_the_first_of_pages_will_be",
                            attachment_filename=attachment.filename,
                            len_attachment_image_urls=len(attachment.image_urls),
                            attachment_pdf_pages=attachment.pdf_pages,
                        ),
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
            ui.notify(
                ui_text("chat.input_bar.upload_error_formatted", error=error),
                type="negative",
            )
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
                    ui.notify(
                        ui_text("chat.input_bar.upload_error_formatted", error=error),
                        type="negative",
                    )
                continue
            if client.is_deleted:
                return
            with client:
                self._accept_attachments([attachment], session_id)
            added_count += 1

        if added_count > 0 and self.session_id == session_id:
            with client:
                ui.notify(
                    ui_text(
                        "chat.input_bar.file_added_to_temporary_storage",
                        added_count=added_count,
                        s_if_added_count_1_else="s" if added_count != 1 else "",
                    ),
                    type="positive",
                )

    def _handle_files_dropped_error(self, e: object) -> None:
        args = getattr(e, "args", {}) if hasattr(e, "args") else {}
        err_msg = (
            args.get("error", app_text("chat.input_bar.file_upload_failed"))
            if isinstance(args, dict)
            else str(args)
        )
        ui.notify(
            ui_text("chat.input_bar.upload_error", err_msg=err_msg), type="negative"
        )

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
                ui.notify(
                    ui_text(
                        "chat.input_bar.unable_to_prepare_pasted_input", error=error
                    ),
                    type="negative",
                )
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
            mode = normalize_tool_approval_mode(str(self.approval_select.value))
            if mode == self.approval_mode:
                return
            self.approval_mode = mode
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
        if self.neural_loader:
            self.neural_loader.set_active(busy)
        if self.stop_button:
            self.stop_button.set_visibility(busy)
        if self.send_button:
            if busy:
                self.send_button.props(
                    UI_PRESENTATION["set_busy.props.icon_schedule_send"]
                )
                if self.send_tooltip:
                    self.send_tooltip.text = ui_text(
                        "chat.input_bar.add_to_queue_enter"
                    )
            else:
                self.send_button.props(
                    UI_PRESENTATION["set_busy.props.icon_arrow_upward"]
                )
                if self.send_tooltip:
                    self.send_tooltip.text = ui_text(
                        "chat.input_bar.send_message_enter"
                    )
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
