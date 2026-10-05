from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from markupsafe import Markup
from nicegui import ui

import exmekh_agent.exmekh_bridge.providers.context_usage as context_usage
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.exmekh_bridge.agents.system_prompt import load_real_hds_system_prompt
from exmekh_agent.exmekh_bridge.history.skill_history import (
    skill_opening_call_id,
    take_skill_for_call,
)
from exmekh_agent.exmekh_bridge.history.tool_calls import parse_tool_calls
from exmekh_agent.exmekh_bridge.models import (
    ChatSession,
    SkillNode,
    ToolEvent,
    record_tool_event,
)
from exmekh_agent.exmekh_bridge.resources import load_prompt
from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.resources import presentation_config, render_html, ui_text
from exmekh_agent.ui.shared.disclosure import DisclosureState

UI_PRESENTATION = presentation_config("controls/raw_history_view")


@dataclass
class _LiveToolCard:
    summary: dict[str, Any]
    container: ui.column
    index: int


def build_raw_history_from_session(
    session: ChatSession,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Show recorded messages, or clearly marked summaries for legacy sessions.

    Execution metadata does not contain the original provider messages and tool
    outputs, so summaries must remain distinguishable from recorded transcripts.
    """
    raw_messages: list[dict[str, Any]] = []
    skill_tree: list[dict[str, Any]] = []
    has_system_message = False

    default_sys_prompt = session.system_prompt
    if not default_sys_prompt:
        if session.agent_mode == "deep_agent":
            default_sys_prompt = load_real_hds_system_prompt(
                capabilities_path_str=session.settings.capabilities_path,
                name_hds=session.settings.name_hds or "main",
                is_sub_agent=False,
            )
        if not default_sys_prompt:
            default_sys_prompt = load_prompt("quick_chat")

    for i, msg in enumerate(session.messages):
        if msg.role == "user":
            # Check if the subsequent assistant message already recorded raw_history that includes this user message
            has_subsequent_recorded_user = False
            if i + 1 < len(session.messages):
                next_msg = session.messages[i + 1]
                if next_msg.role == "assistant" and next_msg.raw_history:
                    for rm in next_msg.raw_history:
                        if rm.get("role") == "user":
                            has_subsequent_recorded_user = True
                            break

            if not has_subsequent_recorded_user:
                # If no system message has been emitted yet, emit system prompt first
                if not has_system_message:
                    raw_messages.append(
                        {"role": "system", "content": default_sys_prompt}
                    )
                    has_system_message = True
                raw_messages.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            if msg.skill_history_tree:
                skill_tree.extend(msg.skill_history_tree)

            if msg.raw_history:
                # If this raw_history doesn't contain a system message and none was emitted yet, emit default system prompt first
                if not has_system_message and not any(
                    rm.get("role") == "system" for rm in msg.raw_history
                ):
                    raw_messages.append(
                        {"role": "system", "content": default_sys_prompt}
                    )
                    has_system_message = True

                new_entries = []
                for rm in msg.raw_history:
                    if rm.get("role") == "system":
                        if not has_system_message:
                            raw_messages.append(rm)
                            has_system_message = True
                    else:
                        new_entries.append(rm)

                # Check if new_entries already includes the prior messages in raw_messages (cumulative dump)
                existing_non_sys = [
                    m for m in raw_messages if m.get("role") != "system"
                ]
                if len(existing_non_sys) > 0 and len(new_entries) >= len(
                    existing_non_sys
                ):
                    match = True
                    for em, nm in zip(
                        existing_non_sys,
                        new_entries[: len(existing_non_sys)],
                        strict=True,
                    ):
                        if em.get("role") != nm.get("role") or em.get(
                            "content"
                        ) != nm.get("content"):
                            match = False
                            break
                    if match:
                        for nm in new_entries[len(existing_non_sys) :]:
                            raw_messages.append(nm)
                    else:
                        for nm in new_entries:
                            raw_messages.append(nm)
                else:
                    for nm in new_entries:
                        raw_messages.append(nm)
            else:
                # Legacy messages have only a summary, not a provider transcript.
                if not has_system_message:
                    raw_messages.append(
                        {"role": "system", "content": default_sys_prompt}
                    )
                    has_system_message = True

                for skill in msg.skills:
                    skill_tree.append(_summarize_skill_tree_node(skill))
                summary: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                    "source": "session_summary",
                }
                if msg.skills:
                    summary["skills"] = [skill.name for skill in msg.skills]
                if msg.tool_events:
                    summary["tool_events"] = [
                        {"name": ev.tool_name, "event_type": ev.event_type}
                        for ev in msg.tool_events
                    ]
                raw_messages.append(summary)

    return raw_messages, skill_tree


def _summarize_skill_tree_node(node: SkillNode) -> dict[str, Any]:
    """Keep hierarchy and observed events without inventing a transcript."""
    return {
        "skill": node.name,
        "id": node.id,
        "opening_tool_call_id": node.opening_tool_call_id,
        "source": "session_summary",
        "status": node.status,
        "history": [
            {
                "source": "session_summary",
                "event": "reasoning",
                "content": block.content,
            }
            for block in node.reasoning_blocks
        ]
        + [
            {
                "source": "session_summary",
                "event": ev.event_type,
                "tool_name": ev.tool_name,
            }
            for ev in node.tool_events
        ],
        "children": [_summarize_skill_tree_node(sub) for sub in node.sub_skills],
    }


class RawHistoryComponent:
    """Component that renders the raw conversation history as colorized JSON,

    inserting vertical collapsible compactions at sub-skill invocation points
    with recursive matryoshka nesting.
    """

    def __init__(self) -> None:
        self.container: ui.column | None = None
        self.content_container: ui.column | None = None
        self.expansions: list[ui.expansion] = []
        self.current_session: ChatSession | None = None
        self.status_label: ui.label | None = None
        self.raw_json_data: list[dict[str, Any]] = []
        self.skill_tree_data: list[dict[str, Any]] = []
        self._skill_disclosures = DisclosureState()
        self._expansion_keys: dict[int, str] = {}

        # Progressive Live Streaming State ("man mano")
        self.is_streaming: bool = False
        self.live_container: ui.column | None = None
        self.live_events_container: ui.column | None = None
        self.live_assistant_card: ui.column | None = None
        self.live_json_html: ui.html | None = None
        self.live_status_badge: ui.row | None = None
        self.live_response_text: str = ""
        self.live_reasoning_text: str = ""
        self.live_tool_events: list[ToolEvent] = []
        self._live_tool_cards: dict[str, _LiveToolCard] = {}
        self.live_skill_nodes: dict[str, SkillNode] = {}
        self.live_turn_index: int = 1
        self._last_token_render_time: float = 0.0

    def build(self) -> ui.column:
        with ui.column().classes(
            UI_PRESENTATION[
                "build.classes.w_full_h_full_flex_grow_px_4_py_4_gap_4_overf"
            ]
        ) as col:
            self.container = col
            self._render_toolbar()
            with ui.column().classes(
                UI_PRESENTATION["build.classes.w_full_gap_3"]
            ) as content:
                self.content_container = content

        return col

    def _render_toolbar(self) -> None:
        with ui.row().classes(
            UI_PRESENTATION[
                "render_toolbar.classes.raw_history_toolbar_w_full_items_center_justi"
            ]
        ):
            with ui.row().classes(
                UI_PRESENTATION["render_toolbar.classes.items_center_gap_2"]
            ):
                ui.icon(
                    UI_PRESENTATION["render_toolbar.icon.data_object"],
                    size=UI_PRESENTATION["render_toolbar.size.sm"],
                ).classes(UI_PRESENTATION["render_toolbar.classes.text_zinc_400"])
                ui.label(
                    ui_text("history.raw_history_view.raw_history_json_inspector")
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.text_sm_font_semibold_tracking_tight_text_zin"
                    ]
                )
                self.status_label = ui.label(
                    ui_text("history.raw_history_view.0_messages")
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.font_mono_text_xs_text_zinc_400_bg_zinc_900_8"
                    ]
                )

            with ui.row().classes(
                UI_PRESENTATION["render_toolbar.classes.items_center_gap_2"]
            ):
                ui.button(
                    ui_text("history.raw_history_view.expand_all"),
                    icon=UI_PRESENTATION["render_toolbar.icon.unfold_more"],
                    on_click=lambda: self.toggle_all(True),
                ).props(
                    UI_PRESENTATION[
                        "render_toolbar.props.flat_dense_no_caps_text_color_grey_4"
                    ]
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.text_xs_px_2_5_py_1_bg_zinc_900_60_hover_bg_z"
                    ]
                )

                ui.button(
                    ui_text("history.raw_history_view.collapse_all"),
                    icon=UI_PRESENTATION["render_toolbar.icon.unfold_less"],
                    on_click=lambda: self.toggle_all(False),
                ).props(
                    UI_PRESENTATION[
                        "render_toolbar.props.flat_dense_no_caps_text_color_grey_4"
                    ]
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.text_xs_px_2_5_py_1_bg_zinc_900_60_hover_bg_z"
                    ]
                )

                ui.button(
                    ui_text("history.raw_history_view.copy_full_json"),
                    icon=UI_PRESENTATION["render_toolbar.icon.content_copy"],
                    on_click=self._handle_copy_full_json,
                ).props(
                    UI_PRESENTATION[
                        "render_toolbar.props.flat_dense_no_caps_text_color_white"
                    ]
                ).classes(
                    UI_PRESENTATION[
                        "render_toolbar.classes.text_xs_px_3_py_1_bg_zinc_800_hover_bg_zinc_7"
                    ]
                )

    def render_session(self, session: ChatSession) -> None:
        if not self.container or not self.content_container:
            return

        self.current_session = session
        self.expansions.clear()
        self._expansion_keys.clear()
        self.content_container.clear()
        self.live_container = None
        self.live_events_container = None
        self.live_assistant_card = None
        self.live_json_html = None
        self.live_status_badge = None

        context_usage.populate_skill_closure_tokens(session)
        raw_messages, skill_tree = build_raw_history_from_session(session)
        self.raw_json_data = raw_messages
        self.skill_tree_data = skill_tree

        if self.status_label:
            count_msg = len(raw_messages)
            count_skills = len(skill_tree)
            self.status_label.text = ui_text(
                "history.raw_history_view.messages_sub_skills",
                count_msg=count_msg,
                count_skills=count_skills,
            )

        with self.content_container:
            if not raw_messages:
                with ui.column().classes(
                    UI_PRESENTATION[
                        "render_session.classes.w_full_py_16_items_center_justify_center_text"
                    ]
                ):
                    ui.icon(
                        UI_PRESENTATION["render_session.icon.terminal"],
                        size=UI_PRESENTATION["render_session.size.md"],
                    ).classes(UI_PRESENTATION["render_session.classes.text_zinc_600"])
                    ui.label(
                        ui_text("history.raw_history_view.no_history_available")
                    ).classes(
                        UI_PRESENTATION[
                            "render_session.classes.text_sm_text_zinc_400_font_mono"
                        ]
                    )
                    ui.label(
                        ui_text(
                            "history.raw_history_view.send_a_message_in_chat_to_generate"
                        )
                    ).classes(
                        UI_PRESENTATION[
                            "render_session.classes.text_xs_text_zinc_600_max_w_sm"
                        ]
                    )
                return

            self._render_message_sequence(raw_messages, skill_tree, depth=0)

    def _render_message_sequence(
        self,
        messages: list[dict[str, Any]],
        skill_tree: list[dict[str, Any]],
        depth: int = 0,
    ) -> None:
        """Renders messages in sequence, inserting vertical compactions between

        skill calls and their respective responses, applying matryoshka nesting.
        """
        remaining_skills = list(skill_tree)
        rendered_call_ids: set[str] = set()

        for idx, msg in enumerate(messages):
            # Render current message JSON card
            self._render_single_raw_message(msg, index=idx + 1, depth=depth)

            for call in parse_tool_calls(msg.get("tool_calls")) or []:
                if call.get_id() in rendered_call_ids:
                    continue
                matched_skill = take_skill_for_call(remaining_skills, call)
                if matched_skill is not None:
                    rendered_call_ids.add(call.get_id())
                    self._render_skill_compaction(
                        matched_skill,
                        depth=depth + 1,
                        prefix_messages=messages[: idx + 1],
                    )

        # Fallback: render any skills in skill_tree that were not matched by inline tool_calls
        for sk in remaining_skills:
            if sk.get("opening_tool_call_id") or sk.get("skill"):
                self._render_skill_compaction(
                    sk,
                    depth=depth + 1,
                    prefix_messages=messages,
                )

    def _render_single_raw_message(
        self, msg: dict[str, Any], index: int, depth: int
    ) -> None:
        """Renders one raw message JSON card with syntax highlighting and copy action."""
        role = str(msg.get("role", "unknown")).lower()

        # Border accent based on role
        if role == "user":
            border_cls = UI_PRESENTATION["message.user.border"]
            role_badge_cls = UI_PRESENTATION["message.user.badge"]
            icon_name = UI_PRESENTATION["render_single_raw_message.icon.person"]
        elif role in {"tool", "function"}:
            border_cls = UI_PRESENTATION["message.tool.border"]
            role_badge_cls = UI_PRESENTATION["message.tool.badge"]
            icon_name = UI_PRESENTATION["render_single_raw_message.icon.build"]
        elif role == "system":
            border_cls = UI_PRESENTATION["message.system.border"]
            role_badge_cls = UI_PRESENTATION["message.system.badge"]
            icon_name = UI_PRESENTATION["render_single_raw_message.icon.settings"]
        else:
            border_cls = UI_PRESENTATION["message.assistant.border"]
            role_badge_cls = UI_PRESENTATION["message.assistant.badge"]
            icon_name = UI_PRESENTATION["render_single_raw_message.icon.smart_toy"]

        with ui.column().classes(
            UI_PRESENTATION["render_single_raw_message.classes_template"].format(
                border_cls=border_cls
            )
        ):
            # Header row with role, index, and copy button
            with ui.row().classes(
                UI_PRESENTATION[
                    "render_single_raw_message.classes.w_full_items_center_justify_between"
                ]
            ):
                with ui.row().classes(
                    UI_PRESENTATION[
                        "render_single_raw_message.classes.items_center_gap_2"
                    ]
                ):
                    ui.icon(
                        icon_name,
                        size=UI_PRESENTATION["render_single_raw_message.size.xs"],
                    ).classes(
                        UI_PRESENTATION[
                            "render_single_raw_message.classes.text_zinc_400"
                        ]
                    )
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_single_raw_message.classes_template_2"
                        ].format(role_badge_cls=role_badge_cls)
                    ):
                        ui.label(role.upper())
                    ui.label(
                        ui_text("history.raw_history_view.message", index=index)
                    ).classes(
                        UI_PRESENTATION[
                            "render_single_raw_message.classes.font_mono_text_11px_text_zinc_500"
                        ]
                    )

                    # If this message is a tool response or contains tool calls, add helper pills
                    if role in {"tool", "function"} and msg.get("name"):
                        ui.label(
                            ui_text(
                                "history.raw_history_view.tool", msg_name=msg["name"]
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_single_raw_message.classes.font_mono_text_11px_text_amber_400_90_bg_ambe"
                            ]
                        )
                    elif msg.get("tool_calls"):
                        ui.label(ui_text("history.raw_history_view.tool_call")).classes(
                            UI_PRESENTATION[
                                "render_single_raw_message.classes.font_mono_text_10px_text_amber_300_bg_amber_9"
                            ]
                        )
                    if msg.get("source") == "session_summary":
                        ui.label(
                            ui_text(
                                "history.raw_history_view.summary_raw_trace_not_available"
                            )
                        ).classes(
                            UI_PRESENTATION[
                                "render_single_raw_message.classes.font_mono_text_10px_text_zinc_400"
                            ]
                        )

                # Copy message button
                msg_json = json.dumps(msg, indent=2, ensure_ascii=False)
                ui.button(
                    icon=UI_PRESENTATION["render_single_raw_message.icon.content_copy"],
                    on_click=lambda _, text=msg_json: self._copy_text(
                        text, app_text("history.raw_history_view.message_copied")
                    ),
                ).props(
                    UI_PRESENTATION[
                        "render_single_raw_message.props.flat_round_dense_size_xs"
                    ]
                ).classes(
                    UI_PRESENTATION[
                        "render_single_raw_message.classes.text_zinc_600_hover_text_zinc_300"
                    ]
                )

            # Formatted and syntax highlighted JSON code block
            highlighted_html = format_json_to_highlighted_html(msg)
            ui.html(
                render_html(
                    "shared/raw_json.html",
                    content=Markup(highlighted_html),
                    extra_classes="",
                )
            ).classes(UI_PRESENTATION["render_single_raw_message.classes.w_full"])

    def _render_skill_compaction(
        self,
        skill_data: dict[str, Any],
        depth: int,
        is_live: bool = False,
        prefix_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Renders a vertical collapsible compaction section between skill call and response.

        When expanded, reveals the inner history with recursive matryoshka styling.
        Preserves user expansion states across renders.
        """
        skill_name = skill_data.get("skill", "sub-skill")
        sub_messages = skill_data.get("history", [])
        children = skill_data.get("children", [])
        msg_count = len(sub_messages)

        # Depth guideline colors: depth 1 = emerald, depth 2 = purple, depth 3+ = amber
        if depth == 1:
            depth_color_border = UI_PRESENTATION["skill.depth.first.border"]
            badge_color = UI_PRESENTATION["skill.depth.first.badge"]
            depth_label = app_text("history.raw_history_view.depth_1")
        elif depth == 2:
            depth_color_border = UI_PRESENTATION["skill.depth.second.border"]
            badge_color = UI_PRESENTATION["skill.depth.second.badge"]
            depth_label = app_text("history.raw_history_view.depth_2_matryoshka")
        else:
            depth_color_border = UI_PRESENTATION["skill.depth.deeper.border"]
            badge_color = UI_PRESENTATION["skill.depth.deeper.badge"]
            depth_label = app_text(
                "history.raw_history_view.depth_depth_matryoshka", depth=depth
            )

        # Vertical compaction wrapper
        with ui.column().classes(
            UI_PRESENTATION["render_skill_compaction.classes.w_full_my_1"]
        ):
            with ui.expansion("").classes(
                UI_PRESENTATION[
                    "render_skill_compaction.classes.w_full_subskill_compaction_rounded_lg_border_"
                ]
            ) as exp:
                # Restore previous expansion state or default to collapsed
                invocation_id = (
                    skill_opening_call_id(skill_data)
                    or skill_data.get("id")
                    or f"legacy:{depth}:{len(self.expansions)}:{skill_name}"
                )
                session_id = self.current_session.id if self.current_session else ""
                key = f"{session_id}:{invocation_id}"
                exp.value = self._skill_disclosures.is_expanded(key)
                self.expansions.append(exp)
                self._expansion_keys[exp.id] = key
                exp.on_value_change(
                    lambda event: self._skill_disclosures.set_expanded(key, event.value)
                )

                # Custom compact header
                with exp.add_slot("header"):
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "render_skill_compaction.classes.items_center_justify_between_w_full_py_1_5_pr"
                        ]
                    ):
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_skill_compaction.classes.items_center_gap_2_5"
                            ]
                        ):
                            ui.icon(
                                UI_PRESENTATION["render_skill_compaction.icon.hub"],
                                size=UI_PRESENTATION["render_skill_compaction.size.xs"],
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.text_zinc_400"
                                ]
                            )
                            ui.label(
                                ui_text(
                                    "history.raw_history_view.sub_skill",
                                    skill_name=skill_name,
                                )
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.font_mono_font_medium_text_zinc_200_text_xs"
                                ]
                            )
                            with ui.row().classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes_template_2"
                                ].format(badge_color=badge_color)
                            ):
                                ui.label(depth_label)
                            ui.label(
                                ui_text(
                                    "history.raw_history_view.messages_in_sub_skill",
                                    msg_count=msg_count,
                                )
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.font_mono_text_11px_text_zinc_400_bg_zinc_900"
                                ]
                            )

                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_skill_compaction.classes.items_center_gap_2"
                            ]
                        ):
                            if is_live:
                                with ui.row().classes(
                                    UI_PRESENTATION[
                                        "render_skill_compaction.classes.items_center_gap_1_5_px_1_5_py_0_5_rounded_te"
                                    ]
                                ):
                                    ui.spinner(
                                        UI_PRESENTATION[
                                            "render_skill_compaction.spinner.dots"
                                        ],
                                        size=UI_PRESENTATION[
                                            "render_skill_compaction.size.xs"
                                        ],
                                        color=UI_PRESENTATION[
                                            "render_skill_compaction.color.emerald_4"
                                        ],
                                    )
                                    ui.label(
                                        ui_text("history.raw_history_view.running")
                                    )
                            elif skill_data.get("status") == "error":
                                ui.label(
                                    ui_text("history.raw_history_view.error")
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_skill_compaction.classes.font_mono_text_10px_text_red_400"
                                    ]
                                )
                            elif skill_data.get("tokens_pending"):
                                ui.label(
                                    ui_text("history.raw_history_view.completed")
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_skill_compaction.classes.font_mono_text_10px_text_emerald_400"
                                    ]
                                )
                            else:
                                before_tok, after_tok = (
                                    context_usage.compute_raw_skill_compaction_tokens(
                                        skill_data, prefix_messages=prefix_messages
                                    )
                                )
                                with (
                                    ui.row()
                                    .classes(
                                        UI_PRESENTATION[
                                            "render_skill_compaction.classes.raw_skill_closure_tokens_items_center_gap_1_p"
                                        ]
                                    )
                                    .props(
                                        f'title="Context tokens before closing: {before_tok:,} • after closing: {after_tok:,}"'
                                    )
                                ):
                                    ui.icon(
                                        UI_PRESENTATION[
                                            "render_skill_compaction.icon.compress"
                                        ],
                                        size=UI_PRESENTATION[
                                            "render_skill_compaction.size.12px"
                                        ],
                                    ).classes(
                                        UI_PRESENTATION[
                                            "render_skill_compaction.classes.text_emerald_400"
                                        ]
                                    )
                                    ui.label(
                                        context_usage.format_skill_closure_tokens(
                                            before_tok, after_tok
                                        )
                                    )
                                ui.label(
                                    ui_text("history.raw_history_view.completed")
                                ).classes(
                                    UI_PRESENTATION[
                                        "render_skill_compaction.classes.font_mono_text_10px_text_emerald_400_bg_emera"
                                    ]
                                )
                            ui.label(
                                ui_text("history.raw_history_view.nested_history")
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.font_mono_text_11px_text_zinc_500"
                                ]
                            )

                # Expanded contents: Indented vertical container with color-coded guideline
                with exp:
                    with ui.column().classes(
                        UI_PRESENTATION[
                            "render_skill_compaction.classes_template"
                        ].format(depth_color_border=depth_color_border)
                    ):
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "render_skill_compaction.classes.items_center_justify_between_w_full_pb_1_bord"
                            ]
                        ):
                            ui.label(
                                ui_text(
                                    "history.raw_history_view.internal_history_of_excluded_from_main_history",
                                    skill_name=skill_name,
                                )
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.font_mono_text_11px_text_zinc_400_italic"
                                ]
                            )
                            # Copy sub-skill json button
                            sub_json = json.dumps(
                                sub_messages, indent=2, ensure_ascii=False
                            )
                            ui.button(
                                ui_text("history.raw_history_view.copy_sub_history"),
                                icon=UI_PRESENTATION[
                                    "render_skill_compaction.icon.content_copy"
                                ],
                                on_click=lambda _, text=sub_json, name=skill_name: (
                                    self._copy_text(
                                        text,
                                        app_text(
                                            "history.raw_history_view.history_of_name_copied",
                                            name=name,
                                        ),
                                    )
                                ),
                            ).props(
                                UI_PRESENTATION[
                                    "render_skill_compaction.props.flat_dense_no_caps_text_color_grey_4_size_xs"
                                ]
                            ).classes(
                                UI_PRESENTATION[
                                    "render_skill_compaction.classes.text_10px_px_2_py_0_5_bg_zinc_900_rounded_bor"
                                ]
                            )

                        # Recursive rendering: matryoshka style for sub-sub-skills!
                        self._render_message_sequence(
                            sub_messages, children, depth=depth
                        )

    def toggle_all(self, expand: bool) -> None:
        """Expands or collapses all sub-skill accordion sections."""
        for exp in self.expansions:
            self._skill_disclosures.set_expanded(self._expansion_keys[exp.id], expand)
            exp.value = expand

    # =========================================================================
    # Progressive Live Streaming API ("man mano")
    # =========================================================================

    def start_live_turn(self, session: ChatSession | None = None) -> None:
        """Prepares the Raw History view for progressive live streaming during an agent turn."""
        if not self.container or not self.content_container:
            return

        self.is_streaming = True
        self.live_response_text = ""
        self.live_reasoning_text = ""
        self.live_tool_events.clear()
        self._live_tool_cards.clear()
        self.live_skill_nodes.clear()
        self._last_token_render_time = 0.0

        if session:
            self.current_session = session
            # Ensure past messages up to current user prompt are rendered
            if not self.raw_json_data or (
                session.messages and len(self.raw_json_data) < len(session.messages)
            ):
                self.render_session(session)

        # Remove previous live container if present
        if self.live_container:
            try:
                self.live_container.delete()
            except Exception:
                pass
            self.live_container = None

        self.live_turn_index = len(self.raw_json_data) + 1

        # Append dedicated live streaming container
        with self.content_container:
            with ui.column().classes(
                UI_PRESENTATION[
                    "start_live_turn.classes.w_full_gap_3_live_stream_wrapper"
                ]
            ) as live_box:
                self.live_container = live_box

                # Container for live tools and skills that execute during this turn
                self.live_events_container = ui.column().classes(
                    UI_PRESENTATION["start_live_turn.classes.w_full_gap_3"]
                )

                # Live assistant response card
                with ui.column().classes(
                    UI_PRESENTATION[
                        "start_live_turn.classes.w_full_bg_111114_border_border_emerald_500_50"
                    ]
                ) as card:
                    self.live_assistant_card = card
                    with ui.row().classes(
                        UI_PRESENTATION[
                            "start_live_turn.classes.w_full_items_center_justify_between"
                        ]
                    ):
                        with ui.row().classes(
                            UI_PRESENTATION[
                                "start_live_turn.classes.items_center_gap_2"
                            ]
                        ):
                            ui.icon(
                                UI_PRESENTATION["start_live_turn.icon.smart_toy"],
                                size=UI_PRESENTATION["start_live_turn.size.xs"],
                            ).classes(
                                UI_PRESENTATION[
                                    "start_live_turn.classes.text_emerald_400"
                                ]
                            )
                            with ui.row().classes(
                                UI_PRESENTATION[
                                    "start_live_turn.classes.items_center_px_1_5_py_0_5_rounded_text_11px_"
                                ]
                            ):
                                ui.label(ui_text("history.raw_history_view.assistant"))
                            ui.label(
                                ui_text(
                                    "history.raw_history_view.message_formatted",
                                    self_live_turn_index=self.live_turn_index,
                                )
                            ).classes(
                                UI_PRESENTATION[
                                    "start_live_turn.classes.font_mono_text_11px_text_zinc_500"
                                ]
                            )
                            with ui.row().classes(
                                UI_PRESENTATION[
                                    "start_live_turn.classes.items_center_gap_1_5_px_2_py_0_5_rounded_text"
                                ]
                            ) as badge:
                                self.live_status_badge = badge
                                ui.spinner(
                                    UI_PRESENTATION["start_live_turn.spinner.dots"],
                                    size=UI_PRESENTATION["start_live_turn.size.xs"],
                                    color=UI_PRESENTATION[
                                        "start_live_turn.color.emerald_4"
                                    ],
                                )
                                ui.label(
                                    ui_text(
                                        "history.raw_history_view.streaming_in_progress"
                                    )
                                )

                        # Copy button for current partial live json
                        ui.button(
                            icon=UI_PRESENTATION["start_live_turn.icon.content_copy"],
                            on_click=lambda _: self._copy_text(
                                json.dumps(
                                    self._build_current_live_assistant_dict(),
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                app_text(
                                    "history.raw_history_view.partial_json_copied"
                                ),
                            ),
                        ).props(
                            UI_PRESENTATION[
                                "start_live_turn.props.flat_round_dense_size_xs"
                            ]
                        ).classes(
                            UI_PRESENTATION[
                                "start_live_turn.classes.text_zinc_600_hover_text_zinc_300"
                            ]
                        )

                    # Live syntax-highlighted pre/code element
                    initial_dict = self._build_current_live_assistant_dict()
                    initial_html = format_json_to_highlighted_html(initial_dict)
                    self.live_json_html = ui.html(
                        render_html(
                            "shared/raw_json.html",
                            content=Markup(initial_html),
                            extra_classes="",
                        )
                    ).classes(UI_PRESENTATION["start_live_turn.classes.w_full"])

        if self.status_label:
            self.status_label.text = ui_text(
                "history.raw_history_view.messages_streaming_in_progress",
                self_live_turn_index=self.live_turn_index,
            )

    def stream_reasoning(
        self,
        text: str,
        skill_path: str = "/",
        depth: int = 0,
        generation_index: int = 1,
    ) -> None:
        """Appends reasoning tokens and updates the live assistant JSON representation."""
        if not self.is_streaming:
            return

        self.live_reasoning_text += text
        self._throttled_update_live_json()

    def add_tool_event(self, ev: ToolEvent) -> None:
        """Progressively displays tool calls and tool responses in the live stream."""
        if not self.is_streaming or not self.live_events_container:
            return

        recorded = record_tool_event(self.live_tool_events, ev)
        event_summary = {
            "role": "event",
            "source": "live_tool_event",
            **recorded.model_dump(mode="json"),
        }
        card = self._live_tool_cards.get(recorded.id)
        if card is not None and card.summary == event_summary:
            return
        if card is None:
            self.raw_json_data.append(event_summary)
            with self.live_events_container:
                container = ui.column().classes(
                    UI_PRESENTATION["add_tool_event.classes.w_full_gap_0"]
                )
            card = _LiveToolCard(event_summary, container, len(self.raw_json_data))
            self._live_tool_cards[recorded.id] = card
        else:
            card.summary.clear()
            card.summary.update(event_summary)
            card.container.clear()
        with card.container:
            self._render_single_raw_message(
                card.summary, index=card.index, depth=recorded.depth
            )

        # Update assistant card index
        self.live_turn_index = len(self.raw_json_data) + 1
        if self.status_label:
            self.status_label.text = ui_text(
                "history.raw_history_view.messages_streaming_in_progress",
                self_live_turn_index=self.live_turn_index,
            )

    def handle_live_skill_event(self, node: SkillNode, action: str) -> None:
        """Progressively handles live skill opening/updating in the Raw History stream."""
        if not self.is_streaming or not self.live_events_container:
            return

        self.live_skill_nodes[node.path] = node

        if action == "opened":
            with self.live_events_container:
                skill_dict = _summarize_skill_tree_node(node)
                self.skill_tree_data.append(skill_dict)
                self._render_skill_compaction(
                    skill_dict, depth=node.depth, is_live=True
                )

    def stream_token(self, token: str) -> None:
        """Appends response tokens and triggers throttled syntax-highlighted updates."""
        if not self.is_streaming:
            return

        self.live_response_text += token
        self._throttled_update_live_json()

    def _throttled_update_live_json(self, force: bool = False) -> None:
        """Throttles DOM updates to ~80ms to avoid DOM thrashing / UI freezing while streaming."""
        if not self.is_streaming or not self.live_json_html:
            return

        now = time.monotonic()
        if not force and (
            now - self._last_token_render_time
            < EXECUTION_DEFAULTS["raw_token_update_interval_seconds"]
        ):
            return

        self._last_token_render_time = now
        msg_dict = self._build_current_live_assistant_dict()
        highlighted = format_json_to_highlighted_html(msg_dict)
        self.live_json_html.content = render_html(
            "shared/raw_json.html", content=Markup(highlighted), extra_classes=""
        )

    def _build_current_live_assistant_dict(self) -> dict[str, Any]:
        """Constructs a dict representing the assistant message currently in generation."""
        msg: dict[str, Any] = {"role": "assistant"}
        if self.live_reasoning_text:
            msg["reasoning"] = self.live_reasoning_text
        msg["content"] = self.live_response_text if self.live_response_text else None
        return msg

    def finalize_live_turn(self) -> None:
        """Finalizes the live stream, flushes pending tokens, and removes active spinners."""
        if not self.is_streaming:
            return

        self._throttled_update_live_json(force=True)
        if self.live_status_badge:
            self.live_status_badge.set_visibility(False)
        self.is_streaming = False

    def _copy_text(self, text: str, notification_msg: str) -> None:
        ui.clipboard.write(text)
        ui.notify(notification_msg, type="positive")

    def _handle_copy_full_json(self) -> None:
        if not self.raw_json_data:
            ui.notify(
                ui_text("history.raw_history_view.no_json_data_to_copy"), type="info"
            )
            return
        full_json = json.dumps(self.raw_json_data, indent=2, ensure_ascii=False)
        self._copy_text(
            full_json,
            app_text("history.raw_history_view.full_raw_history_copied_to_clipboard"),
        )
