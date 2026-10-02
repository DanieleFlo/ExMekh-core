from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass
from typing import Any

from nicegui import ui

import fe.exmekh_bridge.context_usage as context_usage
from fe.exmekh_bridge.models import (
    ChatSession,
    SkillNode,
    ToolEvent,
    record_tool_event,
)
from fe.exmekh_bridge.skill_history import take_skill_for_call
from fe.exmekh_bridge.system_prompt import load_real_hds_system_prompt
from fe.exmekh_bridge.tool_calls import parse_tool_calls


@dataclass
class _LiveToolCard:
    summary: dict[str, Any]
    container: ui.column
    index: int


def format_json_to_highlighted_html(data: Any, indent_level: int = 0) -> str:
    """Recursively formats a Python/JSON data structure into syntax-highlighted HTML.

    Applies clear visual hierarchy:
    - Keys: soft sky blue (#93c5fd)
    - Tool call keys ("tool_calls", "function", "tool_call_id", "arguments"): vibrant amber (#fbbf24)
    - String values: emerald green (#86efac)
    - Number values: magenta/pink (#f472b6)
    - Boolean values: bright yellow (#fde047)
    - Null values: coral red (#f87171)
    - Structural brackets/commas: muted zinc (#71717a)
    """
    indent_space = "  " * indent_level
    inner_indent = "  " * (indent_level + 1)

    if data is None:
        return '<span class="json-val-null">null</span>'
    elif isinstance(data, bool):
        val_str = "true" if data else "false"
        return f'<span class="json-val-bool">{val_str}</span>'
    elif isinstance(data, (int, float)):
        return f'<span class="json-val-num">{data}</span>'
    elif isinstance(data, str):
        # Try to detect if the string itself is a JSON string (e.g. tool_calls dumped string or JSON arguments)
        trimmed = data.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            try:
                parsed = json.loads(trimmed)
                return format_json_to_highlighted_html(
                    parsed, indent_level=indent_level
                )
            except Exception:
                pass
        escaped_str = html.escape(json.dumps(data, ensure_ascii=False))
        return f'<span class="json-val-str">{escaped_str}</span>'

    elif isinstance(data, list):
        if not data:
            return '<span class="json-punct">[]</span>'
        items_html = []
        for item in data:
            formatted_item = format_json_to_highlighted_html(
                item, indent_level=indent_level + 1
            )
            items_html.append(f"{inner_indent}{formatted_item}")
        joined = ",\n".join(items_html)
        return f'<span class="json-punct">[</span>\n{joined}\n{indent_space}<span class="json-punct">]</span>'

    elif isinstance(data, dict):
        if not data:
            return '<span class="json-punct">{}</span>'
        pairs_html = []
        for key, value in data.items():
            key_str = str(key)
            is_tool_key = key_str in (
                "tool_calls",
                "function",
                "tool_call_id",
                "arguments",
            )

            # Determine key styling and potential badge
            if is_tool_key:
                key_class = "json-key json-key-tool"
                badge = (
                    ' <span class="json-tool-badge">TOOL</span>'
                    if key_str == "tool_calls"
                    else ""
                )
            elif key_str == "role":
                key_class = "json-key json-key-role"
                badge = ""
            else:
                key_class = "json-key"
                badge = ""

            escaped_key = html.escape(json.dumps(key_str, ensure_ascii=False))

            # Role value badge enhancement
            if key_str == "role" and isinstance(value, str):
                role_val = str(value).lower()
                role_class = f"role-badge role-badge-{role_val}"
                val_formatted = f'<span class="{role_class}">{html.escape(role_val.upper())}</span> <span class="json-val-str">"{html.escape(value)}"</span>'
            else:
                val_formatted = format_json_to_highlighted_html(
                    value, indent_level=indent_level + 1
                )

            pairs_html.append(
                f'{inner_indent}<span class="{key_class}">{escaped_key}</span>{badge}<span class="json-punct">: </span>{val_formatted}'
            )

        joined = ",\n".join(pairs_html)
        return f'<span class="json-punct">{{</span>\n{joined}\n{indent_space}<span class="json-punct">}}</span>'

    else:
        return f'<span class="json-val-str">"{html.escape(str(data))}"</span>'


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
            default_sys_prompt = (
                "You are a friendly, accurate, and professional AI assistant."
            )

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
        self.expanded_skill_names: set[str] = set()

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
            "w-full h-full flex-grow px-4 py-4 gap-4 overflow-y-auto raw-history-container"
        ) as col:
            self.container = col
            self._render_toolbar()
            with ui.column().classes("w-full gap-3") as content:
                self.content_container = content

        return col

    def _render_toolbar(self) -> None:
        with ui.row().classes(
            "w-full items-center justify-between pb-3 border-b border-zinc-800/80"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("data_object", size="sm").classes("text-zinc-400")
                ui.label("Raw History (JSON Inspector)").classes(
                    "text-sm font-semibold tracking-tight text-zinc-200"
                )
                self.status_label = ui.label("0 messages").classes(
                    "font-mono text-xs text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800"
                )

            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Expand all",
                    icon="unfold_more",
                    on_click=lambda: self.toggle_all(True),
                ).props("flat dense no-caps text-color=grey-4").classes(
                    "text-xs px-2.5 py-1 bg-zinc-900/60 hover:bg-zinc-800 rounded border border-zinc-800/80"
                )

                ui.button(
                    "Collapse all",
                    icon="unfold_less",
                    on_click=lambda: self.toggle_all(False),
                ).props("flat dense no-caps text-color=grey-4").classes(
                    "text-xs px-2.5 py-1 bg-zinc-900/60 hover:bg-zinc-800 rounded border border-zinc-800/80"
                )

                ui.button(
                    "Copy full JSON",
                    icon="content_copy",
                    on_click=self._handle_copy_full_json,
                ).props("flat dense no-caps text-color=white").classes(
                    "text-xs px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded border border-zinc-700"
                )

    def render_session(self, session: ChatSession) -> None:
        if not self.container or not self.content_container:
            return

        self.current_session = session
        self.expansions.clear()
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
            self.status_label.text = f"{count_msg} messages • {count_skills} sub-skills"

        with self.content_container:
            if not raw_messages:
                with ui.column().classes(
                    "w-full py-16 items-center justify-center text-center gap-2"
                ):
                    ui.icon("terminal", size="md").classes("text-zinc-600")
                    ui.label("No history available").classes(
                        "text-sm text-zinc-400 font-mono"
                    )
                    ui.label(
                        "Send a message in chat to generate raw message and tool history."
                    ).classes("text-xs text-zinc-600 max-w-sm")
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
            border_cls = "border-sky-800/40 hover:border-sky-700/60"
            role_badge_cls = "bg-sky-950/80 text-sky-400 border-sky-800/60"
            icon_name = "person"
        elif role == "tool":
            border_cls = "border-amber-700/50 hover:border-amber-600/70"
            role_badge_cls = "bg-amber-950/80 text-amber-400 border-amber-700/60"
            icon_name = "build"
        elif role == "system":
            border_cls = "border-zinc-800/60 hover:border-zinc-700/60"
            role_badge_cls = "bg-zinc-900 text-zinc-400 border-zinc-700"
            icon_name = "settings"
        else:
            border_cls = "border-emerald-800/40 hover:border-emerald-700/60"
            role_badge_cls = "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
            icon_name = "smart_toy"

        with ui.column().classes(
            f"w-full bg-[#111114] border {border_cls} rounded-lg p-3 gap-2 transition-colors duration-150"
        ):
            # Header row with role, index, and copy button
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon_name, size="xs").classes("text-zinc-400")
                    with ui.row().classes(
                        f"items-center px-1.5 py-0.5 rounded text-[11px] font-mono border {role_badge_cls}"
                    ):
                        ui.label(role.upper())
                    ui.label(f"Message #{index}").classes(
                        "font-mono text-[11px] text-zinc-500"
                    )

                    # If this message is a tool response or contains tool calls, add helper pills
                    if role == "tool" and msg.get("name"):
                        ui.label(f"tool: {msg['name']}").classes(
                            "font-mono text-[11px] text-amber-400/90 bg-amber-950/50 px-1.5 py-0.2 rounded border border-amber-800/40"
                        )
                    elif msg.get("tool_calls"):
                        ui.label("TOOL CALL").classes(
                            "font-mono text-[10px] text-amber-300 bg-amber-900/60 px-1.5 py-0.2 rounded font-bold tracking-wider"
                        )
                    if msg.get("source") == "session_summary":
                        ui.label("SUMMARY · raw trace not available").classes(
                            "font-mono text-[10px] text-zinc-400"
                        )

                # Copy message button
                msg_json = json.dumps(msg, indent=2, ensure_ascii=False)
                ui.button(
                    icon="content_copy",
                    on_click=lambda _, text=msg_json: self._copy_text(
                        text, "Message copied"
                    ),
                ).props("flat round dense size=xs").classes(
                    "text-zinc-600 hover:text-zinc-300"
                )

            # Formatted and syntax highlighted JSON code block
            highlighted_html = format_json_to_highlighted_html(msg)
            ui.html(
                f'<pre class="raw-json-block font-mono text-xs leading-relaxed overflow-x-auto m-0 p-2 bg-[#0c0c0e] rounded border border-zinc-900"><code>{highlighted_html}</code></pre>'
            ).classes("w-full")

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
            depth_color_border = "border-emerald-500/50"
            badge_color = "bg-emerald-950/80 text-emerald-300 border-emerald-700/60"
            depth_label = "Depth 1"
        elif depth == 2:
            depth_color_border = "border-purple-500/50"
            badge_color = "bg-purple-950/80 text-purple-300 border-purple-700/60"
            depth_label = "Depth 2 (Matryoshka)"
        else:
            depth_color_border = "border-amber-500/50"
            badge_color = "bg-amber-950/80 text-amber-300 border-amber-700/60"
            depth_label = f"Depth {depth} (Matryoshka)"

        # Vertical compaction wrapper
        with ui.column().classes("w-full my-1"):
            with ui.expansion("").classes(
                "w-full subskill-compaction rounded-lg border border-zinc-800 bg-[#0e0e11] text-xs"
            ) as exp:
                # Restore previous expansion state or default to collapsed
                is_expanded = skill_name in self.expanded_skill_names
                exp.value = is_expanded
                self.expansions.append(exp)

                def _on_val_change(e: Any, name=skill_name):
                    val = getattr(e, "value", e)
                    if val:
                        self.expanded_skill_names.add(name)
                    else:
                        self.expanded_skill_names.discard(name)

                exp.on_value_change(_on_val_change)

                # Custom compact header
                with exp.add_slot("header"):
                    with ui.row().classes(
                        "items-center justify-between w-full py-1.5 pr-2 cursor-pointer"
                    ):
                        with ui.row().classes("items-center gap-2.5"):
                            ui.icon("hub", size="xs").classes("text-zinc-400")
                            ui.label(f"Sub-Skill: {skill_name}").classes(
                                "font-mono font-medium text-zinc-200 text-xs"
                            )
                            with ui.row().classes(
                                f"items-center px-1.5 py-0.5 rounded text-[10px] font-mono border {badge_color}"
                            ):
                                ui.label(depth_label)
                            ui.label(f"{msg_count} messages in sub-skill").classes(
                                "font-mono text-[11px] text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800"
                            )

                        with ui.row().classes("items-center gap-2"):
                            if is_live:
                                with ui.row().classes(
                                    "items-center gap-1.5 px-1.5 py-0.5 rounded text-[10px] font-mono border border-emerald-500/40 bg-emerald-950/40 text-emerald-300 animate-pulse"
                                ):
                                    ui.spinner("dots", size="xs", color="emerald-4")
                                    ui.label("RUNNING")
                            elif skill_data.get("tokens_pending"):
                                ui.label("COMPLETED").classes(
                                    "font-mono text-[10px] text-emerald-400"
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
                                        "raw-skill-closure-tokens items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/40 text-[10px] font-mono text-emerald-300"
                                    )
                                    .props(
                                        f'title="Context tokens before closing: {before_tok:,} • after closing: {after_tok:,}"'
                                    )
                                ):
                                    ui.icon("compress", size="12px").classes(
                                        "text-emerald-400"
                                    )
                                    ui.label(
                                        context_usage.format_skill_closure_tokens(
                                            before_tok, after_tok
                                        )
                                    )
                                ui.label("COMPLETED").classes(
                                    "font-mono text-[10px] text-emerald-400 bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-800/40"
                                )
                            ui.label("Nested History").classes(
                                "font-mono text-[11px] text-zinc-500"
                            )

                # Expanded contents: Indented vertical container with color-coded guideline
                with exp:
                    with ui.column().classes(
                        f"w-full ml-3 pl-3.5 my-2 border-l-2 {depth_color_border} gap-3"
                    ):
                        with ui.row().classes(
                            "items-center justify-between w-full pb-1 border-b border-zinc-800/60"
                        ):
                            ui.label(
                                f"Internal history of '{skill_name}' (excluded from main history):"
                            ).classes("font-mono text-[11px] text-zinc-400 italic")
                            # Copy sub-skill json button
                            sub_json = json.dumps(
                                sub_messages, indent=2, ensure_ascii=False
                            )
                            ui.button(
                                "Copy Sub-History",
                                icon="content_copy",
                                on_click=lambda _, text=sub_json, name=skill_name: (
                                    self._copy_text(text, f"History of {name} copied")
                                ),
                            ).props(
                                "flat dense no-caps text-color=grey-4 size=xs"
                            ).classes(
                                "text-[10px] px-2 py-0.5 bg-zinc-900 rounded border border-zinc-800"
                            )

                        # Recursive rendering: matryoshka style for sub-sub-skills!
                        self._render_message_sequence(
                            sub_messages, children, depth=depth
                        )

    def toggle_all(self, expand: bool) -> None:
        """Expands or collapses all sub-skill accordion sections."""
        for exp in self.expansions:
            exp.value = expand
        if expand:
            for sk in self.skill_tree_data:
                name = sk.get("skill")
                if name:
                    self.expanded_skill_names.add(name)
        else:
            self.expanded_skill_names.clear()

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
            with ui.column().classes("w-full gap-3 live-stream-wrapper") as live_box:
                self.live_container = live_box

                # Container for live tools and skills that execute during this turn
                self.live_events_container = ui.column().classes("w-full gap-3")

                # Live assistant response card
                with ui.column().classes(
                    "w-full bg-[#111114] border border-emerald-500/50 rounded-lg p-3 gap-2 transition-all duration-150 live-assistant-card shadow-lg shadow-emerald-950/10"
                ) as card:
                    self.live_assistant_card = card
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("smart_toy", size="xs").classes("text-emerald-400")
                            with ui.row().classes(
                                "items-center px-1.5 py-0.5 rounded text-[11px] font-mono border bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                            ):
                                ui.label("ASSISTANT")
                            ui.label(f"Message #{self.live_turn_index}").classes(
                                "font-mono text-[11px] text-zinc-500"
                            )
                            with ui.row().classes(
                                "items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono border border-emerald-500/40 bg-emerald-950/40 text-emerald-300 animate-pulse"
                            ) as badge:
                                self.live_status_badge = badge
                                ui.spinner("dots", size="xs", color="emerald-4")
                                ui.label("STREAMING IN PROGRESS...")

                        # Copy button for current partial live json
                        ui.button(
                            icon="content_copy",
                            on_click=lambda _: self._copy_text(
                                json.dumps(
                                    self._build_current_live_assistant_dict(),
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                "Partial JSON copied",
                            ),
                        ).props("flat round dense size=xs").classes(
                            "text-zinc-600 hover:text-zinc-300"
                        )

                    # Live syntax-highlighted pre/code element
                    initial_dict = self._build_current_live_assistant_dict()
                    initial_html = format_json_to_highlighted_html(initial_dict)
                    self.live_json_html = ui.html(
                        f'<pre class="raw-json-block font-mono text-xs leading-relaxed overflow-x-auto m-0 p-2 bg-[#0c0c0e] rounded border border-zinc-900"><code>{initial_html}</code></pre>'
                    ).classes("w-full")

        if self.status_label:
            self.status_label.text = (
                f"{self.live_turn_index} messages • streaming in progress..."
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
                container = ui.column().classes("w-full gap-0")
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
            self.status_label.text = (
                f"{self.live_turn_index} messages • streaming in progress..."
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
        if not force and (now - self._last_token_render_time < 0.08):
            return

        self._last_token_render_time = now
        msg_dict = self._build_current_live_assistant_dict()
        highlighted = format_json_to_highlighted_html(msg_dict)
        self.live_json_html.content = f'<pre class="raw-json-block font-mono text-xs leading-relaxed overflow-x-auto m-0 p-2 bg-[#0c0c0e] rounded border border-zinc-900"><code>{highlighted}</code></pre>'

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
            ui.notify("No JSON data to copy", type="info")
            return
        full_json = json.dumps(self.raw_json_data, indent=2, ensure_ascii=False)
        self._copy_text(full_json, "Full Raw History copied to clipboard")
