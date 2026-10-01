"""Portable conversation exports, with optional nested execution transcripts."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from fe.qs_bridge.models import ChatMessage, ChatSession
from fe.ui.message_text import message_display_text

ExportFormat = Literal["json", "markdown"]


def _json_block(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    # A longer fence also handles literal Markdown fences inside tool outputs.
    fence = "`" * max(
        3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1
    )
    return f"{fence}json\n{text}\n{fence}"


def _markdown_messages(
    messages: list[ChatMessage], *, include_history: bool
) -> list[str]:
    sections = []
    for message in messages:
        for guidance in message.steering_messages:
            sections.append(
                "## User · Steering\n\n"
                + message_display_text(guidance.to_chat_message())
            )
        sections.append(f"## {message.role.title()}\n\n{message_display_text(message)}")
        data = message.export_data(include_history=include_history)
        if data["reasoning_blocks"]:
            sections.append(
                "### Reasoning\n\n"
                + "\n\n".join(block["content"] for block in data["reasoning_blocks"])
            )
        elif data["reasoning_content"]:
            sections.append("### Reasoning\n\n" + data["reasoning_content"])
        if data["tool_events"]:
            sections.append("### Tools\n\n" + _json_block(data["tool_events"]))
        if data["skills"]:
            sections.append("### Skills\n\n" + _json_block(data["skills"]))
        if message.error:
            sections.append(f"Status: {message.error}")
        if include_history and message.skill_history_tree:
            sections.append(
                "### Subskill history\n\n" + _json_block(message.skill_history_tree)
            )
    return sections


def export_chat(
    session: ChatSession, export_format: ExportFormat, *, include_history: bool
) -> str:
    """Export the whole root chat without persisting or changing session state."""
    if export_format == "json":
        return json.dumps(
            session.export_data(include_history=include_history),
            ensure_ascii=False,
            indent=2,
        )
    if export_format != "markdown":
        raise ValueError(f"Unsupported chat export format: {export_format}")

    sections = [f"# {session.title}"]
    if session.system_prompt:
        sections.append("## System\n\n" + session.system_prompt)
    sections.extend(
        _markdown_messages(session.message_history(), include_history=include_history)
    )
    if include_history:
        for child in session.sub_agent_conversations:
            sections.append(f"# Subskill: {child.path} · #{child.order}")
            sections.extend(
                _markdown_messages(child.message_history(), include_history=True)
            )
            if child.raw_history:
                sections.append("### Raw history\n\n" + _json_block(child.raw_history))
    return "\n\n".join(sections) + "\n"
