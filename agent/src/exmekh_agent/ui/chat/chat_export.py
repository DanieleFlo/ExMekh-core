"""Portable conversation exports, with optional nested execution transcripts."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from exmekh_agent.exmekh_bridge.models import ChatMessage, ChatSession
from exmekh_agent.ui.chat.message_text import message_display_text
from exmekh_agent.ui.resources import ui_text

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
                ui_text("chat.chat_export.steering_heading")
                + message_display_text(guidance.to_chat_message())
                + "".join(
                    ui_text("chat.chat_export.steering_image", url=url)
                    for url in guidance.image_urls
                )
            )
        sections.append(f"## {message.role.title()}\n\n{message_display_text(message)}")
        sections.extend(
            ui_text("chat.chat_export.attached_image", index=index + 1, url=url)
            for index, url in enumerate(message.image_urls)
        )
        data = message.export_data(include_history=include_history)
        if data["reasoning_blocks"]:
            sections.append(
                ui_text("chat.chat_export.reasoning_heading")
                + "\n\n".join(block["content"] for block in data["reasoning_blocks"])
            )
        elif data["reasoning_content"]:
            sections.append(
                ui_text("chat.chat_export.reasoning_heading")
                + data["reasoning_content"]
            )
        if data["tool_events"]:
            sections.append(
                ui_text("chat.chat_export.tools_heading")
                + _json_block(data["tool_events"])
            )
        if data["skills"]:
            sections.append(
                ui_text("chat.chat_export.skills_heading") + _json_block(data["skills"])
            )
        if message.error:
            sections.append(ui_text("chat.chat_export.status", error=message.error))
        if include_history and message.skill_history_tree:
            sections.append(
                ui_text("chat.chat_export.subskill_history_heading")
                + _json_block(message.skill_history_tree)
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
        raise ValueError(
            ui_text(
                "chat.chat_export.unsupported_chat_export_format", value_1=export_format
            )
        )

    sections = [f"# {session.title}"]
    if session.system_prompt:
        sections.append(
            ui_text("chat.chat_export.system_heading") + session.system_prompt
        )
    sections.extend(
        _markdown_messages(session.message_history(), include_history=include_history)
    )
    if include_history:
        for child in session.sub_agent_conversations:
            sections.append(
                ui_text(
                    "chat.chat_export.subskill_heading",
                    path=child.path,
                    order=child.order,
                )
            )
            sections.extend(
                _markdown_messages(child.message_history(), include_history=True)
            )
            if child.raw_history:
                sections.append(
                    ui_text("chat.chat_export.raw_history_heading")
                    + _json_block(child.raw_history)
                )
    return "\n\n".join(sections) + "\n"
