"""Presentation text shared by chat bubbles and conversation exports."""

from __future__ import annotations

from exmekh_agent.exmekh_bridge.history.pasted_input import strip_user_input_file_xml
from exmekh_agent.exmekh_bridge.models import ChatMessage


def message_display_text(message: ChatMessage) -> str:
    if message.role == "user" and message.display_content is None:
        return strip_user_input_file_xml(message.visible_text)
    return message.visible_text
