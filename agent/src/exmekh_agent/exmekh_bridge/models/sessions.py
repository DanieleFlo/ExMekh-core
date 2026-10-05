"""Conversation branches, message queues and owned child conversations."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS
from exmekh_agent.exmekh_bridge.models.messages import (
    ChatMessage,
    QueuedMessage,
    SubAgentConversation,
)
from exmekh_agent.exmekh_bridge.models.settings import ChatSettings


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = app_text("defaults.conversation_title")
    project_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: list[ChatMessage] = Field(default_factory=list)
    sub_agent_conversations: list[SubAgentConversation] = Field(default_factory=list)
    queued_messages: list[QueuedMessage] = Field(default_factory=list)
    model: str = PROVIDER_DEFAULTS["default_model"]
    agent_mode: Literal["deep_agent", "quick_chat"] = AGENT_DEFAULTS["agent_mode"]
    system_prompt: str = ""
    settings: ChatSettings = Field(default_factory=ChatSettings)

    def export_data(self, *, include_history: bool) -> dict[str, Any]:
        """Export conversation content without execution settings or pending input."""
        data = {
            "id": self.id,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "messages": [
                message.export_data(include_history=include_history)
                for message in self.messages
            ],
        }
        if include_history:
            data["sub_agent_conversations"] = [
                child.model_dump(mode="json") for child in self.sub_agent_conversations
            ]
        return data

    @property
    def is_empty(self) -> bool:
        return not (
            self.messages or self.queued_messages or self.sub_agent_conversations
        )

    @property
    def has_queued_messages(self) -> bool:
        return bool(self.queued_messages)

    def message_history(self) -> list[ChatMessage]:
        """Return the conversation without sharing its mutable message list."""
        return list(self.messages)

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def find_user_message(self, message_id: str) -> ChatMessage | None:
        for message in self.messages:
            found = message.find_user_message(message_id)
            if found is not None:
                return found
        return None

    def revised_user_message(
        self, message_id: str, replacement: ChatMessage
    ) -> ChatSession | None:
        """Build an independent conversation branch without modifying this session."""
        if replacement.role != "user":
            return None
        revised = self.model_copy(deep=True)
        boundary = revised._revision_boundary(message_id)
        if boundary is None:
            return None
        prefix, original, cutoff = boundary
        revised.messages = prefix + [
            replacement.model_copy(
                deep=True,
                update={
                    "id": original.id,
                    "timestamp": original.timestamp,
                },
            )
        ]
        revised.sub_agent_conversations = [
            child
            for child in revised.sub_agent_conversations
            if child.discard_calls_from(cutoff)
        ]
        revised.queued_messages = []
        revised.settings.clear_command_approvals()
        return revised

    def _revision_boundary(
        self, message_id: str
    ) -> tuple[list[ChatMessage], ChatMessage, float] | None:
        for index, message in enumerate(self.messages):
            if message.role == "user" and message.id == message_id:
                return self.messages[:index], message, message.timestamp
            earlier_guidance = []
            for guidance in message.steering_messages:
                if guidance.id == message_id:
                    # Guidance bubbles precede their owning response. Restart
                    # that response at its initiating input, including child calls.
                    cutoff = next(
                        (
                            item.timestamp
                            for item in reversed(self.messages[:index])
                            if item.role == "user"
                        ),
                        message.timestamp,
                    )
                    return (
                        self.messages[:index] + earlier_guidance,
                        guidance.to_chat_message(),
                        cutoff,
                    )
                earlier_guidance.append(guidance.to_chat_message())
        return None

    def apply_revision(self, revised: ChatSession) -> None:
        """Publish a persisted branch while retaining this managed instance."""
        if revised.id != self.id:
            raise ValueError(
                app_text(
                    "errors_exmekh_bridge.models.revision_belongs_to_another_conversation"
                )
            )
        self.messages = revised.messages
        self.sub_agent_conversations = revised.sub_agent_conversations
        self.queued_messages = revised.queued_messages
        self.settings = revised.settings
        self.updated_at = revised.updated_at

    def find_sub_agent_conversation(
        self, conversation_id: str | None
    ) -> SubAgentConversation | None:
        return next(
            (
                conversation
                for conversation in self.sub_agent_conversations
                if conversation.id == conversation_id
            ),
            None,
        )

    def start_sub_agent_conversation(
        self,
        conversation_id: str,
        path: str,
        persistent_history: bool,
        user_message: str,
    ) -> SubAgentConversation:
        conversation = self.find_sub_agent_conversation(conversation_id)
        if conversation is None:
            order = 1 + sum(item.path == path for item in self.sub_agent_conversations)
            conversation = SubAgentConversation(
                id=conversation_id,
                path=path,
                order=order,
                persistent_history=persistent_history,
            )
            self.sub_agent_conversations.append(conversation)
        elif (
            conversation.path != path
            or conversation.persistent_history != persistent_history
        ):
            raise ValueError(
                app_text(
                    "errors_exmekh_bridge.models.sub_agent_conversation_changed_identity",
                    value_1=conversation_id,
                )
            )
        conversation.start_call(user_message)
        return conversation

    def cleanup_active_sub_agents(self) -> None:
        """Close any unclosed sub-agent conversations after interruption or error."""
        for conversation in self.sub_agent_conversations:
            conversation.interrupt_call()

    def add_queued_message(
        self,
        text: str,
        display_content: str | None = None,
        *,
        image_urls: list[str] | None = None,
        image_paths: list[str] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> QueuedMessage:
        msg = QueuedMessage(
            text=text,
            display_content=display_content,
            image_urls=image_urls or [],
            image_paths=image_paths or [],
            attachment_paths=attachment_paths,
        )
        self.queued_messages.append(msg)
        return msg

    def pop_queued_message(self) -> QueuedMessage | None:
        if self.queued_messages:
            return self.queued_messages.pop(0)
        return None

    def find_queued_message(self, message_id: str) -> QueuedMessage | None:
        return next(
            (message for message in self.queued_messages if message.id == message_id),
            None,
        )

    def remove_queued_message(self, message_id: str) -> QueuedMessage | None:
        for i, m in enumerate(self.queued_messages):
            if m.id == message_id:
                return self.queued_messages.pop(i)
        return None

    def reorder_queued_message(
        self, source_id: str, target_id: str, position: str = "before"
    ) -> bool:
        if source_id == target_id or position not in ("before", "after"):
            return False
        src_idx = next(
            (i for i, m in enumerate(self.queued_messages) if m.id == source_id),
            None,
        )
        if src_idx is None:
            return False
        item = self.queued_messages.pop(src_idx)
        target_idx = next(
            (i for i, m in enumerate(self.queued_messages) if m.id == target_id),
            None,
        )
        if target_idx is None:
            self.queued_messages.insert(src_idx, item)
            return False
        insert_idx = target_idx if position == "before" else target_idx + 1
        self.queued_messages.insert(insert_idx, item)
        return True
