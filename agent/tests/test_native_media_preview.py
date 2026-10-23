"""Tool media use the existing assistant thumbnail path without a vision call."""

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from PIL import Image

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.history.converter import (
    exmekh_message_to_raw_dict,
    exmekh_messages_to_chat_messages,
    raw_dict_to_exmekh_message,
)
from exmekh_agent.exmekh_bridge.history.image_input import (
    serialize_user_content,
    tool_attachment_image_urls,
)


def test_generated_tool_media_are_preserved_as_assistant_previews():
    history = [
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="image-call",
                    function=FunctionCall(
                        name="generate-image", arguments={"prompt": "red"}
                    ),
                )
            ],
        ),
        ToolMessage(
            "<result>file-output</result>",
            "image-call",
            "generate-image",
            attachments=UserMessage(
                ImagesUserMessage(images=[Image.new("RGB", (32, 32), "red")])
            ),
        ),
        AssistantMessage(content="Image ready."),
    ]
    messages = exmekh_messages_to_chat_messages(history)
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].image_urls[0].startswith("data:image/png;base64,")
    assert messages[0].content == "Image ready."
    raw = [exmekh_message_to_raw_dict(item) for item in history]
    restored = raw_dict_to_exmekh_message(raw[1])
    assert restored.get_attachments() is not None
    collector = TurnStreamCollector()
    collector.set_raw_history(raw, [])
    assert collector.build_message().image_urls == messages[0].image_urls
    exported = MessageHistory(history).dump_history()
    collector.set_raw_history(exported, [])
    assert collector.build_message().image_urls == messages[0].image_urls
    restored = raw_dict_to_exmekh_message(exported[1])
    assert restored.get_attachments() is not None


def test_old_ui_attachments_and_library_snapshots_restore_the_same_pixels():
    attachment = UserMessage(ImagesUserMessage([Image.new("RGB", (4, 4), "red")]))
    old = {
        "role": "tool",
        "content": "ready",
        "name": "generate-image",
        "tool_call_id": "image-call",
        "attachments": serialize_user_content(attachment),
    }
    restored = raw_dict_to_exmekh_message(old)
    current = exmekh_message_to_raw_dict(restored)
    assert isinstance(current["attachments"], dict)
    assert tool_attachment_image_urls(old["attachments"]) == tool_attachment_image_urls(
        current["attachments"]
    )
    preview_parts = [None, {"type": "text", "text": "Caption"}, *old["attachments"]]
    assert tool_attachment_image_urls(preview_parts) == tool_attachment_image_urls(
        current["attachments"]
    )
    assert restored.get_attachments() is not None
