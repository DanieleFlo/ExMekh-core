"""Prepare edited user text while retaining the message's explicit attachments."""

from pathlib import Path

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.image_input import IMAGE_SUFFIXES
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    PreparedUserInput,
    StagedAttachment,
    extract_attached_files_from_content,
    format_user_input_file_xml,
    prepare_user_input,
    strip_user_input_file_xml,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.state.temp_storage import TempStorage


def retained_attachment_paths(message: ChatMessage) -> list[str]:
    if message.attachment_paths is not None:
        return list(message.attachment_paths)
    # Compatibility for chats saved before explicit upload paths were recorded.
    # Automatically staged pasted text must be recreated from the edited input.
    plain_content = strip_user_input_file_xml(message.content)
    explicit_uploads = message.display_content is None or plain_content.startswith(
        message.display_content
    )
    return (
        extract_attached_files_from_content(message.content) if explicit_uploads else []
    )


def prepare_message_edit(
    message: ChatMessage,
    text: str,
    *,
    storage: TempStorage,
    use_file_references: bool,
    attachment_paths: list[str] | None = None,
    attachments: list[StagedAttachment] | None = None,
) -> PreparedUserInput:
    if (
        attachment_paths is not None
        and attachment_paths != retained_attachment_paths(message)
    ) or attachments:
        allowed = retained_attachment_paths(message)
        selected_paths = allowed if attachment_paths is None else attachment_paths
        if any(path not in allowed for path in selected_paths):
            raise ValueError(
                app_text(
                    "errors_history.message_edit.an_edited_message_can_only_retain_its_own_attachments"
                )
            )
        retained = [
            StagedAttachment.from_saved_file(path, storage=storage)
            for path in selected_paths
        ]
        return prepare_user_input(
            text,
            storage=storage,
            attachments=[*retained, *(attachments or [])],
            use_file_references=use_file_references,
        )
    paths = retained_attachment_paths(message)
    vision_paths = message.image_paths or [
        path
        for path in paths
        if message.image_urls and Path(path).suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
    ]
    inline_attachments = (
        [
            StagedAttachment.from_saved_file(path, storage=storage)
            for path in paths
            if path not in vision_paths
        ]
        if not use_file_references
        else None
    )
    prepared = prepare_user_input(
        text,
        storage=storage,
        attachments=inline_attachments,
        use_file_references=use_file_references,
    )
    references = [
        format_user_input_file_xml(path)
        for path in paths
        if use_file_references or path in vision_paths
    ]
    # Preserve the PDF page-count notice along with its saved image payloads.
    notices = [
        line
        for line in message.content.splitlines()
        if paths and line.startswith("PDF ") and "are attached as images" in line
    ]
    # Repeated PDF pages can have identical payloads: retain their order and count.
    images = [*message.image_urls, *prepared.image_urls]
    prompt = "\n\n".join(
        part for part in [prepared.prompt, *references, *notices] if part
    )
    return PreparedUserInput(
        prompt=prompt,
        display_content=text if paths or images else prepared.display_content,
        title_text=text or app_text("execution.pasted_input.attached_file"),
        image_urls=images,
        image_paths=list(dict.fromkeys([*vision_paths, *prepared.image_paths])),
        attachment_paths=list(dict.fromkeys([*paths, *prepared.attachment_paths])),
    )
