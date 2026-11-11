"""Read and restore persisted message attachments, including legacy records."""

import re
from dataclasses import dataclass
from pathlib import Path

from exmekh_agent.config.attachments import LEGACY_PDF_PREVIEW_PATTERN
from exmekh_agent.exmekh_bridge.history.image_input import IMAGE_SUFFIXES
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    StagedAttachment,
    extract_attached_files_from_content,
    strip_user_input_file_xml,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.models.messages import ImageDestination
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


def attached_files(message: ChatMessage) -> list[str]:
    """Return file links for attachments that have no separate image preview."""
    paths = dict.fromkeys(
        [
            *retained_attachment_paths(message),
            *extract_attached_files_from_content(message.content),
        ]
    )
    image_paths = message.image_attachment_paths()
    return [
        path
        for path in paths
        if path not in image_paths
        and not (
            message.image_urls
            and not message.image_paths
            and Path(path).suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
        )
    ]


def restore_staged_attachments(
    message: ChatMessage,
    paths: list[str],
    *,
    storage: TempStorage,
    image_destinations: dict[str, ImageDestination] | None = None,
) -> list[StagedAttachment]:
    destinations = {ref.path: ref.destination for ref in message.image_references}
    destinations.update(image_destinations or {})
    filenames = {item.path: item.filename for item in message.image_attachments}
    attachments = []
    for path in paths:
        attachment = StagedAttachment.from_saved_file(
            path, storage=storage, image_destination=destinations.get(path, "llm")
        )
        attachment.filename = filenames.get(path, attachment.filename)
        attachments.append(attachment)
    return attachments


@dataclass(frozen=True)
class SavedImage:
    url: str
    destination: ImageDestination | None
    path: str | None = None
    filename: str | None = None
    file_size: int | None = None


def saved_images(message: ChatMessage) -> list[SavedImage]:
    """Pair the persisted previews with their routes without reading staged files."""
    references = {ref.path: ref for ref in message.image_references}
    metadata = {item.path: item for item in message.image_attachments}
    paths_by_index = {
        item.vision_index: item.path
        for item in message.image_attachments
        if item.vision_index is not None
    }
    if not metadata:
        paths_by_index = _legacy_image_paths(message)

    def describe(url: str, path: str) -> SavedImage:
        item = metadata.get(path)
        ref = references.get(path)
        return SavedImage(
            url=url,
            destination=ref.destination if ref else "llm",
            path=path,
            filename=item.filename if item else Path(path).name,
            file_size=item.file_size if item else None,
        )

    return [
        describe(url, paths_by_index[index])
        if index in paths_by_index
        else SavedImage(url, "llm" if not message.image_paths else None)
        for index, url in enumerate(message.image_urls)
    ] + [
        describe(ref.image_only_preview, ref.path)
        for ref in message.image_references
        if ref.image_only_preview is not None
    ]


def _legacy_image_paths(message: ChatMessage) -> dict[int, str]:
    image_only_paths = {
        ref.path for ref in message.image_references if ref.destination == "image"
    }
    paths = message.image_paths or [
        path
        for path in retained_attachment_paths(message)
        if Path(path).suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
    ]
    pdf_preview_counts = iter(
        int(count) for count in re.findall(LEGACY_PDF_PREVIEW_PATTERN, message.content)
    )
    index = 0
    paths_by_index = {}
    for path in paths:
        if path in image_only_paths:
            continue
        if Path(path).suffix.lower() == ".pdf":
            count = next(pdf_preview_counts, None)
            if count is None:
                break
            index += count
        else:
            paths_by_index[index] = path
            index += 1
    return paths_by_index
