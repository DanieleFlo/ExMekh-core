from __future__ import annotations

import base64
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exmekh_agent.config.attachments import (
    PASTE_TOKEN_THRESHOLD as PASTE_TOKEN_THRESHOLD,
)
from exmekh_agent.config.attachments import XML_TAG_NAME as XML_TAG_NAME
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.image_input import attachment_images
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.providers.context_usage import count_text_tokens
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.utils.filenames import clean_attachment_filename


@dataclass
class StagedAttachment:
    """An input file staged from user paste or large input to be sent as a referenced file."""

    id: str
    content: str
    preview: str
    token_count: int
    file_path: Path | None = None
    relative_path: str | None = None
    filename: str = ""
    is_binary: bool = False
    file_size: int = 0
    data: bytes | None = field(default=None, repr=False)
    image_urls: list[str] = field(default_factory=list)
    pdf_pages: int = 0

    def reference(
        self,
        work_root: Path | str | None = None,
        temp_dirname: str = "tmp",
        *,
        storage: TempStorage | None = None,
    ) -> str:
        """Return a prompt reference, recreating a file moved out of this work root."""
        storage = storage or TempStorage(work_root, temp_dirname)
        if (
            self.file_path is None
            or self.relative_path is None
            or not self.file_path.exists()
            or not storage.contains(self.file_path)
        ):
            data = self.data
            if self.file_path is not None and self.file_path.exists():
                data = self.file_path.read_bytes()
            if data is None:
                if self.is_binary:
                    raise FileNotFoundError(
                        app_text(
                            "errors_history.pasted_input.attachment_is_no_longer_available",
                            value_1=self.filename,
                        )
                    )
                data = self.content.encode("utf-8")
            self.file_path, self.relative_path = storage.write_file(
                self.filename or "user_input.md", data
            )
            if not self.filename:
                self.filename = self.file_path.name
        return format_user_input_file_xml(self.relative_path)

    def upload_payload(self) -> dict[str, Any]:
        """Serialize the descriptor without sending binary data to the browser."""
        return {
            "id": self.id,
            "content": self.content,
            "preview": self.preview,
            "token_count": self.token_count,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "is_binary": self.is_binary,
            "file_size": self.file_size,
            "image_urls": self.image_urls,
            "pdf_pages": self.pdf_pages,
        }

    @classmethod
    def from_upload(
        cls,
        payload: Mapping[str, Any],
        work_root: Path | str | None = None,
        temp_dirname: str = "tmp",
        *,
        storage: TempStorage | None = None,
    ) -> StagedAttachment:
        """Stage inline data or restore a descriptor within this work root."""
        storage = storage or TempStorage(work_root, temp_dirname)
        if "data" in payload:
            data = payload["data"]
            if isinstance(data, str):
                data = base64.b64decode(data.split(",", 1)[-1], validate=True)
            if not isinstance(data, bytes):
                raise ValueError(
                    app_text(
                        "errors_history.pasted_input.uploaded_file_data_must_contain_bytes_or_base64_text"
                    )
                )
            return create_staged_attachment_from_file(
                payload.get("filename", payload.get("name", "uploaded_file")),
                data,
                work_root=work_root,
                storage=storage,
            )
        relative_path = str(payload.get("relative_path", ""))
        if not relative_path:
            raise ValueError(
                app_text(
                    "errors_history.pasted_input.uploaded_file_descriptor_has_no_relative_path"
                )
            )
        file_path = storage.resolve_attachment(relative_path)
        return cls(
            id=str(payload.get("id") or uuid.uuid4().hex[:6]),
            content=str(payload.get("content", "")),
            preview=str(payload.get("preview", payload.get("filename", "file"))),
            token_count=int(payload.get("token_count", 0)),
            file_path=file_path,
            relative_path=relative_path,
            filename=str(payload.get("filename", "file")),
            is_binary=bool(payload.get("is_binary", False)),
            file_size=int(payload.get("file_size", 0)),
            image_urls=list(payload.get("image_urls", [])),
            pdf_pages=int(payload.get("pdf_pages", 0)),
        )

    @classmethod
    def from_saved_file(
        cls, relative_path: str, *, storage: TempStorage
    ) -> StagedAttachment:
        """Restore an explicit upload using the storage's scoped public API."""
        file_path = storage.resolve_attachment(relative_path, allow_legacy=True)
        return cls.from_file_data(
            file_path.name, file_path.read_bytes(), file_path, relative_path
        )

    @classmethod
    def from_file_data(
        cls,
        filename: str,
        data: bytes,
        file_path: Path | None = None,
        relative_path: str | None = None,
    ) -> StagedAttachment:
        """Validate and describe file data before any staging write."""
        image_urls, pdf_pages = attachment_images(filename, data)
        clean_name = clean_attachment_filename(filename)
        is_binary = bool(image_urls) or b"\x00" in data
        text_content = ""
        if not is_binary:
            try:
                text_content = data.decode("utf-8")
            except UnicodeDecodeError:
                is_binary = True
        return cls(
            id=uuid.uuid4().hex[:6],
            content=text_content,
            preview=clean_name,
            token_count=0 if is_binary else count_text_tokens(text_content),
            file_path=file_path,
            relative_path=relative_path,
            filename=clean_name,
            is_binary=is_binary,
            file_size=len(data),
            data=data,
            image_urls=image_urls,
            pdf_pages=pdf_pages,
        )

    def delete_file(self) -> None:
        """Remove the staged file when the user discards or inlines it."""
        if self.file_path is not None:
            try:
                self.file_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.file_path = None
        self.relative_path = None


@dataclass(frozen=True)
class PreparedUserInput:
    prompt: str
    display_content: str | None
    title_text: str
    image_urls: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    attachment_paths: list[str] = field(default_factory=list)

    def to_chat_message(self) -> ChatMessage:
        """Use one mapping for new input and edited input."""
        return ChatMessage(
            role="user",
            content=self.prompt,
            display_content=self.display_content,
            image_urls=list(self.image_urls),
            image_paths=list(self.image_paths),
            attachment_paths=list(self.attachment_paths),
        )


def extract_content_preview(content: str, max_chars: int = 25) -> str:
    """Extract a short single-line preview of the content, suitable for a compact chip icon/box.

    Shows just the first meaningful line (or part of it that fits in the small icon box).
    """
    lines = content.splitlines()
    first_meaningful_line = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        first_meaningful_line = stripped
        break
    if not first_meaningful_line:
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_meaningful_line = stripped
                break
    if not first_meaningful_line:
        first_meaningful_line = "pasted_text"
    if len(first_meaningful_line) > max_chars:
        return first_meaningful_line[:max_chars].rstrip() + "..."
    return first_meaningful_line


def create_staged_attachment(
    content: str,
    work_root: Path | str | None = None,
    temp_dirname: str = "tmp",
    *,
    storage: TempStorage | None = None,
) -> StagedAttachment:
    """Create a temporary markdown file and return a StagedAttachment descriptor."""
    tokens = count_text_tokens(content)
    preview = extract_content_preview(content)
    file_path, rel_path = create_input_markdown_file(
        content, work_root=work_root, temp_dirname=temp_dirname, storage=storage
    )
    uid = uuid.uuid4().hex[:6]
    return StagedAttachment(
        id=uid,
        content=content,
        preview=preview,
        token_count=tokens,
        file_path=file_path,
        relative_path=rel_path,
        filename=file_path.name,
        is_binary=False,
        file_size=len(content.encode("utf-8")),
    )


def create_staged_attachment_from_file(
    filename: str,
    data: bytes,
    work_root: Path | str | None = None,
    temp_dirname: str = "tmp",
    *,
    storage: TempStorage | None = None,
) -> StagedAttachment:
    """Stage an uploaded or dropped file of any type into work_root/tmp.

    Handles binary content and excessively long filenames safely.
    """
    attachment = StagedAttachment.from_file_data(filename, data)
    attachment.reference(work_root, temp_dirname, storage=storage)
    return attachment


def create_staged_attachments_from_text(
    text: str,
    work_root: Path | str | None = None,
    threshold: int = PASTE_TOKEN_THRESHOLD,
    temp_dirname: str = "tmp",
    *,
    storage: TempStorage | None = None,
) -> list[StagedAttachment]:
    """Inspect pasted text and return one or more StagedAttachment objects.

    If multiple fenced code blocks exceed threshold, each becomes an attachment.
    Otherwise, if the text exceeds threshold, the whole text becomes an attachment.
    """
    if not text or not is_large_input(text, threshold=threshold):
        return []

    code_blocks = list(re.finditer(r"```[a-zA-Z0-9_-]*\n[\s\S]*?\n```", text))
    large_blocks = [
        m.group(0)
        for m in code_blocks
        if is_large_input(m.group(0), threshold=threshold)
    ]
    if len(large_blocks) >= 2:
        return [
            create_staged_attachment(
                b, work_root=work_root, temp_dirname=temp_dirname, storage=storage
            )
            for b in large_blocks
        ]

    return [
        create_staged_attachment(
            text, work_root=work_root, temp_dirname=temp_dirname, storage=storage
        )
    ]


def is_large_input(text: str, threshold: int = PASTE_TOKEN_THRESHOLD) -> bool:
    """Return True if text exceeds the token threshold using the exmekh_lib tokenizer."""
    if not text:
        return False
    return count_text_tokens(text) > threshold


def create_input_markdown_file(
    content: str,
    work_root: Path | str | None = None,
    temp_dirname: str = "tmp",
    *,
    storage: TempStorage | None = None,
) -> tuple[Path, str]:
    """Write text content to a markdown (.md) file in a temporary folder inside work_root.

    Returns the absolute path and the relative path (with forward slashes) relative to work_root.
    """
    return (storage or TempStorage(work_root, temp_dirname)).write_file(
        "user_input.md", content.encode("utf-8")
    )


def format_user_input_file_xml(relative_path: str) -> str:
    """Generate an English prompt reference for any staged file."""
    return app_text(
        "execution.pasted_input.file_reference", tag=XML_TAG_NAME, path=relative_path
    )


def prepare_user_input(
    text: str,
    *,
    work_root: Path | str | None = None,
    attachments: list[StagedAttachment] | None = None,
    temp_dirname: str = "tmp",
    use_file_references: bool = True,
    storage: TempStorage | None = None,
) -> PreparedUserInput:
    """Prepare model text and the separate text displayed in the chat."""
    if not use_file_references:
        if any(
            attachment.is_binary and not attachment.image_urls
            for attachment in attachments or []
        ):
            raise ValueError(
                app_text(
                    "errors_history.pasted_input.quick_chat_supports_text_attachments_use_a_project_for_other_fil"
                )
            )
        parts = [
            text,
            *(
                attachment.reference(work_root, temp_dirname, storage=storage)
                if attachment.image_urls
                else app_text(
                    "execution.pasted_input.inline_attachment",
                    filename=attachment.filename,
                    content=attachment.content,
                )
                for attachment in attachments or []
            ),
        ]
        prompt = "\n\n".join(part for part in parts if part)
        display_content = text if attachments else None
    elif attachments is None:
        prompt, display_content, _ = process_user_input_text(
            text, work_root=work_root, temp_dirname=temp_dirname, storage=storage
        )
    else:
        references = [
            attachment.reference(work_root, temp_dirname, storage=storage)
            for attachment in attachments
        ]
        prompt = "\n\n".join(part for part in [text, *references] if part)
        display_content = text if attachments else None
    title_text = text or (
        attachments[0].preview
        if attachments
        else app_text("execution.pasted_input.attached_file")
    )
    image_urls = [
        url for attachment in attachments or [] for url in attachment.image_urls
    ]
    for attachment in attachments or []:
        if attachment.pdf_pages:
            prompt += app_text(
                "execution.pasted_input.pdf_page_notice",
                prefix=app_text("execution.pasted_input.pdf_notice_prefix"),
                marker=app_text("execution.pasted_input.pdf_notice_marker"),
                filename=attachment.filename,
                sent_pages=len(attachment.image_urls),
                total_pages=attachment.pdf_pages,
            )
            if attachment.pdf_pages > len(attachment.image_urls):
                prompt += app_text("execution.pasted_input.pdf_remaining_pages")
    image_paths = [
        attachment.relative_path
        for attachment in attachments or []
        if attachment.image_urls and attachment.relative_path is not None
    ]
    # Editing a queued message returns its original references to the composer.
    # Rebuild vision parts from those scoped files instead of losing the images.
    reference_storage = storage or TempStorage(work_root, temp_dirname)
    for relative_path in extract_attached_files_from_content(text):
        if relative_path in image_paths:
            continue
        try:
            file_path = reference_storage.resolve_attachment(
                relative_path, allow_legacy=True
            )
        except ValueError:
            continue
        if not file_path.is_file():
            continue
        urls, _ = attachment_images(file_path.name, file_path.read_bytes())
        if urls:
            image_urls.extend(urls)
            image_paths.append(relative_path)
    attachment_paths = list(
        dict.fromkeys(
            [
                *(
                    attachment.relative_path
                    for attachment in attachments or []
                    if attachment.relative_path
                ),
                *extract_attached_files_from_content(text),
            ]
        )
    )
    return PreparedUserInput(
        prompt, display_content, title_text, image_urls, image_paths, attachment_paths
    )


def resolve_attached_file_path(relative_path: str, work_root: Path | str) -> Path:
    """Resolve a chat attachment only within the work root's staging folder."""
    return TempStorage(work_root).resolve_attachment(relative_path)


def process_user_input_text(
    text: str,
    work_root: Path | str | None = None,
    threshold: int = PASTE_TOKEN_THRESHOLD,
    temp_dirname: str = "tmp",
    *,
    storage: TempStorage | None = None,
) -> tuple[str, str | None, list[Path]]:
    """Inspect text for one or more input blocks longer than threshold tokens.

    If any parts (or the whole text) exceed threshold:
    - Generates an MD text file for each large block in a temporary path within work_root.
    - In its place in the chat, appends an XML tag in English referencing the relative path.
    - Preserves the original user text for display to the user in the UI.

    Returns (processed_prompt, display_content, created_file_paths).
    """
    if not text or not is_large_input(text, threshold=threshold):
        return text, None, []

    created_files: list[Path] = []

    # 1. Check for fenced code blocks
    code_blocks = list(re.finditer(r"```[a-zA-Z0-9_-]*\n[\s\S]*?\n```", text))
    replacements: list[tuple[int, int, str]] = []

    for match in code_blocks:
        block_text = match.group(0)
        if is_large_input(block_text, threshold=threshold):
            file_path, rel_path = create_input_markdown_file(
                block_text,
                work_root=work_root,
                temp_dirname=temp_dirname,
                storage=storage,
            )
            created_files.append(file_path)
            xml_block = format_user_input_file_xml(rel_path)
            replacements.append((match.start(), match.end(), xml_block))

    if replacements:
        processed = text
        for start, end, rep in reversed(replacements):
            processed = processed[:start] + rep + processed[end:]
        return processed, text, created_files

    # 2. Check for paragraphs / blocks separated by double newlines (\n\n)
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        new_paragraphs: list[str] = []
        replaced_any = False
        for p in paragraphs:
            if is_large_input(p, threshold=threshold):
                file_path, rel_path = create_input_markdown_file(
                    p, work_root=work_root, temp_dirname=temp_dirname, storage=storage
                )
                created_files.append(file_path)
                new_paragraphs.append(format_user_input_file_xml(rel_path))
                replaced_any = True
            else:
                new_paragraphs.append(p)
        if replaced_any:
            return "\n\n".join(new_paragraphs), text, created_files

    # 3. Default: the whole text exceeds threshold
    file_path, rel_path = create_input_markdown_file(
        text, work_root=work_root, temp_dirname=temp_dirname, storage=storage
    )
    created_files.append(file_path)
    xml_block = format_user_input_file_xml(rel_path)
    return xml_block, text, created_files


def extract_attached_files_from_content(content: str) -> list[str]:
    """Extract relative paths of user_input_file XML tags from message content."""
    if not content:
        return []
    return re.findall(rf'<{XML_TAG_NAME}\s+path="([^"]+)">', content)


def strip_user_input_file_xml(content: str) -> str:
    """Remove user_input_file XML tags and their inner content from text."""
    if not content:
        return ""
    pattern = rf'<{XML_TAG_NAME}\s+path="[^"]*">[\s\S]*?</{XML_TAG_NAME}>'
    cleaned = re.sub(pattern, "", content)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
