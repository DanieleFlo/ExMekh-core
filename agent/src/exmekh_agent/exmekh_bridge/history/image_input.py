"""Local image/PDF decoding and lossless ExMekh-Lib multimodal history conversion."""

from __future__ import annotations

import base64
from contextlib import closing
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import pypdfium2 as pdfium
from exmekh_lib.core.cache.user_message_codec import (
    deserialize_user_message,
)
from exmekh_lib.core.models.user_message import (
    ImagesUserMessage,
    UserMessage,
    UserMessageContent,
)
from PIL import Image, ImageOps, UnidentifiedImageError

from exmekh_agent.config.attachments import (
    DEFAULTS as ATTACHMENTS_DEFAULTS,
)
from exmekh_agent.config.attachments import (
    IMAGE_SUFFIXES as IMAGE_SUFFIXES,
)
from exmekh_agent.config.attachments import (
    MAX_IMAGE_SIDE as MAX_IMAGE_SIDE,
)
from exmekh_agent.config.attachments import (
    MAX_PDF_PAGES as MAX_PDF_PAGES,
)
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.exmekh_bridge.history.recorded_history import RecordedHistory
from exmekh_agent.exmekh_bridge.history.tool_attachments import (
    tool_attachment_image_urls as tool_attachment_image_urls,
)

# PDFium is not thread-safe; uploads run in the shared executor.
_pdf_render_lock = Lock()


def image_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(
        "ascii"
    )


def image_from_url(url: str) -> Image.Image:
    if not url.startswith("data:image/png;base64,"):
        raise ValueError(
            app_text("errors_history.image_input.saved_images_must_be_png_data_urls")
        )
    data = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(BytesIO(data)) as source:
        return source.copy()


def _normalized_url(source: Image.Image) -> str:
    oriented = ImageOps.exif_transpose(source).convert("RGBA")
    image = Image.new("RGB", oriented.size, ATTACHMENTS_DEFAULTS["image_background"])
    image.paste(oriented, mask=oriented.getchannel("A"))
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image_data_url(image)


def attachment_images(filename: str, data: bytes) -> tuple[list[str], int]:
    """Render up to ten PDF pages, or decode the first frame of a raster image."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" or data.startswith(b"%PDF-"):
        try:
            with _pdf_render_lock, pdfium.PdfDocument(data) as document:
                total = len(document)
                if total == 0:
                    raise ValueError(
                        app_text("errors_history.image_input.pdf_has_no_pages")
                    )
                urls = []
                for index in range(min(total, MAX_PDF_PAGES)):
                    with closing(document[index]) as page:
                        width, height = page.get_size()
                        scale = min(
                            ATTACHMENTS_DEFAULTS["pdf_render_scale"],
                            MAX_IMAGE_SIDE / max(width, height),
                        )
                        with closing(page.render(scale=scale)) as bitmap:
                            urls.append(_normalized_url(bitmap.to_pil()))
                return urls, total
        except pdfium.PdfiumError as error:
            raise ValueError(
                app_text(
                    "errors_history.image_input.unable_to_read_pdf_it_may_be_damaged_or_password_protected"
                )
            ) from error
    try:
        with Image.open(BytesIO(data)) as source:
            return [_normalized_url(source)], 0
    except UnidentifiedImageError as error:
        if suffix in IMAGE_SUFFIXES:
            raise ValueError(
                app_text("errors_history.image_input.unable_to_read_image")
            ) from error
        return [], 0
    except Image.DecompressionBombError as error:
        raise ValueError(
            app_text("errors_history.image_input.image_dimensions_are_too_large")
        ) from error


def user_content(text: str, image_urls: list[str]) -> UserMessageContent:
    if not image_urls:
        return text
    return [text, ImagesUserMessage([image_from_url(url) for url in image_urls])]


def serialize_user_content(message: UserMessage) -> str | list[dict[str, Any]]:
    """ExMekh-Lib dumps only image hashes; persist actual parts so reload keeps vision."""
    content = message.get_message()
    if isinstance(content, str):
        return content
    parts = []
    for part in message.iter_content_parts():
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            parts.append(
                {"type": "image_url", "image_url": {"url": image_data_url(part)}}
            )
        else:
            raise ValueError(
                app_text(
                    "errors_history.image_input.exmekh_agent_converts_documents_to_images_before_inference"
                )
            )
    return parts


def deserialize_user_content(content: Any) -> UserMessageContent:
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for part in content:
        if part["type"] == "text":
            parts.append(str(part["text"]))
        elif part["type"] == "image_url":
            parts.append(ImagesUserMessage([image_from_url(part["image_url"]["url"])]))
        else:
            raise ValueError(
                app_text(
                    "errors_history.image_input.unsupported_saved_user_content_part"
                )
            )
    return parts


def deserialize_tool_attachments(content: Any) -> UserMessage | None:
    """Accept lossless library snapshots and previously saved UI payloads."""
    if not content:
        return None
    if isinstance(content, dict):
        return deserialize_user_message(content)
    return UserMessage(deserialize_user_content(content))


def content_text(content: Any) -> str:
    """Extract model text without counting or displaying image base64 as prose."""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if part.get("type") == "text"
        )
    return str(content or "")


def restore_image_history(
    history: list[dict[str, Any]],
    tree: list[dict[str, Any]],
    content_by_dump: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore complete invocation and image payloads on an independent snapshot."""
    return RecordedHistory(history, tree).restored_snapshot(content_by_dump)
