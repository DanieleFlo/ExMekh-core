"""Local image/PDF decoding and lossless ExMekh-Lib multimodal history conversion."""

from __future__ import annotations

import base64
from contextlib import closing
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import pypdfium2 as pdfium
from exmekh_lib.core.models.user_message import (
    ImagesUserMessage,
    UserMessage,
    UserMessageContent,
)
from PIL import Image, ImageOps, UnidentifiedImageError

# PDFium is not thread-safe; uploads run in the shared executor.
_pdf_render_lock = Lock()

MAX_PDF_PAGES = 10
MAX_IMAGE_SIDE = 2048
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".ico",
}


def image_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(
        "ascii"
    )


def image_from_url(url: str) -> Image.Image:
    if not url.startswith("data:image/png;base64,"):
        raise ValueError("Saved images must be PNG data URLs")
    data = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(BytesIO(data)) as source:
        return source.copy()


def _normalized_url(source: Image.Image) -> str:
    oriented = ImageOps.exif_transpose(source).convert("RGBA")
    image = Image.new("RGB", oriented.size, "white")
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
                    raise ValueError("PDF has no pages")
                urls = []
                for index in range(min(total, MAX_PDF_PAGES)):
                    with closing(document[index]) as page:
                        width, height = page.get_size()
                        scale = min(2.0, MAX_IMAGE_SIDE / max(width, height))
                        with closing(page.render(scale=scale)) as bitmap:
                            urls.append(_normalized_url(bitmap.to_pil()))
                return urls, total
        except pdfium.PdfiumError as error:
            raise ValueError(
                "Unable to read PDF; it may be damaged or password protected"
            ) from error
    try:
        with Image.open(BytesIO(data)) as source:
            return [_normalized_url(source)], 0
    except UnidentifiedImageError as error:
        if suffix in IMAGE_SUFFIXES:
            raise ValueError("Unable to read image") from error
        return [], 0
    except Image.DecompressionBombError as error:
        raise ValueError("Image dimensions are too large") from error


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
                "ExMekh Agent converts documents to images before inference"
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
            raise ValueError("Unsupported saved user content part")
    return parts


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
    """Replace ExMekh-Lib image hashes in snapshots without mutating runtime history."""
    history, tree = deepcopy((history, tree))
    histories = [history]
    pending = list(tree)
    for entry in pending:
        histories.append(entry.get("history", []))
        pending.extend(entry.get("children", []))
    for items in histories:
        for item in items:
            content = item.get("content")
            if (
                item.get("role") == "user"
                and isinstance(content, str)
                and content in content_by_dump
            ):
                item["content"] = deepcopy(content_by_dump[content])
    return history, tree
