from __future__ import annotations

from pathlib import Path

from exmekh_agent.ui.resources import presentation_config


def format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


FILE_ICON_MAP: dict[str, tuple[str, str]] = {
    suffix: tuple(icon) for suffix, icon in presentation_config("file_icons").items()
}


def get_file_icon(filename: str, mime_type: str = "") -> tuple[str, str, str]:
    """Return (icon_name, color_class, ext_label) tailored for the given filename or MIME type."""
    path = Path(filename)
    suffix = ".env" if path.name.lower() == ".env" else path.suffix.lower()
    if suffix in FILE_ICON_MAP:
        icon, color = FILE_ICON_MAP[suffix]
        return icon, color, suffix

    if mime_type:
        m = mime_type.lower()
        if m.startswith("image/"):
            return "image", "text-purple-400", suffix or ".img"
        if m.startswith("audio/"):
            return "audiotrack", "text-pink-400", suffix or ".audio"
        if m.startswith("video/"):
            return "videocam", "text-red-400", suffix or ".video"
        if "pdf" in m:
            return "picture_as_pdf", "text-rose-400", ".pdf"
        if "text/" in m or "json" in m or "xml" in m:
            return "article", "text-sky-400", suffix or ".txt"

    return "insert_drive_file", "text-zinc-400", suffix or ".file"
