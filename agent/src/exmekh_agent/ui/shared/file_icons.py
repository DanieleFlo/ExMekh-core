from __future__ import annotations

from pathlib import Path

from exmekh_agent.ui.resources import presentation_config
from exmekh_agent.utils.formatting import format_file_size as format_file_size

FALLBACKS = presentation_config("file_icon_fallbacks")

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

    lowered_mime = mime_type.lower()
    for rule in FALLBACKS["rules"]:
        matches = any(
            lowered_mime.startswith(pattern)
            if rule["mode"] == "prefix"
            else pattern in lowered_mime
            for pattern in rule["matches"]
        )
        if matches:
            extension = (
                rule["extension"]
                if rule.get("force_extension")
                else suffix or rule["extension"]
            )
            return rule["icon"], rule["color"], extension

    default = FALLBACKS["default"]
    return default["icon"], default["color"], suffix or default["extension"]
