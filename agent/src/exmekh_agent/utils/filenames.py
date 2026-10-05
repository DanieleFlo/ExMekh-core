"""Portable, bounded attachment names shared by staging and uploads."""

from __future__ import annotations

import re
from pathlib import Path

from exmekh_agent.config.storage import DEFAULTS as STORAGE_DEFAULTS


def clean_attachment_filename(filename: str) -> str:
    """Keep a display name that can also be used safely on disk."""
    name = Path(filename).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return (
        name
        if name.replace(".", "").strip()
        else STORAGE_DEFAULTS["fallback_upload_name"]
    )


def sanitize_attachment_filename(
    filename: str, max_stem_len: int = STORAGE_DEFAULTS["filename_stem_limit"]
) -> tuple[str, str]:
    """Return a bounded stem and suffix for the stored file name."""
    path = Path(clean_attachment_filename(filename))
    stem = path.stem[:max_stem_len].rstrip(". ") or STORAGE_DEFAULTS["fallback_stem"]
    return stem, path.suffix[: STORAGE_DEFAULTS["filename_suffix_limit"]]
