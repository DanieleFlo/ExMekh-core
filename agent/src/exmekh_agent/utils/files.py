"""Atomic text writes with cleanup after failed writes or replacement."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def atomic_write_text(
    destination: Path, content: str, *, temporary_path: Path | None = None
) -> None:
    temporary_path = temporary_path or destination.with_suffix(
        f".{uuid4().hex}{destination.suffix}.tmp"
    )
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
