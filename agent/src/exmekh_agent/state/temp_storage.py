from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from exmekh_agent.config import config

TEMP_STORAGE_ALERT_THRESHOLD_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class TempFileInfo:
    name: str
    size: int
    modified: float


@dataclass(frozen=True)
class TempStorageStats:
    folder_path: Path
    total_files: int
    total_size: int
    files: list[TempFileInfo]

    @property
    def is_over_limit(self) -> bool:
        return self.total_size >= TEMP_STORAGE_ALERT_THRESHOLD_BYTES


def clean_attachment_filename(filename: str) -> str:
    """Keep a display name that can also be used safely on disk."""
    name = Path(filename).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name if name.replace(".", "").strip() else "uploaded_file"


def sanitize_attachment_filename(
    filename: str, max_stem_len: int = 60
) -> tuple[str, str]:
    """Return a bounded stem and suffix for the stored file name."""
    path = Path(clean_attachment_filename(filename))
    stem = path.stem[:max_stem_len].rstrip(". ") or "file"
    return stem, path.suffix[:15]


class TempStorage:
    """Own the naming, writing, inspection, and cleanup of staged files."""

    def __init__(
        self, work_root: Path | str | None = None, temp_dirname: str = "tmp"
    ) -> None:
        self._root = Path(work_root or config.work_root).resolve()
        self._directory = (self._root / temp_dirname).resolve()
        if self._directory == self._root or not self._directory.is_relative_to(
            self._root
        ):
            raise ValueError("Temporary storage must be a folder inside the work root")

    def contains(self, file_path: Path) -> bool:
        return file_path.resolve().is_relative_to(self._directory)

    @property
    def work_root(self) -> Path:
        return self._root

    def resolve_attachment(
        self, relative_path: str, *, allow_legacy: bool = False
    ) -> Path:
        """Resolve an attachment only within this storage folder."""
        path = (self._root / relative_path).resolve()
        legacy_file = allow_legacy and path.parent == self._root / "tmp"
        if not self.contains(path) and not legacy_file:
            raise ValueError("Attachment path is outside the staging folder")
        return path

    def write_file(self, filename: str, data: bytes) -> tuple[Path, str]:
        self._directory.mkdir(parents=True, exist_ok=True)
        stem, suffix = sanitize_attachment_filename(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = (
            self._directory / f"{stem}_{timestamp}_{uuid.uuid4().hex[:8]}{suffix}"
        )
        file_path.write_bytes(data)
        return file_path, file_path.relative_to(self._root).as_posix()

    def _entries(self) -> list[Path]:
        try:
            return sorted(self._directory.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            return []

    def stats(self) -> TempStorageStats:
        files: list[TempFileInfo] = []
        for entry in self._entries():
            try:
                if entry.is_file():
                    stat = entry.stat()
                    files.append(TempFileInfo(entry.name, stat.st_size, stat.st_mtime))
            except OSError:
                continue
        return TempStorageStats(
            folder_path=self._directory,
            total_files=len(files),
            total_size=sum(file.size for file in files),
            files=files,
        )

    def clean(self) -> tuple[int, int]:
        """Remove files only, returning their count and freed bytes."""
        deleted_count = 0
        freed_bytes = 0
        for entry in self._entries():
            try:
                if entry.is_file():
                    size = entry.stat().st_size
                    entry.unlink()
                    deleted_count += 1
                    freed_bytes += size
            except OSError:
                continue
        return deleted_count, freed_bytes
