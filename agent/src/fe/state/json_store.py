"""Atomic JSON persistence, independent of chat and project state."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JsonRecordStore[T: BaseModel]:
    def __init__(self, directory: Path, model_type: type[T]) -> None:
        self.directory = directory
        self.model_type = model_type
        self._write_lock = Lock()
        directory.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[T]:
        records = []
        for path in self.directory.glob("*.json"):
            try:
                records.append(
                    self.model_type.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError) as error:
                logger.warning(
                    "Could not load %s from %s: %s",
                    self.model_type.__name__,
                    path,
                    error,
                )
        return records

    def save(self, record_id: str, record: T) -> None:
        path = self._path(record_id)
        temporary_path = path.with_suffix(f".{uuid4().hex}.json.tmp")
        with self._write_lock:
            try:
                temporary_path.write_text(
                    record.model_dump_json(indent=2), encoding="utf-8"
                )
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def delete(self, record_id: str) -> None:
        with self._write_lock:
            self._path(record_id).unlink(missing_ok=True)

    def _path(self, record_id: str) -> Path:
        if not record_id or Path(record_id).name != record_id:
            raise ValueError("Record ID must be a file name")
        return self.directory / f"{record_id}.json"
