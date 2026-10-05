"""Atomic JSON persistence, independent of chat and project state."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.formatting import DEFAULTS as FORMAT_DEFAULTS
from exmekh_agent.utils.files import atomic_write_text

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
                    app_text("errors_state.json_store.could_not_load_s_from_s_s"),
                    self.model_type.__name__,
                    path,
                    error,
                )
        return records

    def save(self, record_id: str, record: T) -> None:
        path = self._path(record_id)
        with self._write_lock:
            atomic_write_text(
                path, record.model_dump_json(indent=FORMAT_DEFAULTS["indent_spaces"])
            )

    def delete(self, record_id: str) -> None:
        with self._write_lock:
            self._path(record_id).unlink(missing_ok=True)

    def _path(self, record_id: str) -> Path:
        if not record_id or Path(record_id).name != record_id:
            raise ValueError(
                app_text("errors_state.json_store.record_id_must_be_a_file_name")
            )
        return self.directory / f"{record_id}.json"
