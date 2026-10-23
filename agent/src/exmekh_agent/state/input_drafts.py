"""Persist unsent text independently of potentially large conversation histories."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from exmekh_agent.state.json_store import JsonRecordStore


class InputDraft(BaseModel):
    session_id: str
    text: str


class InputDraftStore:
    def __init__(self, directory: Path) -> None:
        self._store = JsonRecordStore(directory, InputDraft)
        self._drafts = {draft.session_id: draft for draft in self._store.load_all()}
        self._saved = dict(self._drafts)
        self._write_lock = Lock()
        self._deleted: set[str] = set()

    def text(self, session_id: str) -> str:
        draft = self._drafts.get(session_id)
        return draft.text if draft else ""

    async def save_text(self, session_id: str, text: str) -> None:
        """Retain the latest edit and persist it, including retries after failure."""
        if session_id in self._deleted:
            return
        if text:
            self._drafts[session_id] = InputDraft(session_id=session_id, text=text)
        else:
            self._drafts.pop(session_id, None)
        await asyncio.to_thread(self._save_latest, session_id)

    def delete(self, session_id: str) -> None:
        self._deleted.add(session_id)
        self._drafts.pop(session_id, None)
        self._save_latest(session_id)

    def _save_latest(self, session_id: str) -> None:
        # Queued writes read the latest text while serialized, so older work
        # cannot overwrite a newer edit or recreate a sent/deleted draft.
        with self._write_lock:
            draft = self._drafts.get(session_id)
            if draft == self._saved.get(session_id):
                return
            if draft is None:
                self._store.delete(session_id)
                self._saved.pop(session_id, None)
            else:
                self._store.save(session_id, draft)
                self._saved[session_id] = draft
