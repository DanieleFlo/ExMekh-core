from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from exmekh_agent.exmekh_bridge.models import ChatSession
from exmekh_agent.state.json_store import JsonRecordStore


def test_failed_replacement_preserves_previous_record_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    store = JsonRecordStore(tmp_path, ChatSession)
    chat = ChatSession(id="chat", title="Saved")
    store.save(chat.id, chat)
    with patch.object(Path, "replace", side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError):
            store.save(chat.id, chat.model_copy(update={"title": "Unsaved"}))
    assert store.load_all()[0].title == "Saved"
    assert not list(tmp_path.glob("*.tmp"))
    store.delete(chat.id)
    store.delete(chat.id)
    assert store.load_all() == []


def test_concurrent_writes_use_separate_temporary_files(tmp_path: Path) -> None:
    store = JsonRecordStore(tmp_path, ChatSession)
    records = [ChatSession(id="chat", title=f"Version {index}") for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda chat: store.save(chat.id, chat), records))
    assert store.load_all()[0].title in {chat.title for chat in records}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("record_id", ["", "../outside", "nested/file"])
def test_record_ids_cannot_escape_the_store(tmp_path: Path, record_id: str) -> None:
    store = JsonRecordStore(tmp_path, ChatSession)
    with pytest.raises(ValueError, match="file name"):
        store.save(record_id, ChatSession())
    with pytest.raises(ValueError, match="file name"):
        store.delete(record_id)
