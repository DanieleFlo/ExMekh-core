from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from exmekh_agent.state.input_drafts import InputDraftStore
from exmekh_agent.state.session_manager import SessionManager


async def test_text_drafts_survive_restart_and_empty_text_removes_saved_draft(tmp_path):
    store = InputDraftStore(tmp_path)
    text = "  Bozza non inviata: caffè ☕\nSeconda riga\n  "
    await asyncio.gather(
        store.save_text("first", text), store.save_text("second", "Another chat")
    )
    restored = InputDraftStore(tmp_path)
    assert restored.text("first") == text
    assert restored.text("second") == "Another chat"
    assert restored.text("missing") == ""
    await restored.save_text("first", "")
    assert InputDraftStore(tmp_path).text("first") == ""
    assert not (tmp_path / "first.json").exists()
    assert InputDraftStore(tmp_path).text("second") == "Another chat"


async def test_pending_saves_keep_latest_edit_and_do_not_recreate_deleted_drafts(
    tmp_path,
):
    store = InputDraftStore(tmp_path)
    pending = []
    for index in range(50):
        pending.append(asyncio.create_task(store.save_text("first", f"Draft {index}")))
    await asyncio.gather(*pending)
    assert InputDraftStore(tmp_path).text("first") == "Draft 49"
    pending_save = asyncio.create_task(store.save_text("first", "About to close"))
    store.delete("first")
    await pending_save
    assert not (tmp_path / "first.json").exists()


async def test_session_drafts_are_shared_by_views_without_rewriting_chat_history(
    tmp_path,
):
    sessions = SessionManager(tmp_path / "chats")
    first = sessions.get_current_session()
    second = sessions.create_session("Other conversation")
    history = {path: path.read_bytes() for path in sessions.storage_dir.glob("*.json")}
    updated_at = first.updated_at
    view = sessions.for_view(first.id)
    assert not await view.save_draft_text("missing", "Ignore this")
    assert await view.save_draft_text(first.id, "Unsent first message")
    assert await view.save_draft_text(first.id, "Unsent first message")
    assert sessions.draft_text(first.id) == "Unsent first message"
    assert sessions.draft_text(second.id) == ""
    assert sessions.draft_text("missing") == ""
    assert first.updated_at == updated_at
    assert history == {
        path: path.read_bytes() for path in sessions.storage_dir.glob("*.json")
    }
    reloaded = SessionManager(tmp_path / "chats")
    assert reloaded.draft_text(first.id) == "Unsent first message"
    reloaded.delete_session(first.id)
    assert not (tmp_path / "drafts" / f"{first.id}.json").exists()
    assert reloaded.draft_text(first.id) == ""


async def test_failed_draft_save_can_retry_the_same_text(tmp_path):
    sessions = SessionManager(tmp_path / "chats")
    chat = sessions.get_current_session()
    with patch.object(Path, "replace", side_effect=OSError("read only")):
        assert not await sessions.save_draft_text(chat.id, "Keep this draft")
    assert sessions.draft_text(chat.id) == "Keep this draft"
    assert await sessions.save_draft_text(chat.id, "Keep this draft")
    assert SessionManager(tmp_path / "chats").draft_text(chat.id) == "Keep this draft"


async def test_draft_cleanup_failure_does_not_leave_a_deleted_chat_in_memory(tmp_path):
    sessions = SessionManager(tmp_path / "chats")
    chat = sessions.get_current_session()
    await sessions.save_draft_text(chat.id, "Draft")
    unlink = Path.unlink

    def fail_draft_cleanup(path, *args, **kwargs):
        if path.parent.name == "drafts":
            raise OSError("locked draft")
        return unlink(path, *args, **kwargs)

    with patch.object(Path, "unlink", fail_draft_cleanup):
        sessions.delete_session(chat.id)
    assert sessions.get_session(chat.id) is None
    assert sessions.draft_text(chat.id) == ""
    assert SessionManager(tmp_path / "chats").get_session(chat.id) is None
