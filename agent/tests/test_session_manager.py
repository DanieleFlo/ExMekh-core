from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.models import ChatMessage, ChatSession, ChatSettings
from fe.state.active_turns import active_turns
from fe.state.session_manager import CODE_WORK_ROOT, SessionManager


@pytest.fixture
def isolated_session_manager(tmp_path: Path) -> Generator[SessionManager]:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        yield SessionManager()


def test_new_chat_reuses_empty_session_and_preserves_settings(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    empty = manager.get_current_session()
    empty.title = "Prepared chat"
    empty.agent_mode = "quick_chat"
    empty.settings = ChatSettings(
        name_hds="custom", tool_approval_mode="manual", work_root=str(CODE_WORK_ROOT)
    )
    manager.save_session(empty)
    occupied = manager.create_session("Existing conversation")
    occupied.messages.append(ChatMessage(role="user", content="Hello"))
    manager.save_session(occupied)
    saved_empty = (manager.storage_dir / f"{empty.id}.json").read_bytes()

    assert manager.get_or_create_empty_session() is empty
    assert manager.get_current_session() is empty
    assert manager.get_or_create_empty_session() is empty
    assert len(manager.list_sessions_sorted()) == 2
    assert (manager.storage_dir / f"{empty.id}.json").read_bytes() == saved_empty

    manager.load_all()
    restored = manager.get_or_create_empty_session()
    assert restored.id == empty.id
    assert restored.title == "Prepared chat"
    assert restored.agent_mode == "quick_chat"
    assert restored.settings == empty.settings
    assert len(list(manager.storage_dir.glob("*.json"))) == 2


def test_new_chat_selects_most_recent_empty_session(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    older = manager.get_current_session()
    recent = manager.create_session("Recent empty chat")
    occupied = manager.create_session("Occupied chat")
    occupied.messages.append(ChatMessage(role="user", content="Hello"))
    older.updated_at = 1.0
    recent.updated_at = 2.0
    occupied.updated_at = 3.0

    assert manager.get_or_create_empty_session() is recent
    assert manager.get_current_session() is recent
    assert len(manager.list_sessions_sorted()) == 3


@pytest.mark.parametrize("content", ["message", "queued_message", "sub_agent"])
def test_new_chat_creates_session_when_existing_chat_has_content(
    isolated_session_manager: SessionManager, content: str
) -> None:
    manager = isolated_session_manager
    occupied = manager.get_current_session()
    if content == "message":
        occupied.messages.append(ChatMessage(role="user", content="Hello"))
    elif content == "queued_message":
        occupied.add_queued_message("Waiting to send")
    else:
        occupied.start_sub_agent_conversation("child", "/child", True, "Hello")
    original = occupied.model_dump()

    ready = manager.get_or_create_empty_session()

    assert not occupied.is_empty
    assert ready.is_empty
    assert ready.id != occupied.id
    assert manager.get_current_session() is ready
    assert len(manager.list_sessions_sorted()) == 2
    assert occupied.model_dump() == original
    assert (manager.storage_dir / f"{ready.id}.json").exists()


@pytest.mark.asyncio
async def test_new_chat_skips_session_with_active_turn(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    active_session = manager.get_current_session()
    task = asyncio.create_task(asyncio.sleep(10))
    active_turns.start(active_session.id, task, TurnStreamCollector(), started_at=0.0)
    try:
        ready = manager.get_or_create_empty_session()
        assert ready.id != active_session.id
        assert manager.get_current_session() is ready
        assert not task.cancelled()
        assert active_turns.is_running(active_session.id)
        ready.messages.append(ChatMessage(role="user", content="Hello"))
    finally:
        active_turns.cancel_and_forget(active_session.id)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert manager.get_or_create_empty_session() is active_session


def test_managed_updates_persist_and_history_list_is_independent(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    session = manager.get_current_session()
    message = ChatMessage(role="user", content="Inspect this repository")
    assert manager.add_message(session, message)
    history = session.message_history()
    history.clear()
    assert session.message_history() == [message]
    assert manager.update_title(session, "Repository review")
    raw_history = [{"role": "user", "content": "git status"}]
    assert manager.record_command_approval(session, "git status", raw_history)
    assert manager.record_command_approval(session, "git status", raw_history)

    restored_manager = SessionManager(manager.storage_dir)
    restored = restored_manager.get_session(session.id)
    assert restored is not None
    assert restored.message_history() == [message]
    assert restored.title == "Repository review"
    assert restored.settings.approved_commands_history == ["git status"]
    assert restored.settings.tool_approval_raw_history == raw_history
    assert restored_manager.is_current_session(session.id)


@pytest.mark.parametrize("state", ["copied", "reloaded", "deleted"])
def test_late_updates_reject_stale_session_instances(
    isolated_session_manager: SessionManager, state: str
) -> None:
    manager = isolated_session_manager
    session = manager.get_current_session()
    if state == "copied":
        session = session.model_copy(deep=True)
    elif state == "reloaded":
        manager.load_all()
    else:
        manager.delete_session(session.id)
    original = session.model_dump()
    saved_files = {
        path: path.read_bytes() for path in manager.storage_dir.glob("*.json")
    }

    assert not manager.owns_session(session)
    assert not manager.add_message(
        session, ChatMessage(role="assistant", content="Late")
    )
    assert not manager.update_title(session, "Late title")
    assert not manager.record_command_approval(session, "pwd", [])

    assert session.model_dump() == original
    assert {
        path: path.read_bytes() for path in manager.storage_dir.glob("*.json")
    } == saved_files


def test_empty_title_does_not_change_a_managed_session(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    session = manager.get_current_session()
    original = session.model_dump()

    assert not manager.update_title(session, "")
    assert session.model_dump() == original


def test_reload_empty_storage_clears_selection(
    isolated_session_manager: SessionManager,
) -> None:
    manager = isolated_session_manager
    session = manager.get_current_session()
    assert manager.is_current_session(session.id)
    for file_path in manager.storage_dir.glob("*.json"):
        file_path.unlink()

    manager.load_all()

    assert not manager.is_current_session(session.id)
    assert manager.get_session(session.id) is None
    assert manager.list_sessions_sorted() == []


def test_session_manager_init_empty(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        assert len(sm.sessions) == 1
        curr = sm.get_current_session()
        assert curr.title == "New Conversation"


def test_session_manager_create_and_switch(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "quick_chat"
        sm = SessionManager()
        s1 = sm.create_session("Chat 1")
        s2 = sm.create_session("Chat 2")

        assert sm.current_session_id == s2.id
        switched = sm.switch_session(s1.id)
        assert switched is not None
        assert switched.id == s1.id
        assert sm.current_session_id == s1.id

        # Switch to unknown
        assert sm.switch_session("unknown_id") is None


def test_session_manager_get_current_session_fallback(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        sm.create_session("Chat 1")

        # Invalidate current_session_id
        sm.current_session_id = None
        curr = sm.get_current_session()
        assert curr.id in sm.sessions

        # Empty sessions entirely
        sm.sessions.clear()
        sm.current_session_id = None
        new_curr = sm.get_current_session()
        assert new_curr is not None
        assert new_curr.id in sm.sessions


def test_session_manager_delete_session(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        sm.sessions.clear()
        s1 = sm.create_session("Chat 1")
        s2 = sm.create_session("Chat 2")

        # Delete active session s2 -> current falls back to s1
        sm.delete_session(s2.id)
        assert s2.id not in sm.sessions
        assert sm.current_session_id == s1.id

        # Delete non-existent session (no-op)
        sm.delete_session("non-existent")

        # Delete remaining session s1 -> sessions becomes empty, triggers create_session()
        sm.delete_session(s1.id)
        assert len(sm.sessions) == 1
        assert sm.current_session_id is not None
        assert sm.current_session_id in sm.sessions


def test_session_manager_delete_session_unlink_error(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        s1 = sm.create_session("Chat 1")

        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                sm.delete_session(s1.id)
            assert s1.id in sm.sessions


@pytest.mark.asyncio
async def test_deleting_session_cancels_active_turn(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        manager = SessionManager()
        session = manager.get_current_session()
        task = asyncio.create_task(asyncio.sleep(10))
        active_turns.start(session.id, task, TurnStreamCollector(), started_at=0.0)

        manager.delete_session(session.id)

        with pytest.raises(asyncio.CancelledError):
            await task
        assert not active_turns.is_running(session.id)

        completed_session = manager.create_session("Completed")
        completed_task = asyncio.create_task(asyncio.sleep(0))
        await completed_task
        active_turns.start(
            completed_session.id, completed_task, TurnStreamCollector(), started_at=0.0
        )
        manager.delete_session(completed_session.id)
        assert completed_task.cancelled() is False
        assert not active_turns.is_running(completed_session.id)


def test_session_manager_save_session_error(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        s = ChatSession(id="test_err")
        with patch.object(
            Path, "write_text", side_effect=PermissionError("Cannot write")
        ):
            # Should not raise exception, logs error
            sm.save_session(s)

        assert not (sm.storage_dir / "test_err.json").exists()
        assert not (sm.storage_dir / "test_err.json.tmp").exists()


def test_save_session_is_atomic_and_updates_timestamp(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        session = ChatSession(id="atomic", updated_at=1.0)

        sm.save_session(session)

        saved = ChatSession.model_validate_json(
            (sm.storage_dir / "atomic.json").read_text(encoding="utf-8")
        )
        assert saved.updated_at > 1.0
        assert session.updated_at == saved.updated_at
        assert not (sm.storage_dir / "atomic.json.tmp").exists()


def test_session_manager_load_all_corrupted_file(tmp_path: Path) -> None:
    chats_dir = tmp_path / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    # Valid session
    valid = ChatSession(id="valid1", title="Valid Chat", updated_at=2000.0)
    (chats_dir / "valid1.json").write_text(valid.model_dump_json(), encoding="utf-8")

    # Corrupted JSON file
    (chats_dir / "bad.json").write_text("{corrupted json", encoding="utf-8")

    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        sm = SessionManager()
        assert "valid1" in sm.sessions
        assert sm.current_session_id == "valid1"


def test_session_manager_list_sessions_sorted(tmp_path: Path) -> None:
    with patch("fe.state.session_manager.config") as mock_cfg:
        mock_cfg.data_dir = tmp_path
        mock_cfg.default_model = "test-model"
        mock_cfg.agent_mode = "deep_agent"
        sm = SessionManager()
        sm.sessions.clear()

        s1 = ChatSession(id="s1", title="First", updated_at=100.0)
        s2 = ChatSession(id="s2", title="Second", updated_at=200.0)
        sm.sessions["s1"] = s1
        sm.sessions["s2"] = s2

        sorted_list = sm.list_sessions_sorted()
        assert len(sorted_list) == 2
        assert sorted_list[0].id == "s2"
        assert sorted_list[1].id == "s1"


def test_legacy_chat_provider_options_are_not_local_settings() -> None:
    session = ChatSession.model_validate(
        {"settings": {"base_url": "http://old-provider", "name_hds": "custom"}}
    )
    assert session.settings == ChatSettings(name_hds="custom")
