from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from exmekh_agent.exmekh_bridge.models import ChatMessage, ChatSession, ProjectSettings
from exmekh_agent.state.session_manager import SessionManager


def test_execution_settings_are_independent_and_shared_updates_preserve_chat_order(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    options = ProjectSettings(name_hds="original")
    project = manager.create_project(tmp_path, options)
    options.name_hds = "caller mutation"
    chat = manager.create_session(project_id=project.id)
    snapshot = manager.execution_settings(chat)
    snapshot.name_hds = "snapshot mutation"
    snapshot.approved_commands_history.append("unrelated")
    assert chat.settings.name_hds == project.settings.name_hds == "original"
    assert chat.settings.approved_commands_history == []
    path = manager.storage_dir / f"{chat.id}.json"
    before = path.read_bytes()
    timestamp = chat.updated_at
    order = [item.id for item in manager.list_sessions_sorted()]

    manager.save_project_settings(project.id, ProjectSettings(name_hds="updated"))

    assert path.read_bytes() == before
    assert chat.updated_at == timestamp
    assert [item.id for item in manager.list_sessions_sorted()] == order
    restored = SessionManager(manager.storage_dir)
    assert restored.get_session(chat.id).settings.name_hds == "updated"


def test_missing_project_is_recovered_without_losing_chat_identity(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    project = manager.create_project(tmp_path, ProjectSettings(name_hds="saved"))
    chat = manager.create_session(project_id=project.id)
    manager.add_message(chat, ChatMessage(role="user", content="Retain this"))
    (manager.projects_dir / f"{project.id}.json").unlink()

    restored = SessionManager(manager.storage_dir)

    recovered = restored.get_session(chat.id)
    assert recovered.project_id == project.id
    assert recovered.messages == chat.messages
    assert recovered.updated_at == chat.updated_at
    assert restored.get_project(project.id).settings.name_hds == "saved"
    assert SessionManager(manager.storage_dir).get_project(project.id) is not None


def test_migration_write_failure_keeps_legacy_chat_visible(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    legacy = ChatSession(
        id="legacy", messages=[ChatMessage(role="user", content="History")]
    )
    legacy.settings.work_root = str(tmp_path)
    data = legacy.model_dump(mode="json")
    del data["project_id"]
    path = manager.storage_dir / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with patch.object(manager, "save_project", side_effect=OSError("read-only")):
        manager.load_all()
    chat = manager.get_session(legacy.id)
    assert chat.messages == legacy.messages
    assert manager.get_project(chat.project_id) is not None
    assert "project_id" not in json.loads(path.read_text(encoding="utf-8"))


def test_partial_project_deletion_selects_a_surviving_chat(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    project = manager.create_project(tmp_path)
    first = manager.create_session(project_id=project.id)
    selected = manager.create_session(project_id=project.id)
    real_unlink = Path.unlink

    def unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == f"{first.id}.json":
            raise PermissionError("locked")
        real_unlink(path, *args, **kwargs)

    with patch.object(Path, "unlink", unlink):
        with pytest.raises(PermissionError, match="locked"):
            manager.delete_project(project.id)

    assert manager.get_session(selected.id) is None
    assert manager.get_project(project.id) is project
    assert manager.get_session(first.id) is first
    assert manager.get_current_session() is first
    manager.delete_session(first.id)
    manager.delete_session(manager.get_current_session().id)
    assert manager.get_current_session().project_id is None


def test_approval_mode_save_failure_rolls_back_the_change(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    chat = manager.get_current_session()
    assert not manager.set_tool_approval_mode("missing", "Approve All")
    assert manager.set_tool_approval_mode(chat.id, "Approve All")
    timestamp = chat.updated_at
    with patch.object(Path, "write_text", side_effect=OSError("read-only")):
        assert not manager.set_tool_approval_mode(chat.id, "Manual")
    assert chat.settings.tool_approval_mode == "Approve All"
    assert chat.updated_at == timestamp
    assert (
        SessionManager(manager.storage_dir)
        .get_session(chat.id)
        .settings.tool_approval_mode
        == "Approve All"
    )


def test_live_approval_policy_changes_only_after_successful_persistence(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    chat = manager.get_current_session()
    observations = []
    save_session = manager.save_session

    def persist(updated):
        observations.append(chat.settings.tool_approval_mode)
        return save_session(updated)

    with patch.object(manager, "save_session", side_effect=persist):
        assert manager.set_tool_approval_mode(chat.id, "Approve All")
    assert observations == ["Manual"]
    assert chat.settings.tool_approval_mode == "Approve All"


def test_deleting_a_background_project_preserves_the_selected_quick_chat(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "data" / "chats")
    quick = manager.get_current_session()
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    manager.switch_session(quick.id)
    manager.delete_project(project.id)
    assert manager.get_current_session() is quick
    assert manager.get_project(project.id) is None
    assert manager.get_session(chat.id) is None
