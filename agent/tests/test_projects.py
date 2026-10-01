from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.converter import qs_messages_to_session, session_to_qs_messages
from fe.qs_bridge.models import ChatMessage, ChatSession, Project, ProjectSettings
from fe.qs_bridge.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
    prepare_user_input,
)
from fe.state.active_turns import active_turns
from fe.state.session_manager import CODE_WORK_ROOT, SessionManager
from fe.state.temp_storage import TempStorage


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "data" / "chats")


def test_project_folder_name_is_immutable_and_persisted(
    manager: SessionManager, tmp_path: Path
) -> None:
    folder = tmp_path / "My project"
    folder.mkdir()
    project = manager.create_project(folder)
    assert project.name == "My project"
    assert manager.list_projects() == [project]
    assert manager.get_project("missing") is None
    assert manager.get_project(None) is None
    with pytest.raises(ValidationError, match="frozen"):
        project.work_root = str(tmp_path)
    assert Project(work_root=folder.anchor).name == folder.anchor
    with pytest.raises(ValueError, match="already uses"):
        manager.create_project(folder / ".")
    with pytest.raises(ValueError, match="existing"):
        manager.create_project(folder / "missing")
    with pytest.raises(ValueError, match="not found"):
        manager.create_session(project_id="missing")
    restored = SessionManager(manager.storage_dir)
    assert restored.get_project(project.id) == project


def test_quick_chat_uses_code_folder_and_project_chats_use_deep_agent(
    manager: SessionManager, tmp_path: Path
) -> None:
    quick = manager.get_current_session()
    assert quick.agent_mode == "quick_chat"
    assert quick.project_id is None
    assert Path(quick.settings.work_root) == CODE_WORK_ROOT
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    assert chat.agent_mode == "deep_agent"
    assert chat.settings.work_root == project.work_root
    assert manager.get_or_create_empty_session() is quick
    assert manager.get_or_create_empty_session(project.id) is chat
    chat.add_message(ChatMessage(role="user", content="Occupied"))
    assert manager.get_or_create_empty_session(project.id).project_id == project.id


def test_shared_settings_and_approval_defaults_are_independent(
    manager: SessionManager, tmp_path: Path
) -> None:
    project = manager.create_project(
        tmp_path, ProjectSettings(name_hds="first", tool_approval_mode="Approve All")
    )
    first = manager.create_session(project_id=project.id)
    second = manager.create_session(project_id=project.id)
    quick = manager.create_session()
    first.settings.set_tool_approval_mode("Manual")
    manager.record_command_approval(
        first, "git status", [{"role": "user", "content": "approved"}]
    )
    settings = ProjectSettings(
        name_hds="new shared prompt",
        capabilities_path="custom",
        coding_agent_persistent_history=False,
        tool_approval_mode="Approve via LLM",
    )
    assert manager.save_project_settings(project.id, settings)
    assert not manager.save_project_settings("missing", settings)
    assert first.settings.name_hds == second.settings.name_hds == "new shared prompt"
    assert quick.settings.name_hds is None
    assert first.settings.tool_approval_mode == "Manual"
    assert second.settings.tool_approval_mode == "Approve All"
    assert second.settings.approved_commands_history == []
    new = manager.create_session(project_id=project.id)
    assert new.settings.tool_approval_mode == "Approve via LLM"
    assert new.settings.coding_agent_persistent_history is False
    settings.name_hds = "outside mutation"
    assert project.settings.name_hds == "new shared prompt"
    manager.load_all()
    assert manager.get_session(first.id).settings.approved_commands_history == [
        "git status"
    ]
    assert manager.get_session(second.id).settings.tool_approval_mode == "Approve All"
    assert (
        manager.get_project(project.id).settings.tool_approval_mode == "Approve via LLM"
    )


async def test_delete_project_cancels_its_chats_and_preserves_other_projects(
    manager: SessionManager, tmp_path: Path
) -> None:
    first = manager.create_project(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = manager.create_project(other_root)
    preserved = manager.create_session(project_id=other.id)
    chats = [manager.create_session(project_id=first.id) for _ in range(2)]
    task = asyncio.create_task(asyncio.sleep(60))
    active_turns.start(chats[0].id, task, TurnStreamCollector(), started_at=0)
    manager.delete_project(first.id)
    with pytest.raises(asyncio.CancelledError):
        await task
    manager.delete_project("missing")
    assert manager.get_project(first.id) is None
    assert all(manager.get_session(chat.id) is None for chat in chats)
    assert manager.get_session(preserved.id) is preserved
    assert manager.get_current_session().project_id != first.id
    assert tmp_path.is_dir()
    restored = SessionManager(manager.storage_dir)
    assert restored.get_project(first.id) is None
    assert restored.get_project(other.id) is not None
    assert all(restored.get_session(chat.id) is None for chat in chats)


def test_delete_last_project_chat_leaves_empty_project(
    manager: SessionManager, tmp_path: Path
) -> None:
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    manager.delete_session(chat.id)
    assert manager.get_project(project.id) is project
    assert manager.get_or_create_empty_session(project.id).project_id == project.id


def test_legacy_sessions_migrate_once_with_newest_shared_settings(
    manager: SessionManager, tmp_path: Path
) -> None:
    older = ChatSession(title="Older", updated_at=1)
    older.settings.name_hds = "older-prompt"
    older.settings.work_root = str(tmp_path)
    newer = older.model_copy(deep=True, update={"id": "newer", "updated_at": 2})
    newer.settings.name_hds = "newer-prompt"
    newer.settings.tool_approval_mode = "Approve All"
    newer.add_message(ChatMessage(role="user", content="Preserved history"))
    quick = ChatSession(agent_mode="quick_chat")
    for chat in (older, newer, quick):
        data = chat.model_dump(mode="json")
        del data["project_id"]
        (manager.storage_dir / f"{chat.id}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
    manager.load_all()
    project_id = manager.get_session(newer.id).project_id
    assert manager.get_session(older.id).project_id == project_id
    assert manager.get_session(older.id).settings.name_hds == "newer-prompt"
    assert manager.get_session(newer.id).messages == newer.messages
    assert manager.get_session(newer.id).updated_at == 2
    assert manager.get_session(quick.id).project_id is None
    manager.load_all()
    assert len(manager.projects) == 1
    assert manager.get_session(newer.id).project_id == project_id


def test_legacy_default_work_root_and_unreadable_project(
    manager: SessionManager,
) -> None:
    legacy = ChatSession(id="legacy-default")
    data = legacy.model_dump(mode="json")
    del data["project_id"]
    (manager.storage_dir / "legacy-default.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    (manager.projects_dir / "corrupt.json").write_text("{broken", encoding="utf-8")
    manager.load_all()
    assert (
        Path(manager.get_project(manager.get_session(legacy.id).project_id).work_root)
        == CODE_WORK_ROOT
    )


def test_project_save_failure_does_not_change_shared_options(
    manager: SessionManager, tmp_path: Path
) -> None:
    project = manager.create_project(tmp_path)
    with patch.object(Path, "write_text", side_effect=PermissionError("read-only")):
        with pytest.raises(PermissionError):
            manager.save_project_settings(
                project.id, ProjectSettings(name_hds="unsaved")
            )
    assert project.settings.name_hds is None
    assert not list(manager.projects_dir.glob("*.tmp"))


def test_project_attachment_namespaces_and_cleanup_are_isolated(
    manager: SessionManager, tmp_path: Path
) -> None:
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    namespace = manager.temp_dirname(chat)
    attachment = create_staged_attachment_from_file(
        "code.py", b"print(42)", tmp_path, namespace
    )
    storage = TempStorage(tmp_path, namespace)
    other = TempStorage(tmp_path, "tmp/projects/other")
    preserved, _ = other.write_file("keep.txt", b"keep")
    assert storage.stats().total_files == 1
    descriptor = StagedAttachment.from_upload(
        attachment.upload_payload(), tmp_path, namespace
    )
    assert descriptor.file_path == attachment.file_path
    with pytest.raises(ValueError, match="outside"):
        StagedAttachment.from_upload(
            attachment.upload_payload(), tmp_path, "tmp/projects/other"
        )
    inline_upload = StagedAttachment.from_upload(
        {"data": b"source", "filename": "source.txt"}, tmp_path, namespace
    )
    assert storage.contains(inline_upload.file_path)
    assert storage.clean()[0] == 2
    assert preserved.read_bytes() == b"keep"
    legacy, legacy_relative = TempStorage(tmp_path).write_file("legacy.txt", b"old")
    assert storage.resolve_attachment(legacy_relative, allow_legacy=True) == legacy
    with pytest.raises(ValueError, match="outside"):
        storage.resolve_attachment(legacy_relative)
    with pytest.raises(ValueError, match="outside"):
        storage.resolve_attachment("tmp/projects/other/keep.txt", allow_legacy=True)
    assert manager.temp_dirname(manager.create_session()) == "tmp/quickchats"


def test_staged_files_move_to_destination_project(
    manager: SessionManager, tmp_path: Path
) -> None:
    project = manager.create_project(tmp_path)
    namespace = manager.temp_dirname(manager.create_session(project_id=project.id))
    attachment = create_staged_attachment_from_file("code.py", b"print(42)", tmp_path)
    prepared = prepare_user_input(
        "Inspect", work_root=tmp_path, attachments=[attachment], temp_dirname=namespace
    )
    assert f"tmp/projects/{project.id}/" in prepared.prompt
    assert TempStorage(tmp_path, namespace).contains(attachment.file_path)


def test_quick_chat_keeps_long_text_inline_and_includes_text_attachments(
    tmp_path: Path,
) -> None:
    text = "long text " * 1000
    plain = prepare_user_input(text, work_root=tmp_path, use_file_references=False)
    assert plain.prompt == text
    assert not (tmp_path / "tmp").exists()
    attachment = create_staged_attachment_from_file("code.py", b"print(42)", tmp_path)
    prepared = prepare_user_input(
        "Read", work_root=tmp_path, attachments=[attachment], use_file_references=False
    )
    assert "print(42)" in prepared.prompt
    assert "<user_input_file" not in prepared.prompt
    binary = create_staged_attachment_from_file("image.png", b"\x00data", tmp_path)
    with pytest.raises(ValueError, match="text attachments"):
        prepare_user_input(
            "Read", work_root=tmp_path, attachments=[binary], use_file_references=False
        )


def test_project_identity_survives_qs_conversion(tmp_path: Path) -> None:
    session = ChatSession(
        project_id="project", messages=[ChatMessage(role="user", content="Hello")]
    )
    restored = qs_messages_to_session(session_to_qs_messages(session))
    assert restored.project_id == "project"
