from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exmekh_agent.config import AppConfig, config
from exmekh_agent.state.session_manager import SessionManager, session_manager
from tests import conftest as test_fixtures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_PROBE = """
import json
import pytest
import tests
from exmekh_agent.config import config
from exmekh_agent.state.session_manager import session_manager

project = session_manager.create_project(config.data_dir)
chat = session_manager.create_session(project_id=project.id)
print(json.dumps({
    "data_dir": str(config.data_dir),
    "chats_dir": str(session_manager.storage_dir),
    "projects_dir": str(session_manager.projects_dir),
    "chat_id": chat.id,
}))
"""


def test_fixture_preserves_user_records_changed_while_tests_run(
    tmp_path: Path,
) -> None:
    application_dir = tmp_path / "application"
    user_manager = SessionManager(application_dir / "data" / "chats")
    chat = user_manager.get_current_session()
    user_manager.create_project(application_dir)

    with pytest.MonkeyPatch.context() as fixture_patch:
        fixture = test_fixtures.cleanup_test_chats.__wrapped__(
            tmp_path / "test-run", fixture_patch
        )
        next(fixture)
        try:
            assert session_manager.storage_dir == config.data_dir / "chats"
            assert config.data_dir.is_relative_to(tmp_path / "test-run")
            assert SessionManager().storage_dir == session_manager.storage_dir

            chat.title = "Updated by the user while tests were running"
            assert user_manager.save_session(chat)
            user_manager.create_session("Created while tests were running")
            expected = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (application_dir / "data").glob("*/*.json")
            }
        finally:
            with pytest.raises(StopIteration):
                next(fixture)

    actual = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (application_dir / "data").glob("*/*.json")
    }
    assert actual == expected


def test_manager_restores_live_objects_and_selection_after_storage_scope(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "original" / "chats")
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    chat.add_queued_message("Unsaved guidance")
    manager.create_session()
    manager.switch_session(chat.id)

    with pytest.raises(RuntimeError, match="caller failed"):
        with manager.using_storage(tmp_path / "isolated" / "chats"):
            assert manager.get_session(chat.id) is None
            assert manager.get_project(project.id) is None
            manager.create_session("Isolated chat")
            raise RuntimeError("caller failed")

    assert manager.get_current_session() is chat
    assert manager.get_project(project.id) is project
    assert chat.has_queued_messages
    assert manager.storage_dir == tmp_path / "original" / "chats"
    assert manager.save_session(chat)
    assert SessionManager(manager.storage_dir).get_session(chat.id).has_queued_messages


def test_manager_restores_storage_when_switching_directory_fails(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "original" / "chats")
    chat = manager.get_current_session()
    invalid_directory = tmp_path / "file"
    invalid_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(OSError):
        with manager.using_storage(invalid_directory):
            pytest.fail("invalid storage must not enter the scope")
    assert manager.get_current_session() is chat
    assert manager.storage_dir == tmp_path / "original" / "chats"


def test_default_settings_saves_use_an_isolated_env_file(tmp_path: Path) -> None:
    default_path = tmp_path / ".env"
    explicit_path = tmp_path / "explicit.env"
    with pytest.MonkeyPatch.context() as fixture_patch:
        test_fixtures.preserve_env_file.__wrapped__(tmp_path, fixture_patch)
        AppConfig(base_url="http://isolated.test").save_to_env()
        AppConfig(base_url="http://explicit.test").save_to_env(explicit_path)

    assert "BASE_URL=http://isolated.test" in default_path.read_text(encoding="utf-8")
    assert "BASE_URL=http://explicit.test" in explicit_path.read_text(encoding="utf-8")


def run_collection_probe() -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, "-c", COLLECTION_PROBE],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_concurrent_pytest_processes_have_independent_startup_storage() -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_collection_probe)
        second = executor.submit(run_collection_probe)
        records = [first.result(), second.result()]

    assert records[0]["data_dir"] != records[1]["data_dir"]
    assert records[0]["chat_id"] != records[1]["chat_id"]
    for record in records:
        data_dir = Path(record["data_dir"])
        assert data_dir.is_relative_to(PROJECT_ROOT / ".test-tmp")
        assert Path(record["chats_dir"]) == data_dir / "chats"
        assert Path(record["projects_dir"]) == data_dir / "projects"
