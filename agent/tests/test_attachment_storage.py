from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from fe.exmekh_bridge.pasted_input import (
    StagedAttachment,
    create_staged_attachment_from_file,
)
from fe.state.session_manager import session_manager
from fe.state.temp_storage import TempStorage
from fe.ui.file_upload import upload_router


def test_binary_attachment_retains_bytes_and_extension_across_work_roots(
    tmp_path: Path,
) -> None:
    data = b"\x00\xfforiginal binary contents"
    attachment = create_staged_attachment_from_file(
        "archive.bin", data, tmp_path / "first"
    )
    descriptor = attachment.upload_payload()
    assert "data" not in descriptor and "file_path" not in descriptor
    restored = StagedAttachment.from_upload(descriptor, tmp_path / "first")
    restored.reference(tmp_path / "second")
    assert restored.file_path.read_bytes() == data
    assert restored.file_path.suffix == ".bin"
    assert restored.filename == "archive.bin"

    attachment.file_path.unlink()
    attachment.reference(tmp_path / "second")
    assert attachment.file_path.read_bytes() == data


def test_missing_binary_attachment_does_not_create_an_empty_replacement(
    tmp_path: Path,
) -> None:
    attachment = create_staged_attachment_from_file(
        "archive.zip", b"PK\x00data", tmp_path
    )
    restored = StagedAttachment.from_upload(attachment.upload_payload(), tmp_path)
    attachment.file_path.unlink()
    with pytest.raises(FileNotFoundError, match="no longer available"):
        restored.reference(tmp_path)
    assert not list((tmp_path / "tmp").iterdir())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"relative_path": "../outside.txt"},
        {"data": 123},
        {"data": "invalid base64!"},
    ],
)
def test_upload_descriptor_rejects_invalid_data_and_paths(
    tmp_path: Path, payload: dict
) -> None:
    with pytest.raises(ValueError):
        StagedAttachment.from_upload(payload, tmp_path)


def test_minimal_upload_descriptor_uses_only_a_path_within_storage(
    tmp_path: Path,
) -> None:
    attachment = StagedAttachment.from_upload(
        {"relative_path": "tmp/test.txt", "file_path": str(tmp_path / "outside.txt")},
        tmp_path,
    )
    assert attachment.file_path == tmp_path / "tmp" / "test.txt"
    assert attachment.id


@pytest.mark.parametrize("folder", [".", "..", "../outside"])
def test_temp_storage_rejects_paths_outside_its_work_root(
    tmp_path: Path, folder: str
) -> None:
    with pytest.raises(ValueError, match="inside the work root"):
        TempStorage(tmp_path, folder)


def test_inaccessible_file_is_ignored_when_scanning_storage(tmp_path: Path) -> None:
    storage = TempStorage(tmp_path)
    storage.write_file("locked.txt", b"locked")
    with patch.object(Path, "is_file", side_effect=PermissionError("unreadable")):
        assert storage.stats().total_files == 0
        assert storage.clean() == (0, 0)


def test_upload_is_bound_to_its_original_conversation(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    current_root = tmp_path / "current"
    current_root.mkdir()
    original_project = session_manager.create_project(original_root)
    current_project = session_manager.create_project(current_root)
    original = session_manager.create_session("Upload original", original_project.id)
    session_manager.create_session("Upload current", current_project.id)
    app = FastAPI()
    app.include_router(upload_router)
    with TestClient(app) as client:
        response = client.post(
            "/_exmekh/upload",
            data={"session_id": original.id},
            files={"files": ("source.py", b"print(42)")},
        )
        assert response.status_code == 200
        descriptor = response.json()[0]
        assert (
            tmp_path / "original" / descriptor["relative_path"]
        ).read_bytes() == b"print(42)"
        assert not (current_root / "tmp").exists()
        assert (
            client.post("/_exmekh/upload", data={"session_id": "deleted"}).status_code
            == 404
        )
        assert (
            client.post(
                "/_exmekh/upload",
                data={"session_id": original.id, "files": "not a file"},
            ).json()
            == []
        )
        assert (
            client.post("/_exmekh/upload", data={"session_id": original.id}).json()
            == []
        )


def test_upload_without_a_filename_receives_a_safe_default(tmp_path: Path) -> None:
    session = session_manager.create_session("Nameless upload")
    session.settings.work_root = str(tmp_path)
    app = FastAPI()
    app.include_router(upload_router)
    body = (
        f'--upload\r\nContent-Disposition: form-data; name="session_id"\r\n\r\n{session.id}\r\n'
        '--upload\r\nContent-Disposition: form-data; name="file"; filename=""\r\n'
        "Content-Type: application/octet-stream\r\n\r\ncontents\r\n--upload--\r\n"
    )
    with TestClient(app) as client:
        response = client.post(
            "/_exmekh/upload",
            content=body,
            headers={"Content-Type": "multipart/form-data; boundary=upload"},
        )
    assert response.status_code == 200
    assert response.json()[0]["filename"] == "uploaded_file"
