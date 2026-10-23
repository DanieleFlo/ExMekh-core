from pathlib import Path

import pytest
from exmekh_lib.utils.xml import stringify_tool_result

from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent
from tests.live_file_support import verify_file_effects


def creation_history(call_id: str, *, ok: bool = True) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": stringify_tool_result(
                    {
                        "ok": ok,
                        "results": [{"index": 1, "path": "notes.txt", "ok": ok}],
                    }
                ),
            }
        ],
    )


def test_frontend_observation_uses_emitted_arguments(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    event = ToolEvent(
        tool_name="new-file",
        tool_call_id="creation",
        arguments={"document": '<new-exmekh-file path="notes.txt"/>MODEL_CHOICE'},
    )
    history = creation_history("creation")
    path.write_bytes(b"MODEL_CHOICE")
    assert verify_file_effects([event], tmp_path, {path: None}, history=history) == [
        event
    ]
    path.write_bytes(b"CORRUPT")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects([event], tmp_path, {path: None}, history=history)


@pytest.mark.parametrize("status", ["completed", "error"])
def test_frontend_valid_creation_failure_remains_a_bug(tmp_path: Path, status) -> None:
    event = ToolEvent(
        tool_name="new-file",
        tool_call_id="creation",
        status=status,
        arguments={"document": '<new-exmekh-file path="notes.txt"/>VALID'},
    )
    with pytest.raises(AssertionError, match="valid new-file call failed"):
        verify_file_effects(
            [event],
            tmp_path,
            {tmp_path / "notes.txt": None},
            history=creation_history("creation", ok=False),
        )


def test_model_repeated_creation_is_a_valid_rejection(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    args = {"document": '<new-exmekh-file path="notes.txt"/>CREATED'}
    first = ToolEvent(tool_name="new-file", tool_call_id="first", arguments=args)
    repeated = ToolEvent(tool_name="new-file", tool_call_id="repeat", arguments=args)
    history = creation_history("first")
    history.raw_history.extend(creation_history("repeat", ok=False).raw_history)
    path.write_bytes(b"CREATED")
    assert verify_file_effects(
        [first, repeated], tmp_path, {path: None}, history=history
    ) == [first]


def test_batch_deletion_checks_each_receipt_and_directory_descendants(tmp_path):
    child = tmp_path / "folder" / "child.txt"
    event = ToolEvent(
        tool_name="delete-file",
        tool_call_id="batch",
        arguments={"paths": ["folder", "missing"]},
    )
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "tool",
                "tool_call_id": "batch",
                "content": stringify_tool_result(
                    {
                        "ok": False,
                        "results": [
                            {"path": "folder", "ok": True},
                            {"path": "missing", "ok": False},
                        ],
                    }
                ),
            }
        ],
    )
    assert (
        verify_file_effects([event], tmp_path, {child: b"OLD"}, history=message) == []
    )
    child.parent.mkdir()
    child.write_bytes(b"OLD")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects([event], tmp_path, {child: b"OLD"}, history=message)


def test_creation_preserves_successful_entries_when_another_block_is_rejected(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    path = folder / "notes.txt"
    path.write_bytes(b"CREATED")
    rejected = tmp_path / "short.txt"
    event = ToolEvent(
        tool_name="new-file",
        tool_call_id="batch-creation",
        arguments={
            "document": '<new-exmekh-dir path="folder"/>'
            '<new-exmekh-file path="folder/notes.txt"/>CREATED'
            '<new-exmekh-file path="short.txt"/>x'
        },
    )
    history = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "tool",
                "tool_call_id": "batch-creation",
                "content": stringify_tool_result(
                    {
                        "ok": False,
                        "results": [
                            {"index": 1, "path": "folder", "ok": True},
                            {"index": 2, "path": "folder/notes.txt", "ok": True},
                            {"index": 3, "path": "short.txt", "ok": False},
                        ],
                    }
                ),
            }
        ],
    )
    assert (
        verify_file_effects(
            [event], tmp_path, {path: None, rejected: None}, history=history
        )
        == []
    )
    path.write_bytes(b"CORRUPT")
    with pytest.raises(AssertionError, match="file bytes differ"):
        verify_file_effects(
            [event], tmp_path, {path: None, rejected: None}, history=history
        )
