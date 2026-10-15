from pathlib import Path

import pytest
from exmekh_lib.utils.xml import stringify_tool_result

from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent
from tests.live_file_support import verify_file_effects


def test_frontend_observation_uses_emitted_arguments(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    event = ToolEvent(
        tool_name="new-file", arguments={"path": "notes.txt", "content": "MODEL_CHOICE"}
    )
    path.write_bytes(b"MODEL_CHOICE")
    assert verify_file_effects([event], tmp_path, {path: None}) == [event]
    path.write_bytes(b"CORRUPT")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects([event], tmp_path, {path: None})


def test_frontend_valid_creation_failure_remains_a_bug(tmp_path: Path) -> None:
    event = ToolEvent(
        tool_name="new-file",
        status="error",
        arguments={"path": "notes.txt", "content": "VALID"},
    )
    with pytest.raises(AssertionError, match="valid new-file call failed"):
        verify_file_effects([event], tmp_path, {tmp_path / "notes.txt": None})


def test_model_repeated_creation_is_a_valid_rejection(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    args = {"path": "notes.txt", "content": "CREATED"}
    first = ToolEvent(tool_name="new-file", tool_call_id="first", arguments=args)
    repeated = ToolEvent(
        tool_name="new-file", tool_call_id="repeat", status="error", arguments=args
    )
    path.write_bytes(b"CREATED")
    assert verify_file_effects([first, repeated], tmp_path, {path: None}) == [first]


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
