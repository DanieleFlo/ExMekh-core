"""Restore steering IDs in their own skill scope without duplicating descendants."""

from __future__ import annotations

from pathlib import Path

import pytest
from qs.core.cache.session_cache import SessionCache
from qs.core.tool_calls.execution_history import ExecutionHistory

from fe.qs_bridge.converter import reconstruct_execution_history_and_cache
from fe.qs_bridge.models import ChatMessage
from fe.qs_bridge.recorded_output import RecordedToolOutput
from fe.qs_bridge.steering_history import direct_steering_output_ids


def header(output_id: str, references: list[tuple[str, str]]) -> str:
    items = "".join(
        f"<item><name>{name}</name><id>{ref}</id></item>" for name, ref in references
    )
    return (
        f"<header><name>skill</name><type>DeepSkill</type><id>{output_id}</id>"
        f'<used-tools type="array">{items}</used-tools></header>\ndone'
    )


def opening(name: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": {}}},
        ],
    }


@pytest.mark.parametrize(
    "output",
    [
        "legacy output",
        "<header><used-tools></header>",
        "<header><used-tools><item><name>x</name></item></used-tools></header>",
        "<header><id>old</id></header>",
    ],
)
def test_missing_or_invalid_references_do_not_invent_steering_ids(output: str) -> None:
    parsed = RecordedToolOutput.parse(output)
    assert parsed is None or parsed.used_tools is None
    assert direct_steering_output_ids(parsed, []) == []


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("nested_output_from_reply", [False, True])
def test_parent_and_nested_steering_keep_order_ids_and_scope(
    tmp_path: Path, use_cache: bool, nested_output_from_reply: bool
) -> None:
    nested_refs = [("steering_message", "nested-guidance")]
    parent_refs = [
        ("steering_message", "first-guidance"),
        *nested_refs,
        ("steering_message", "last-guidance"),
    ]
    nested_output = header("nested-output", nested_refs)
    nested = {
        "skill": "nested",
        "opening_tool_call_id": "open-nested",
        "history": [
            opening("nested", "open-nested"),
            {"role": "tool", "content": "opened"},
            {"role": "user", "content": "Repeated text"},
        ],
        "children": [],
        "output": "" if nested_output_from_reply else nested_output,
    }
    parent_output = header("parent-output", parent_refs)
    parent = {
        "skill": "parent",
        "opening_tool_call_id": "open-parent",
        "history": [
            opening("parent", "open-parent"),
            {"role": "tool", "content": "opened"},
            {"role": "user", "content": "Repeated text"},
            opening("nested", "open-nested"),
            {"role": "tool", "tool_call_id": "open-nested", "content": nested_output},
            {"role": "user", "content": "Repeated text"},
        ],
        "children": [nested],
        "output": parent_output,
    }
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "user", "content": "Root guidance stays in history"},
            opening("parent", "open-parent"),
            {"role": "tool", "tool_call_id": "open-parent", "content": parent_output},
        ],
        skill_history_tree=[parent],
    )
    session_cache = SessionCache(tmp_path)
    try:
        cache = session_cache.node("root") if use_cache else None
        history = ExecutionHistory()
        reconstruct_execution_history_and_cache([message], history, cache)
        assert [(e.name, e.call_id) for e in history.entries] == [
            ("parent", "open-parent"),
            ("steering_message", "first-guidance"),
            ("nested", "open-nested"),
            ("steering_message", "nested-guidance"),
            ("steering_message", "last-guidance"),
        ]
        if cache is not None:
            for output_id, scope in [
                ("first-guidance", "/parent"),
                ("nested-guidance", "/parent/nested"),
                ("last-guidance", "/parent"),
            ]:
                record = cache.get(output_id)
                assert record.source == "UserMessage"
                assert record.skill_path == scope
                assert record.result == "Repeated text"
                assert record.user_message["parts"] == [
                    {"type": "text", "text": "Repeated text"}
                ]
            assert [
                (r.name, r.id) for r in cache.get("parent-output").used_tools
            ] == parent_refs
    finally:
        session_cache.close()


def test_a_saved_read_with_missing_original_guidance_does_not_fabricate_it(
    tmp_path: Path,
) -> None:
    session_cache = SessionCache(tmp_path)
    try:
        cache = session_cache.node("root")
        message = ChatMessage(
            role="assistant",
            raw_history=[
                opening("read_tool_output", "read"),
                {
                    "role": "tool",
                    "tool_call_id": "read",
                    "content": "<header><name>steering_message</name><type>UserMessage</type><id>missing</id></header>text",
                },
            ],
        )
        history = ExecutionHistory()
        reconstruct_execution_history_and_cache([message], history, cache)
        assert history.count == 1
        assert cache.get("missing") is None
    finally:
        session_cache.close()


def test_rereading_a_tool_preserves_its_original_invocation_metadata(
    tmp_path: Path,
) -> None:
    output = "<header><name>work</name><type>Tool</type><id>original-output</id></header>result"
    call = opening("work", "original-call")
    call["tool_calls"][0]["function"]["arguments"] = {"path": "report.txt"}
    message = ChatMessage(
        role="assistant",
        raw_history=[
            call,
            {"role": "tool", "tool_call_id": "original-call", "content": output},
            opening("read_tool_output", "read-call"),
            {"role": "tool", "tool_call_id": "read-call", "content": output},
        ],
    )
    session_cache = SessionCache(tmp_path)
    try:
        cache = session_cache.node("root")
        history = ExecutionHistory()
        reconstruct_execution_history_and_cache([message], history, cache)
        record = cache.get("original-output")
        assert record.name == "work"
        assert record.tool_call_id == "original-call"
        assert record.args == {"path": "report.txt"}
        assert [entry.saved_output_id for entry in history.entries] == [
            "original-output",
            "original-output",
        ]
    finally:
        session_cache.close()


@pytest.mark.parametrize(
    "references",
    [
        "<item><name>work</name></item>",
        "<item><name> </name><id>output</id></item>",
    ],
)
def test_malformed_references_do_not_discard_a_valid_output(references: str) -> None:
    output = RecordedToolOutput.parse(
        f"<header><id>saved</id><used-tools>{references}</used-tools></header>body"
    )
    assert output is not None
    assert output.id == "saved"
    assert output.body == "body"
    assert output.used_tools is None


def test_an_explicit_empty_reference_list_remains_distinct_from_legacy_metadata() -> (
    None
):
    output = RecordedToolOutput.parse(
        "<header><id>saved</id><used-tools /></header>\nbody"
    )
    assert output == RecordedToolOutput("", "Tool", "saved", "body", ())


def test_a_skill_read_without_original_history_retains_the_output_identity(
    tmp_path: Path,
) -> None:
    output = header("saved-skill", [("work", "work-output")])
    message = ChatMessage(
        role="assistant",
        raw_history=[
            opening("read_tool_output", "read-call"),
            {"role": "tool", "tool_call_id": "read-call", "content": output},
        ],
    )
    session_cache = SessionCache(tmp_path)
    try:
        cache = session_cache.node("root")
        history = ExecutionHistory()
        reconstruct_execution_history_and_cache([message], history, cache)
        record = cache.get("saved-skill")
        assert record.name == "skill"
        assert record.source == "DeepSkill"
        assert record.result == {"text": "done"}
        assert record.tool_call_id is None
        assert record.args == {}
        assert [(ref.name, ref.id) for ref in record.used_tools] == [
            ("work", "work-output")
        ]
        assert history.entries[0].recoverable_output_id(cache) == "saved-skill"
    finally:
        session_cache.close()
