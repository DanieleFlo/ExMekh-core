from __future__ import annotations

import html
from copy import deepcopy

import pytest

from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.shared.output_format import (
    format_output_html,
    format_raw_output_text,
)


def output_for_call(root, tree, call_id):
    return ChatMessage(
        role="assistant", raw_history=root, skill_history_tree=tree
    ).output_for_call(call_id)


def test_outputs_correlate_ids_in_root_nested_and_empty_results():
    root = [
        {"role": "assistant", "tool_call_id": "wrong", "content": "Not output"},
        {"role": "tool", "tool_call_id": "first", "content": None},
        {"role": "tool", "tool_call_id": "first", "content": "Correct result"},
    ]
    tree = [
        {
            "history": None,
            "children": [
                {
                    "history": [
                        {
                            "role": "tool",
                            "tool_call_id": "inner",
                            "content": {"answer": 42},
                        }
                    ]
                }
            ],
        }
    ]
    assert output_for_call(root, tree, "first") == "Correct result"
    assert output_for_call(root, tree, "inner") == '{\n  "answer": 42\n}'
    assert output_for_call(root, tree, "wrong") is None
    assert output_for_call(root, tree, None) is None
    assert output_for_call([], [], "absent") is None
    assert (
        output_for_call(
            [{"role": "tool", "tool_call_id": "empty", "content": ""}], [], "empty"
        )
        == ""
    )


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "plain\ntext",
        '{"items":[1,2]}',
        "<result><line>value</line></result>",
        "<unclosed>",
        '<script>alert("bad")</script>',
        "<header><name>x</name><type>Tool</type><id>one</id><thinking>reason\nmore</thinking></header>\nresult",
        "<header><name>x</name><id>two</id></header>\nresult",
    ],
)
def test_readable_outputs_never_insert_untrusted_html(text):
    result = format_output_html(text)
    assert "<script>" not in result
    if text and "<script>" in text:
        assert "&lt;script&gt;" in result
    if text and "<thinking>" in text:
        assert "output-thinking" in result and "reason\nmore" in result
    if text == '{"items":[1,2]}':
        assert '\n  "items": [' in html.unescape(result)
    if text == "<result><line>value</line></result>":
        assert "\n  &lt;line&gt;" in result


def test_raw_multiline_strings_and_single_line_headers_use_readable_output():
    text = "first line\nsecond line"
    formatted = format_json_to_highlighted_html({"content": text})
    assert "json-readable-string" in formatted and text in formatted
    header = "<header><name>read</name><id>abc</id></header>result"
    assert "&lt;header&gt;" in format_json_to_highlighted_html(header)
    assert "output-header" not in format_json_to_highlighted_html(header)


def test_legacy_function_outputs_use_the_same_invocation_lookup():
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "FUNCTION", "tool_call_id": "legacy", "content": "Legacy result"},
            {"role": "tool", "content": "No invocation identity"},
        ],
    )
    assert message.output_for_call("legacy") == "Legacy result"
    assert message.output_for_call(None) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_archived_read_tool_output_keeps_original_header_and_matches_summary_mode(
    reverse,
):
    calls = [
        {
            "id": mode,
            "function": {
                "name": "read_tool_output",
                "arguments": {"output_id": "cached", "summary_mode": mode},
            },
        }
        for mode in ("none", "brief", "detailed")
    ]
    outputs = [
        {
            "role": "tool",
            "content": (
                "<header><name>read-web-page</name><type>Tool</type><id>cached</id>"
                + (f"<summary-mode>{mode}</summary-mode>" if mode != "none" else "")
                + "</header>\n"
                + mode.upper()
                + " RESULT"
            ),
        }
        for mode in ("none", "brief", "detailed")
    ]
    history = [
        {"role": "assistant", "tool_calls": calls},
        *(reversed(outputs) if reverse else outputs),
    ]
    before = deepcopy(history)
    message = ChatMessage(role="assistant", raw_history=history)
    for mode in ("none", "brief", "detailed"):
        assert message.output_for_call(mode).endswith(mode.upper() + " RESULT")
    assert history == before
    assert all("tool_call_id" not in output for output in outputs)


def test_cached_result_with_other_id_is_not_assigned_to_read_tool_output():
    message = ChatMessage(
        role="assistant",
        raw_history=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "read",
                        "function": {
                            "name": "read_tool_output",
                            "arguments": {"output_id": "wanted"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "<header><name>origin</name><id>different</id></header>Other output",
            },
        ],
    )
    assert message.output_for_call("read") is None


def test_raw_header_keeps_xml_args_thinking_and_body_without_metadata_cards():
    output = (
        "<header><name>origin</name><id>cached</id><args><url>https://example.com</url></args>"
        "<thinking>Recorded thinking</thinking></header>\n"
        '{"answer":42}'
    )
    rendered = format_json_to_highlighted_html({"content": output})
    readable = html.unescape(rendered)
    assert "<header>\n  <name>origin</name>" in readable
    assert "  <args>\n    <url>https://example.com</url>\n  </args>" in readable
    assert "<thinking>Recorded thinking</thinking>" in readable
    assert '</header>\n{\n  "answer": 42\n}' in readable
    assert "output-header" not in rendered and "output-thinking" not in rendered
    assert "output-header" in format_output_html(output)


@pytest.mark.parametrize(
    "text",
    [
        "<header><name>incomplete",
        "<header><name>broken</header>\nUnchanged body",
    ],
)
def test_raw_streaming_or_malformed_xml_header_is_preserved(text):
    assert format_raw_output_text(text) == text
