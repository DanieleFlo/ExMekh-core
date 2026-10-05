from __future__ import annotations

from pathlib import Path

import pytest
from exmekh_lib.core.agents.exit_tool.data import ExitToolData
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.models.tool import Tool

from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    TOOL_APPROVAL_ALL,
    CommandApprovalHook,
)
from tests.test_exit_agent_misuse_live import SkillExecutionTrace


def finish(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"text": "attempt"}, "missing a required argument"),
        ({"text": "attempt", "items": "abc"}, "items must be a list"),
    ],
)
def test_exmekh_exit_tool_rejects_exact_malformed_arguments_before_execution(
    tmp_path: Path, arguments: dict, error: str
) -> None:
    tool = ExitTool(
        finish,
        name="exit_agent",
        description="Finish an agent turn",
        instructions_path=tmp_path / "instructions.md",
        instructions="Return a non-empty answer and the selected output IDs.",
        folder_path=tmp_path,
    )
    with pytest.raises(TypeError, match=error):
        tool.execute_with_hooks(arguments, work_root=tmp_path, history=())
    assert tool.execute_with_hooks(
        {"text": "Recovered", "items": []}, work_root=tmp_path, history=()
    ) == {"text": "Recovered", "items": []}


def test_exmekh_tool_executes_only_after_command_approval(tmp_path: Path) -> None:
    executed: list[str] = []

    def run_command(command: str) -> str:
        executed.append(command)
        return "executed"

    declined = Tool(
        run_command,
        name="bash",
        before=CommandApprovalHook(request_manual_approval=lambda _: False),
    )
    result = declined.execute_with_hooks(
        {"command": "git status"}, work_root=tmp_path, history=()
    )
    assert "declined" in result.lower()
    assert executed == []

    approved = Tool(
        run_command, name="bash", before=CommandApprovalHook(mode=TOOL_APPROVAL_ALL)
    )
    assert (
        approved.execute_with_hooks(
            {"command": "git status"}, work_root=tmp_path, history=()
        )
        == "executed"
    )
    assert executed == ["git status"]


@pytest.mark.parametrize(
    ("case", "arguments"),
    [
        ("missing_items", {"text": "attempt"}),
        ("wrong_items_type", {"text": "attempt", "items": "abc"}),
        ("unknown_output_id", {"text": "attempt", "items": ["id-inesistente-xyz"]}),
        ("duplicate_output_id", {"text": "attempt", "items": ["saved", "saved"]}),
        (
            "placeholder_without_item",
            {"text": "attempt {{id-inesistente-xyz}}", "items": []},
        ),
    ],
)
def test_live_trace_requires_the_requested_malformed_call(
    case: str, arguments: dict
) -> None:
    trace = SkillExecutionTrace(
        history=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "exit_agent", "arguments": arguments}}
                ],
            }
        ],
        output=None,
    )
    assert trace.has_requested_misuse(case)
    valid_trace = SkillExecutionTrace(
        history=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exit_agent",
                            "arguments": {"text": "Recovered", "items": []},
                        }
                    }
                ],
            }
        ],
        output="Recovered",
    )
    assert not valid_trace.has_requested_misuse(case)
