from __future__ import annotations

from unittest.mock import patch

import pytest
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall

from fe.exmekh_bridge.agent_runner import convert_to_exmekh_messages
from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.converter import raw_dict_to_exmekh_message
from fe.exmekh_bridge.event_handlers import DeepAgentTurnEventHandler
from fe.exmekh_bridge.models import (
    ChatMessage,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from fe.exmekh_bridge.sub_agents import SubAgentEventRouter


@pytest.mark.parametrize("status", ["completed", "error"])
def test_skill_closes_nested_work_and_preserves_existing_results(status: str) -> None:
    completed = SkillNode(status="completed", start_time=1, end_time=3)
    active = SkillNode(
        start_time=2,
        reasoning_blocks=[ReasoningBlock(timestamp=3)],
        tool_events=[ToolEvent(tool_name="nested", status="running", timestamp=4)],
    )
    root = SkillNode(start_time=1, sub_skills=[completed, active])
    root.finalize_timing(status, now=10)
    assert root.status == active.status == status
    assert completed.status == "completed"
    assert completed.end_time == 3
    assert completed.duration_seconds == 2
    assert active.duration_seconds == 8
    assert active.reasoning_blocks[0].duration_seconds == 7
    assert active.tool_events[0].status == status
    assert active.tool_events[0].duration_seconds == 6
    before = root.model_dump()
    root.finalize_timing("error", now=20)
    assert root.model_dump() == before


def test_timing_normalization_does_not_close_active_work() -> None:
    open_tool = ToolEvent(tool_name="open", status="running")
    finished_tool = ToolEvent(tool_name="finished", timestamp=1, end_time=4)
    instant_tool = ToolEvent(tool_name="instant")
    open_block = ReasoningBlock(timestamp=1)
    finished_block = ReasoningBlock(timestamp=1, end_time=4)
    skill = SkillNode(
        tool_events=[open_tool, finished_tool, instant_tool],
        reasoning_blocks=[open_block, finished_block],
    )
    skill.finalize_timing(now=10)
    assert skill.status == "active"
    assert open_tool.end_time is None
    assert open_tool.duration_seconds is None
    assert open_block.end_time is None
    assert finished_tool.duration_seconds == finished_block.duration_seconds == 3
    assert instant_tool.duration_seconds == 0
    with patch("fe.exmekh_bridge.models.time.time", return_value=10):
        open_tool.finalize_timing("error")
        open_block.finalize_timing()
        skill.finalize_timing("completed")
    assert open_tool.status == "error"
    assert open_block.end_time == skill.end_time == 10


def test_message_completion_closes_all_work_using_one_timestamp() -> None:
    message = ChatMessage(
        role="assistant",
        timestamp=1,
        is_streaming=True,
        skills=[SkillNode(start_time=2)],
        reasoning_blocks=[ReasoningBlock(timestamp=3)],
        tool_events=[ToolEvent(tool_name="tool", status="running", timestamp=4)],
    )
    message.finish("error", now=10)
    assert message.duration_seconds == 9
    assert not message.is_streaming
    assert message.skills[0].status == message.tool_events[0].status == "error"
    assert message.reasoning_blocks[0].end_time == 10


def test_handler_failure_publishes_failed_skill_and_pending_root_tool() -> None:
    tools = []
    skills = []
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None),
        lambda _: None,
        lambda *_: None,
        tools.append,
        lambda node, action: skills.append((node, action)),
        None,
    )
    skill = SkillNode()
    tool = ToolEvent(tool_name="root", status="running")
    handler.skill_stack.append(skill)
    handler.active_tools["call"] = tool
    handler.finish("error")
    assert skills == [(skill, "error")]
    assert tools == [tool]
    assert tool.status == skill.status == "error"
    assert not handler.active_tools
    handler.finish("error")
    assert len(tools) == 1


def test_missing_tool_name_matches_id_instead_of_batch_order() -> None:
    pending = [
        ToolCall(id="first", function=FunctionCall(name="first_tool", arguments={})),
        ToolCall(id="second", function=FunctionCall(name="second_tool", arguments={})),
    ]
    message = raw_dict_to_exmekh_message(
        {"role": "tool", "tool_call_id": "second", "content": "result"},
        pending,
    )
    assert message.get_name() == "second_tool"
    assert [call.id for call in pending] == ["first"]


@pytest.mark.parametrize("next_role", ["user", "assistant"])
def test_interrupted_tool_batch_is_closed_before_later_messages(next_role: str) -> None:
    messages = convert_to_exmekh_messages(
        [
            {"role": "user", "content": "first turn"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "interrupted",
                        "function": {"name": "tool", "arguments": {}},
                    },
                ],
            },
            {"role": next_role, "content": "later message"},
        ]
    )
    assert messages[2].get_role_name() == "tool"
    assert messages[2].get_tool_call_id() == "interrupted"
    assert messages[-1].get_message() == "later message"


@pytest.mark.parametrize("status", ["completed", "error"])
def test_handler_finish_updates_collected_tool_without_duplicating_it(
    status: str,
) -> None:
    collector = TurnStreamCollector()
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None),
        collector.append_token,
        collector.append_reasoning,
        collector.append_tool,
        collector.append_skill,
        None,
    )
    event = ToolEvent(tool_name="root", event_type="tool_start", status="running")
    collector.append_tool(event)
    handler.active_tools["call"] = event
    handler.finish(status)
    message = collector.build_message(error="failed" if status == "error" else None)
    assert len(message.tool_events) == 1
    assert message.tool_events[0].status == status
    assert message.tool_events[0].event_type == "tool_end"


@pytest.mark.parametrize("status", ["completed", "error", "running"])
@pytest.mark.parametrize("has_start", [False, True])
def test_tool_end_result_is_consistent_in_root_and_sub_agent(
    status: str, has_start: bool
) -> None:
    collector = TurnStreamCollector()
    conversation = SubAgentConversation(path="main/worker")
    conversation.start_call("task")
    if has_start:
        start = ToolEvent(
            tool_name="tool",
            tool_call_id="call",
            event_type="tool_start",
            timestamp=1,
        )
        conversation.add_tool_event(start)
        collector.append_tool(start)
    end = ToolEvent(
        tool_name="tool",
        tool_call_id="call",
        event_type="tool_end",
        status=status,
        timestamp=3,
    )
    conversation.add_tool_event(end)
    collector.append_tool(end)
    event = conversation.messages[-1].tool_events[0]
    assert len(conversation.messages[-1].tool_events) == 1
    assert event.status == ("error" if status == "error" else "completed")
    assert event.duration_seconds == (2 if has_start else 0)
    assert len(collector.tools) == 1
    assert collector.tools[0].model_dump() == event.model_dump()
