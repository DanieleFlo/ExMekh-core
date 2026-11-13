from __future__ import annotations

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    SkillNode,
    ToolEvent,
    format_duration,
)


def test_turn_stream_collector_token_accumulation() -> None:
    collector = TurnStreamCollector()
    collector.append_token("Hello ")
    collector.append_token("world!")
    assert collector.get_final_content() == "Hello world!"
    assert collector.get_final_content("fallback") == "Hello world!"


def test_turn_stream_collector_fallback_content() -> None:
    collector = TurnStreamCollector()
    assert collector.get_final_content("fallback text") == "fallback text"


def test_final_response_without_deltas_follows_intermediate_text_and_tool() -> None:
    collector = TurnStreamCollector()
    intermediate = collector.append_token("Looking up the result.")
    tool = ToolEvent(tool_name="lookup")
    collector.append_tool(tool)
    message = collector.build_message(response="Here is the result.")
    progress, lookup, answer = message.execution_timeline(
        include_stream=True
    ).root_activity()
    assert progress.id == intermediate.id
    assert lookup.id == tool.id
    assert answer.content == "Here is the result."
    assert message.content == answer.content
    assert collector.snapshot_message().content == intermediate.content


def test_final_response_reconciles_only_the_last_stream_block() -> None:
    collector = TurnStreamCollector()
    collector.append_token("Looking up the result.")
    collector.append_tool(ToolEvent(tool_name="lookup"))
    partial = collector.append_token("Partial answer")
    message = collector.build_message(response="Complete answer.")
    assert [(block.id, block.content) for block in message.content_blocks][-1] == (
        partial.id,
        "Complete answer.",
    )
    assert message.content_blocks[0].content == "Looking up the result."
    assert partial.content == "Partial answer"


def test_response_token_finishes_current_reasoning_step() -> None:
    collector = TurnStreamCollector()
    first = collector.append_reasoning("First thought", skill_path="/")
    collector.append_token("Partial answer")
    second = collector.append_reasoning("Another thought", skill_path="/")

    assert second is not first
    assert [block.content for block in collector.reasoning_blocks] == [
        "First thought",
        "Another thought",
    ]


def test_stream_order_survives_snapshot_and_reload() -> None:
    collector = TurnStreamCollector()
    plan = collector.append_reasoning("Plan")
    tool = ToolEvent(
        tool_name="lookup", tool_call_id="lookup-1", event_type="tool_start"
    )
    collector.append_tool(tool)
    follow_up = collector.append_reasoning("Follow-up")
    collector.append_tool(tool.model_copy(update={"event_type": "tool_end"}))
    assert collector.current_reasoning_block is follow_up
    before_skill = collector.append_token("Starting the skill. ")

    skill = SkillNode(name="review", path="/review", depth=1)
    collector.append_skill(skill)
    skill_plan = collector.append_reasoning(
        "Review plan", skill_path=skill.path, depth=1
    )
    skill_text = collector.append_token("Review ", skill.path, 1)
    collector.append_skill(skill, "updated")
    assert collector.append_token("complete.", skill.path, 1) is skill_text
    nested_tool = ToolEvent(tool_name="check", skill_path=skill.path, depth=1)
    skill.add_tool_event(nested_tool)
    collector.append_tool(nested_tool)
    collector.append_skill(skill, "completed")
    final_text = collector.append_token("All done.")

    messages = [
        collector.snapshot_message(),
        ChatMessage.model_validate_json(collector.build_message().model_dump_json()),
    ]
    for message in messages:
        timeline = message.execution_timeline(include_stream=True)
        assert [activity.id for activity in timeline.root_activity()] == [
            plan.id,
            tool.id,
            follow_up.id,
            before_skill.id,
            skill.id,
            final_text.id,
        ]
        assert [activity.id for activity in timeline.skill_activity(skill)] == [
            skill_plan.id,
            skill_text.id,
            nested_tool.id,
        ]
        assert message.content == "Starting the skill. All done."
        assert message.content_blocks[1].content == "Review complete."


def test_turn_stream_collector_reasoning_segmentation() -> None:
    collector = TurnStreamCollector()

    # Block 1: root reasoning
    b1 = collector.append_reasoning("Reasoning 1a ", skill_path="/", depth=0)
    assert b1.generation_index == 1
    assert b1.skill_path == "/"
    assert b1.depth == 0

    # Continuation of Block 1
    b1_cont = collector.append_reasoning("Reasoning 1b", skill_path="/", depth=0)
    assert b1_cont is b1
    assert b1.content == "Reasoning 1a Reasoning 1b"
    assert len(collector.reasoning_blocks) == 1

    # Block 2: transition to skill path
    b2 = collector.append_reasoning(
        "Skill reasoning", skill_path="/mock-skill", depth=1
    )
    assert b2.generation_index == 2
    assert b2.skill_path == "/mock-skill"
    assert b2.depth == 1
    assert len(collector.reasoning_blocks) == 2

    # Append tool -> resets current block so next reasoning becomes new generation
    tool_ev = ToolEvent(tool_name="test_tool", skill_path="/mock-skill", depth=1)
    collector.append_tool(tool_ev)
    assert len(collector.tools) == 1
    assert collector.current_reasoning_block is None

    # Block 3: next reasoning after tool
    b3 = collector.append_reasoning(
        "After tool reasoning", skill_path="/mock-skill", depth=1
    )
    assert b3.generation_index == 3
    assert len(collector.reasoning_blocks) == 3

    assert "Reasoning 1a" in collector.get_reasoning_content()
    assert "Skill reasoning" in collector.get_reasoning_content()
    assert "After tool reasoning" in collector.get_reasoning_content()


def test_turn_stream_collector_skills() -> None:
    collector = TurnStreamCollector()

    s1 = SkillNode(name="root_skill", path="/root_skill", depth=1)
    s2 = SkillNode(name="sub_skill", path="/root_skill/sub", depth=2)

    collector.append_skill(s2)
    assert collector.skills == []

    collector.append_skill(s1)
    # Duplicate top-level skill
    collector.append_skill(s1)
    collector.append_skill(s2)

    assert len(collector.skills) == 1
    assert collector.skills[0].name == "root_skill"
    assert collector.skill_snapshot(s2.path).id == s2.id


def test_nested_skill_updates_attach_to_the_latest_parent_invocation() -> None:
    collector = TurnStreamCollector()
    first = SkillNode(
        name="parent", path="/parent", depth=1, opening_tool_call_id="first"
    )
    second = first.model_copy(
        update={"id": "parent-2", "opening_tool_call_id": "second"}
    )
    child = SkillNode(
        name="child", path="/parent/child", depth=2, opening_tool_call_id="child"
    )
    collector.append_skill(first)
    collector.append_skill(second)
    collector.append_skill(child)
    assert (
        collector.skill_snapshot(first.path, "first").find_invocation(child.path)
        is None
    )
    assert (
        collector.skill_snapshot(second.path, "second").find_invocation(child.path).id
        == child.id
    )
    assert collector.skill_snapshot(child.path, "missing") is None


def test_turn_stream_collector_build_message() -> None:
    collector = TurnStreamCollector()
    collector.append_token("Response body")
    collector.append_reasoning("Reasoning text")
    ev = ToolEvent(tool_name="cmd", skill_path="/", depth=0)
    collector.append_tool(ev)

    msg = collector.build_message(role="assistant", error="Optional error")
    assert msg.role == "assistant"
    assert msg.content == "Response body"
    assert msg.reasoning_content == "Reasoning text"
    assert len(msg.reasoning_blocks) == 1
    assert msg.reasoning_blocks[0].duration_seconds is not None
    assert len(msg.tool_events) == 1
    assert msg.tool_events[0].duration_seconds is not None
    assert msg.error == "Optional error"


def test_format_duration_seconds_minutes_hours() -> None:
    assert format_duration(None) == "0s"
    assert format_duration(-5) == "0s"
    assert format_duration(0) == "0s"
    assert format_duration(30) == "30s"
    assert format_duration(59) == "59s"
    assert format_duration(60) == "1m 0s"
    assert format_duration(70) == "1m 10s"
    assert format_duration(3599) == "59m 59s"
    assert format_duration(3600) == "1h 0m 0s"
    assert format_duration(7460) == "2h 4m 20s"


def test_turn_stream_collector_tool_start_end_and_skill_timing() -> None:
    collector = TurnStreamCollector()

    # Tool start followed by tool end with same tool_call_id
    start_ev = ToolEvent(
        tool_call_id="call_42",
        event_type="tool_start",
        tool_name="search_tool",
        skill_path="/",
        depth=0,
        timestamp=100.0,
    )
    collector.append_tool(start_ev)
    assert len(collector.tools) == 1
    assert collector.tools[0].status == "running"

    end_ev = ToolEvent(
        tool_call_id="call_42",
        event_type="tool_end",
        tool_name="search_tool",
        skill_path="/",
        depth=0,
        timestamp=170.0,
        end_time=170.0,
    )
    collector.append_tool(end_ev)
    assert len(collector.tools) == 1
    assert collector.tools[0].status == "completed"
    assert collector.tools[0].duration_seconds == 70.0

    # Skill open and close preserves start_time and computes open-to-close duration
    opened_skill = SkillNode(
        name="analysis",
        path="/analysis",
        depth=1,
        status="active",
        start_time=200.0,
    )
    collector.append_skill(opened_skill, action="opened")
    assert len(collector.skills) == 1
    assert collector.skills[0].status == "active"

    completed_skill = SkillNode(
        name="analysis",
        path="/analysis",
        depth=1,
        status="completed",
        start_time=205.0,
        end_time=7660.0,
    )
    collector.append_skill(completed_skill, action="completed")
    assert len(collector.skills) == 1
    assert collector.skills[0].status == "completed"
    assert collector.skills[0].start_time == 200.0
    assert collector.skills[0].duration_seconds == 7460.0
