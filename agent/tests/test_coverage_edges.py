from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.system_message import SystemMessage

from fe.config import AppConfig
from fe.exmekh_bridge.agent_runner import convert_to_exmekh_messages
from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.exmekh_bridge.converter import (
    chat_message_to_exmekh_messages,
    exmekh_messages_to_session,
    session_to_exmekh_messages,
)
from fe.exmekh_bridge.event_handlers import DeepAgentTurnEventHandler
from fe.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    ReasoningBlock,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from fe.exmekh_bridge.sub_agents import ObservedNode, SubAgentEvent, SubAgentEventRouter


def test_save_to_env_recovers_when_existing_file_cannot_be_read(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("BASE_URL=old\n", encoding="utf-8")
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == env_path:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    with patch.object(Path, "read_text", read_text):
        AppConfig(base_url="new").save_to_env(env_path)

    assert env_path.read_text(encoding="utf-8").startswith("BASE_URL=new\n")


@pytest.mark.parametrize("operation", ["write", "replace"])
def test_save_to_env_cleans_temporary_file_after_write_error(
    tmp_path: Path, operation: str
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("CUSTOM=preserved\n", encoding="utf-8")
    temporary_path = env_path.with_suffix(".env.tmp")
    method = "write_text" if operation == "write" else "replace"
    original = getattr(Path, method)

    def fail_for_temporary(path: Path, *args: object, **kwargs: object):
        if path == temporary_path:
            raise OSError("disk error")
        return original(path, *args, **kwargs)

    with patch.object(Path, method, fail_for_temporary):
        AppConfig().save_to_env(env_path)

    assert env_path.read_text(encoding="utf-8") == "CUSTOM=preserved\n"
    assert not temporary_path.exists()


def test_skill_finalization_closes_nested_timing() -> None:
    child = SkillNode(
        name="child",
        path="/root/child",
        depth=2,
        status="completed",
        start_time=10,
        end_time=15,
    )
    root = SkillNode(
        name="root",
        path="/root",
        depth=1,
        status="completed",
        start_time=1,
        end_time=20,
    )
    root.reasoning_blocks.append(ReasoningBlock(content="thinking", timestamp=2))
    root.tool_events.extend(
        [
            ToolEvent(tool_name="running", status="running", timestamp=3),
            ToolEvent(
                tool_name="finished", status="completed", timestamp=4, end_time=8
            ),
            ToolEvent(tool_name="instant", status="completed", timestamp=9),
        ]
    )
    root.sub_skills.append(child)

    root.finalize_timing()

    assert root.duration_seconds == 19
    assert root.reasoning_blocks[0].end_time == 20
    assert root.reasoning_blocks[0].duration_seconds == 18
    assert root.tool_events[0].status == "completed"
    assert root.tool_events[0].duration_seconds == 17
    assert root.tool_events[1].duration_seconds == 4
    assert root.tool_events[2].duration_seconds == 0
    assert child.duration_seconds == 5


def test_tool_end_without_matching_start_is_preserved() -> None:
    collector = TurnStreamCollector()
    collector.append_tool(
        ToolEvent(
            tool_name="other", event_type="tool_start", tool_call_id="a", timestamp=1
        )
    )
    collector.append_tool(
        ToolEvent(
            tool_name="target", event_type="tool_end", tool_call_id="b", timestamp=3
        )
    )
    assert len(collector.tools) == 2
    assert collector.tools[-1].status == "completed"
    assert collector.tools[-1].duration_seconds == 0


def test_tool_end_matches_by_name_without_call_id() -> None:
    collector = TurnStreamCollector()
    collector.append_tool(
        ToolEvent(tool_name="target", event_type="tool_start", timestamp=1)
    )
    collector.append_tool(
        ToolEvent(
            tool_name="target", event_type="tool_end", timestamp=4, duration_seconds=2
        )
    )
    assert len(collector.tools) == 1
    assert collector.tools[0].duration_seconds == 2


def test_skill_update_replaces_observed_nested_content() -> None:
    collector = TurnStreamCollector()
    old = SkillNode(name="root", path="/root", depth=1, start_time=1)
    collector.append_skill(old)
    replacement = SkillNode(
        name="root",
        path="/root",
        depth=1,
        status="completed",
        start_time=2,
        end_time=10,
    )
    replacement.reasoning_blocks.append(ReasoningBlock(content="reason"))
    replacement.tool_events.append(ToolEvent(tool_name="tool"))
    replacement.sub_skills.append(SkillNode(name="child", path="/root/child", depth=2))
    collector.append_skill(replacement, action="completed")
    collected = collector.skill_snapshot("/root")
    assert collected.id == old.id
    assert collected.start_time == 1
    assert collected.reasoning_blocks[0].content == "reason"
    assert collected.tool_events[0].tool_name == "tool"
    assert collected.sub_skills[0].name == "child"
    assert old.status == "active" and not old.sub_skills


def test_build_message_marks_unfinished_tool_as_error() -> None:
    collector = TurnStreamCollector()
    collector.append_tool(ToolEvent(tool_name="tool", event_type="tool_start"))
    message = collector.build_message(error="failed")
    assert message.tool_events[0].status == "error"
    assert message.tool_events[0].end_time is not None
    assert message.tool_events[0].duration_seconds is not None


def test_sub_agent_tool_events_finish_and_fail() -> None:
    conversation = SubAgentConversation(path="main/child")
    conversation.start_call("first")
    conversation.add_tool_event(
        ToolEvent(
            tool_name="compile", event_type="tool_start", tool_call_id="a", timestamp=1
        )
    )
    conversation.add_tool_event(
        ToolEvent(
            tool_name="compile",
            event_type="tool_end",
            tool_call_id="a",
            timestamp=4,
            end_time=4,
        )
    )
    conversation.add_tool_event(
        ToolEvent(
            tool_name="orphan",
            event_type="tool_end",
            tool_call_id="missing",
            timestamp=5,
        )
    )
    conversation.add_tool_event(
        ToolEvent(tool_name="pending", event_type="tool_start", timestamp=6)
    )
    conversation.complete_call("done", [{"role": "assistant"}])
    assert conversation.status == "completed"
    assert [event.status for event in conversation.messages[-1].tool_events] == [
        "completed"
    ] * 3
    assert conversation.messages[-1].tool_events[0].duration_seconds == 3
    assert conversation.messages[-1].tool_events[1].duration_seconds == 0
    assert conversation.messages[-1].tool_events[2].duration_seconds is not None
    assert conversation.raw_history == [{"role": "assistant"}]

    conversation.start_call("second")
    conversation.add_tool_event(ToolEvent(tool_name="compile", event_type="tool_start"))
    conversation.fail_call()
    assert conversation.status == "error"
    assert conversation.messages[-1].tool_events[0].status == "error"


def test_sub_agent_tool_matching_by_name_and_unmatched_execution() -> None:
    conversation = SubAgentConversation(path="main/child")
    conversation.start_call("task")
    conversation.add_tool_event(
        ToolEvent(tool_name="read", event_type="tool_start", timestamp=1)
    )
    conversation.add_tool_event(
        ToolEvent(
            tool_name="read", event_type="tool_end", timestamp=3, duration_seconds=1
        )
    )
    conversation.add_tool_event(ToolEvent(tool_name="exec", end_time=8, timestamp=4))
    conversation.add_tool_event(ToolEvent(tool_name="instant", timestamp=4))
    assert [
        event.duration_seconds for event in conversation.messages[-1].tool_events
    ] == [1, 4, 0]


def test_sub_agent_identity_change_is_rejected() -> None:
    session = ChatSession()
    session.start_sub_agent_conversation("same", "main/child", True, "first")
    session.find_sub_agent_conversation("same").complete_call("done", [])
    with pytest.raises(ValueError, match="changed identity"):
        session.start_sub_agent_conversation("same", "main/other", True, "second")
    with pytest.raises(ValueError, match="changed identity"):
        session.start_sub_agent_conversation("same", "main/child", False, "second")


def test_sub_agent_router_tracks_active_child_and_detects_bad_stack() -> None:

    observed = []
    router = SubAgentEventRouter(observed.append)
    first = MagicMock()
    second = MagicMock()
    assert router.route(MagicMock()) is False
    router.enter(first)
    assert router.route("event") is True
    first.forward_stream_event.assert_called_once_with("event")
    router.enter(second)
    with pytest.raises(RuntimeError, match="out of order"):
        router.leave(first)
    assert router.route("second is still active") is True
    second.forward_stream_event.assert_called_once_with("second is still active")
    router.leave(second)
    router.leave(first)
    router.publish("published")
    assert observed == ["published"]


def test_observed_child_reports_failure_and_resets_active_call() -> None:

    observed = []
    router = SubAgentEventRouter(observed.append)
    child = ObservedNode(
        name_hds="child",
        path="main/child",
        persistent_history=False,
        event_router=router,
    )
    with patch.object(Node, "run", side_effect=RuntimeError("child failed")):
        with pytest.raises(RuntimeError, match="child failed"):
            child.run("task")
    assert [event.action for event in observed] == ["started", "error"]
    assert router.route(MagicMock()) is False
    assert child._active_call_id is None


def test_finalize_active_skill_preserves_live_timing() -> None:
    skill = SkillNode(name="active", status="active")
    skill.reasoning_blocks.append(ReasoningBlock(content="live"))
    skill.tool_events.append(ToolEvent(tool_name="running", status="running"))
    skill.finalize_timing()
    assert skill.end_time is None
    assert skill.reasoning_blocks[0].end_time is None
    assert skill.tool_events[0].status == "running"
    assert skill.tool_events[0].duration_seconds is None


def test_finalize_completed_skill_supplies_missing_end_time() -> None:
    skill = SkillNode(name="done", status="completed", start_time=0)
    skill.finalize_timing()
    assert skill.end_time is not None
    assert skill.duration_seconds == skill.end_time


def test_collector_finds_nested_skill_and_keeps_existing_block_timing() -> None:
    collector = TurnStreamCollector()
    child = SkillNode(name="child", path="/root/child", depth=2)
    root = SkillNode(name="root", path="/root", depth=1, sub_skills=[child])
    collector.append_skill(root)
    replacement = SkillNode(
        name="child", path="/root/child", depth=2, status="completed", end_time=10
    )
    collector.append_skill(replacement)
    collected = collector.skill_snapshot("/root/child")
    assert collected.status == "completed"
    assert collected.end_time == 10
    assert child.status == "active" and child.end_time is None

    block = collector.append_reasoning("thinking")
    block.end_time = block.timestamp + 1
    block.duration_seconds = 1
    collector.append_token("answer")
    assert block.duration_seconds == 1


def test_collector_preserves_supplied_tool_duration_and_end_time() -> None:
    collector = TurnStreamCollector()
    collector.append_tool(
        ToolEvent(
            tool_name="orphan",
            event_type="tool_end",
            timestamp=1,
            end_time=3,
            duration_seconds=2,
        )
    )
    collector.append_tool(
        ToolEvent(tool_name="execution", timestamp=4, end_time=6, duration_seconds=2)
    )
    assert [event.duration_seconds for event in collector.tools] == [2, 2]


def test_collector_finalizes_manually_added_unfinished_events() -> None:
    collector = TurnStreamCollector()
    collector.reasoning_blocks.append(ReasoningBlock(content="reason"))
    collector.tools.append(ToolEvent(tool_name="event", status="completed"))
    message = collector.build_message()
    assert message.reasoning_blocks[0].end_time is not None
    assert message.reasoning_blocks[0].duration_seconds is not None
    assert message.tool_events[0].duration_seconds == 0


def test_collector_preserves_skill_duration_and_updates_live_skill() -> None:
    finished = SkillNode(
        name="done",
        path="/done",
        depth=1,
        status="completed",
        start_time=1,
        end_time=3,
        duration_seconds=2,
    )
    finished.finalize_timing()
    assert finished.duration_seconds == 2

    collector = TurnStreamCollector()
    collector.append_skill(SkillNode(name="live", path="/live", depth=1))
    collector.append_skill(SkillNode(name="live", path="/live", depth=1))
    collector.append_skill(finished)
    collector.set_raw_history([{"role": "assistant"}], [{"skill": "done"}])
    message = collector.build_message()
    assert len(message.skills) == 2
    assert message.skills[0].status == "completed"
    assert message.skills[1].duration_seconds == 2
    assert message.raw_history == [{"role": "assistant"}]
    assert message.skill_history_tree == [{"skill": "done"}]


def test_observed_child_restores_only_persistent_history() -> None:
    conversation = SubAgentConversation(
        id="prior",
        path="main/child",
        messages=[ChatMessage(role="user", content="Earlier")],
    )
    router = SubAgentEventRouter(None)
    fresh = ObservedNode(
        name_hds="child",
        path="main/child",
        persistent_history=False,
        event_router=router,
    )
    persistent = ObservedNode(
        name_hds="child",
        path="main/child",
        persistent_history=True,
        event_router=router,
    )
    prior_exmekh = convert_to_exmekh_messages(conversation.replay_history())
    with patch("fe.exmekh_bridge.sub_agents.restore_node_history") as restore:
        fresh.restore_conversation(conversation, prior_exmekh=prior_exmekh)
        persistent.restore_conversation(conversation, prior_exmekh=[])
        persistent.restore_conversation(conversation, prior_exmekh=prior_exmekh)
    restore.assert_called_once_with(
        persistent, conversation.message_history(), prior_exmekh=prior_exmekh
    )
    assert persistent._conversation_id == "prior"
    assert fresh._conversation_id != "prior"


def test_observed_child_forwards_provider_and_tool_events() -> None:
    published = []
    node = ObservedNode(
        name_hds="child",
        path="main/child",
        persistent_history=True,
        event_router=SubAgentEventRouter(published.append),
    )
    response = DeepAgentStreamEvent(
        provider_event=ProviderStreamEvent(
            type=ProviderStreamEventType.RESPONSE, content="answer"
        ),
        depth=1,
        skill_path="/child",
    )
    node.forward_stream_event(response)
    assert published == []
    node._active_call_id = "call"
    node.forward_stream_event(response)
    node.forward_stream_event(
        DeepAgentStreamEvent(
            provider_event=ProviderStreamEvent(
                type=ProviderStreamEventType.REASONING, content="thinking"
            ),
            depth=1,
            skill_path="/child",
        )
    )
    other = MagicMock(spec=DeepAgentStreamEvent)
    other.provider_event = ProviderStreamEvent(type=ProviderStreamEventType.END)
    node.forward_stream_event(other)
    tool = MagicMock(spec=DeepAgentExecutionEvent)
    tool.type = "tool_start"
    tool.tool_call_id = "tool-call"
    tool.tool_name = "read"
    tool.depth = 1
    tool.skill_path = "/child"
    node.forward_stream_event(tool)
    node.forward_stream_event(object())
    assert [event.action for event in published] == ["token", "reasoning", "tool"]
    assert published[0].token == "answer"
    assert published[1].token == "thinking"
    assert published[2].tool_event.status == "running"


def test_router_without_callback_accepts_event() -> None:
    SubAgentEventRouter(None).publish(
        SubAgentEvent(
            action="started", id="id", path="main/child", persistent_history=False
        )
    )


def test_sub_agent_preserves_supplied_tool_duration_and_finished_events() -> None:
    conversation = SubAgentConversation(path="main/child")
    conversation.start_call("task")
    conversation.add_tool_event(
        ToolEvent(
            tool_name="orphan",
            event_type="tool_end",
            tool_call_id="absent",
            timestamp=1,
            duration_seconds=2,
        )
    )
    conversation.add_tool_event(
        ToolEvent(tool_name="completed", event_type="execution", duration_seconds=3)
    )
    conversation.add_tool_event(ToolEvent(tool_name="pending", event_type="tool_start"))
    conversation.fail_call()
    assert [event.status for event in conversation.messages[-1].tool_events] == [
        "completed",
        "completed",
        "error",
    ]
    assert [
        event.duration_seconds for event in conversation.messages[-1].tool_events[:2]
    ] == [
        2,
        3,
    ]


def test_queued_messages_operations_and_converter_preservation() -> None:
    session = ChatSession(title="Queue test")
    # Empty pop
    assert session.pop_queued_message() is None

    # Add messages
    m1 = session.add_queued_message("First message")
    m2 = session.add_queued_message("Second message")
    m3 = session.add_queued_message("Third message")
    assert len(session.queued_messages) == 3
    assert session.queued_messages[0].text == "First message"

    # Reorder edge cases
    # 1. source_id == target_id
    assert session.reorder_queued_message(m1.id, m1.id, "before") is False
    # 2. source_id not found
    assert session.reorder_queued_message("nonexistent", m2.id, "before") is False
    # 3. target_id not found
    assert session.reorder_queued_message(m1.id, "nonexistent", "before") is False
    assert len(session.queued_messages) == 3
    # A rejected reorder leaves every message in its original position.
    assert [m.id for m in session.queued_messages] == [m1.id, m2.id, m3.id]

    # 4. Valid reorder: move m1 before m2 -> [m1, m2, m3]
    assert session.reorder_queued_message(m1.id, m2.id, "before") is True
    assert [m.id for m in session.queued_messages] == [m1.id, m2.id, m3.id]

    # 5. Valid reorder: move m1 after m2 -> [m2, m1, m3]
    assert session.reorder_queued_message(m1.id, m2.id, "after") is True
    assert [m.id for m in session.queued_messages] == [m2.id, m1.id, m3.id]

    # Pop message
    popped = session.pop_queued_message()
    assert popped is not None and popped.id == m2.id
    assert len(session.queued_messages) == 2

    # Remove message: existing
    removed = session.remove_queued_message(m1.id)
    assert removed is not None and removed.id == m1.id
    assert len(session.queued_messages) == 1

    # Remove message: nonexistent
    assert session.remove_queued_message("unknown") is None
    assert len(session.queued_messages) == 1

    # Converter round-trip preservation
    session.add_queued_message("Another queued")
    exmekh_msgs = session_to_exmekh_messages(session)
    restored = exmekh_messages_to_session(exmekh_msgs)
    assert len(restored.queued_messages) == 2
    assert restored.queued_messages[0].text == m3.text
    assert restored.queued_messages[1].text == "Another queued"

    # Test already validated QueuedMessage instance in session_meta
    dummy_meta = {
        "id": "meta_id",
        "queued_messages": [QueuedMessage(text="Direct")],
    }
    sys_m = SystemMessage(content="sys")
    sys_m.set_metadata("_fe_session_metadata", dummy_meta)
    restored2 = exmekh_messages_to_session([sys_m])
    assert len(restored2.queued_messages) == 1
    assert restored2.queued_messages[0].text == "Direct"


def test_convert_to_exmekh_messages_unclosed_tool_call() -> None:
    raw = [
        {"role": "user", "content": "run tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_pending",
                    "function": {"name": "test_tool", "arguments": {}},
                }
            ],
        },
    ]
    exmekh_lib = convert_to_exmekh_messages(raw)
    assert len(exmekh_lib) == 4
    assert exmekh_lib[0].get_role_name() == "user"
    assert exmekh_lib[1].get_role_name() == "assistant"
    assert exmekh_lib[2].get_role_name() == "tool"
    assert exmekh_lib[2].get_message() == "[Execution interrupted]"
    assert exmekh_lib[3].get_role_name() == "assistant"


def test_chat_message_to_exmekh_messages_unclosed_tool_call() -> None:
    msg = ChatMessage(
        role="assistant",
        raw_history=[
            {"role": "user", "content": "prompt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c_unclosed",
                        "function": {"name": "orphan_tool", "arguments": {}},
                    }
                ],
            },
        ],
    )
    exmekh_lib = chat_message_to_exmekh_messages(msg)
    assert any(
        m.get_role_name() == "tool" and m.get_message() == "[Execution interrupted]"
        for m in exmekh_lib
    )
    assert exmekh_lib[-1].get_role_name() == "assistant"


def test_complete_skill_recursively_completes_active_sub_skills() -> None:
    events = []
    handler = DeepAgentTurnEventHandler(
        event_router=SubAgentEventRouter(None),
        on_token=lambda _: None,
        on_reasoning=lambda *_: None,
        on_tool_event=lambda _: None,
        on_skill_event=lambda node, action: events.append((node.path, action)),
        on_raw_history=None,
    )
    child = SkillNode(name="child", path="/root/child", depth=2, status="active")
    parent = SkillNode(
        name="root", path="/root", depth=1, status="active", sub_skills=[child]
    )
    handler.skill_stack.append(parent)
    handler.finish()
    assert parent.status == "completed"
    assert child.status == "completed"
    assert child.end_time is not None

    # Test with preset end_time and duration
    parent2 = SkillNode(
        name="root2",
        path="/root2",
        depth=1,
        status="active",
        end_time=100.0,
        duration_seconds=5.0,
    )
    handler.skill_stack.append(parent2)
    handler.finish()
    assert parent2.duration_seconds == 5.0


def test_sub_agent_fail_call_finalizes_active_skills_and_cleanup_method() -> None:
    conv = SubAgentConversation(path="main/coding-agent")
    conv.start_call("user task")
    skill1 = SkillNode(
        name="active_skill1", path="/active_skill1", depth=1, status="active"
    )
    skill2 = SkillNode(
        name="active_skill2",
        path="/active_skill2",
        depth=1,
        status="active",
        end_time=50.0,
        duration_seconds=10.0,
    )
    skill3 = SkillNode(
        name="completed_skill",
        path="/completed_skill",
        depth=1,
        status="completed",
        end_time=40.0,
        duration_seconds=5.0,
    )
    conv.messages[-1].skills.extend([skill1, skill2, skill3])
    conv.fail_call()
    assert conv.status == "error"
    assert skill1.status == "error"
    assert skill1.duration_seconds is not None
    assert skill2.duration_seconds == 10.0
    assert skill3.status == "completed"

    session = ChatSession(sub_agent_conversations=[conv])
    session.cleanup_active_sub_agents()
    assert conv.status == "error"

    # Test cleanup_active_sub_agents when conv is actively streaming (covers line 333)
    conv_streaming = SubAgentConversation(path="main/streaming-agent")
    conv_streaming.start_call("streaming sub task")
    assert conv_streaming.messages[-1].is_streaming is True
    session.sub_agent_conversations.append(conv_streaming)
    session.cleanup_active_sub_agents()
    assert conv_streaming.status == "error"
    assert conv_streaming.messages[-1].is_streaming is False

    # Also test cleanup_active_sub_agents when conv is active but has no messages
    conv2 = SubAgentConversation(path="main/research", status="active", messages=[])
    session.sub_agent_conversations.append(conv2)
    session.cleanup_active_sub_agents()
    assert conv2.status == "error"
