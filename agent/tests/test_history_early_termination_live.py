from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from nicegui.client import Client as NiceGUIClient
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    convert_to_exmekh_messages,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEvent
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.state.sub_agent_conversations import apply_sub_agent_event
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.graph.graph_data import build_graph_data_from_session
from exmekh_agent.ui.history.raw_history_view import build_raw_history_from_session
from tests.mock_client import (
    MockProvider,
    PipelineStep,
    ScenarioPipeline,
    install_mock_provider,
    uninstall_mock_provider,
)

pytestmark = [pytest.mark.live, pytest.mark.mock]


@pytest.fixture(autouse=True)
def live_mock_env() -> Generator[MockProvider]:
    """Install a fresh mock provider for live testing and clean up afterwards."""
    provider = MockProvider()
    install_mock_provider(provider)
    yield provider
    uninstall_mock_provider()


@pytest.fixture
def ui_page_context() -> Generator[NiceGUIClient]:
    """Fixture providing a NiceGUI Client context for live UI controller tests."""
    client = NiceGUIClient(page("/test-early-termination-live"))
    with client:
        yield client


# =============================================================================
# SCENARIO 1: Early termination inside an active Root Skill
# =============================================================================


async def test_live_early_termination_in_root_skill(
    live_mock_env: MockProvider,
) -> None:
    """Verifies that an unclosed active skill is properly finalized when early termination occurs,
    computing valid end_time, duration, and that the next turn resumes cleanly."""
    root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="active",
        start_time=time.time() - 2.5,
        tool_events=[
            ToolEvent(
                tool_name="skill-tool",
                depth=1,
                skill_path="/mock-skill",
                status="running",
                timestamp=time.time() - 1.0,
            )
        ],
    )

    step_abort = PipelineStep(
        name="step_skill_aborted",
        skill_events=[(root_skill, "opened")],
        tool_events=[root_skill.tool_events[0]],
        reasoning_chunks=[
            ("Exploring inside skill...", "/mock-skill", 1),
        ],
        response_tokens=["Partial output before interruption "],
        exception=asyncio.CancelledError("Turn cancelled by user"),
    )
    live_mock_env.pipeline = ScenarioPipeline("early_skill_abort", [step_abort])

    collector = TurnStreamCollector()
    session = ChatSession(id="test-session-skill-abort")
    session.messages.append(ChatMessage(role="user", content="Execute skill task"))

    with pytest.raises(asyncio.CancelledError):
        await run_agent_turn(
            user_prompt="Execute skill task",
            model="mock-pipeline-model",
            on_token=collector.append_token,
            on_reasoning=collector.append_reasoning,
            on_tool_event=collector.append_tool,
            on_skill_event=collector.append_skill,
            agent_mode="deep_agent",
        )

    # Collector builds message on cancellation / error
    msg = collector.build_message(
        role="assistant",
        fallback_content=collector.get_final_content() or "[Interrupted]",
        error="cancelled",
    )
    session.messages.append(msg)

    # 1. Assert skill was finalized and not left dangling in active status
    assert len(msg.skills) == 1
    final_skill = msg.skills[0]
    assert final_skill.status == "error"
    assert final_skill.end_time is not None
    assert final_skill.duration_seconds is not None
    assert final_skill.duration_seconds >= 2.0

    # 2. Assert tool events inside the skill were finalized from running to error
    assert len(final_skill.tool_events) == 1
    assert final_skill.tool_events[0].status == "error"
    assert final_skill.tool_events[0].end_time is not None
    assert final_skill.tool_events[0].duration_seconds is not None

    # 3. Assert reasoning blocks were finalized with timestamps and durations
    assert len(msg.reasoning_blocks) == 1
    assert msg.reasoning_blocks[0].end_time is not None
    assert msg.reasoning_blocks[0].duration_seconds is not None

    # 4. Turn 2: verify the session can continue immediately without errors
    step_recovery = PipelineStep(
        name="step_recovery",
        final_response="Turn 2 successfully resumed after interrupted skill.",
        response_tokens=["Turn 2 successfully resumed after interrupted skill."],
    )
    live_mock_env.pipeline.add_step(step_recovery)

    collector_turn2 = TurnStreamCollector()
    response_turn2 = await run_agent_turn(
        user_prompt="Continue after stop",
        model="mock-pipeline-model",
        on_token=collector_turn2.append_token,
        on_reasoning=collector_turn2.append_reasoning,
        on_tool_event=collector_turn2.append_tool,
        agent_mode="deep_agent",
        history=list(session.messages),
    )
    assert "Turn 2 successfully resumed" in response_turn2


# =============================================================================
# SCENARIO 2: Early termination inside a Nested Sub-Skill (Matryoshka Depth 2)
# =============================================================================


async def test_live_early_termination_in_nested_sub_skill(
    live_mock_env: MockProvider,
) -> None:
    """Verifies that nested sub-skills (depth 2) and their parents are both finalized recursively
    when interrupted, and build_raw_history_from_session renders clean matryoshka trees."""
    sub_sub = SkillNode(
        name="nested-sub-skill",
        path="/mock-skill/nested-sub-skill",
        depth=2,
        status="active",
        start_time=time.time() - 1.5,
        tool_events=[
            ToolEvent(
                tool_name="leaf-tool",
                depth=2,
                skill_path="/mock-skill/nested-sub-skill",
                status="running",
                timestamp=time.time() - 0.8,
            )
        ],
    )
    root_skill = SkillNode(
        name="mock-skill",
        path="/mock-skill",
        depth=1,
        status="active",
        start_time=time.time() - 3.0,
        sub_skills=[sub_sub],
    )

    collector = TurnStreamCollector()
    collector.append_skill(root_skill, action="opened")
    collector.append_skill(sub_sub, action="opened")
    collector.append_reasoning(
        "Deep reasoning at depth 2",
        skill_path="/mock-skill/nested-sub-skill",
        depth=2,
    )

    # Interruption occurs: build_message with error="cancelled"
    msg = collector.build_message(
        role="assistant",
        fallback_content="Interrupted at sub-skill",
        error="cancelled",
    )

    session = ChatSession(id="session-sub-skill-abort")
    session.messages = [
        ChatMessage(role="user", content="Run nested sub-skill"),
        msg,
    ]

    # 1. Assert root skill AND child sub-skill are finalized recursively
    assert len(msg.skills) == 1
    final_root = msg.skills[0]
    assert final_root.status == "error"
    assert final_root.end_time is not None
    assert final_root.duration_seconds is not None

    assert len(final_root.sub_skills) == 1
    final_sub = final_root.sub_skills[0]
    assert final_sub.status == "error"
    assert final_sub.end_time is not None
    assert final_sub.duration_seconds is not None

    # 2. Child tool inside sub-skill is finalized
    assert len(final_sub.tool_events) == 1
    assert final_sub.tool_events[0].status == "error"
    assert final_sub.tool_events[0].end_time is not None

    # 3. Raw history synthesis cleanly reconstructs both levels
    raw_msgs, skill_tree = build_raw_history_from_session(session)
    assert len(raw_msgs) == 3  # system, user, assistant
    assert len(skill_tree) == 1
    assert skill_tree[0]["skill"] == "mock-skill"
    assert skill_tree[0]["status"] == "error"
    assert len(skill_tree[0]["children"]) == 1
    assert skill_tree[0]["children"][0]["skill"] == "nested-sub-skill"
    assert skill_tree[0]["children"][0]["status"] == "error"


# =============================================================================
# SCENARIO 3: Early termination inside a Sub-Agent (Fresh and Persistent)
# =============================================================================


async def test_live_early_termination_in_sub_agent_and_clean_next_turn() -> None:
    """Verifies that an interrupted sub-agent conversation is closed with status='error',
    does NOT stay active/streaming, and does not crash subsequent calls with 'already active'."""
    session = ChatSession(id="session-subagent-abort")

    # Start sub-agent delegation
    sub_id = "coding-agent-live-call"
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="started",
            id=sub_id,
            path="main/coding-agent",
            persistent_history=True,
            user_message="Implement search algorithm",
        ),
    )
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="reasoning",
            id=sub_id,
            path="main/coding-agent",
            persistent_history=True,
            token="Analyzing requirements...",
        ),
    )
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="tool",
            id=sub_id,
            path="main/coding-agent",
            persistent_history=True,
            tool_event=ToolEvent(
                tool_name="write_file",
                depth=0,
                status="running",
                timestamp=time.time() - 1.0,
            ),
        ),
    )

    conv = session.find_sub_agent_conversation(sub_id)
    assert conv is not None
    assert conv.status == "active"
    assert conv.messages[-1].is_streaming is True
    assert conv.messages[-1].tool_events[0].status == "running"

    # Simulate early termination (e.g. stop generation or exception)
    session.cleanup_active_sub_agents()

    # 1. Assert sub-agent was closed cleanly
    assert conv.status == "error"
    assert conv.messages[-1].is_streaming is False
    assert conv.messages[-1].tool_events[0].status == "error"
    assert conv.messages[-1].tool_events[0].end_time is not None
    assert conv.messages[-1].duration_seconds is not None

    # 2. CRUCIAL: In the next turn, call the same persistent sub-agent again
    # Must NOT raise ValueError("Sub-agent conversation ... is already active")
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="started",
            id=sub_id,
            path="main/coding-agent",
            persistent_history=True,
            user_message="Retry algorithm implementation",
        ),
    )
    assert conv.status == "active"
    assert conv.messages[-1].is_streaming is True

    # Complete the second call successfully
    apply_sub_agent_event(
        session,
        SubAgentEvent(
            action="completed",
            id=sub_id,
            path="main/coding-agent",
            persistent_history=True,
            response="Algorithm successfully implemented.",
            raw_history=[
                {"role": "user", "content": "Retry algorithm implementation"},
                {
                    "role": "assistant",
                    "content": "Algorithm successfully implemented.",
                },
            ],
        ),
    )
    assert conv.status == "completed"
    assert conv.messages[-1].is_streaming is False
    assert conv.messages[-1].content == "Algorithm successfully implemented."
    # Total messages in persistent sub-agent: 2 from first aborted call + 2 from second call = 4
    assert len(conv.messages) == 4


# =============================================================================
# SCENARIO 4: Early termination during a Tool Call (Unresponded Tool Call)
# =============================================================================


def test_live_early_termination_unresponded_tool_call_history_resilience() -> None:
    """Verifies that an assistant message with unresponded tool calls from an interrupted turn
    is converted with synthetic tool closure, preventing provider BadRequest errors."""
    session = ChatSession(id="session-tool-abort")
    raw_interrupted = [
        {"role": "user", "content": "Query database"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_db_search",
                    "function": {
                        "name": "db_search",
                        "arguments": {"query": "SELECT *"},
                    },
                }
            ],
        },
        # Tool response was NEVER emitted because turn was interrupted!
    ]

    session.messages = [
        ChatMessage(role="user", content="Query database"),
        ChatMessage(
            role="assistant",
            content="[Generation stopped]",
            raw_history=raw_interrupted,
            error="cancelled",
        ),
    ]

    # Convert to ExMekh-Lib messages for next turn
    exmekh_messages = convert_to_exmekh_messages(
        session.messages, current_user_prompt="Next question"
    )

    # 1. Assert unresponded tool call was closed with synthetic tool message
    tool_messages = [m for m in exmekh_messages if m.get_role_name() == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].get_message() == "[Execution interrupted]"
    assert tool_messages[0].get_tool_call_id() == "call_db_search"

    # 2. Assert history ends with a valid AssistantMessage so provider allows next UserMessage
    assert exmekh_messages[-1].get_role_name() == "assistant"
    assert (
        exmekh_messages[-1].get_tool_calls() is None
        or len(exmekh_messages[-1].get_tool_calls()) == 0
    )


# =============================================================================
# SCENARIO 5: Full UI PageController Stop Generation Flow
# =============================================================================


@pytest.mark.parametrize("switch_session", [False, True])
async def test_live_ui_controller_stop_generation_flow(
    ui_page_context: NiceGUIClient,
    switch_session: bool,
) -> None:
    """Tests the full PageController UI stop generation mechanism while a streaming turn is active,
    verifying partial message preservation, sub-agent cleanup, and queue stability."""
    with ui_page_context:
        session = session_manager.create_session("Live Stop Test")
        session_manager.save_session(session)
        session_manager.current_session_id = session.id

        controller = ChatPageController(ui_page_context)
        controller.build()
        input_bar = controller.input_bar
        assert input_bar is not None

        # Mock a slow live agent turn that simulates streaming
        async def fake_slow_turn(*args: Any, **kwargs: Any) -> str:
            on_token = kwargs.get("on_token")
            on_skill = kwargs.get("on_skill_event")
            if on_token:
                on_token("Streaming beginning... ")
            if on_skill:
                on_skill(
                    SkillNode(name="live-skill", path="/live-skill", depth=1),
                    "opened",
                )
            # Sleep to allow controller.stop() to intervene
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                if on_token:
                    on_token("Late token must be ignored")
                on_sub_agent = kwargs.get("on_sub_agent_event")
                if on_sub_agent:
                    on_sub_agent(
                        SubAgentEvent(
                            action="started",
                            id="late",
                            path="main/late",
                            persistent_history=True,
                        )
                    )
                raise
            return "Should not finish"

        with patch(
            "exmekh_agent.ui.chat.page_controller.run_agent_turn",
            side_effect=fake_slow_turn,
        ):
            controller.start_send("Start long process")
            await asyncio.sleep(0.05)

            # Assert UI state during execution
            assert input_bar.is_busy is True
            assert controller._active_task is not None
            assert not controller._active_task.done()

            session.add_queued_message("Keep queued after stop")
            if switch_session:
                other_session = session_manager.create_session("Other chat")
                other_conversation = other_session.start_sub_agent_conversation(
                    "other-call", "main/other", True, "Other task"
                )
                controller.session_switched(other_session.id)
                controller.stop()
                await asyncio.sleep(0)
                assert active_turns.is_running(session.id)
                assert not controller._active_task.cancelling()
                assert other_conversation.status == "active"
                session_manager.switch_session(session.id)
                controller.session_switched(session.id)
            # User clicks STOP in the chat that owns the running turn.
            controller.stop()
            await asyncio.sleep(0.05)

            # Assert UI state after stop
            assert input_bar.is_busy is False
            assert controller._active_task is None

            # Assert partial message is preserved in session
            current_sess = session
            assert len(current_sess.messages) >= 2
            assistant_msg = current_sess.messages[-1]
            assert assistant_msg.role == "assistant"
            assert "Streaming beginning..." in assistant_msg.content
            assert assistant_msg.error == "cancelled"
            assert "Late token" not in assistant_msg.content
            assert not session.sub_agent_conversations
            assert [item.text for item in session.queued_messages] == [
                "Keep queued after stop"
            ]
            if switch_session:
                assert other_conversation.status == "active"
                assert other_conversation.messages[-1].is_streaming
                assert not other_session.messages

            # Assert skills were finalized
            assert len(assistant_msg.skills) == 1
            assert assistant_msg.skills[0].status == "error"
            assert assistant_msg.skills[0].end_time is not None

            # Assert UI renders the finalized assistant message with closed skill immediately
            if not switch_session:
                assert controller.chat_view is not None
                assert controller.chat_view.current_stream_card is None
                assert (
                    len(controller.chat_view.chat_container.default_slot.children) == 2
                )


# =============================================================================
# SCENARIO 6: Graph View and Raw History after Multiple Early Terminations
# =============================================================================


def test_live_graph_and_raw_history_rendering_after_multiple_early_terminations(
    ui_page_context: NiceGUIClient,
) -> None:
    """Verifies that Graph View and Raw History render cleanly without crashing or showing
    perpetual spinning states when sessions contain multiple interrupted turns, skills, and sub-agents."""
    with ui_page_context:
        session = ChatSession(title="Multi Interrupted Session")

        # Turn 1: Interrupted skill
        s1 = SkillNode(
            name="skill_interrupted",
            path="/skill_interrupted",
            depth=1,
            status="error",
            duration_seconds=1.2,
        )
        # Interrupted sub-agent
        sub_conv = SubAgentConversation(
            id="sub_aborted",
            path="main/coding-agent",
            order=1,
            status="error",
            messages=[
                ChatMessage(role="user", content="Sub task"),
                ChatMessage(
                    role="assistant",
                    content="Partial sub task",
                    is_streaming=False,
                    tool_events=[
                        ToolEvent(
                            tool_name="write_code",
                            depth=0,
                            status="error",
                            duration_seconds=0.5,
                        )
                    ],
                ),
            ],
        )
        session.sub_agent_conversations.append(sub_conv)

        session.messages = [
            ChatMessage(role="user", content="Task 1"),
            ChatMessage(
                role="assistant",
                content="Partial response 1",
                skills=[s1],
                error="cancelled",
                duration_seconds=1.5,
            ),
            ChatMessage(role="user", content="Task 2"),
            ChatMessage(
                role="assistant",
                content="Completed response 2",
                duration_seconds=2.0,
            ),
        ]

        # 1. Test build_graph_data_from_session
        graph_data = build_graph_data_from_session(session)
        assert graph_data is not None
        assert len(graph_data.histories) >= 1

        # Check interaction node for sub-agent
        main_hist = next((h for h in graph_data.histories if h.id == "main"), None)
        assert main_hist is not None
        inter_node = next(
            (n for n in main_hist.nodes if n.kind == "sub_agent_bundle"), None
        )
        assert inter_node is not None
        # Assert tool event status in the triplet is 'error', NOT perpetually 'running'
        assert len(inter_node.events) >= 2
        exec_event = next(
            (ev for ev in inter_node.events if ev.get("role") == "execution"), None
        )
        assert exec_event is not None
        assert exec_event.get("status") == "error"

        # 2. Test build_raw_history_from_session
        raw_msgs, skill_tree = build_raw_history_from_session(session)
        assert len(raw_msgs) > 0
        assert len(skill_tree) == 1
        assert skill_tree[0]["status"] == "error"
