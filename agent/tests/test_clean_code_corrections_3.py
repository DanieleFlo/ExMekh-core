from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.interfaces.tool_hook import ToolContext, ToolResponse
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.history.recorded_history import RecordedHistory
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SkillNode,
    SteeringInsertion,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    ApprovalModeChanged,
    CommandApprovalDecision,
    CommandApprovalEvaluator,
    CommandApprovalHook,
)
from exmekh_agent.state.agent_design import AgentDesign, DesignNode
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.graph.graph_data import build_graph_data_from_session
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from tests.mock_client import MockProvider


def insertion(
    identifier: str,
    *,
    after: str | None = None,
    path: str = "/",
    timestamp: float | None = 15,
) -> SteeringInsertion:
    return SteeringInsertion(
        message_id=identifier,
        after_tool_call_id=after,
        tool_name=None,
        skill_path=path,
        timestamp=timestamp,
    )


def test_missing_tool_boundaries_use_confirmation_time_and_duplicate_confirmations_are_ignored():
    before = ToolEvent(tool_name="before", timestamp=10, tool_call_id="before")
    after = ToolEvent(tool_name="after", timestamp=20, tool_call_id="after")
    guidance = insertion("one", after="missing")
    message = ChatMessage(
        role="assistant",
        tool_events=[before, after],
        steering_insertions=[guidance, guidance],
    )
    timeline = message.execution_timeline()
    assert timeline.root_activity() == [
        before.normalized_copy(),
        guidance,
        after.normalized_copy(),
    ]
    timeline.root_activity().clear()
    assert len(timeline.root_activity()) == 3
    assert message.tool_events == [before, after]


def test_steering_anchors_are_scoped_and_keep_fifo_order():
    root_tool = ToolEvent(tool_name="root", tool_call_id="reused", timestamp=30)
    child_tool = ToolEvent(
        tool_name="child", tool_call_id="reused", timestamp=10, skill_path="/thinking"
    )
    skill = SkillNode(
        name="thinking",
        path="/thinking",
        depth=1,
        tool_events=[child_tool],
        start_time=5,
    )
    first = insertion("first", after="reused", path=skill.path)
    second = insertion("second", after="reused", path=skill.path)
    unknown = insertion("unknown", path="/deleted-skill", timestamp=40)
    timeline = ChatMessage(
        role="assistant",
        tool_events=[root_tool],
        skills=[skill],
        steering_insertions=[first, second, unknown],
    ).execution_timeline()
    assert timeline.skill_activity(skill) == [
        child_tool.normalized_copy(),
        first,
        second,
    ]
    assert timeline.root_activity() == [skill, root_tool.normalized_copy(), unknown]


def test_repeated_skill_paths_place_guidance_in_the_matching_invocation():
    first = SkillNode(name="thinking", path="/thinking", depth=1, start_time=5)
    second = SkillNode(name="thinking", path="/thinking", depth=1, start_time=20)
    guidance = insertion("later", path=first.path, timestamp=25)
    timeline = ChatMessage(
        role="assistant", skills=[first, second], steering_insertions=[guidance]
    ).execution_timeline()
    assert timeline.skill_activity(first) == []
    assert timeline.skill_activity(second) == [guidance]
    # A legacy confirmation without timing still uses the earliest invocation.
    legacy = insertion("legacy", path=first.path, timestamp=None)
    early = insertion("early", path=first.path, timestamp=1)
    timeline = ChatMessage(
        role="assistant", skills=[first, second], steering_insertions=[legacy, early]
    ).execution_timeline()
    assert timeline.skill_activity(first) == [legacy, early]


def test_shared_manager_views_keep_selection_independent_and_share_record_updates(
    tmp_path: Path,
):
    manager = SessionManager(tmp_path / "chats")
    first = manager.get_current_session()
    second = manager.create_session("Second")
    first_view = manager.for_view(first.id)
    second_view = first_view.for_view(second.id)
    assert first_view.get_current_session() is first
    assert second_view.get_current_session() is second
    assert manager.get_current_session() is second
    assert second_view.update_title(first, "Renamed")
    assert first_view.get_current_session().title == "Renamed"
    third = second_view.create_session("Third")
    assert first_view.get_current_session() is first
    assert manager.get_current_session() is third
    assert second_view.set_tool_approval_mode(first.id, "Approve All")
    assert first_view.get_current_session().settings.tool_approval_mode == "Approve All"
    second_view.delete_session(first.id)
    assert first_view.get_current_session() is manager.list_sessions_sorted()[0]
    assert first_view.get_current_session() is not first


@pytest.mark.parametrize("action", ["title", "approval_history"])
def test_failed_persistence_keeps_the_live_record_and_timestamp(
    tmp_path: Path, action: str
):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    original = session.model_dump()
    with patch.object(Path, "write_text", side_effect=OSError("read-only")):
        saved = (
            manager.update_title(session, "Changed")
            if action == "title"
            else manager.record_command_approval(
                session, "pwd", [{"role": "user", "content": "pwd"}]
            )
        )
    assert not saved
    assert session.model_dump() == original
    assert (
        SessionManager(manager.storage_dir).get_session(session.id).model_dump()
        == original
    )


def test_positioned_addition_is_atomic_for_the_board_and_callers_card():
    root = DesignNode(kind="hds", name="root")
    design = AgentDesign(nodes=[root], root_id=root.id)
    duplicate = DesignNode(kind="tool", name="duplicate", x=root.x, y=root.y)
    snapshot = duplicate.model_dump()
    with pytest.raises(ValueError):
        design.add_positioned_node(duplicate, "missing-parent")
    assert design.nodes == [root]
    assert duplicate.model_dump() == snapshot


def test_history_owns_multimodal_restoration_and_handles_absent_child_histories():
    images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    child = {"history": [{"role": "user", "content": "image"}], "children": None}
    recorded = RecordedHistory([], [{"history": None, "children": [child]}])
    recorded.restore_user_content({"image": images})
    assert child["history"][0]["content"] == images
    images[0]["image_url"]["url"] = "changed"
    assert child["history"][0]["content"][0]["image_url"]["url"] != "changed"


def test_obsolete_llm_approval_does_not_enter_the_committed_history():
    mode = "Approve via LLM"
    provider = MockProvider()
    evaluator = CommandApprovalEvaluator(history=["pwd"], api_type="responses")
    original = evaluator.dump_history()

    def evaluate(request):
        nonlocal mode
        mode = "Manual"
        return StructuredProviderResponse(
            StructuredAssistantMessage(
                CommandApprovalDecision(approved=True, reason="Safe")
            )
        )

    hook = CommandApprovalHook(
        get_mode=lambda: mode,
        evaluator=evaluator,
        request_manual_approval=lambda _: False,
    )
    with (
        patch(
            "exmekh_agent.exmekh_bridge.tools.tool_approval.LocalProvider",
            return_value=provider,
        ),
        patch.object(provider, "call_structured", side_effect=evaluate),
    ):
        assert isinstance(
            hook.handle(
                ToolContext(
                    arguments={"command": "git status"},
                    work_root=Path.cwd(),
                    history=(),
                )
            ),
            ToolResponse,
        )
    assert evaluator.dump_history() == original


def test_manual_policy_change_is_retried_even_if_mode_changes_back():
    calls = 0

    def manual(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ApprovalModeChanged()
        return True

    hook = CommandApprovalHook(mode="Manual", request_manual_approval=manual)
    assert (
        hook.handle(
            ToolContext(arguments={"command": "pwd"}, work_root=Path.cwd(), history=())
        )
        is None
    )
    assert calls == 2
    assert hook.history == ["pwd"]


def test_routes_use_the_supplied_repository_and_reject_nonlocal_urls(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    routes = ChatRoutes(manager)
    path = routes.path_for(session)
    assert routes.resolve(path) is session
    assert routes.resolve("/chat/project/--" + session.id) is None
    assert routes.resolve("https://example.com" + path) is None
    assert routes.resolve("//example.com" + path) is None


def test_setting_approval_in_the_view_does_not_emit_a_user_change():
    client = Client(page("/approval-value-synchronization"))
    changes = []
    try:
        with client:
            bar = InputBarComponent(
                lambda _: None, lambda: None, on_approval_mode_changed=changes.append
            )
            bar.build()
            bar.set_approval_mode("Approve All")
            bar.set_approval_mode("Approve All")
            assert changes == []
            bar.approval_select.value = "Manual"
            assert changes == ["Manual"]
    finally:
        client.delete()


def test_graph_and_chat_share_steering_order_without_raw_history():
    before = ToolEvent(tool_name="before", timestamp=10, tool_call_id="before")
    after = ToolEvent(tool_name="after", timestamp=20, tool_call_id="after")
    inner = ToolEvent(
        tool_name="inner", timestamp=30, tool_call_id="inner", skill_path="/thinking"
    )
    skill = SkillNode(
        name="thinking", path="/thinking", depth=1, start_time=25, tool_events=[inner]
    )
    guidance = QueuedMessage(text="Root correction")
    nested = QueuedMessage(text="Skill correction")
    root_insertion = insertion(guidance.id, timestamp=15)
    child_insertion = insertion(nested.id, path=skill.path, timestamp=28)
    message = ChatMessage(
        role="assistant",
        tool_events=[before, after],
        skills=[skill],
        steering_messages=[guidance, nested],
        steering_insertions=[root_insertion, child_insertion],
    )
    timeline = message.execution_timeline()
    assert timeline.root_activity() == [
        before.normalized_copy(),
        root_insertion,
        after.normalized_copy(),
        skill,
    ]
    assert timeline.skill_activity(skill) == [child_insertion, inner.normalized_copy()]
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    nodes = graph.histories[0].nodes
    assert [
        node.kind for node in nodes if node.kind not in {"system", "assistant"}
    ] == [
        "tool_bundle",
        "steering",
        "tool_bundle",
        "skill_bundle",
        "steering",
        "tool_bundle",
    ]
    assert [
        node.events[0]["message_id"] for node in nodes if node.kind == "steering"
    ] == [guidance.id, nested.id]


def test_skill_completion_anchors_and_reused_call_ids_keep_the_original_boundary():
    first = ToolEvent(tool_name="first", tool_call_id="shared", timestamp=10)
    future = ToolEvent(tool_name="future", tool_call_id="shared", timestamp=40)
    skill = SkillNode(
        name="thinking",
        path="/thinking",
        depth=1,
        start_time=20,
        opening_tool_call_id="open",
    )
    early = insertion("early", after="shared", timestamp=15)
    completed = insertion("completed", after="open", path=skill.path, timestamp=30)
    legacy = insertion("legacy", after="open", path=skill.path, timestamp=None)
    timeline = ChatMessage(
        role="assistant",
        tool_events=[first, future],
        skills=[skill],
        steering_insertions=[early, completed, legacy],
    ).execution_timeline()
    assert timeline.root_activity() == [
        first.normalized_copy(),
        early,
        skill,
        completed,
        legacy,
        future.normalized_copy(),
    ]


def test_graph_handles_confirmation_without_a_retained_guidance_message():
    message = ChatMessage(role="assistant", steering_insertions=[insertion("missing")])
    graph = build_graph_data_from_session(ChatSession(messages=[message]))
    assert all(node.kind != "steering" for node in graph.histories[0].nodes)
