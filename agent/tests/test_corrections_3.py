from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.interfaces.tool_hook import ToolContext, ToolResponse
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    SkillNode,
    SteeringInsertion,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.tools.tool_approval import (
    ApprovalModeChanged,
    CommandApprovalDecision,
    CommandApprovalHook,
)
from exmekh_agent.state.agent_design import BOARD_DEFAULTS
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes


@pytest.mark.parametrize("profile", ["generic", "programmer", "creative", "custom"])
def test_default_boards_have_no_overlapping_cards(profile):
    design = CapabilityRepository().default_design(profile)
    assert any(node.name == "deep-thinking" for node in design.nodes)
    for index, node in enumerate(design.nodes):
        for other in design.nodes[index + 1 :]:
            assert (
                abs(node.x - other.x) >= BOARD_DEFAULTS["node_width"]
                or abs(node.y - other.y) >= BOARD_DEFAULTS["node_height"]
            ), (node.name, other.name)


def test_new_dependencies_avoid_existing_cards_without_moving_them():
    repository = CapabilityRepository()
    catalog = repository.catalog()
    design = repository.default_design()
    source = next(
        node
        for node in catalog
        if node.kind == "tool" and not design.capability(node.kind, node.name)
    )
    source = source.model_copy(update={"x": design.nodes[0].x, "y": design.nodes[0].y})
    coordinates = {node.id: (node.x, node.y) for node in design.nodes}
    added = repository.add_to_design(design, source, design.root_id, catalog)
    new_node = added.capability(source.kind, source.name)
    assert new_node.y >= design.nodes[0].y + BOARD_DEFAULTS["node_height"]
    assert all(
        (added.node(key).x, added.node(key).y) == value
        for key, value in coordinates.items()
    )


def test_chat_urls_are_unique_and_old_names_keep_resolving(tmp_path: Path):
    folder = tmp_path / "Prògetto uno"
    folder.mkdir()
    project = session_manager.create_project(folder)
    first = session_manager.create_session("Chat à uno", project_id=project.id)
    second = session_manager.create_session(first.title, project_id=project.id)
    original = ChatRoutes(session_manager).path_for(first)
    assert original.startswith("/chat/progetto-uno/chat-a-uno--")
    assert original != ChatRoutes(session_manager).path_for(second)
    first.title = "Renamed"
    assert ChatRoutes(session_manager).resolve(original) is first
    assert (
        ChatRoutes(session_manager).resolve(ChatRoutes(session_manager).path_for(first))
        is first
    )
    quick = session_manager.create_session("💬")
    assert (
        ChatRoutes(session_manager)
        .path_for(quick)
        .startswith("/chat/quick-chats/chat--")
    )
    emoji_folder = tmp_path / "💬"
    emoji_folder.mkdir()
    emoji_project = session_manager.create_project(emoji_folder)
    emoji_chat = session_manager.create_session(project_id=emoji_project.id)
    assert ChatRoutes(session_manager).path_for(emoji_chat).startswith("/chat/project/")
    for invalid in [
        "/",
        "/chat/empty/missing",
        "/chat/empty/chat--missing",
        "/chat//chat--missing",
        "/chat/one/two/three",
    ]:
        assert ChatRoutes(session_manager).resolve(invalid) is None


@pytest.mark.parametrize("new_mode", ["Approve All", "Approve via LLM"])
def test_changing_mode_rechecks_an_already_pending_command(new_mode):
    mode = "Manual"
    evaluator = MagicMock()
    evaluator.evaluate.return_value = CommandApprovalDecision(approved=True)

    def manual(_):
        nonlocal mode
        mode = new_mode
        return False

    hook = CommandApprovalHook(
        get_mode=lambda: mode, evaluator=evaluator, request_manual_approval=manual
    )
    context = ToolContext(
        arguments={"command": "pwd"}, work_root=Path.cwd(), history=()
    )
    assert hook.handle(context) is None
    assert evaluator.evaluate.call_count == (new_mode == "Approve via LLM")
    mode = "Manual"
    hook.request_manual_approval = lambda _: False
    assert isinstance(hook.handle(context), ToolResponse)


def test_changing_mode_discards_an_inflight_llm_decision():
    mode = "Approve via LLM"
    evaluator = MagicMock()

    def evaluate(**_):
        nonlocal mode
        mode = "Manual"
        return CommandApprovalDecision(approved=True)

    evaluator.evaluate.side_effect = evaluate
    hook = CommandApprovalHook(
        get_mode=lambda: mode,
        evaluator=evaluator,
        request_manual_approval=lambda _: False,
    )
    context = ToolContext(
        arguments={"command": "pwd"}, work_root=Path.cwd(), history=()
    )
    assert isinstance(hook.handle(context), ToolResponse)
    evaluator.record_approved_command.assert_not_called()


def test_controller_releases_pending_manual_approval_on_mode_change():
    client = Client(page("/corrections-3-approval"))
    try:
        with client:
            controller = ChatPageController(client)
            controller.input_bar = MagicMock()
            session = session_manager.get_current_session()
            decision = Future()
            controller._show_manual_approval("pwd", session, decision, lambda: True)
            with patch("exmekh_agent.ui.chat.page_controller.ui.notify"):
                controller.approval_mode_changed("Approve All")
            with pytest.raises(ApprovalModeChanged):
                decision.result(timeout=1)
            assert controller._pending_approval is None
            controller.input_bar.hide_approval_prompt.assert_called_once()
            late_decision = Future()
            controller._show_manual_approval(
                "pwd", session, late_decision, lambda: True
            )
            with pytest.raises(ApprovalModeChanged):
                late_decision.result(timeout=1)
    finally:
        client.delete()


def test_approval_change_from_another_page_releases_the_owner_prompt():
    client = Client(page("/corrections-3-approval-other-page"))
    try:
        with client:
            owner = ChatPageController(client)
            owner.input_bar = MagicMock()
            other = ChatPageController(client)
            session = session_manager.get_current_session()
            decision = Future()
            owner._show_manual_approval("pwd", session, decision, lambda: True)
            with patch("exmekh_agent.ui.chat.page_controller.ui.notify"):
                other.approval_mode_changed("Approve All")
            assert not decision.done()
            owner._sync_active_turn()
            with pytest.raises(ApprovalModeChanged):
                decision.result(timeout=1)
            owner.input_bar.hide_approval_prompt.assert_called_once()
    finally:
        client.delete()


@pytest.mark.parametrize("scope", ["/", "/deep-thinking"])
@pytest.mark.parametrize("timestamp", [None, 15])
def test_steering_without_a_tool_boundary_stays_before_later_tools(scope, timestamp):
    client = Client(page("/corrections-3-steering"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            tools = [
                ToolEvent(
                    tool_name=name, tool_call_id=name, timestamp=time, skill_path=scope
                )
                for name, time in [("before", 10), ("after", 20)]
            ]
            insertion = SteeringInsertion(
                message_id="guidance",
                after_tool_call_id=None,
                tool_name=None,
                skill_path=scope,
                timestamp=timestamp,
            )
            skills = (
                [
                    SkillNode(
                        name="deep-thinking", path=scope, depth=1, tool_events=tools
                    )
                ]
                if scope != "/"
                else []
            )
            message = ChatMessage(
                role="assistant",
                skills=skills,
                tool_events=tools if not skills else [],
                steering_insertions=[insertion],
            )
            order = []
            with patch.object(
                view,
                "_render_tool_event",
                side_effect=lambda event: order.append(event.tool_name),
            ):
                original = view._render_steering_insertion

                def render_insertion(value):
                    order.append("steering")
                    original(value)

                with patch.object(
                    view, "_render_steering_insertion", side_effect=render_insertion
                ):
                    view._render_turn_activity(message)
            assert order == (
                ["steering", "before", "after"]
                if timestamp is None
                else ["before", "steering", "after"]
            )
    finally:
        client.delete()


def test_navigation_history_restores_selected_chat(browser_history):
    client = Client(page("/corrections-3-routes"))
    try:
        with client:
            controller = ChatPageController(client)
            controller.chat_view = MagicMock()
            controller.sidebar = MagicMock()
            first = session_manager.get_current_session()
            controller._sync_chat_url(replace=True)
            second = session_manager.create_session("Second")
            with (
                patch.object(controller, "update_context_usage"),
                patch.object(controller, "_sync_active_turn"),
            ):
                controller.session_switched(second.id)
                browser_history.push.assert_called_once_with(
                    ChatRoutes(session_manager).path_for(second)
                )
                controller._handle_chat_navigation(
                    SimpleNamespace(
                        args={"path": ChatRoutes(session_manager).path_for(first)}
                    )
                )
                assert session_manager.get_current_session() is first
                browser_history.replace.assert_called_with(
                    ChatRoutes(session_manager).path_for(first)
                )
                controller._handle_chat_navigation(
                    SimpleNamespace(args={"path": "/chat/missing/chat--missing"})
                )
                controller._handle_chat_navigation(SimpleNamespace(args=[]))
            assert browser_history.push.call_count == 1
    finally:
        client.delete()
