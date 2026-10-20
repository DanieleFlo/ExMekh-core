from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    SubAgentConversation,
)
from exmekh_agent.state.design_history import DesignHistory, DesignSnapshot
from exmekh_agent.state.design_selection import DesignSelection
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.agents.agent_designer import AgentDesigner
from tests.test_design_history import make_design


def test_appended_snapshot_keeps_the_source_message_list_independent():
    user = ChatMessage(role="user", content="Original task")
    response = ChatMessage(role="assistant", content="Reply")
    source = ChatSession(messages=[user])
    snapshot = source.with_message(response)
    snapshot.add_message(ChatMessage(role="user", content="Next task"))
    assert source.message_history() == [user]
    assert snapshot.message_history()[:2] == [user, response]
    assert snapshot.id == source.id


def test_selection_has_one_primary_and_survives_partial_removal_and_undo():
    design = make_design()
    selection = DesignSelection.for_nodes(design, ["tool", "skill", "skill"])
    assert selection.node_ids == ("skill", "tool")
    assert selection.primary_id == "skill"
    snapshot = DesignSnapshot.capture(design, [], selection)
    history = DesignHistory()
    design.remove_nodes(["skill"])
    history.record(snapshot, design, [])
    retained = selection.retain_existing(design)
    assert retained.node_ids == ("tool",)
    assert retained.primary_id == "tool"
    design.remove_nodes(["tool"])
    assert retained.retain_existing(design).node_ids == ("root",)
    assert DesignSelection().retain_existing(design).primary_id == ""
    restored = history.undo()
    assert restored.selection is selection
    assert restored.design.node_ids() == snapshot.design.node_ids()
    assert selection.node_ids == ("skill", "tool")
    with pytest.raises(ValueError, match="Unknown"):
        DesignSelection.for_nodes(design, ["root", "missing"])


@pytest.mark.parametrize("operation", ["remove_nodes", "disconnect_nodes"])
def test_collective_graph_edits_validate_all_targets_and_keep_the_root(operation):
    design = make_design()
    before = design.model_dump()
    with pytest.raises(ValueError, match="Unknown"):
        getattr(design, operation)(["skill", "missing"])
    assert design.model_dump() == before
    getattr(design, operation)([])
    assert design.model_dump() == before
    getattr(design, operation)(["root", "skill", "skill"])
    assert design.has_node(design.root_id)
    assert len(design.connections("root")) == (
        0 if operation == "disconnect_nodes" else 2
    )
    assert design.has_node("skill") is (operation == "disconnect_nodes")
    assert design.has_node("tool")
    design.validate_graph()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("axis", [0, 1])
def test_invalid_coordinate_never_partially_moves_a_group(invalid, axis):
    design = make_design()
    before = design.model_dump()
    coordinates = [300, 400]
    coordinates[axis] = invalid
    with pytest.raises(ValueError, match="finite"):
        design.move_nodes({"root": (100, 200), "skill": tuple(coordinates)})
    assert design.model_dump() == before
    with pytest.raises(ValueError, match="finite"):
        design.move_node("root", *coordinates)
    assert design.model_dump() == before


@pytest.mark.parametrize("incomplete", ["user", "streaming", "after-cutoff"])
def test_child_history_truncation_never_keeps_an_incomplete_call(incomplete):
    previous = [
        ChatMessage(role="user", content="Earlier task", timestamp=1),
        ChatMessage(
            role="assistant",
            content="Earlier reply",
            timestamp=2,
            raw_history=[{"role": "assistant", "content": "Earlier reply"}],
        ),
    ]
    current = [ChatMessage(role="user", content="Unfinished task", timestamp=3)]
    if incomplete != "user":
        current.append(
            ChatMessage(
                role="assistant",
                content="Unfinished reply",
                timestamp=10 if incomplete == "after-cutoff" else 4,
                is_streaming=incomplete == "streaming",
            )
        )
    child = SubAgentConversation(path="main/worker", messages=[*previous, *current])
    assert child.discard_calls_from(5)
    assert child.message_history() == previous
    assert child.raw_history == previous[-1].raw_history
    assert child.status == "completed"
    child = SubAgentConversation(path="main/worker", messages=current)
    assert not child.discard_calls_from(5)
    assert child.message_history() == [] and child.raw_history == []


@pytest.mark.parametrize("save_succeeds", [False, True])
async def test_cancelled_split_finishes_disk_write_without_orphans_or_number_collisions(
    tmp_path, monkeypatch, save_succeeds
):
    manager = SessionManager(tmp_path / "chats")
    source = manager.get_current_session()
    response = ChatMessage(role="assistant", content="A completed response")
    source.add_message(response)
    manager.save_session(source)
    started, release = Event(), Event()
    saved_ids = []
    save_session = manager.save_session

    def save_with_gate(branch):
        saved_ids.append(branch.id)
        if len(saved_ids) == 1:
            started.set()
            assert release.wait(5)
            if not save_succeeds:
                return False
        return save_session(branch)

    monkeypatch.setattr(manager, "save_session", save_with_gate)
    cancelled = asyncio.create_task(manager.split_session_async(source.id, response.id))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        cancelled.cancel()
        second = asyncio.create_task(
            manager.for_view().split_session_async(source.id, response.id)
        )
        await asyncio.sleep(0)
        assert len(saved_ids) == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    branch = await second
    assert branch.split_number == 1
    assert branch.title.endswith(" (1)")
    assert manager.get_session(saved_ids[0]) is None
    assert not (manager.storage_dir / f"{saved_ids[0]}.json").exists()
    assert manager.get_session(source.id) is source
    assert SessionManager(manager.storage_dir).get_session(branch.id) == branch


async def test_invalid_collective_ui_action_keeps_design_selection_and_undo_unchanged():
    client = Client(page("/clean-code-board"))
    try:
        with client:
            designer = AgentDesigner()
            designer.build()
            await designer.open(CapabilityRepository().default_design(), lambda _: None)
            before = designer._design.model_dump()
            selection = designer._selection
            root = designer._design.root_id
            with patch.object(ui, "notify") as notify:
                designer._handle_action(
                    SimpleNamespace(
                        args={
                            "scope": designer._scope,
                            "action": "move_many",
                            "positions": [{"id": root, "x": 100, "y": 200}],
                            "ids": ["missing"],
                        }
                    )
                )
                notify.assert_called_once()
            assert designer._design.model_dump() == before
            assert designer._selection == selection
            assert not designer._history.can_undo
            designer._handle_action(SimpleNamespace(args=None))
            designer._handle_action(SimpleNamespace(args={"scope": designer._scope}))
    finally:
        client.delete()
