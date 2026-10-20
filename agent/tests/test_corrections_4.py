from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent

from exmekh_agent.exmekh_bridge.agents.agent_runner import run_agent_turn
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.history.compaction import ModelContextCheckpoint
from exmekh_agent.exmekh_bridge.history.converter import convert_to_exmekh_messages
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    ProjectSettings,
    QueuedMessage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.state.agent_design import BOARD_DEFAULTS, AgentDesign, DesignNode
from exmekh_agent.state.design_history import DesignHistory, DesignSnapshot
from exmekh_agent.state.design_selection import DesignSelection
from exmekh_agent.state.session_manager import SessionManager
from tests.mock_client import MockProvider


@pytest.mark.parametrize("profile", ["generic", "creative", "programmer", "custom"])
def test_arrangement_orders_shared_dag_without_overlaps_and_is_undoable(
    tmp_path, profile
):
    design = CapabilityRepository(tmp_path).default_design(profile)
    free = DesignNode(kind="hds", name="unconnected")
    design.add_node(free)
    shared = design.capability("tool", "read-file")
    design.connect(free.id, shared.id)
    design.move_nodes({node.id: (0, 0) for node in design.nodes})
    before = DesignSnapshot.capture(
        design, [], DesignSelection.for_nodes(design, [free.id, shared.id])
    )
    history = DesignHistory()
    design.arrange()
    history.record(before, design, [])
    for parent, child in design.connected_nodes():
        assert child.x > parent.x
    for index, node in enumerate(design.nodes):
        for other in design.nodes[index + 1 :]:
            assert (
                abs(node.x - other.x) >= BOARD_DEFAULTS["node_width"]
                or abs(node.y - other.y) >= BOARD_DEFAULTS["node_height"]
            )
    arranged = design.model_dump()
    design.arrange()
    assert design.model_dump() == arranged
    restored = history.undo()
    assert restored.design == before.design
    assert set(restored.selection.node_ids) == {free.id, shared.id}
    with pytest.raises(ValueError, match="Unknown"):
        design.move_nodes({free.id: (99, 99), "missing": (0, 0)})
    assert design.model_dump() == arranged
    design.move_nodes({free.id: (-5, -10)})
    assert (free.x, free.y) == (0, 0)


def test_dependency_warning_uses_each_skill_scope_and_respects_custom_tools(tmp_path):
    repository = CapabilityRepository(tmp_path)
    root = DesignNode(kind="hds", name="root")
    skill = DesignNode(kind="skill", name="skill")
    read = DesignNode(kind="tool", name="read-file")
    inspect = DesignNode(kind="tool", name="inspect-file")
    custom = DesignNode(kind="tool", name="edit-file", source=str(tmp_path / "main.py"))
    unknown = DesignNode(kind="tool", name="custom-tool")
    design = AgentDesign(nodes=[root], root_id=root.id)
    design.add_node(skill, root.id)
    design.add_node(read, skill.id)
    design.add_node(inspect, root.id)
    design.add_node(custom, skill.id)
    design.add_node(unknown, skill.id)
    warnings = repository.dependency_warnings(design)
    assert len(warnings) == 1
    assert "skill: read-file requires inspect-file" in warnings[0]
    design.connect(skill.id, inspect.id)
    assert repository.dependency_warnings(design) == []
    assert not any("mock" in item.name for item in repository.catalog())


def test_optional_dependencies_can_be_absent_and_explicit_links_stay_authoritative(
    tmp_path,
):
    repository = CapabilityRepository(tmp_path)
    root = DesignNode(kind="hds", name="root")
    skill = DesignNode(
        kind="skill",
        name="optional",
        content=(
            "---\noptional-sub-skills:\n- missing-skill\noptional-sub-tools:\n- absent\n- calculate\n---\nUse available tools."
        ),
    )
    design = AgentDesign(nodes=[root], root_id=root.id)
    design.add_node(skill, root.id)
    expanded = repository.expand_skill_connections(design)
    calculate = expanded.capability("tool", "calculate")
    assert calculate in expanded.children(skill.id)
    expanded.remove_node(calculate.id)
    compiled = repository.compile(expanded)
    saved = (compiled.capabilities_for(root.id) / "deep-skill/optional.md").read_text()
    assert "optional-sub-tools" not in saved
    assert "absent" not in saved


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("profile", ["generic", "custom"])
async def test_schema_switch_reaches_real_runtime_for_root_and_children(
    tmp_path, enabled, profile
):
    manager = SessionManager(tmp_path / "chats")
    project = manager.create_project(
        tmp_path,
        ProjectSettings(
            agent_profile=profile, parameter_descriptions_in_schema=enabled
        ),
    )
    session = manager.create_session(project_id=project.id)
    settings = manager.execution_settings(session)
    assert settings.parameter_descriptions_in_schema is enabled
    assert (
        SessionManager(tmp_path / "chats")
        .execution_settings(session)
        .parameter_descriptions_in_schema
        is enabled
    )
    captured = []

    def create_agent(**kwargs):
        agent = DeepAgent(**kwargs)
        captured.extend(agent.capabilities_by_node.values())
        return agent

    with (
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.DeepAgent",
            side_effect=create_agent,
        ),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=MockProvider(),
        ),
    ):
        await run_agent_turn(
            "Hello",
            "test",
            lambda _: None,
            lambda *_: None,
            lambda _: None,
            agent_mode="deep_agent",
            settings=settings,
        )
    assert len(captured) == 2
    assert all(data.parameter_descriptions_in_schema is enabled for data in captured)


def branch_source() -> ChatSession:
    first_raw = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "First reply"},
    ]
    return ChatSession(
        title="Conversation",
        system_prompt="Instructions",
        messages=[
            ChatMessage(id="u1", role="user", content="First", timestamp=10),
            ChatMessage(
                id="r1",
                role="assistant",
                content="First reply",
                timestamp=20,
                raw_history=first_raw,
            ),
            ChatMessage(id="u2", role="user", content="Second", timestamp=30),
            ChatMessage(
                id="r2",
                role="assistant",
                content="Second reply",
                timestamp=40,
                raw_history=[
                    *first_raw,
                    {"role": "user", "content": "Second"},
                    {"role": "assistant", "content": "Second reply"},
                ],
            ),
        ],
        sub_agent_conversations=[
            SubAgentConversation(
                id="child",
                path="main/coding-agent",
                status="completed",
                messages=[
                    ChatMessage(role="user", content="Old child input", timestamp=11),
                    ChatMessage(
                        role="assistant",
                        content="Old child reply",
                        timestamp=12,
                        raw_history=[
                            {"role": "assistant", "content": "Old child reply"}
                        ],
                    ),
                    ChatMessage(
                        role="user", content="Future child input", timestamp=31
                    ),
                    ChatMessage(
                        role="assistant", content="Future child reply", timestamp=32
                    ),
                ],
            )
        ],
        queued_messages=[QueuedMessage(text="Pending input")],
        settings=ChatSettings(approved_commands_history=["future command"]),
    )


@pytest.mark.parametrize("boundary,count", [("r1", 2), ("r2", 4)])
def test_split_preserves_prefix_and_its_child_history_without_sharing_mutable_data(
    boundary, count
):
    source = branch_source()
    source.messages[1].model_context = ModelContextCheckpoint(
        history=source.messages[1].raw_history
    )
    source.messages[1].skills = [SkillNode(name="deep-thinking", path="/deep-thinking")]
    source.messages[1].tool_events = [
        ToolEvent(tool_name="calculate", arguments={"expression": "1 + 1"})
    ]
    source.messages[0].image_paths = ["shared-image.png"]
    before = source.model_dump()
    branch = source.split_after(boundary, 1)
    assert branch.id != source.id
    assert branch.title == "Conversation (1)"
    assert len(branch.messages) == count
    assert branch.messages == source.messages[:count]
    replay = str(
        [
            message.message_dump()
            for message in convert_to_exmekh_messages(branch.messages)
        ]
    )
    assert ("Second reply" in replay) is (count == 4)
    assert not branch.queued_messages
    assert not branch.settings.approved_commands_history
    child = branch.sub_agent_conversations[0]
    assert len(child.messages) == count
    assert child.raw_history == child.messages[-1].raw_history
    branch.messages[1].raw_history.clear()
    branch.sub_agent_conversations[0].messages[1].content = "Changed"
    assert source.model_dump() == before
    for invalid in ("u1", "missing"):
        with pytest.raises(ValueError, match="completed"):
            source.split_after(invalid, 2)
    source.messages[-1].is_streaming = True
    with pytest.raises(ValueError, match="completed"):
        source.split_after("r2", 2)
    source.messages[-1].is_streaming = False
    source.messages[-1].error = "failure"
    with pytest.raises(ValueError, match="completed"):
        source.split_after("r2", 2)


async def test_split_numbering_persistence_failure_and_concurrent_views(
    tmp_path, monkeypatch
):
    manager = SessionManager(tmp_path / "chats")
    project = manager.create_project(tmp_path)
    source = branch_source()
    source.project_id = project.id
    source.settings = manager.execution_settings(source)
    manager.sessions[source.id] = source
    manager.save_session(source)
    view = manager.for_view()
    first, second = await asyncio.gather(
        manager.split_session_async(source.id, "r1"),
        view.split_session_async(source.id, "r2"),
    )
    assert {first.title, second.title} == {"Conversation (1)", "Conversation (2)"}
    third = await manager.split_session_async(first.id, "r1")
    assert third.title == "Conversation (3)"
    assert third.project_id == project.id
    assert manager.current_session_id != third.id
    restored = SessionManager(tmp_path / "chats")
    assert restored.get_session(third.id) == third
    assert await manager.split_session_async("missing", "r1") is None
    before = set(manager.sessions)
    monkeypatch.setattr(manager, "save_session", lambda _: False)
    assert await manager.split_session_async(source.id, "r1") is None
    assert set(manager.sessions) == before


async def test_deleted_source_during_split_does_not_reappear(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path / "chats")
    source = branch_source()
    manager.sessions[source.id] = source
    save = manager.save_session
    saved = []

    def delete_source(branch):
        saved.append(branch.id)
        manager.sessions.pop(source.id)
        return save(branch)

    monkeypatch.setattr(manager, "save_session", delete_source)
    assert await manager.split_session_async(source.id, "r1") is None
    assert not (manager.storage_dir / f"{saved[0]}.json").exists()
