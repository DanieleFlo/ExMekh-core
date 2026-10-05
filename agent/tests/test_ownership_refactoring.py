from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.capabilities.agent_design import validate_agent_design
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
)
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.agent_design import AgentDesign, DesignLink, DesignNode
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.agents.agent_designer import AgentDesigner
from exmekh_agent.ui.agents.capability_editor import CapabilityEditor
from exmekh_agent.ui.settings.settings_drawer import SettingsDrawerComponent
from exmekh_agent.ui.shared.disclosure import Disclosure


@pytest.fixture
def client():
    instance = Client(page("/ownership-refactoring"))
    try:
        yield instance
    finally:
        if not instance.is_deleted:
            instance.delete()


async def test_board_save_validates_an_independent_snapshot_and_blocks_repeated_events(
    client,
):
    design = CapabilityRepository().default_design("generic")
    saved = []
    started, release = Event(), Event()

    def validate(snapshot):
        started.set()
        assert release.wait(5)
        snapshot.validate_connected()

    with client:
        designer = AgentDesigner()
        designer.build()
        await designer.open(design, saved.append)
        with patch(
            "exmekh_agent.ui.agents.agent_designer.validate_agent_design",
            side_effect=validate,
        ) as validation:
            task = asyncio.create_task(designer._save())
            try:
                assert await asyncio.get_running_loop().run_in_executor(
                    None, started.wait, 5
                )
                assert designer._shell.props["inert"] is True
                designer._handle_action(
                    SimpleNamespace(
                        args={
                            "scope": designer._scope,
                            "action": "move",
                            "id": design.root_id,
                            "x": 999,
                            "y": 999,
                        }
                    )
                )
                await designer._save()
            finally:
                release.set()
                await task
            validation.assert_called_once()
        assert len(saved) == 1
        assert saved[0].model_dump() == design.model_dump()
        assert saved[0] is not design
        assert not designer._dialog.value
        assert "inert" not in designer._shell.props


async def test_failed_board_validation_keeps_the_editor_available(client):
    with client:
        designer = AgentDesigner()
        designer.build()
        saved = []
        await designer.open(None, saved.append)
        with patch(
            "exmekh_agent.ui.agents.agent_designer.validate_agent_design",
            side_effect=ValueError("invalid capability"),
        ):
            await designer._save()
        assert saved == []
        assert designer._dialog.value
        assert "inert" not in designer._shell.props
        assert "persistent" not in designer._dialog.props


def test_design_validation_loads_actual_capabilities_and_rejects_disconnected_nodes(
    tmp_path,
):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design("generic")
    with patch.object(config, "data_dir", tmp_path / "runtime"):
        validate_agent_design(design)
        assert list((tmp_path / "runtime" / "agent-designs").iterdir())
    design.add_node(DesignNode(kind="tool", name="unused"))
    with patch.object(config, "data_dir", tmp_path / "invalid-runtime"):
        with pytest.raises(ValueError, match="Connect every board node"):
            validate_agent_design(design)
    assert not (tmp_path / "invalid-runtime" / "agent-designs").exists()


async def test_capability_editor_preserves_the_board_and_recovers_from_invalid_description(
    client, tmp_path
):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design()
    before = design.model_dump()
    saved = []
    with client:
        editor = CapabilityEditor(
            repository, lambda node_id, node: saved.append((node_id, node))
        )
        editor.open(design, design.root_id)
        editor._description.value = " "
        await editor._save()
        assert saved == []
        assert editor._dialog.value
        assert editor._save_button.enabled
        editor._description.value = "An independent description"
        editor._name.value = "edited-root"
        editor._instructions.value = "Clear instructions"
        await editor._save()
        assert not editor._dialog.value
    assert design.model_dump() == before
    node_id, copied = saved[0]
    assert node_id == design.root_id
    assert copied.name == "edited-root"
    assert (
        repository.editable_values(copied).description == "An independent description"
    )
    assert repository.editable_values(copied).instructions == "Clear instructions"
    assert Path(copied.source).is_file()


def test_legacy_duplicate_cards_become_shared_capabilities_without_losing_connections():
    root = DesignNode(kind="hds", name="root")
    child = DesignNode(kind="hds", name="child")
    tool = DesignNode(kind="tool", name="calculate")
    duplicate = tool.model_copy(update={"id": "duplicate"})
    design = AgentDesign(
        nodes=[root, child, tool, duplicate],
        root_id=root.id,
        links=[
            DesignLink(parent=root.id, child=child.id),
            DesignLink(parent=root.id, child=tool.id),
            DesignLink(parent=root.id, child=duplicate.id),
            DesignLink(parent=child.id, child=duplicate.id),
        ],
    )
    migrated = CapabilityRepository().expand_skill_connections(design)
    migrated.validate_connected()
    assert len(migrated.nodes) == 3
    assert migrated.children(root.id, kind="tool") == [tool]
    assert migrated.children(child.id, kind="tool") == [tool]
    assert len(migrated.links) == 3
    assert len(design.nodes) == 4


@pytest.mark.parametrize("change", [{"source": "other"}, {"content": "Different"}])
def test_conflicting_capability_definitions_do_not_mutate_the_design(change):
    root = DesignNode(kind="hds", name="root")
    tool = DesignNode(kind="tool", name="calculate")
    duplicate = tool.model_copy(update={"id": "duplicate", **change})
    design = AgentDesign(nodes=[root, tool, duplicate], root_id=root.id)
    before = design.model_dump()
    with pytest.raises(ValueError, match="Conflicting definitions"):
        design.consolidate_capabilities()
    assert design.model_dump() == before


def test_consolidation_rolls_back_when_duplicate_skills_would_form_a_cycle():
    root = DesignNode(kind="hds", name="root")
    skill = DesignNode(kind="skill", name="analysis")
    duplicate = skill.model_copy(update={"id": "duplicate"})
    design = AgentDesign(
        nodes=[root, skill, duplicate],
        root_id=root.id,
        links=[
            DesignLink(parent=root.id, child=skill.id),
            DesignLink(parent=skill.id, child=duplicate.id),
        ],
    )
    before = design.model_dump()
    with pytest.raises(ValueError, match="different existing nodes"):
        design.consolidate_capabilities()
    assert design.model_dump() == before


@pytest.mark.parametrize("kind", ["skill", "tool"])
def test_duplicate_capability_names_are_rejected_atomically(kind):
    root = DesignNode(kind="hds", name="root")
    node = DesignNode(kind=kind, name="capability")
    design = AgentDesign(nodes=[root, node], root_id=root.id)
    design.connect(root.id, node.id)
    before = design.model_dump()
    with pytest.raises(ValueError, match="unique"):
        design.add_node(DesignNode(kind=kind, name=node.name), root.id)
    assert design.model_dump() == before
    with pytest.raises(ValueError, match="unique"):
        design.add_node(root)
    assert design.model_dump() == before


def test_disclosure_owns_accessible_label_escaping(client):
    with client:
        header, panel = ui.row(), ui.column()
        label = 'Inspect "quoted" output & café'
        disclosure = Disclosure(
            header, panel, label=label, expanded=False, on_change=lambda _: None
        )
        assert header.props["aria-label"] == label
        disclosure.toggle()
        assert header.props["aria-expanded"] == "true"


@pytest.mark.parametrize("deleted", [False, True])
async def test_async_project_save_preserves_late_approvals_and_concurrent_deletions(
    tmp_path, deleted
):
    manager = SessionManager(tmp_path / "data" / "chats")
    settings = ProjectSettings(name_hds="updated-prompt")
    assert not await manager.save_project_settings_async("missing", settings)
    project = manager.create_project(tmp_path)
    chat = manager.create_session(project_id=project.id)
    started, release = Event(), Event()
    save_project = manager.save_project
    caller_thread = get_ident()
    threads = []

    def persist(updated):
        threads.append(get_ident())
        started.set()
        assert release.wait(5)
        save_project(updated)

    with patch.object(manager, "save_project", side_effect=persist):
        task = asyncio.create_task(
            manager.save_project_settings_async(project.id, settings)
        )
        try:
            assert await asyncio.get_running_loop().run_in_executor(
                None, started.wait, 5
            )
            if deleted:
                manager.delete_project(project.id)
            else:
                assert manager.record_command_approval(chat, "git status", [])
                assert manager.set_tool_approval_mode(chat.id, "Approve All")
        finally:
            release.set()
            result = await task
    assert len(threads) == 1 and threads[0] != caller_thread
    assert result is not deleted
    restored = SessionManager(tmp_path / "data" / "chats")
    if deleted:
        assert manager.get_project(project.id) is None
        assert restored.get_project(project.id) is None
        assert not (manager.projects_dir / f"{project.id}.json").exists()
    else:
        assert project.settings.name_hds == chat.settings.name_hds == "updated-prompt"
        assert chat.settings.approved_commands_history == ["git status"]
        assert chat.settings.tool_approval_mode == "Approve All"
        assert restored.get_session(chat.id).settings.approved_commands_history == [
            "git status"
        ]


async def test_provider_refresh_ignores_deleted_clients(client):
    refreshed = []
    drawer = SettingsDrawerComponent(
        lambda: None, on_models_refreshed=lambda: refreshed.append(True)
    )

    async def health(*_):
        client.delete()
        return True, ["new-model"]

    with (
        client,
        patch(
            "exmekh_agent.ui.settings.settings_drawer.check_provider_health",
            side_effect=health,
        ),
    ):
        await drawer.refresh_models()
    assert refreshed == []
