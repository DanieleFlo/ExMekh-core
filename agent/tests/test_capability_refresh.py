from __future__ import annotations

import asyncio
from threading import Event, get_ident
from unittest.mock import patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRefresh,
    CapabilityRepository,
)
from exmekh_agent.state.agent_design import AgentDesign, DesignNode
from exmekh_agent.ui.agents.agent_designer import AgentDesigner


def make_design(*nodes: DesignNode) -> AgentDesign:
    root = DesignNode(kind="hds", name="root")
    design = AgentDesign(
        nodes=[root, *nodes], root_id=root.id, skill_connections_explicit=True
    )
    for node in nodes:
        design.connect(root.id, node.id)
    return design


def test_catalog_reload_compares_definitions_and_respects_undone_local_copies(tmp_path):
    repository = CapabilityRepository(tmp_path)
    tool = DesignNode(kind="tool", name="tool", source="tool.py", content="old")
    copy = DesignNode(kind="tool", name="copy", source="copy.py")
    design = make_design(tool)
    before = design.model_dump()
    with patch.object(
        repository,
        "catalog",
        return_value=[copy, tool.model_copy(update={"id": "fresh"})],
    ):
        assert repository.refresh_from_disk(design, [tool], {copy.source}) is None
    latest = tool.model_copy(update={"content": "new"})
    with patch.object(repository, "catalog", return_value=[copy, latest]):
        refreshed = repository.refresh_from_disk(design, [tool, copy], {copy.source})
    assert refreshed is not None
    assert refreshed.catalog == [copy, latest]
    assert refreshed.design.node(tool.id).content == "new"
    assert design.model_dump() == before


@pytest.mark.parametrize(
    "metadata", ["sub-tools: read", "sub-tools: false", "sub-skills: [5]"]
)
def test_refresh_rejects_invalid_dependencies_without_changing_the_board(
    tmp_path, metadata
):
    skill = DesignNode(
        kind="skill", name="skill", source="skill.md", content="Instructions"
    )
    design = make_design(skill)
    before = design.model_dump()
    latest = skill.model_copy(
        update={"content": "---\n" + metadata + "\n---\nNew instructions"}
    )
    with pytest.raises(ValueError, match="list of capability names"):
        CapabilityRepository(tmp_path).refresh_design(design, [latest])
    assert design.model_dump() == before


def test_linking_an_existing_skill_keeps_its_customized_dependencies(tmp_path):
    parent = DesignNode(
        kind="skill", name="parent", source="parent.md", content="Parent"
    )
    nested = DesignNode(
        kind="skill",
        name="nested",
        source="nested.md",
        content="---\nsub-tools: [leaf]\n---\nNested",
    )
    leaf = DesignNode(kind="tool", name="leaf")
    design = make_design(parent, nested, leaf)
    latest = parent.model_copy(
        update={"content": "---\nsub-skills: [nested]\n---\nParent"}
    )
    repository = CapabilityRepository(tmp_path)
    refreshed = repository.refresh_design(design, [latest, nested, leaf])
    assert refreshed.node(nested.id) in refreshed.children(parent.id)
    assert refreshed.children(nested.id) == []
    wrapper = DesignNode(
        kind="skill", name="wrapper", content="---\nsub-skills: [nested]\n---\nWrapper"
    )
    extended = repository.add_to_design(
        design, wrapper, design.root_id, [nested, leaf, wrapper]
    )
    assert extended.children(nested.id) == []


def test_refresh_prunes_obsolete_skill_even_if_its_source_also_changed(tmp_path):
    parent = DesignNode(
        kind="skill",
        name="parent",
        source="parent.md",
        content="---\nsub-skills: [nested]\n---\nParent",
    )
    nested = DesignNode(
        kind="skill", name="nested", source="nested.md", content="Nested"
    )
    root = DesignNode(kind="hds", name="root")
    design = AgentDesign(nodes=[root, parent, nested], root_id=root.id)
    design.connect(root.id, parent.id)
    design.connect(parent.id, nested.id)
    latest_parent = parent.model_copy(update={"content": "Parent without dependencies"})
    latest_nested = nested.model_copy(
        update={"content": "---\nsub-tools: [leaf]\n---\nNested"}
    )
    leaf = DesignNode(kind="tool", name="leaf")
    refreshed = CapabilityRepository(tmp_path).refresh_design(
        design, [latest_parent, latest_nested, leaf]
    )
    assert {node.name for node in refreshed.nodes} == {"root", "parent"}
    assert refreshed.has_node(root.id) and not refreshed.has_node(nested.id)
    assert refreshed.connected_nodes() == [
        (refreshed.node(root.id), refreshed.node(parent.id))
    ]
    refreshed.validate_connected()


@pytest.mark.parametrize("action", ["reopen", "close", "edit", "delete"])
async def test_delayed_reload_cannot_overwrite_newer_editor_state(action):
    client = Client(page("/capability-reload"))
    started, release = Event(), Event()
    ui_thread = get_ident()
    task = None
    try:
        with client:
            designer = AgentDesigner()
            designer.build()
            await designer.open(None, lambda _: None, profile="creative")
            initial = designer._design.model_copy(deep=True)
            catalog = list(designer._catalog)

            def refresh(*_):
                assert get_ident() != ui_thread
                started.set()
                assert release.wait(5)
                stale = initial.model_copy(deep=True)
                stale.node(stale.root_id).content = "Outdated worker result"
                return CapabilityRefresh(stale, catalog)

            with patch.object(
                designer._capabilities, "refresh_from_disk", side_effect=refresh
            ):
                task = asyncio.create_task(designer._refresh_capabilities())
                assert await asyncio.get_running_loop().run_in_executor(
                    None, started.wait, 5
                )
                # A timer tick must not start a second overlapping filesystem read.
                await designer._refresh_capabilities()
                if action == "reopen":
                    with patch.object(
                        designer._capabilities, "catalog", return_value=catalog
                    ):
                        await designer.open(initial, lambda _: None, profile="creative")
                elif action == "close":
                    designer._dialog.close()
                elif action == "edit":
                    designer._design.move_node(initial.root_id, 777, 888)
                else:
                    client.delete()
                release.set()
                await task
                assert (
                    designer._design.node(initial.root_id).content
                    != "Outdated worker result"
                )
                if action == "edit":
                    assert designer._design.node(initial.root_id).x == 777
    finally:
        release.set()
        if task is not None:
            await task
        if not client.is_deleted:
            client.delete()
