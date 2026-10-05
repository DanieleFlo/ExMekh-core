from __future__ import annotations

from copy import deepcopy

import pytest

from exmekh_agent.state.agent_design import AgentDesign, DesignLink, DesignNode
from exmekh_agent.state.design_history import DesignHistory, DesignSnapshot


def make_design():
    nodes = [
        DesignNode(id="root", kind="hds", name="root"),
        DesignNode(id="worker", kind="hds", name="worker"),
        DesignNode(id="free-worker", kind="hds", name="free-worker"),
        DesignNode(id="skill", kind="skill", name="skill"),
        DesignNode(id="nested", kind="skill", name="nested"),
        DesignNode(id="tool", kind="tool", name="tool"),
        DesignNode(id="free-tool", kind="tool", name="free-tool"),
    ]
    return AgentDesign(
        nodes=nodes,
        root_id="root",
        skill_connections_explicit=True,
        links=[
            DesignLink(parent="root", child="worker"),
            DesignLink(parent="root", child="skill"),
            DesignLink(parent="skill", child="nested"),
            DesignLink(parent="root", child="tool"),
        ],
    )


@pytest.mark.parametrize(
    "parent,expected",
    [
        ("root", {"free-worker", "nested", "free-tool"}),
        ("worker", {"free-worker", "skill", "nested", "tool", "free-tool"}),
        ("skill", {"tool", "free-tool"}),
        ("nested", {"tool", "free-tool"}),
        ("tool", set()),
    ],
)
def test_connection_candidates_respect_types_ownership_cycles_and_existing_links(
    parent, expected
):
    design = make_design()
    before = design.model_dump()
    targets = design.connection_targets(parent)
    assert {node.id for node in targets} == expected
    assert design.model_dump() == before
    for node in targets:
        candidate = design.model_copy(deep=True)
        candidate.connect(parent, node.id)
        candidate.validate_graph()


def test_shared_skills_and_tools_remain_available_but_owned_hds_does_not():
    design = make_design()
    design.connect("worker", "nested")
    assert "worker" not in {
        node.id for node in design.connection_targets("free-worker")
    }
    assert {"nested", "tool"} <= {
        node.id for node in design.connection_targets("free-worker")
    }
    design.connect("worker", "free-worker")
    assert "free-worker" not in {node.id for node in design.connection_targets("root")}
    with pytest.raises(ValueError, match="Unknown"):
        design.connection_targets("missing")


def test_every_edit_restores_nodes_links_root_history_catalog_and_selection():
    design = make_design()
    catalog = [DesignNode(kind="tool", name="catalog-tool", content="original source")]
    history = DesignHistory()
    snapshots = []

    def edit(operation, selected="root"):
        before = DesignSnapshot.capture(design, catalog, selected)
        snapshots.append(deepcopy(before))
        operation()
        history.record(before, design, catalog)

    assert history.undo() is None and not history.can_undo
    edit(lambda: design.move_node("skill", 400, 600), "skill")
    edit(lambda: design.disconnect(design.connections("worker")[0]), "worker")
    edit(lambda: design.connect("worker", "free-tool"), "worker")
    edit(lambda: design.set_persistent_history("worker", False), "worker")
    edit(lambda: design.set_root("worker"))
    edit(lambda: design.remove_node("tool"), "tool")
    edit(
        lambda: design.add_node(DesignNode(id="new", kind="hds", name="new"), "worker")
    )
    edit(
        lambda: design.replace_node("skill", DesignNode(kind="skill", name="edited")),
        "skill",
    )
    edit(
        lambda: catalog.append(
            DesignNode(kind="tool", name="copy", content="copied source")
        )
    )
    assert history.can_undo
    for expected in reversed(snapshots):
        restored = history.undo()
        assert restored == expected
        restored.design.validate_graph()
    assert history.undo() is None and not history.can_undo


def test_undo_ignores_noop_changes_and_reset_discards_previous_editor_session():
    design = make_design()
    catalog = []
    history = DesignHistory()
    before = DesignSnapshot.capture(design, catalog, "root")
    design.move_node("root", design.node("root").x, design.node("root").y)
    history.record(before, design, catalog)
    assert not history.can_undo
    design.move_node("root", 200, 300)
    history.record(before, design, catalog)
    assert history.can_undo
    assert before.design.node("root").x == 40
    history.clear()
    assert history.undo() is None
