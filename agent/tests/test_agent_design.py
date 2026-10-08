from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from exmekh_lib.core.agents.node import DefaultCapabilities, Node, ToolHooks

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEventRouter
from exmekh_agent.exmekh_bridge.capabilities.agent_design import build_designed_agent
from exmekh_agent.exmekh_bridge.capabilities.capability_repository import (
    CapabilityRepository,
    markdown_parts,
)
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.state.agent_design import (
    AgentDesign,
    DesignLink,
    DesignNode,
)
from exmekh_agent.state.session_manager import SessionManager


@pytest.mark.parametrize("profile", ["generic", "programmer", "creative", "custom"])
def test_profiles_compile_real_library_hierarchy(tmp_path: Path, profile: str):
    with patch.object(config, "data_dir", tmp_path):
        design = CapabilityRepository().default_design(profile)
        root, children = build_designed_agent(design, SubAgentEventRouter(None), {})
        assert isinstance(root, Node)
        for node in [root, *children]:
            data = node.load_capabilities(work_root=tmp_path)
            assert data.get_hds(node.name_hds).body
            assert (
                "deep-thinking" in {skill.name for skill in data.deep_skills}
                or node != root
            )
        assert bool(children) == (profile != "creative")
        if children:
            assert children[0].persistent_history
            assert children[0].path == "agent-root/coding-agent"


def test_chat_override_is_persistent_independent_and_new_chats_inherit(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    project = manager.create_project(
        tmp_path, ProjectSettings(agent_profile="programmer")
    )
    chat = manager.create_session(project_id=project.id)
    assert manager.execution_settings(chat).agent_profile == "programmer"
    assert manager.save_chat_agent(chat.id, "creative")
    assert manager.execution_settings(chat).agent_profile == "creative"
    assert (
        manager.create_session(project_id=project.id).settings.agent_profile
        == "programmer"
    )
    assert manager.save_project_settings(
        project.id, ProjectSettings(agent_profile="generic")
    )
    assert manager.execution_settings(chat).agent_profile == "creative"
    restored = SessionManager(tmp_path / "chats").get_session(chat.id)
    assert (
        restored.settings.agent_override
        and restored.settings.agent_profile == "creative"
    )
    assert not manager.save_chat_agent("missing", "generic")
    with patch.object(manager, "save_session", return_value=False):
        assert not manager.save_chat_agent(chat.id, "generic")
    assert chat.settings.agent_profile == "creative"


def test_graph_rejects_invalid_ownership_root_names_cycles_and_disconnected_nodes():
    root = DesignNode(kind="hds", name="root")
    child = DesignNode(kind="hds", name="child")
    tool = DesignNode(kind="tool", name="bash")
    design = AgentDesign(nodes=[root, child, tool], root_id=root.id)
    design.connect(root.id, child.id)
    design.connect(root.id, child.id)
    design.connect(child.id, tool.id)
    design.validate_connected()
    for parent, target in [
        (tool.id, child.id),
        (child.id, root.id),
        (root.id, root.id),
        ("missing", tool.id),
    ]:
        with pytest.raises(ValueError):
            design.connect(parent, target)
    second = DesignNode(kind="hds", name="second")
    design.nodes.append(second)
    with pytest.raises(ValueError, match="single parent"):
        design.connect(second.id, child.id)
    with pytest.raises(ValueError, match="Connect every"):
        design.validate_connected()
    design.connect(root.id, second.id)


def test_graph_errors_and_cycles():
    root = DesignNode(kind="hds", name="root")
    tool = DesignNode(kind="tool", name="tool")
    with pytest.raises(ValueError, match="unique"):
        AgentDesign(nodes=[root, root], root_id=root.id).validate_graph()
    with pytest.raises(ValueError, match="unique"):
        AgentDesign(
            nodes=[root, root.model_copy(update={"id": "other"})], root_id=root.id
        ).validate_graph()
    for nodes, root_id in [([root], "missing"), ([tool], tool.id)]:
        with pytest.raises(ValueError, match="root"):
            AgentDesign(nodes=nodes, root_id=root_id).validate_graph()
    invalid = root.model_copy(update={"name": "../escape"})
    with pytest.raises(ValueError, match="letters"):
        AgentDesign(nodes=[invalid], root_id=invalid.id).validate_graph()
    first = DesignNode(kind="hds", name="first")
    second = DesignNode(kind="hds", name="second")
    design = AgentDesign(nodes=[root, first, second], root_id=root.id)
    design.connect(first.id, second.id)
    with pytest.raises(ValueError, match="cycles"):
        design.connect(second.id, first.id)
    with pytest.raises(ValueError, match="different existing"):
        design.connect(root.id, "missing")


def test_copies_preserve_sources_and_compile_dependency_closure(tmp_path: Path):
    with patch.object(config, "data_dir", tmp_path / "app"):
        catalog = CapabilityRepository().catalog()
        skill = next(node for node in catalog if node.name == "deep-thinking")
        tool = next(node for node in catalog if node.name == "mock-tool")
        original_skill, original_tool = (
            Path(skill.source).read_bytes(),
            Path(tool.source).read_bytes(),
        )
        edited_skill = CapabilityRepository().copy_capability(
            skill, "my-thinking", skill.content
        )
        edited_tool = CapabilityRepository().copy_capability(
            tool, "my-tool", tool.content
        )
        assert Path(skill.source).read_bytes() == original_skill
        assert Path(tool.source).read_bytes() == original_tool
        assert "name='my-tool'" in edited_tool.content
        root = DesignNode(kind="hds", name="root", content="Write clear answers.")
        custom_root = CapabilityRepository().copy_capability(
            root, "my-root", "Answer with a distinct style."
        )
        design = AgentDesign(
            nodes=[custom_root, edited_skill, edited_tool], root_id=custom_root.id
        )
        design.connect(custom_root.id, edited_skill.id)
        design.connect(custom_root.id, edited_tool.id)
        agent, _ = build_designed_agent(design, SubAgentEventRouter(None), {})
        data = agent.load_capabilities(work_root=tmp_path)
        assert "my-thinking" in {item.name for item in data.deep_skills}
        assert "my-tool" in {item.get_name() for item in data.tools}
        assert "calculate" in {item.get_name() for item in data.tools}
        available = {
            (item.kind, item.name) for item in CapabilityRepository().catalog()
        }
        assert {
            ("skill", "deep-thinking"),
            ("skill", "my-thinking"),
            ("hds", "my-root"),
        } <= available
        for name in ["my-thinking", "deep-thinking", "../invalid"]:
            with pytest.raises(ValueError):
                CapabilityRepository().copy_capability(skill, name, skill.content)
        runtime = DesignNode(kind="tool", name="bash")
        with pytest.raises(ValueError, match="Runtime tools"):
            CapabilityRepository().copy_capability(runtime, "my-bash", "pass")
        with pytest.raises(SyntaxError):
            CapabilityRepository().copy_capability(tool, "bad-tool", "invalid python(")


def test_user_paths_missing_dependencies_recursive_skills_and_catalog_ignored_entries(
    tmp_path: Path,
):
    path = tmp_path / "capabilities"
    (path / "deep-skill").mkdir(parents=True)
    (path / "tools" / "incomplete").mkdir(parents=True)
    (path / "hds").mkdir(parents=True)
    (path / "hds" / "README.txt").write_text("Ignored")
    (path / "deep-skill" / "wrapper.md").write_text(
        "---\ndescription: Wrapper\nsub-skills:\n  - deep-thinking\n  - deep-thinking\n---\nPerform analysis."
    )
    skill = next(
        node
        for node in CapabilityRepository().catalog(["", str(path)])
        if node.name == "wrapper"
    )
    root = DesignNode(kind="hds", name="root", content="Use the wrapper skill.")
    design = AgentDesign(
        nodes=[root, skill], root_id=root.id, capability_paths=[str(path)]
    )
    design.connect(root.id, skill.id)
    folder = CapabilityRepository().compile(design)
    assert (
        folder.capabilities_for(root.id) / "deep-skill" / "deep-thinking.md"
    ).is_file()
    skill.content = (
        "---\ndescription: Broken\nsub-tools:\n  - absent\n---\nInvestigate."
    )
    with pytest.raises(ValueError, match="Missing capability"):
        CapabilityRepository().compile(design)
    assert markdown_parts("No metadata") == ({}, "No metadata")
    assert markdown_parts("---\nUnclosed") == ({}, "---\nUnclosed")
    assert markdown_parts("---\n\n---\nBody") == ({}, "Body")


def test_nested_observed_nodes_history_options_and_copy_defaults(tmp_path: Path):
    design = CapabilityRepository().default_design("programmer")
    child = next(node for node in design.nodes if node.name == "coding-agent")
    child.persistent_history = False
    grandchild = DesignNode(kind="hds", name="grandchild", content="Review the result.")
    design.nodes.append(grandchild)
    design.connect(child.id, grandchild.id)
    root, children = build_designed_agent(
        design, SubAgentEventRouter(None), {"bash": ToolHooks()}, "context"
    )
    assert root.agent_context_id == "context"
    assert [node.path for node in children] == [
        "agent-root/coding-agent/grandchild",
        "agent-root/coding-agent",
    ]
    assert not children[-1].persistent_history
    manager = SessionManager(tmp_path / "chats")
    project = manager.create_project(tmp_path, ProjectSettings(agent_design=design))
    chat = manager.create_session(project_id=project.id)
    inherited = manager.execution_settings(chat)
    inherited.agent_design.nodes[0].content = "Mutated"
    assert project.settings.agent_design.nodes[0].content != "Mutated"
    assert manager.save_chat_agent(chat.id, "custom", design)
    design.nodes[0].content = "Changed source"
    assert (
        manager.execution_settings(chat).agent_design.nodes[0].content
        != "Changed source"
    )


def test_invalid_node_ids_and_metadata_are_rejected_before_writing(tmp_path: Path):
    root = DesignNode(id="../escape", kind="hds", name="root")
    with pytest.raises(ValueError, match="safe node IDs"):
        CapabilityRepository().compile(AgentDesign(nodes=[root], root_id=root.id))
    for header in ["description: [", "- one\n- two"]:
        with pytest.raises(ValueError, match="metadata"):
            markdown_parts("---\n" + header + "\n---\nBody")


def test_board_operations_preserve_invariants_and_roll_back_failed_mutations():
    root = DesignNode(kind="hds", name="root")
    child = DesignNode(kind="hds", name="child")
    tool = DesignNode(kind="tool", name="bash")
    design = AgentDesign(nodes=[root], root_id=root.id)
    design.add_node(child, root.id)
    design.add_node(tool)
    design.connect(child.id, tool.id)
    assert design.children(root.id, kind="hds") == [child]
    assert design.children(child.id) == [tool]
    assert len(design.connections(child.id)) == 2
    design.move_node(child.id, 5000, -20)
    assert child.x == 5000 and child.y == 0
    design.move_node(child.id, -20, 1500)
    assert child.x == 0 and child.y == 1500
    design.set_persistent_history(child.id, False)
    assert not child.persistent_history
    with pytest.raises(ValueError, match="Only HDS"):
        design.set_persistent_history(tool.id, False)
    with pytest.raises(ValueError, match="Unknown"):
        design.node("missing")
    before = design.model_dump()
    with pytest.raises(ValueError):
        design.add_node(DesignNode(kind="hds", name="root"), root.id)
    with pytest.raises(ValueError):
        design.replace_node(child.id, root)
    assert design.model_dump() == before
    replacement = child.model_copy(update={"id": "new-id", "name": "renamed"})
    design.replace_node(child.id, replacement)
    assert design.node(child.id).name == "renamed"
    assert design.connections(child.id)
    with pytest.raises(ValueError, match="another root"):
        design.remove_node(root.id)
    with pytest.raises(ValueError, match="HDS"):
        design.set_root(tool.id)
    design.disconnect(design.connections(tool.id)[0])
    design.set_root(child.id)
    design.connect(child.id, root.id)
    design.remove_node(root.id)
    design.remove_node(tool.id)
    design.validate_connected()
    assert len(design.nodes) == 1
    invalid = DesignLink(parent=child.id, child="missing")
    design.links.append(invalid)
    before = design.model_dump()
    with pytest.raises(ValueError):
        design.set_root(child.id)
    assert design.model_dump() == before


@pytest.mark.parametrize("header", ["false", "[]", "0"])
def test_falsey_front_matter_still_requires_a_mapping(header):
    with pytest.raises(ValueError, match="mapping"):
        markdown_parts("---\n" + header + "\n---\nBody")


@pytest.mark.parametrize(
    "declaration", ["name='declared-tool',", "", "name=None,", "name=TOOL_NAME,"]
)
def test_tool_identity_and_copy_follow_decorator_metadata_not_folder_name(
    tmp_path: Path, declaration: str
):
    folder = tmp_path / "capabilities" / "tools" / "folder-name"
    folder.mkdir(parents=True)
    content = (
        """# Preserve this comment and non-ASCII text: caffè
from pathlib import Path
from exmekh_lib.core.agents.tools.decorator import deep_tool
TOOL_NAME = None
unrelated = dict(name='declared-tool')
@deep_tool("""
        + declaration
        + """ description='Test tool')
def run(work_root: Path) -> str:
    return 'result'
"""
    )
    (folder / "main.py").write_text(content)
    repository = CapabilityRepository(tmp_path / "data")
    tool = next(
        item
        for item in repository.catalog([str(folder.parents[1])])
        if item.source == str(folder / "main.py")
    )
    expected = "declared-tool" if "name='" in declaration else "folder-name"
    assert tool.name == expected
    copied = repository.copy_capability(tool, "copied-tool", content)
    assert "dict(name='declared-tool')" in copied.content and "caffè" in copied.content
    assert (folder / "main.py").read_text() == content
    root = DesignNode(kind="hds", name="root", content="Run the available tool.")
    design = AgentDesign(nodes=[root], root_id=root.id)
    design.add_node(copied, root.id)
    with patch.object(config, "data_dir", tmp_path / "runtime"):
        agent, _ = build_designed_agent(design, SubAgentEventRouter(None), {})
        assert "copied-tool" in {
            item.get_name()
            for item in agent.load_capabilities(work_root=tmp_path).tools
        }


def test_user_tool_sharing_a_builtin_name_is_not_replaced_by_the_library(
    tmp_path: Path,
):
    folder = tmp_path / "tools" / "calculate"
    folder.mkdir(parents=True)
    content = "from pathlib import Path\nfrom exmekh_lib.core.agents.tools.decorator import deep_tool\n@deep_tool(name='calculate', description='User calculate')\ndef run(work_root: Path) -> str:\n    return 'user result'\n"
    source = folder / "main.py"
    source.write_text(content)
    root = DesignNode(kind="hds", name="root", content="Run calculate.")
    tool = DesignNode(
        kind="tool", name="calculate", source=str(source), content=content
    )
    design = AgentDesign(nodes=[root], root_id=root.id)
    design.add_node(tool, root.id)
    repository = CapabilityRepository(tmp_path / "data")
    compiled = repository.compile(design)
    assert "calculate" not in compiled.default_names_for(root.id)
    assert (
        compiled.capabilities_for(root.id) / "tools" / "calculate" / "main.py"
    ).read_text() == content
    default_tool = next(
        item for item in repository.catalog() if item.name == "mock-tool"
    )
    with pytest.raises(ValueError, match="already exists"):
        repository.copy_capability(default_tool, "calculate", default_tool.content)


def test_invalid_profile_updates_do_not_write_or_mutate_a_chat(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    chat = manager.get_current_session()
    before = chat.model_dump()
    with pytest.raises(ValueError):
        manager.save_chat_agent(chat.id, "unsupported")
    assert chat.model_dump() == before
    assert (
        SessionManager(tmp_path / "chats").get_session(chat.id).model_dump() == before
    )


def test_skill_dependencies_are_visible_shared_and_migrated_only_once(tmp_path: Path):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design("generic")
    thinking = design.capability("skill", "deep-thinking")
    calculate = design.capability("tool", "calculate")
    assert design.skill_connections_explicit
    assert calculate in design.children(thinking.id)
    assert calculate not in design.children(design.root_id)
    catalog = repository.catalog()
    mock = next(item for item in catalog if item.name == "mock-skill")
    expanded = repository.add_to_design(design, mock, design.root_id, catalog)
    mock = expanded.capability("skill", "mock-skill")
    nested = expanded.capability("skill", "nested-skill")
    shared = expanded.capability("tool", "shared-tool")
    assert nested in expanded.children(mock.id)
    assert shared in expanded.children(mock.id) and shared in expanded.children(
        nested.id
    )
    assert len([item for item in expanded.nodes if item.name == "shared-tool"]) == 1
    assert design.capability("skill", "mock-skill") is None
    before = expanded.model_dump()
    connected = repository.add_to_design(expanded, calculate, mock.id, catalog)
    assert connected.capability("tool", "calculate") in connected.children(mock.id)
    assert expanded.model_dump() == before
    connected.disconnect(
        next(link for link in connected.connections(mock.id) if link.child == nested.id)
    )
    connected.connect(connected.root_id, nested.id)
    reopened = repository.expand_skill_connections(connected)
    assert nested not in reopened.children(mock.id)
    assert reopened.model_dump() == connected.model_dump()
    legacy = AgentDesign.model_validate(
        {
            "nodes": [design.node(design.root_id).model_dump(), mock.model_dump()],
            "links": [{"parent": design.root_id, "child": mock.id}],
            "root_id": design.root_id,
        }
    )
    migrated = repository.expand_skill_connections(legacy)
    assert migrated.capability("skill", "nested-skill") in migrated.children(mock.id)
    assert not legacy.skill_connections_explicit and len(legacy.nodes) == 2


def test_skill_edits_save_prose_and_description_with_dependencies_from_connections(
    tmp_path: Path,
):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design("generic")
    catalog = repository.catalog()
    mock = next(item for item in catalog if item.name == "mock-skill")
    original = Path(mock.source).read_bytes()
    design = repository.add_to_design(design, mock, design.root_id, catalog)
    mock = design.capability("skill", "mock-skill")
    shared = design.capability("tool", "shared-tool")
    nested = design.capability("skill", "nested-skill")
    design.disconnect(
        next(link for link in design.connections(mock.id) if link.child == shared.id)
    )
    date_time = design.capability("tool", "date-time")
    design.connect(mock.id, date_time.id)
    values = repository.editable_values(mock)
    assert (
        not values.instructions.startswith("---")
        and "sub-skills:" not in values.instructions
    )
    edited = repository.copy_for_design(
        design,
        mock.id,
        "edited-skill",
        "A new description",
        "Answer clearly.\n\nKeep this instruction.",
    )
    metadata, body = markdown_parts(edited.content)
    assert metadata["description"] == "A new description"
    assert metadata["sub-skills"] == [nested.name]
    assert metadata["sub-tools"] == ["mock-sub-tool", "date-time"]
    assert body == "Answer clearly.\n\nKeep this instruction."
    assert Path(mock.source).read_bytes() == original
    design.replace_node(mock.id, edited)
    with patch.object(config, "data_dir", tmp_path / "runtime"):
        agent, _ = build_designed_agent(design, SubAgentEventRouter(None), {})
        data = agent.load_capabilities(work_root=tmp_path)
    loaded = next(item for item in data.deep_skills if item.name == "edited-skill")
    assert loaded.description == "A new description"
    assert loaded.metadata.get_sub_tool_names() == ("mock-sub-tool", "date-time")
    assert loaded.metadata.get_sub_skill_names() == ("nested-skill",)
    with pytest.raises(ValueError, match="description"):
        repository.copy_for_design(design, mock.id, "empty-description", " ", "Body")


def test_builtin_skill_connections_override_yaml_without_altering_original(
    tmp_path: Path,
):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design("generic")
    skill = design.capability("skill", "deep-thinking")
    original = Path(skill.source).read_bytes()
    calculate = design.capability("tool", "calculate")
    design.remove_node(calculate.id)
    compiled = repository.compile(design)
    content = (
        compiled.capabilities_for(design.root_id) / "deep-skill" / "deep-thinking.md"
    ).read_text()
    metadata, _ = markdown_parts(content)
    assert "calculate" not in metadata["sub-tools"]
    assert "calculate" not in compiled.default_names_for(design.root_id)
    assert Path(skill.source).read_bytes() == original
    with patch.object(config, "data_dir", tmp_path / "runtime"):
        agent, _ = build_designed_agent(design, SubAgentEventRouter(None), {})
        data = agent.load_capabilities(work_root=tmp_path)
    loaded = next(item for item in data.deep_skills if item.name == "deep-thinking")
    assert "calculate" not in loaded.metadata.get_sub_tool_names()
    assert "calculate" not in {item.get_name() for item in data.tools}


@pytest.mark.parametrize(
    "dependencies",
    ["sub-tools: calculate", "sub-skills: [4]", "sub-tools: false", "sub-skills: 0"],
)
def test_invalid_skill_dependency_metadata_is_rejected(dependencies: str):
    root = DesignNode(kind="hds", name="root")
    skill = DesignNode(
        kind="skill",
        name="broken",
        content="---\ndescription: Broken\n" + dependencies + "\n---\nBody",
    )
    design = AgentDesign(nodes=[root, skill], root_id=root.id)
    design.connect(root.id, skill.id)
    with pytest.raises(ValueError, match="list of capability names"):
        CapabilityRepository().expand_skill_connections(design)


def test_skill_connections_reject_hds_and_recursive_cycles_without_partial_changes():
    repository = CapabilityRepository()
    design = repository.default_design("generic")
    skill = design.capability("skill", "deep-thinking")
    child = design.capability("hds", "coding-agent")
    before = design.model_dump()
    with pytest.raises(ValueError, match="sub-skills and tools"):
        design.connect(skill.id, child.id)
    assert design.model_dump() == before
    looping = DesignNode(
        kind="skill",
        name="looping",
        content="---\ndescription: Recursive\nsub-skills: [looping]\n---\nBody",
    )
    with pytest.raises(ValueError, match="different existing"):
        repository.add_to_design(design, looping, design.root_id, [looping])
    assert design.model_dump() == before
    nested = DesignNode(kind="skill", name="nested", content="Nested body")
    design.add_node(nested, skill.id)
    with pytest.raises(ValueError, match="cycles"):
        design.connect(nested.id, skill.id)


def test_adding_a_tool_does_not_import_unrelated_skill_dependencies(tmp_path: Path):
    repository = CapabilityRepository(tmp_path / "data")
    design = repository.default_design("generic")
    catalog = repository.catalog()
    tool = next(
        item for item in catalog if item.kind == "tool" and item.name == "mock-sub-tool"
    )
    before = design.model_dump()
    updated = repository.add_to_design(design, tool, design.root_id, catalog)
    added = updated.capability("tool", tool.name)
    assert added in updated.children(updated.root_id)
    assert len(updated.nodes) == len(design.nodes) + 1
    assert updated.capability("skill", "mock-skill") is None
    assert design.model_dump() == before
    compiled = repository.compile(updated)
    assert tool.name in compiled.default_names_for(updated.root_id)


def test_default_board_uses_current_harness_selection(tmp_path: Path):
    repository = CapabilityRepository(tmp_path)
    catalog = repository.catalog()
    added = DesignNode(kind="tool", name="new-harness-tool")
    names = DefaultCapabilities().selected_names | {added.name}
    with (
        patch.object(repository, "catalog", return_value=[*catalog, added]),
        patch.object(
            DefaultCapabilities,
            "selected_names",
            new_callable=PropertyMock,
            return_value=names,
        ),
    ):
        design = repository.default_design("creative")
    assert design.capability("tool", added.name) in design.children(design.root_id)
    assert design.capability("skill", "file-processing") is not None
    assert design.capability("tool", "bash") is None


def test_refresh_source_dependencies_preserves_manual_connections(tmp_path: Path):
    repository = CapabilityRepository(tmp_path)
    root = DesignNode(kind="hds", name="root", source="root.md", content="Old root")
    skill = DesignNode(
        kind="skill",
        name="skill",
        source="skill.md",
        content="---\nsub-tools: [old, shared, removed-manually]\n---\nOld instructions",
    )
    old = DesignNode(kind="tool", name="old", source="old.py", content="old source")
    shared = DesignNode(kind="tool", name="shared")
    manual = DesignNode(kind="tool", name="manual")
    design = AgentDesign(
        nodes=[root, skill, old, shared, manual],
        root_id=root.id,
        skill_connections_explicit=True,
    )
    for child in [skill, shared]:
        design.connect(root.id, child.id)
    for child in [old, shared, manual]:
        design.connect(skill.id, child.id)
    latest_skill = skill.model_copy(
        update={
            "content": "---\nsub-tools: [new, removed-manually]\n---\nNew instructions"
        }
    )
    added = DesignNode(kind="tool", name="new")
    catalog = [
        root.model_copy(update={"content": "New root"}),
        latest_skill,
        old.model_copy(update={"content": "new source"}),
        added,
    ]
    before = design.model_dump()
    updated = repository.refresh_design(design, catalog)
    assert design.model_dump() == before
    assert updated.node(root.id).content == "New root"
    assert updated.node(skill.id).content == latest_skill.content
    assert updated.capability("tool", "old") is None
    assert updated.capability("tool", "shared") in updated.children(root.id)
    assert {node.name for node in updated.children(skill.id)} == {"manual", "new"}
    assert updated.capability("tool", "removed-manually") is None
    assert updated.node(skill.id).id == skill.id
    assert repository.refresh_design(updated, catalog) == updated


def test_refresh_removes_obsolete_shared_dependency_subtree(tmp_path: Path):
    repository = CapabilityRepository(tmp_path)
    root = DesignNode(kind="hds", name="root")
    parents = [
        DesignNode(
            kind="skill",
            name=name,
            source=f"{name}.md",
            content="---\nsub-skills: [nested]\n---\nInstructions",
        )
        for name in ("first", "second")
    ]
    nested = DesignNode(kind="skill", name="nested")
    leaf = DesignNode(kind="tool", name="leaf")
    design = AgentDesign(nodes=[root, *parents, nested, leaf], root_id=root.id)
    for parent in parents:
        design.connect(root.id, parent.id)
        design.connect(parent.id, nested.id)
    design.connect(nested.id, leaf.id)
    catalog = [
        parent.model_copy(update={"content": "---\nsub-skills: []\n---\nUpdated"})
        for parent in parents
    ]
    updated = repository.refresh_design(design, catalog)
    assert {node.name for node in updated.nodes} == {"root", "first", "second"}
    updated.validate_connected()
