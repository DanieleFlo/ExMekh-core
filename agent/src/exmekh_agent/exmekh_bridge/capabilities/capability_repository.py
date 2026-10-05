"""Own capability discovery, immutable copies and framework configuration files."""

from __future__ import annotations

import ast
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capabilities.defaults import (
    HARNESS_NAMES,
    MOCK_NAMES,
    RUNTIME_NAMES,
)
from exmekh_lib.core.agents.node import DefaultCapabilities

from exmekh_agent.config import config
from exmekh_agent.config.agents import DEFAULTS as AGENT_DEFAULTS
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.catalogs import presentation_settings
from exmekh_agent.config.paths import BUNDLED_CAPABILITIES_DIR
from exmekh_agent.exmekh_bridge.resources import load_prompt
from exmekh_agent.state.agent_design import AgentDesign, AgentProfile, DesignNode
from exmekh_agent.utils.markdown import markdown_parts as markdown_parts

BOARD_DEFAULTS = presentation_settings("agent_board")

RUNTIME_TOOLS = RUNTIME_NAMES


@dataclass(frozen=True)
class CompiledAgentDesign:
    """Public access to a generated design without exposing its file layout."""

    _directory: Path

    def capabilities_for(self, node_id: str) -> Path:
        return self._directory / node_id / "capabilities"

    def default_names_for(self, node_id: str) -> list[str]:
        return (
            (self._directory / node_id / "defaults.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )


@dataclass(frozen=True)
class CapabilityEditorValues:
    description: str
    instructions: str


class CapabilityRepository:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or config.data_dir
        self._library_path = DefaultCapabilities().capabilities_path

    @staticmethod
    def folder_reference(path: str) -> str:
        """Resolve a capabilities folder, accepting its parent project folder."""
        folder = Path(path).expanduser().resolve()
        if folder.name != CAPABILITY_FORMAT.root_folder:
            folder = folder / CAPABILITY_FORMAT.root_folder
        if not folder.is_dir():
            raise ValueError(
                app_text(
                    "agents.capability_editor.invalid_folder_reference",
                    name=CAPABILITY_FORMAT.root_folder,
                )
            )
        return str(folder)

    def catalog(self, paths: list[str] | None = None) -> list[DesignNode]:
        roots = [
            self._library_path,
            BUNDLED_CAPABILITIES_DIR,
        ]
        roots.extend(Path(path).expanduser() for path in paths or [] if path)
        roots.append(self._data_dir / "agent-capabilities")
        catalog: dict[tuple[str, str], DesignNode] = {}
        for root in roots:
            for kind, folder in (
                ("hds", "hds"),
                ("skill", "deep-skill"),
                ("tool", "tools"),
            ):
                for entry in sorted((root / folder).glob("*")):
                    source = entry / "main.py" if kind == "tool" else entry
                    if source.suffix not in {".py", ".md"} or not source.is_file():
                        continue
                    content = source.read_text(encoding="utf-8")
                    literal = (
                        self._tool_name_literal(content) if kind == "tool" else None
                    )
                    name = literal.value if literal is not None else entry.stem
                    catalog.setdefault(
                        (kind, name),
                        DesignNode(
                            kind=kind,
                            name=name,
                            source=str(source),
                            content=content,
                        ),
                    )
        for name in sorted(RUNTIME_TOOLS):
            catalog[("tool", name)] = DesignNode(kind="tool", name=name)
        return list(catalog.values())

    def default_design(
        self, profile: AgentProfile = "generic", paths: list[str] | None = None
    ) -> AgentDesign:
        catalog = self.catalog(paths)
        root = DesignNode(
            kind="hds",
            name=AGENT_DEFAULTS["root_name"],
            content=load_prompt(f"profile_{profile}"),
            x=BOARD_DEFAULTS["root_x"],
            y=BOARD_DEFAULTS["root_y"],
        )
        design = AgentDesign(
            nodes=[root], root_id=root.id, capability_paths=paths or []
        )
        selected = list(AGENT_DEFAULTS["standard_capabilities"])
        if profile in AGENT_DEFAULTS["coding_profiles"]:
            selected.append(AGENT_DEFAULTS["shell_tool"])
        for index, name in enumerate(selected):
            item = next(
                node for node in catalog if node.name == name and node.kind != "hds"
            ).model_copy(deep=True)
            item.x, item.y = (
                BOARD_DEFAULTS["profile_node_x"],
                BOARD_DEFAULTS["profile_node_y"]
                + index * BOARD_DEFAULTS["profile_node_vertical_gap"],
            )
            design.add_node(item, root.id)
        if profile in AGENT_DEFAULTS["coding_profiles"]:
            child = next(
                node
                for node in catalog
                if node.name == AGENT_DEFAULTS["coding_capability"]
            ).model_copy(deep=True)
            child.x, child.y = (
                BOARD_DEFAULTS["sub_agent_x"],
                BOARD_DEFAULTS["sub_agent_y"],
            )
            design.add_node(child, root.id)
            bash = next(
                node
                for node in design.nodes
                if node.name == AGENT_DEFAULTS["shell_tool"]
            )
            design.connect(child.id, bash.id)
        return self.expand_skill_connections(design)

    def expand_skill_connections(self, design: AgentDesign) -> AgentDesign:
        """Migrate implicit YAML dependencies once, preserving later board edits."""
        expanded = design.model_copy(deep=True)
        expanded.consolidate_capabilities()
        if not expanded.skill_connections_explicit:
            catalog = {
                (item.kind, item.name): item
                for item in self.catalog(design.capability_paths)
            }
            catalog.update({(item.kind, item.name): item for item in expanded.nodes})
            self._expand_dependencies(
                expanded,
                catalog,
                [item for item in expanded.nodes if item.kind == "skill"],
            )
            expanded.skill_connections_explicit = True
        expanded.validate_graph()
        return expanded

    def add_to_design(
        self,
        design: AgentDesign,
        capability: DesignNode,
        parent_id: str,
        catalog: list[DesignNode],
    ) -> AgentDesign:
        """Add a capability with visible dependencies, reusing shared cards atomically."""
        updated = design.model_copy(deep=True)
        node = updated.capability(capability.kind, capability.name)
        if node is None:
            node = capability.model_copy(deep=True, update={"id": uuid4().hex})
            updated.add_node(node, parent_id)
            if node.kind == "skill":
                available = {(item.kind, item.name): item for item in catalog}
                available.update(
                    {(item.kind, item.name): item for item in updated.nodes}
                )
                self._expand_dependencies(updated, available, [node])
        else:
            updated.connect(parent_id, node.id)
        return updated

    def _expand_dependencies(
        self,
        design: AgentDesign,
        catalog: dict[tuple[str, str], DesignNode],
        skills: list[DesignNode],
    ) -> None:
        processed: set[str] = set()
        pending = list(skills)
        while pending:
            skill = pending.pop(0)
            if skill.id in processed:
                continue
            processed.add(skill.id)
            metadata, _ = markdown_parts(skill.content)
            index = 0
            for kind, key in (("skill", "sub-skills"), ("tool", "sub-tools")):
                names = metadata.get(key)
                if names is None:
                    names = []
                if not isinstance(names, list) or any(
                    not isinstance(name, str) for name in names
                ):
                    raise ValueError(
                        app_text(
                            "errors_capabilities.capability_repository.must_be_a_list_of_capability_names",
                            value_1=key,
                        )
                    )
                for name in names:
                    child = design.capability(kind, name)
                    if child is None:
                        source = catalog.get((kind, name))
                        if source is None:
                            raise ValueError(
                                app_text(
                                    "errors_capabilities.capability_repository.missing_capability_required_by",
                                    value_1=skill.name,
                                    value_2=name,
                                )
                            )
                        child = source.model_copy(
                            deep=True,
                            update={
                                "id": uuid4().hex,
                                "x": skill.x + BOARD_DEFAULTS["skill_child_x_offset"],
                                "y": skill.y
                                + index * BOARD_DEFAULTS["skill_child_vertical_gap"],
                            },
                        )
                        design.add_node(child, skill.id)
                    else:
                        design.connect(skill.id, child.id)
                    if child.kind == "skill":
                        pending.append(child)
                    index += 1

    def editable_values(self, node: DesignNode) -> CapabilityEditorValues:
        metadata, instructions = markdown_parts(node.content)
        return CapabilityEditorValues(
            str(metadata.get("description", node.name)), instructions
        )

    def copy_for_design(
        self,
        design: AgentDesign,
        node_id: str,
        name: str,
        description: str,
        instructions: str,
    ) -> DesignNode:
        """Edit prose while deriving dependency metadata from board connections."""
        if not description.strip():
            raise ValueError(
                app_text(
                    "errors_capabilities.capability_repository.enter_a_capability_description"
                )
            )
        node = design.node(node_id)
        metadata, _ = markdown_parts(node.content)
        metadata["description"] = description.strip()
        content = self._connected_markdown(design, node, metadata, instructions)
        return self.copy_capability(node, name, content)

    @staticmethod
    def _connected_markdown(
        design: AgentDesign, node: DesignNode, metadata: dict, instructions: str
    ) -> str:
        metadata = dict(metadata)
        metadata["sub-skills"] = [
            item.name for item in design.children(node.id, kind="skill")
        ]
        metadata["sub-tools"] = [
            item.name
            for item in design.children(node.id, kind="tool")
            if node.kind == "skill" or item.name not in RUNTIME_TOOLS
        ]
        metadata.setdefault("description", node.name)
        if node.kind == "hds":
            metadata["exit-tool"] = "exit_agent"
        return (
            "---\n"
            + yaml.safe_dump(metadata, sort_keys=False)
            + "---\n"
            + instructions.strip()
            + "\n"
        )

    def copy_capability(self, node: DesignNode, name: str, content: str) -> DesignNode:
        """Copy and rename a resource; never write into the library or source path."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name == node.name:
            raise ValueError(
                app_text(
                    "errors_capabilities.capability_repository.choose_a_new_capability_name_using_letters_numbers_or"
                )
            )
        root = self._data_dir / "agent-capabilities"
        folder = {"hds": "hds", "skill": "deep-skill", "tool": "tools"}[node.kind]
        destination = root / folder / name
        if node.kind != "tool":
            destination = destination.with_suffix(".md")
        if destination.exists() or any(
            item.kind == node.kind and item.name == name for item in self.catalog()
        ):
            raise ValueError(
                app_text(
                    "errors_capabilities.capability_repository.a_capability_with_this_name_already_exists"
                )
            )
        if node.kind == "tool":
            if node.name in RUNTIME_TOOLS:
                raise ValueError(
                    app_text(
                        "errors_capabilities.capability_repository.runtime_tools_are_configured_by_connections_their_implementation"
                    )
                )
            content = self._rename_tool(content, name)
            shutil.copytree(
                Path(node.source).parent,
                destination,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            destination = destination / "main.py"
        else:
            metadata, body = markdown_parts(content)
            metadata.setdefault("description", name)
            if node.kind == "hds":
                metadata.setdefault("exit-tool", "exit_agent")
            content = (
                "---\n"
                + yaml.safe_dump(metadata, sort_keys=False)
                + "---\n"
                + body
                + "\n"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return node.model_copy(
            update={"name": name, "source": str(destination), "content": content},
            deep=True,
        )

    def compile(self, design: AgentDesign) -> CompiledAgentDesign:
        """Generate a versioned capabilities folder per HDS from a validated design."""
        design = self.expand_skill_connections(design)
        design.validate_connected()
        version = self._data_dir / "agent-designs" / uuid4().hex
        for hds in [node for node in design.nodes if node.kind == "hds"]:
            folder = version / hds.id / "capabilities"
            for kind in ("hds", "deep-skill", "tools", "exit-tool"):
                (folder / kind).mkdir(parents=True, exist_ok=True)
            selected = [node for node in design.children(hds.id) if node.kind != "hds"]
            self._write_hds(folder, hds, design)
            included = self._write_dependencies(folder, selected, design)
            if design.children(hds.id, kind="hds"):
                included.add("call_sub_agent")
            (folder.parent / "defaults.txt").write_text(
                "\n".join(sorted(included)), encoding="utf-8"
            )
        return CompiledAgentDesign(version)

    def _is_bundled(self, node: DesignNode) -> bool:
        if node.name in RUNTIME_TOOLS and not node.source:
            return True
        return node.name in HARNESS_NAMES | MOCK_NAMES and Path(
            node.source
        ).is_relative_to(self._library_path)

    def _write_hds(self, folder: Path, hds: DesignNode, design: AgentDesign) -> None:
        metadata, body = markdown_parts(hds.content)
        (folder / "hds" / f"{hds.name}.md").write_text(
            self._connected_markdown(design, hds, metadata, body), encoding="utf-8"
        )

    def _write_dependencies(
        self, folder: Path, selected: list[DesignNode], design: AgentDesign
    ) -> set[str]:
        processed: set[str] = set()
        pending = list(selected)
        included = {"exit_agent"}
        while pending:
            item = pending.pop()
            if item.id in processed:
                continue
            processed.add(item.id)
            if item.kind == "skill":
                metadata, body = markdown_parts(item.content)
                (folder / "deep-skill" / f"{item.name}.md").write_text(
                    self._connected_markdown(design, item, metadata, body),
                    encoding="utf-8",
                )
                pending.extend(design.children(item.id))
            elif self._is_bundled(item):
                included.add(item.name)
            else:
                shutil.copytree(
                    Path(item.source).parent,
                    folder / "tools" / item.name,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
        return included

    @staticmethod
    def _tool_name_literal(content: str) -> ast.Constant | None:
        """Read a declared tool name without importing or executing user code."""
        for node in ast.walk(ast.parse(content)):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func).rsplit(".", 1)[-1] != "deep_tool":
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    return keyword.value
        return None

    @classmethod
    def _rename_tool(cls, content: str, name: str) -> str:
        literal = cls._tool_name_literal(content)
        if literal is None:
            # Without an explicit name, ExMekh-Lib uses the copied folder name.
            return content
        lines = content.encode("utf-8").splitlines(keepends=True)
        start = (
            sum(len(line) for line in lines[: literal.lineno - 1]) + literal.col_offset
        )
        end = (
            sum(len(line) for line in lines[: literal.end_lineno - 1])
            + literal.end_col_offset
        )
        encoded = content.encode("utf-8")
        return (encoded[:start] + repr(name).encode("utf-8") + encoded[end:]).decode(
            "utf-8"
        )
