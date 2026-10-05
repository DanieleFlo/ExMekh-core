"""Own capability discovery, immutable copies and framework configuration files."""

from __future__ import annotations

import ast
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml
from exmekh_lib.core.agents.capabilities.defaults import (
    HARNESS_NAMES,
    MOCK_NAMES,
    RUNTIME_NAMES,
)
from exmekh_lib.core.agents.node import DefaultCapabilities

from exmekh_agent.config import config
from exmekh_agent.state.agent_design import AgentDesign, AgentProfile, DesignNode

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


def markdown_parts(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        header, separator, body = text[4:].partition("\n---")
        if separator:
            try:
                metadata = yaml.safe_load(header)
                if metadata is None:
                    metadata = {}
            except yaml.YAMLError as error:
                raise ValueError("Invalid capability metadata") from error
            if not isinstance(metadata, dict):
                raise ValueError("Capability metadata must be a mapping")
            return metadata, body.strip()
    return {}, text.strip()


class CapabilityRepository:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or config.data_dir
        self._library_path = DefaultCapabilities().capabilities_path

    def catalog(self, paths: list[str] | None = None) -> list[DesignNode]:
        roots = [
            self._library_path,
            Path(__file__).resolve().parents[2] / "capabilities",
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
        instructions = {
            "generic": "You are a clear, accurate and collaborative assistant. Use the available capabilities when useful.",
            "programmer": "You are a programming assistant. Inspect project instructions, write clear maintainable code, and verify changes with tests.",
            "creative": "You are a creative assistant. Explore original ideas, adapt tone and style to the user's goals, and refine drafts with concrete examples.",
            "custom": "You are a clear, accurate and collaborative assistant. Use the available capabilities when useful.",
        }
        root = DesignNode(
            kind="hds",
            name="agent-root",
            content=instructions[profile] + "\nRespond in the user's language.",
            x=50,
            y=170,
        )
        design = AgentDesign(
            nodes=[root], root_id=root.id, capability_paths=paths or []
        )
        selected = [
            "deep-thinking",
            "get_execution_history",
            "search_tool_outputs",
            "read_tool_output",
            "control_tool_execution",
        ]
        if profile in {"generic", "programmer", "custom"}:
            selected.append("bash")
        for index, name in enumerate(selected):
            item = next(
                node for node in catalog if node.name == name and node.kind != "hds"
            ).model_copy(deep=True)
            item.x, item.y = 390, 30 + index * 90
            design.add_node(item, root.id)
        if profile in {"generic", "programmer", "custom"}:
            child = next(
                node for node in catalog if node.name == "coding-agent"
            ).model_copy(deep=True)
            child.x, child.y = 50, 420
            design.add_node(child, root.id)
            bash = next(node for node in design.nodes if node.name == "bash")
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
                    raise ValueError(f"{key} must be a list of capability names")
                for name in names:
                    child = design.capability(kind, name)
                    if child is None:
                        source = catalog.get((kind, name))
                        if source is None:
                            raise ValueError(
                                f"Missing capability required by {skill.name}: {name}"
                            )
                        child = source.model_copy(
                            deep=True,
                            update={
                                "id": uuid4().hex,
                                "x": skill.x + 300,
                                "y": skill.y + index * 90,
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
            raise ValueError("Enter a capability description")
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
                "Choose a new capability name using letters, numbers, _ or -"
            )
        root = self._data_dir / "agent-capabilities"
        folder = {"hds": "hds", "skill": "deep-skill", "tool": "tools"}[node.kind]
        destination = root / folder / name
        if node.kind != "tool":
            destination = destination.with_suffix(".md")
        if destination.exists() or any(
            item.kind == node.kind and item.name == name for item in self.catalog()
        ):
            raise ValueError("A capability with this name already exists")
        if node.kind == "tool":
            if node.name in RUNTIME_TOOLS:
                raise ValueError(
                    "Runtime tools are configured by connections; their implementation is managed by the library"
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
