from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.node import DefaultCapabilities, Node

from exmekh_agent.config import config


@lru_cache(maxsize=32)
def load_real_hds_system_prompt(
    capabilities_path_str: str = "",
    name_hds: str = "main",
    is_sub_agent: bool = False,
    capability_paths: tuple[str, ...] | None = None,
) -> str:
    """Load the system prompt constructed from the selected HDS capabilities."""
    work_dir = Path(config.work_root)
    target = (name_hds or "main").strip("/").split("/")[-1] or "main"
    capabilities_dir = Path(capabilities_path_str or config.capabilities_path)
    selected_paths = (
        [Path(path) for path in capability_paths]
        if capability_paths
        else [capabilities_dir]
    )
    fallback_dir = Path(__file__).resolve().parents[2] / "capabilities"

    candidates: list[tuple[list[Path], DefaultCapabilities | None, str]] = [
        (selected_paths, None, target),
        ([fallback_dir], DefaultCapabilities(), target),
        (selected_paths, None, "main"),
    ]
    for paths, defaults, hds_name in candidates:
        try:
            node = Node(
                capabilities_paths=paths,
                default_capabilities=defaults,
                name_hds=hds_name,
            )
            capabilities = node.load_capabilities(work_root=work_dir)
            hds = capabilities.get_hds(hds_name)
            content = f"{hds.get_hds()}\n\n{RUNTIME.harness_output_guidance}"
            if is_sub_agent:
                exit_tool = capabilities.get_exit_tool(hds.get_exit_tool_name())
                content = f"{content}\n\n{exit_tool.render_prompt_section()}"
            return content
        except Exception:
            continue
    return ""
