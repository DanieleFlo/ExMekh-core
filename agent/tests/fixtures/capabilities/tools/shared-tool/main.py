from pathlib import Path

from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="shared-tool",
    description="Returns a result shared by the mock and nested skills.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return "Shared tool result: {text}".format(text=text)
