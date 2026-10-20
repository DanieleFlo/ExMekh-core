from pathlib import Path

from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="mock-sub-tool",
    description="Returns a result available only in the mock skill.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return "Mock sub-tool result: {text}".format(text=text)
