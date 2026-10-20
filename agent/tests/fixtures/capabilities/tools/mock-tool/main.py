from pathlib import Path

from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="mock-tool",
    description="Returns the received text as a demonstration result.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return "Mock tool result: {text}".format(text=text)
