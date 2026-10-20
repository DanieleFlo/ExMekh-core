from pathlib import Path

from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="nested-sub-tool",
    description="Returns a result available only in the nested skill.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return "Nested sub-tool result: {text}".format(text=text)
