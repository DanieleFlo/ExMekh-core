"""The context estimator must use the native server's literal tool instructions."""

import ast
import re
from pathlib import Path

import pytest

from exmekh_agent.exmekh_bridge.resources import load_prompt


@pytest.mark.parametrize("part", ["prefix", "suffix"])
def test_qwen_tool_instructions_match_native_server(part):
    header = Path(__file__).resolve().parents[2] / "ds4_prompt_settings.h"
    macro = f"DS4_PROMPT_QWEN_TOOLS_{part.upper()}"
    body = header.read_text().split(f"#define {macro} \\\n", 1)[1]
    body = body.split("\n#define ", 1)[0]
    # These prose macros contain ordinary C string literals. Concatenation
    # and the shared escape syntax can be checked without a C toolchain.
    literals = re.findall(r'"(?:\\.|[^"\\])*"', body)
    assert literals, macro
    native = "".join(ast.literal_eval(literal) for literal in literals)
    assert load_prompt(f"ds4_tools_{part}") == native
