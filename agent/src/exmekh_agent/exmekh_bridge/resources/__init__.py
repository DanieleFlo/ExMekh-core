"""Packaged instructions used at the model integration boundary."""

from functools import cache
from pathlib import Path


@cache
def load_prompt(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.md").read_text(
        encoding="utf-8"
    )
