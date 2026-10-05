"""Packaged instructions used at the model integration boundary."""

from pathlib import Path

from exmekh_agent.utils.resources import read_text


def load_prompt(name: str) -> str:
    return read_text(Path(__file__).parent / "prompts" / f"{name}.md")
