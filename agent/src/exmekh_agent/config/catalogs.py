"""Owned configuration snapshots and parameterized application messages."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from exmekh_agent.utils.resources import read_json

CONFIG_DIR = Path(__file__).parent


def settings(name: str) -> dict[str, Any]:
    return deepcopy(read_json(CONFIG_DIR / "resources" / f"{name}.json"))


def presentation_settings(name: str) -> dict[str, Any]:
    return deepcopy(read_json(CONFIG_DIR / "presentation" / f"{name}.json"))


def message(key: str, **values: object) -> str:
    catalog, entry = key.split(".", 1)
    text = read_json(CONFIG_DIR / "texts" / f"{catalog}.json")[entry]
    return text.format(**values) if values else text
