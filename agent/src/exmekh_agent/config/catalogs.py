"""Owned configuration snapshots and parameterized application messages."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from exmekh_agent.package_paths import CONFIG_DIR
from exmekh_agent.utils.resources import catalog_message, read_json


def settings(name: str) -> dict[str, Any]:
    return deepcopy(read_json(CONFIG_DIR / "resources" / f"{name}.json"))


def presentation_settings(name: str) -> dict[str, Any]:
    return deepcopy(read_json(CONFIG_DIR / "presentation" / f"{name}.json"))


def message(key: str, **values: object) -> str:
    return catalog_message(CONFIG_DIR, key, **values)
