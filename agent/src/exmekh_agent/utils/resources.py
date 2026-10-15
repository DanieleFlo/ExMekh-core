"""Read packaged resources independently of the process working directory."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any


@cache
def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def catalog_message(directory: Path, key: str, **values: object) -> str:
    """Resolve one named message without importing runtime configuration."""
    catalog, entry = key.split(".", 1)
    text = read_json(directory / "texts" / f"{catalog}.json")[entry]
    return text.format(**values) if values else text
