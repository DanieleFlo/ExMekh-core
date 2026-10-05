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
