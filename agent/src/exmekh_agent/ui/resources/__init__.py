"""Load presentation resources from the installed package, independently of cwd."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

RESOURCE_DIR = Path(__file__).parent
_TEMPLATES = Environment(
    loader=FileSystemLoader(RESOURCE_DIR / "templates"),
    autoescape=True,
    undefined=StrictUndefined,
    keep_trailing_newline=False,
)


@cache
def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((RESOURCE_DIR / relative_path).read_text(encoding="utf-8"))


def presentation_config(name: str) -> dict[str, Any]:
    """Return an owned copy of a named presentation configuration."""
    return deepcopy(_read_json(f"config/{name}.json"))


def ui_text(key: str, **values: object) -> str:
    """Resolve a catalog entry, formatting dynamic values only when supplied."""
    catalog, entry = key.split(".", 1)
    text = _read_json(f"text/{catalog}.json")[entry]
    return text.format(**values) if values else text


def render_html(template: str, **values: object) -> str:
    """Escape values by default; only sanitized fragments may be marked safe."""
    return _TEMPLATES.get_template(template).render(**values)
