"""Load presentation resources from the installed package, independently of cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.catalogs import presentation_settings

RESOURCE_DIR = Path(__file__).parent
_TEMPLATES = Environment(
    loader=FileSystemLoader(RESOURCE_DIR / "templates"),
    autoescape=True,
    undefined=StrictUndefined,
    keep_trailing_newline=False,
)


def presentation_config(name: str) -> dict[str, Any]:
    """Return an owned copy of a named presentation configuration."""
    return presentation_settings(name)


def ui_text(key: str, **values: object) -> str:
    """Resolve a catalog entry, formatting dynamic values only when supplied."""
    return app_text(key, **values)


def render_html(template: str, **values: object) -> str:
    """Escape values by default; only sanitized fragments may be marked safe."""
    return _TEMPLATES.get_template(template).render(**values)
