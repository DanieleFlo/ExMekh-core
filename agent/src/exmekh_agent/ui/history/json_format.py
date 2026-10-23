"""Syntax highlighting for recorded JSON, independent of NiceGUI controls."""

from __future__ import annotations

import html
import json
from typing import Any

from markupsafe import Markup

from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput
from exmekh_agent.ui.resources import render_html, ui_text
from exmekh_agent.ui.shared.output_format import format_raw_output_html


def _span(css_class: str, content: str | Markup) -> str:
    return render_html("shared/span.html", css_class=css_class, content=content)


def _quoted(value: object) -> Markup:
    return Markup(html.escape(json.dumps(value, ensure_ascii=False)))


def format_json_to_highlighted_html(data: Any, indent_level: int = 0) -> str:
    """Render JSON with role badges, readable tool outputs and safe nested fragments."""
    indent = "  " * indent_level
    inner_indent = "  " * (indent_level + 1)
    if data is None:
        return _span("json-val-null", "null")
    if isinstance(data, bool):
        return _span("json-val-bool", "true" if data else "false")
    if isinstance(data, (int, float)):
        return _span("json-val-num", str(data))
    if isinstance(data, str):
        trimmed = data.strip()
        if trimmed.startswith("data:image/") and "," in trimmed:
            return render_html(
                "history/image_data.html",
                data=data,
                label=ui_text("history.raw_history_view.copy_image_data"),
            )
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            try:
                return format_json_to_highlighted_html(
                    json.loads(trimmed), indent_level
                )
            except ValueError:
                pass
        if "\n" in data or trimmed.startswith("<") or RecordedToolOutput.parse(data):
            return _span("json-readable-string", Markup(format_raw_output_html(data)))
        return _span("json-val-str", _quoted(data))
    if isinstance(data, (list, dict)):
        opening, closing = ("[", "]") if isinstance(data, list) else ("{", "}")
        if not data:
            return _span("json-punct", opening + closing)
        if isinstance(data, list):
            items = [
                inner_indent + format_json_to_highlighted_html(item, indent_level + 1)
                for item in data
            ]
        else:
            items = [
                _format_pair(str(key), value, inner_indent, indent_level + 1)
                for key, value in data.items()
            ]
        return render_html(
            "history/container.html",
            opening=opening,
            closing=closing,
            items=Markup(",\n".join(items)),
            indent=indent,
        )
    return _span("json-val-str", _quoted(str(data)))


def _format_pair(key: str, value: Any, indent: str, indent_level: int) -> str:
    css_class = "json-key"
    badge = ""
    if key in {"tool_calls", "function", "tool_call_id", "arguments"}:
        css_class += " json-key-tool"
        if key == "tool_calls":
            badge = " " + _span("json-tool-badge", "TOOL")
    elif key == "role":
        css_class += " json-key-role"
    if key == "role" and isinstance(value, str):
        formatted = render_html(
            "history/role.html",
            css_role="tool" if value.lower() == "function" else value.lower(),
            role=value,
        )
    else:
        formatted = format_json_to_highlighted_html(value, indent_level)
    return render_html(
        "history/pair.html",
        indent=indent,
        css_class=css_class,
        key=_quoted(key),
        badge=Markup(badge),
        value=Markup(formatted),
    )
