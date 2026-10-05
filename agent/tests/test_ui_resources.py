from __future__ import annotations

import ast
import json
from pathlib import Path
from string import Formatter

import pytest
from jinja2 import TemplateNotFound, UndefinedError
from markupsafe import Markup

from exmekh_agent.config.catalogs import CONFIG_DIR
from exmekh_agent.ui.history.json_format import format_json_to_highlighted_html
from exmekh_agent.ui.resources import (
    presentation_config,
    render_html,
    ui_text,
)


def test_presentation_config_is_owned_and_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = presentation_config("agent_board")
    config["colors"]["tool"] = "changed"
    assert presentation_config("agent_board")["colors"]["tool"] == "#34d399"
    assert ui_text("shared.output_format.empty") == "Empty output"
    assert ui_text("shared.output_format.empty", unused="value") == "Empty output"
    assert "output-unavailable" in render_html(
        "shared/output.html", unavailable="missing", recorded=None, body=""
    )


def test_templates_escape_values_and_require_explicit_safe_fragments():
    output = render_html(
        "shared/span.html", css_class='x" onclick="bad', content="<script>bad</script>"
    )
    assert "<script>" not in output and "&lt;script&gt;" in output
    assert 'class="x"' not in output
    output = render_html(
        "shared/span.html", css_class="safe", content=Markup("<strong>safe</strong>")
    )
    assert "<strong>safe</strong>" in output
    with pytest.raises(UndefinedError):
        render_html("shared/span.html")
    with pytest.raises(TemplateNotFound):
        render_html("missing.html")


def test_all_literal_text_keys_and_json_resources_are_valid():
    catalogs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (CONFIG_DIR / "texts").glob("*.json")
    }
    for path in (CONFIG_DIR / "presentation").rglob("*.json"):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    for path in CONFIG_DIR.parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"ui_text", "app_text"}
                and isinstance(node.args[0], ast.Constant)
            ):
                key = ast.literal_eval(node.args[0])
                catalog, entry = key.split(".", 1)
                assert entry in catalogs[catalog], (path, key)
                fields = {
                    field
                    for _, field, _, _ in Formatter().parse(catalogs[catalog][entry])
                    if field is not None
                }
                assert fields == {kw.arg for kw in node.keywords}, (path, key, fields)


def test_raw_highlighter_preserves_malformed_json_and_unknown_values():
    malformed = format_json_to_highlighted_html("{invalid}")
    unknown = format_json_to_highlighted_html(Path("<unknown>"))
    assert "{invalid}" in malformed
    assert "<unknown>" not in unknown and "&lt;unknown&gt;" in unknown
    assert "role-badge-tool" in format_json_to_highlighted_html({"role": "function"})
