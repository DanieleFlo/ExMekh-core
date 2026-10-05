"""Configuration ownership, utility failure behavior and extraction regressions."""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Formatter
from unittest.mock import patch

import pytest

from exmekh_agent.config import AppConfig
from exmekh_agent.config.catalogs import (
    CONFIG_DIR,
    message,
    presentation_settings,
    settings,
)
from exmekh_agent.config.environment import ENVIRONMENT_FIELDS
from exmekh_agent.utils.files import atomic_write_text
from exmekh_agent.utils.resources import read_json, read_text


def test_configuration_resources_are_independent_and_work_outside_repo(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    owned = settings("provider")
    owned["api_types"].append("changed")
    assert "changed" not in settings("provider")["api_types"]
    controls = presentation_settings("controls/agent_designer")
    controls.clear()
    assert presentation_settings("controls/agent_designer")
    assert message("settings.settings_drawer.online", count=3) == "Online (3 models)"
    assert message("shared.output_format.empty") == "Empty output"


def test_every_configuration_file_is_valid_and_catalogs_have_no_duplicate_keys():
    def unique_pairs(pairs):
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys))
        return dict(pairs)

    for folder in ("resources", "presentation", "texts"):
        paths = list((CONFIG_DIR / folder).rglob("*.json"))
        assert paths
        for path in paths:
            value = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
            assert isinstance(value, dict) and value


def test_environment_defaults_preserve_runtime_overrides(monkeypatch, tmp_path):
    values = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "default_model": "resource-test-model",
        "context_length": "32768",
        "api_type": "chat_completions",
        "reasoning_effort": "low",
        "host": "0.0.0.0",
        "port": "9011",
        "app_title": "Resource test",
        "coding_agent_persistent_history": "yes",
    }
    for field, value in values.items():
        monkeypatch.setenv(ENVIRONMENT_FIELDS[field], value)
    configured = AppConfig()
    assert configured.context_length == 32768 and configured.port == 9011
    assert configured.coding_agent_persistent_history
    assert configured.app_title == values["app_title"]
    path = tmp_path / ".env"
    path.write_text("# Keep me\nCUSTOM=value\n")
    configured.save_to_env(path)
    saved = path.read_text()
    assert "# Keep me\nCUSTOM=value" in saved
    assert "EXMEKH_AGENT_PORT=9011" in saved
    assert "CODING_AGENT_PERSISTENT_HISTORY=true" in saved


def test_literal_control_styles_and_ui_copy_stay_outside_python():
    for path in (CONFIG_DIR.parent / "ui").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr in {"classes", "props", "style"} and node.args:
                assert not isinstance(node.args[0], ast.Constant), (path, node.lineno)
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ui"
                and node.func.attr
                in {
                    "label",
                    "input",
                    "textarea",
                    "button",
                    "notify",
                    "badge",
                    "icon",
                    "spinner",
                }
                and node.args
            ):
                assert (
                    not isinstance(node.args[0], ast.Constant)
                    or node.args[0].value == ""
                ), (
                    path,
                    node.lineno,
                )


def test_atomic_write_default_temporary_paths_are_unique_and_removed(tmp_path):
    directory = tmp_path / "atomic"
    directory.mkdir()
    destination = directory / "record.json"
    payloads = [str(index) * 10000 for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(lambda content: atomic_write_text(destination, content), payloads)
        )
    assert destination.read_text() in payloads
    assert sorted(path.name for path in directory.iterdir()) == ["record.json"]


@pytest.mark.parametrize("operation", ["write_text", "replace"])
def test_atomic_write_failure_preserves_destination_and_cleans_up(tmp_path, operation):
    destination = tmp_path / "record.json"
    temporary = tmp_path / "pending.tmp"
    destination.write_text("previous")
    original = getattr(Path, operation)

    def fail(path, *args, **kwargs):
        if path == temporary:
            raise OSError("Cannot persist")
        return original(path, *args, **kwargs)

    with (
        patch.object(Path, operation, fail),
        pytest.raises(OSError, match="Cannot persist"),
    ):
        atomic_write_text(destination, "new", temporary_path=temporary)
    assert destination.read_text() == "previous"
    assert not temporary.exists()


def test_resource_readers_cache_without_mutating_files_and_report_missing_resources(
    tmp_path,
):
    document = tmp_path / "settings.json"
    document.write_text('{"setting": true}')
    assert read_json(document) == {"setting": True}
    assert read_text(document) == '{"setting": true}'
    # Reads are reused, and neither reader changes the file.
    assert read_json(document) == {"setting": True}
    assert document.read_text() == '{"setting": true}'
    with pytest.raises(FileNotFoundError):
        settings("absent")
    with pytest.raises(FileNotFoundError):
        presentation_settings("absent")
    with pytest.raises(KeyError):
        message("shared.absent")
    with pytest.raises(KeyError):
        message("settings.settings_drawer.online", incorrect=1)


def test_control_configuration_references_and_template_parameters_match():
    """Every control style lookup resolves, and each template gets its named values."""
    for path in (CONFIG_DIR.parent / "ui").rglob("*.py"):
        config_path = CONFIG_DIR / "presentation" / "controls" / f"{path.stem}.json"
        if not config_path.exists():
            continue
        controls = json.loads(config_path.read_text())
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "UI_PRESENTATION"
                and isinstance(node.slice, ast.Constant)
            ):
                assert node.slice.value in controls, (path, node.slice.value)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Subscript)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "UI_PRESENTATION"
            ):
                continue
            key = ast.literal_eval(node.func.value.slice)
            fields = {
                field
                for _, field, _, _ in Formatter().parse(controls[key])
                if field is not None
            }
            assert fields == {kw.arg for kw in node.keywords}, (path, key, fields)
