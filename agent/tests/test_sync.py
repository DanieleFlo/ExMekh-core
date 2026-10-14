from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from exmekh_agent import sync


@pytest.fixture
def source_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    exmekh_dir = tmp_path / "exmekh_lib"
    exmekh_dir.mkdir()
    (exmekh_dir / "pyproject.toml").write_text('[project]\nname = "exmekh-lib"\n')
    monkeypatch.setattr(sync, "APPLICATION_ROOT", agent_dir)
    monkeypatch.setenv("UV_BIN", "/custom/bin/uv")
    return agent_dir


def test_sync_requires_exmekh_source_project(source_projects: Path) -> None:
    (source_projects.parent / "exmekh_lib/pyproject.toml").unlink()
    with patch("exmekh_agent.sync.subprocess.run") as run:
        assert sync.sync_wheel() == 1
    run.assert_not_called()


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0"])
def test_sync_builds_current_source_and_installs_new_wheel(
    source_projects: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    wheels_dir = source_projects / "wheels"
    wheels_dir.mkdir()
    old_wheel = wheels_dir / "exmekh_lib-0.1.0-py3-none-any.whl"
    old_wheel.write_bytes(b"old wheel")
    stale_wheel = wheels_dir / "exmekh_lib-0.0.8-py3-none-any.whl"
    stale_wheel.write_bytes(b"stale dependency")
    unrelated = wheels_dir / "another-package.whl"
    unrelated.write_bytes(b"keep")
    calls: list[tuple[list[str], Path]] = []

    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        assert check
        calls.append((command, cwd))
        if command[1] == "build":
            output = Path(command[-1])
            (output / f"exmekh_lib-{version}-py3-none-any.whl").write_bytes(
                b"new source wheel"
            )

    monkeypatch.setattr(sync.subprocess, "run", run)
    assert sync.sync_wheel() == 0
    assert [command[1] for command, _ in calls] == ["build", "add", "lock", "sync"]
    assert calls[0][0][:4] == ["/custom/bin/uv", "build", "--wheel", "--out-dir"]
    assert calls[0][1] == source_projects.parent / "exmekh_lib"
    assert calls[1] == (
        [
            "/custom/bin/uv",
            "add",
            "--no-sync",
            f"wheels/exmekh_lib-{version}-py3-none-any.whl",
        ],
        source_projects,
    )
    assert calls[2] == (
        [
            "/custom/bin/uv",
            "lock",
            "--upgrade-package",
            "exmekh-lib",
            "--refresh-package",
            "exmekh-lib",
        ],
        source_projects,
    )
    assert calls[3] == (
        ["/custom/bin/uv", "sync", "--locked", "--reinstall-package", "exmekh-lib"],
        source_projects,
    )
    assert (
        wheels_dir / f"exmekh_lib-{version}-py3-none-any.whl"
    ).read_bytes() == b"new source wheel"
    if version != "0.1.0":
        assert not old_wheel.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not stale_wheel.exists()
    assert not Path(calls[0][0][-1]).exists()


@pytest.mark.parametrize("stage", ["build", "add", "lock", "sync"])
def test_sync_stops_on_uv_failure_and_preserves_previous_wheels(
    source_projects: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    wheels_dir = source_projects / "wheels"
    wheels_dir.mkdir()
    old_wheel = wheels_dir / "exmekh_lib-0.0.9-py3-none-any.whl"
    old_wheel.write_bytes(b"previous wheel")
    stale_wheel = wheels_dir / "exmekh_lib-0.0.8-py3-none-any.whl"
    stale_wheel.write_bytes(b"stale dependency")
    calls: list[str] = []

    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        calls.append(command[1])
        if command[1] == stage:
            raise subprocess.CalledProcessError(1, command)
        if command[1] == "build":
            (Path(command[-1]) / "exmekh_lib-0.1.0-py3-none-any.whl").write_bytes(
                b"new"
            )

    monkeypatch.setattr(sync.subprocess, "run", run)
    assert sync.sync_wheel() == 1
    assert (
        calls
        == ["build", "add", "lock", "sync"][
            : ["build", "add", "lock", "sync"].index(stage) + 1
        ]
    )
    assert old_wheel.read_bytes() == b"previous wheel"
    assert stale_wheel.read_bytes() == b"stale dependency"


@pytest.mark.parametrize(
    "wheel_names", [[], ["exmekh_lib-0.1.whl", "exmekh_lib-0.2.whl"]]
)
def test_sync_rejects_missing_or_ambiguous_build_output(
    source_projects: Path, monkeypatch: pytest.MonkeyPatch, wheel_names: list[str]
) -> None:
    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        for name in wheel_names:
            (Path(command[-1]) / name).write_bytes(b"wheel")

    with patch("exmekh_agent.sync.subprocess.run", side_effect=run) as subprocess_run:
        assert sync.sync_wheel() == 1
    subprocess_run.assert_called_once()
    assert not (source_projects / "wheels").exists()


def test_sync_reports_missing_uv(source_projects: Path) -> None:
    with patch(
        "exmekh_agent.sync.subprocess.run", side_effect=OSError("uv unavailable")
    ):
        assert sync.sync_wheel() == 1


def test_main_returns_sync_result() -> None:
    with (
        patch("exmekh_agent.sync.sync_wheel", return_value=1),
        patch("sys.exit") as exit_mock,
    ):
        sync.main()
    exit_mock.assert_called_once_with(1)


@pytest.mark.parametrize("script", ["run.sh", "sync_exmekh_lib.sh"])
def test_wsl_launchers_resolve_paths_and_forward_arguments(
    tmp_path: Path, script: str
) -> None:
    agent_dir = Path(__file__).resolve().parents[1]
    uv = tmp_path / "uv executable"
    uv.write_text('#!/bin/bash\nprintf "%s\n" "$PWD" "$UV_BIN" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        [str(agent_dir / script), "argument with spaces"],
        cwd=tmp_path,
        env=dict(os.environ, UV_BIN=str(uv)),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert lines[:2] == [str(agent_dir), str(uv)]
    if script == "run.sh":
        assert lines[2:] == ["run", "--frozen", "exmekh-agent", "argument with spaces"]
    else:
        assert lines[2:] == [
            "run",
            "--no-project",
            "--python",
            "3.13",
            str(agent_dir / "src/exmekh_agent/sync.py"),
            "argument with spaces",
        ]


@pytest.mark.parametrize("script", ["run.sh", "sync_exmekh_lib.sh"])
def test_wsl_launchers_report_invalid_explicit_uv(tmp_path: Path, script: str) -> None:
    agent_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(agent_dir / script)],
        cwd=tmp_path,
        env=dict(os.environ, UV_BIN=str(tmp_path / "missing-uv")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 127
    assert "uv executable not found" in result.stderr


def test_installed_frontend_uses_renamed_distribution_and_console_scripts() -> None:
    metadata = importlib.metadata.distribution("exmekh-agent")
    assert metadata.metadata["Name"] == "exmekh-agent"
    requirements = metadata.requires or []
    assert any(
        requirement.lower().startswith("exmekh-lib") for requirement in requirements
    )
    assert not any(
        requirement == "qs" or requirement.startswith("qs ")
        for requirement in requirements
    )
    scripts = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="console_scripts")
    }
    assert scripts["exmekh-agent"] == "exmekh_agent.main:main"
    assert "fe" not in scripts
    assert importlib.util.find_spec("fe") is None
    assert scripts["exmekh-lib"] == "exmekh_lib.main:main"
    assert scripts["sync-exmekh-lib"] == "exmekh_agent.sync:main"
    assert "sync-qs" not in scripts
    assert "qs" not in scripts
    assert importlib.util.find_spec("qs") is None
