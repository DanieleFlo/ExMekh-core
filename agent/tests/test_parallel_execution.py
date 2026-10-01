from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.parallel_execution import pytest_xdist_auto_num_workers

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_isolated_suite(
    directory: Path, workers: int
) -> subprocess.CompletedProcess[str]:
    """Exercise the scheduler in real pytest workers without application fixtures."""
    (directory / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    serial: exclusive test\n"
        "    live: real provider\n    mock: mock provider\n",
        encoding="utf-8",
    )
    conftest = directory / "conftest.py"
    if not conftest.exists():
        conftest.write_text(
            'pytest_plugins = ["tests.parallel_execution"]\n', encoding="utf-8"
        )
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            str(workers),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(directory / "pytest-temp"),
        ],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_parallel_mock_tests_overlap_and_exclusive_tests_wait_for_teardown(
    tmp_path: Path,
) -> None:
    (tmp_path / "conftest.py").write_text(
        """
import time
from pathlib import Path

import pytest

pytest_plugins = ["tests.parallel_execution"]
ROOT = Path(__file__).parent

@pytest.fixture(scope="session", autouse=True)
def worker_lifetime(worker_id):
    yield
    time.sleep(0.2)
    (ROOT / f"teardown-{worker_id}").touch()
""",
        encoding="utf-8",
    )
    for name, peer, marker in [
        ("a", "b", ""),
        ("b", "a", "@pytest.mark.live\n@pytest.mark.mock"),
    ]:
        (tmp_path / f"test_parallel_{name}.py").write_text(
            f"""
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent

@pytest.fixture(autouse=True)
def active_test():
    active = ROOT / "active-{name}"
    active.touch()
    yield
    time.sleep(0.2)
    active.unlink()
    (ROOT / "done-{name}").touch()

{marker}
def test_parallel(active_test):
    deadline = time.monotonic() + 10
    while not (ROOT / "active-{peer}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (ROOT / "active-{peer}").exists(), "independent tests did not overlap"
    assert not (ROOT / "exclusive-started").exists()
    (ROOT / "overlap-{name}").touch()
    while not (ROOT / "overlap-{peer}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (ROOT / "overlap-{peer}").exists()
""",
            encoding="utf-8",
        )
    (tmp_path / "test_exclusive.py").write_text(
        """
from pathlib import Path

import pytest

ROOT = Path(__file__).parent

@pytest.fixture(autouse=True)
def exclusive_test():
    assert not list(ROOT.glob("active-*"))
    assert (ROOT / "done-a").exists() and (ROOT / "done-b").exists()
    assert len(list(ROOT.glob("teardown-*"))) == 2
    (ROOT / "exclusive-started").touch()
    active = ROOT / "exclusive-active"
    assert not active.exists()
    active.touch()
    yield
    active.unlink()

@pytest.mark.serial
def test_serial():
    pass

@pytest.mark.live
def test_real_provider_marker():
    # This synthetic test contacts no provider; it checks live scheduling.
    pass
""",
        encoding="utf-8",
    )
    result = run_isolated_suite(tmp_path, 3)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout


@pytest.mark.parametrize(
    ("workers", "parallel"), [(0, True), (1, True), (3, True), (3, False)]
)
def test_exclusive_phase_handles_single_items_and_worker_counts(
    tmp_path: Path, workers: int, parallel: bool
) -> None:
    if parallel:
        (tmp_path / "test_a_parallel.py").write_text(
            """
from pathlib import Path

import pytest

ROOT = Path(__file__).parent

@pytest.fixture(autouse=True)
def finish_parallel():
    yield
    (ROOT / "parallel-finished").touch()

def test_parallel():
    pass
""",
            encoding="utf-8",
        )
    (tmp_path / "test_z_serial.py").write_text(
        f"""
from pathlib import Path

import pytest

@pytest.mark.serial
def test_serial():
    if {parallel!r}:
        assert (Path(__file__).parent / "parallel-finished").exists()
""",
        encoding="utf-8",
    )
    result = run_isolated_suite(tmp_path, workers)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{1 + parallel} passed" in result.stdout


def test_replacement_worker_cannot_overlap_serial_tests_with_recovered_work(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_parallel.py").write_text(
        """
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent

def test_a_crashes_once():
    marker = ROOT / "crashed"
    if not marker.exists():
        marker.touch()
        os._exit(7)

def test_b_recovered_work():
    # Let the replacement worker finish collection while this module runs.
    time.sleep(2)
    assert not (ROOT / "exclusive-started").exists()

def test_c_more_recovered_work():
    assert not (ROOT / "exclusive-started").exists()

def test_d_finish_recovered_work():
    (ROOT / "parallel-finished").touch()
""",
        encoding="utf-8",
    )
    (tmp_path / "test_serial.py").write_text(
        """
from pathlib import Path

import pytest

@pytest.mark.serial
def test_serial():
    root = Path(__file__).parent
    assert (root / "parallel-finished").exists()
    (root / "exclusive-started").touch()
""",
        encoding="utf-8",
    )
    result = run_isolated_suite(tmp_path, 2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 5 passed" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("cpu_count", [None, 1, 8, 32])
def test_auto_workers_use_available_logical_cpus(monkeypatch, cpu_count):
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setattr("os.process_cpu_count", lambda: cpu_count)
    assert pytest_xdist_auto_num_workers(None) == (cpu_count or 1)


def test_auto_worker_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "16")
    monkeypatch.setattr("os.process_cpu_count", lambda: 8)
    assert pytest_xdist_auto_num_workers(None) == 16
