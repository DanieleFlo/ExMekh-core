from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from fe.config import AppConfig, config
from fe.state.active_turns import active_turns
from fe.state.session_manager import session_manager
from tests.execution_policy import is_live_provider_test

pytest_plugins = ["tests.parallel_execution"]

_LIVE_TEST_ENV_VAR = "EXMEKH_LIB_RUN_LIVE_TESTS"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    live_group = parser.getgroup("live tests")
    live_group.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that contact the configured live model provider",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    env_value = os.getenv(_LIVE_TEST_ENV_VAR, "").strip().lower()
    if config.getoption("--run-live") or env_value in _TRUTHY_VALUES:
        return

    skip_live = pytest.mark.skip(
        reason=f"live test disabled; pass --run-live or set {_LIVE_TEST_ENV_VAR}=1"
    )
    for item in items:
        if is_live_provider_test(item):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def cleanup_test_chats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None]:
    """Give every test independent state without touching user records."""
    data_dir = tmp_path / "isolated-state"
    monkeypatch.setattr(config, "data_dir", data_dir)
    with session_manager.using_storage(data_dir / "chats"):
        try:
            yield
        finally:
            active_turns.cancel_all()


@pytest.fixture(autouse=True)
def preserve_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect default settings saves to a test file, preserving explicit paths."""
    save_to_env = AppConfig.save_to_env

    def save_test_settings(self: AppConfig, env_path: Path | None = None) -> None:
        save_to_env(self, env_path if env_path is not None else tmp_path / ".env")

    monkeypatch.setattr(AppConfig, "save_to_env", save_test_settings)
