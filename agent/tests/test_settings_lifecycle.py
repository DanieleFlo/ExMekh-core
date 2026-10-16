"""Settings persistence and UI lifetime regressions with real async boundaries."""

from __future__ import annotations

import asyncio
from threading import Event, get_ident
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.config import AppConfig, config
from exmekh_agent.exmekh_bridge.providers.provider_health import check_provider_health
from exmekh_agent.ui.settings.settings_drawer import SettingsDrawerComponent


@pytest.fixture
def settings_view():
    client = Client(page("/settings-lifecycle"))
    callback = AsyncMock()
    drawer = SettingsDrawerComponent(callback)
    for name, value in {
        "input_base_url": "http://example.test/v1",
        "input_api_key": "test-key",
        "select_api_type": "responses",
        "select_reasoning": "low",
        "model_select": "test-model",
    }.items():
        setattr(drawer, name, MagicMock(value=value))
    drawer.save_button = MagicMock()
    drawer.close = MagicMock()
    drawer.refresh_btn = MagicMock()
    drawer.status_spinner = MagicMock()
    drawer.status_badge = MagicMock()
    with client:
        try:
            yield client, drawer, callback
        finally:
            if not client.is_deleted:
                client.delete()


async def in_client(client, operation):
    """Enter the event's client explicitly, including tasks created by the test."""
    with client:
        return await operation()


async def test_async_settings_apply_on_caller_and_persist_an_independent_snapshot(
    monkeypatch,
):
    configuration = AppConfig(model_context_lengths={"test-model": 8192})
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    caller_thread = get_ident()
    saved = []

    def save(snapshot, env_path=None):
        saved.append((get_ident(), snapshot.port, snapshot.default_model))
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(AppConfig, "save_to_env", save)
    task = asyncio.create_task(
        configuration.update_async(
            port=9079,
            default_model="test-model",
            save_to_env="must not replace a method",
            unknown_field="ignored",
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert configuration.port == 9079
        assert configuration.context_length == 8192
        assert callable(configuration.save_to_env)
        assert not hasattr(configuration, "unknown_field")
        # A later runtime change cannot change the snapshot being persisted.
        configuration.port = 9080
    finally:
        release.set()
        result = await task
    assert result
    assert saved[0][0] != caller_thread
    assert saved[0][1:] == (9079, "test-model")


async def test_failed_save_keeps_dialog_open_and_allows_retry(settings_view):
    client, drawer, callback = settings_view
    with (
        patch.object(
            config, "update_async", AsyncMock(side_effect=[False, True])
        ) as save,
        patch("exmekh_agent.ui.settings.settings_drawer.ui.notify") as notify,
    ):
        await in_client(client, drawer._save_settings)
        assert notify.call_args.kwargs["type"] == "negative"
        drawer.close.assert_not_called()
        assert not drawer._saving
        await in_client(client, drawer._save_settings)
        assert notify.call_args.kwargs["type"] == "positive"
    assert save.await_count == 2
    assert callback.await_count == 2
    drawer.close.assert_called_once()
    drawer.save_button.set_enabled.assert_called_with(True)


@pytest.mark.parametrize("phase", ["persistence", "callback"])
async def test_duplicate_save_and_client_deletion_do_not_update_dead_ui(
    settings_view,
    phase,
):
    client, drawer, callback = settings_view
    started = asyncio.Event()
    release = asyncio.Event()

    async def pending(**kwargs):
        started.set()
        await release.wait()
        return True

    save = AsyncMock(
        side_effect=pending if phase == "persistence" else None, return_value=True
    )
    if phase == "callback":
        callback.side_effect = pending
    with (
        patch.object(config, "update_async", save),
        patch("exmekh_agent.ui.settings.settings_drawer.ui.notify") as notify,
    ):
        task = asyncio.create_task(in_client(client, drawer._save_settings))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await in_client(client, drawer._save_settings)
            assert save.await_count == 1
            client.delete()
        finally:
            release.set()
            await task
        notify.assert_not_called()
    drawer.close.assert_not_called()
    assert not drawer._saving
    # Only the initial disabling happened; no controls are touched after deletion.
    drawer.save_button.set_enabled.assert_called_once_with(False)
    assert callback.await_count == (0 if phase == "persistence" else 1)


@pytest.mark.parametrize("outcome", ["changed_endpoint", "deleted_client", "error"])
async def test_model_refresh_is_exclusive_and_ignores_stale_results(
    settings_view,
    monkeypatch,
    outcome,
):
    client, drawer, _ = settings_view
    drawer.on_models_refreshed = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def health(*args):
        started.set()
        await release.wait()
        if outcome == "error":
            raise RuntimeError("probe failed")
        return True, ["stale-model"]

    with patch(
        "exmekh_agent.ui.settings.settings_drawer.check_provider_health",
        AsyncMock(side_effect=health),
    ) as probe:
        task = asyncio.create_task(in_client(client, drawer.refresh_models))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await in_client(client, drawer.refresh_models)
            assert probe.await_count == 1
            if outcome == "changed_endpoint":
                monkeypatch.setattr(config, "base_url", "http://new.test/v1")
            elif outcome == "deleted_client":
                client.delete()
        finally:
            release.set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="probe failed"):
                    await task
            else:
                await task
    assert not drawer._refreshing_models
    drawer.on_models_refreshed.assert_not_called()
    assert drawer.status_badge.text != "Online (1 models)"
    if not client.is_deleted:
        drawer.refresh_btn.props.assert_any_call(remove="loading")
        drawer.status_spinner.set_visibility.assert_called_with(False)
    else:
        drawer.status_spinner.set_visibility.assert_called_once_with(True)


async def test_storage_cleanup_rechecks_client_after_stats_refresh(settings_view):
    client, drawer, _ = settings_view
    storage = MagicMock()
    storage.clean.return_value = (1, 100)
    drawer.temp_storage_getter = lambda: storage

    async def close_during_refresh():
        client.delete()

    drawer.refresh_tmp_stats = AsyncMock(side_effect=close_during_refresh)
    with patch("exmekh_agent.ui.settings.settings_drawer.ui.notify") as notify:
        await in_client(client, drawer._handle_clean_tmp)
        notify.assert_not_called()
    storage.clean.assert_called_once()


async def test_provider_probe_respects_an_explicit_empty_api_key(monkeypatch):
    monkeypatch.setattr(config, "api_key", "configured-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": []}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)) as request:
        assert await check_provider_health("http://example.test/v1", "") == (True, [])
    assert request.call_args.kwargs["headers"] == {}


async def test_old_provider_response_does_not_replace_current_context_capacity(
    monkeypatch,
):
    monkeypatch.setattr(config, "base_url", "http://first.test/v1")
    monkeypatch.setattr(config, "api_key", "test-key")
    monkeypatch.setattr(config, "default_model", "Qwen-shared-model")
    monkeypatch.setattr(config, "context_length", 8192)
    monkeypatch.setattr(config, "model_context_lengths", {"Qwen-shared-model": 8192})
    monkeypatch.setattr(config, "model_owners", {})
    started = asyncio.Event()
    release = asyncio.Event()
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [
            {"id": "Qwen-shared-model", "owned_by": "ds4.c", "context_length": 32768}
        ]
    }

    async def request(*args, **kwargs):
        started.set()
        await release.wait()
        return response

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=request)):
        task = asyncio.create_task(check_provider_health())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            config.base_url = "http://second.test/v1"
        finally:
            release.set()
            assert await task == (True, ["Qwen-shared-model"])
    assert config.get_context_length() == 8192
    assert config.uses_ds4_qwen_prompt("Qwen-shared-model", "http://first.test/v1")
    assert not config.uses_ds4_qwen_prompt("Qwen-shared-model", config.base_url)


async def test_explicit_provider_probe_caches_capacity_for_the_probed_endpoint(monkeypatch):
    monkeypatch.setattr(config, "base_url", "http://current.test/v1")
    monkeypatch.setattr(config, "model_context_lengths", {})
    monkeypatch.setattr(config, "server_context_lengths", {})
    monkeypatch.setattr(config, "default_model", "shared")
    config.set_model_context_length("shared", 8192)
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": [{"id": "shared", "context_length": 32768}]}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
        assert await check_provider_health("http://other.test/v1/", "") == (True, ["shared"])
    assert config.get_context_length("shared") == 8192
    assert config.get_server_context_length("shared", config.base_url) == 8192
    assert config.get_server_context_length("shared", "http://other.test/v1") == 32768
    config.set_model_context_length("invalid", True)
    assert config.get_server_context_length("invalid", config.base_url) is None
