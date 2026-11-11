"""Settings persistence and UI lifetime regressions with real async boundaries."""

from __future__ import annotations

import asyncio
from threading import Event, get_ident
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exmekh_agent.config import AppConfig, config
from exmekh_agent.exmekh_bridge.providers.provider_health import check_provider_health


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


async def test_provider_probe_respects_an_explicit_empty_api_key(monkeypatch):
    monkeypatch.setattr(config, "api_key", "configured-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": []}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)) as request:
        assert await check_provider_health("http://example.test/v1", "") == (True, [])
    assert request.call_args.kwargs["headers"] == {}


async def test_legacy_discovery_keeps_valid_models_after_bad_metadata(monkeypatch):
    monkeypatch.setattr(config, "model_owners", {})
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [
            {"id": "malformed", "ds4": {"operations": None}},
            {"id": "chat-model"},
        ]
    }
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
        assert await check_provider_health("http://legacy.test/v1", "") == (
            True,
            ["chat-model"],
        )


def test_selecting_same_model_on_new_endpoint_uses_new_capacity():
    configuration = AppConfig(base_url="http://first.test/v1", default_model="shared")
    configuration.set_model_context_length("shared", 8192)
    configuration.set_model_context_length(
        "shared", 32768, base_url="http://second.test/v1"
    )
    configuration.base_url = "http://second.test/v1"
    configuration.select_available_model(["shared"])
    assert configuration.get_context_length() == 32768


async def test_concurrent_settings_saves_keep_the_newest_snapshot(monkeypatch):
    configuration = AppConfig()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    saves = []

    def persist(snapshot, env_path=None):
        saves.append((snapshot.base_url, snapshot.reasoning_effort))
        if len(saves) == 1:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(AppConfig, "save_to_env", persist)
    first = asyncio.create_task(
        configuration.update_async(base_url="http://first.test/v1")
    )
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(configuration.update_async(reasoning_effort="low"))
        await asyncio.sleep(0)
        assert len(saves) == 1
    finally:
        release.set()
        await first
        if second:
            await second
    assert saves[-1] == ("http://first.test/v1", "low")


async def test_cancelled_settings_save_holds_lock_until_write_finishes(monkeypatch):
    configuration = AppConfig()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    saves = []

    def persist(snapshot, env_path=None):
        saves.append(snapshot.reasoning_effort)
        if len(saves) == 1:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(AppConfig, "save_to_env", persist)
    first = asyncio.create_task(configuration.update_async(reasoning_effort="low"))
    await asyncio.wait_for(started.wait(), timeout=2)
    first.cancel()
    second = asyncio.create_task(configuration.update_async(reasoning_effort="medium"))
    try:
        await asyncio.sleep(0)
        assert saves == ["low"]
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second
    assert saves == ["low", "medium"]


def test_discovery_preserves_capacity_when_provider_omits_context():
    configuration = AppConfig(default_model="chat", context_length=4096)
    configuration.select_available_model(["chat"])
    assert configuration.get_context_length() == 4096


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


async def test_explicit_provider_probe_caches_capacity_for_the_probed_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(config, "base_url", "http://current.test/v1")
    monkeypatch.setattr(config, "model_context_lengths", {})
    monkeypatch.setattr(config, "server_context_lengths", {})
    monkeypatch.setattr(config, "default_model", "shared")
    config.set_model_context_length("shared", 8192)
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": [{"id": "shared", "context_length": 32768}]}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
        assert await check_provider_health("http://other.test/v1/", "") == (
            True,
            ["shared"],
        )
    assert config.get_context_length("shared") == 8192
    assert config.get_server_context_length("shared", config.base_url) == 8192
    assert config.get_server_context_length("shared", "http://other.test/v1") == 32768
    config.set_model_context_length("invalid", True)
    assert config.get_server_context_length("invalid", config.base_url) is None
