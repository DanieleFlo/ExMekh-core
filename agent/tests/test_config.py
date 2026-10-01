from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from qs.core.agents.node import DefaultCapabilities

from fe.config import AppConfig


def test_app_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.base_url.startswith("http")
    assert cfg.port > 0
    assert cfg.host != ""
    assert cfg.agent_mode in ["deep_agent", "quick_chat"]
    assert cfg.name_hds == "main"
    assert cfg.capabilities_path == str(DefaultCapabilities().capabilities_path)
    assert cfg.reasoning_effort == "xhigh"
    assert isinstance(cfg.data_dir, Path)


def test_model_owner_is_scoped_to_provider_endpoint() -> None:
    cfg = AppConfig()
    cfg.set_model_owner("Qwen-test", "ds4.c", "https://first.example/v1/")

    assert cfg.uses_ds4_qwen_prompt("Qwen-test", "https://first.example/v1")
    assert not cfg.uses_ds4_qwen_prompt("Qwen-test", "https://second.example/v1")
    assert not cfg.uses_ds4_qwen_prompt("other-model", "https://first.example/v1")
    cfg.set_model_owner("other-model", "ds4.c", "https://first.example/v1")
    assert not cfg.uses_ds4_qwen_prompt("other-model", "https://first.example/v1")
    cfg.set_model_owner("Qwen-test", "another-provider", "https://second.example/v1")
    assert not cfg.uses_ds4_qwen_prompt("Qwen-test", "https://second.example/v1")


def test_app_config_update(monkeypatch) -> None:
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("API_KEY", "dummy-key")
    monkeypatch.setenv("REASONING_EFFORT", "xhigh")
    cfg = AppConfig()
    with patch.object(AppConfig, "save_to_env"):
        cfg.update(
            base_url="http://custom-host:9999/v1",
            api_key="secret-key-123",
            port=9090,
            reasoning_effort="low",
            unknown_field="should_be_ignored",
        )
    assert cfg.base_url == "http://custom-host:9999/v1"
    assert cfg.api_key == "secret-key-123"
    assert cfg.port == 9090
    assert cfg.reasoning_effort == "low"
    assert not hasattr(cfg, "unknown_field")
    assert os.environ["BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert os.environ["API_KEY"] == "dummy-key"
    assert os.environ["REASONING_EFFORT"] == "xhigh"


def test_save_to_env_updates_known_keys_atomically(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# preserved\nBASE_URL=http://old/v1\nCUSTOM=value\n",
        encoding="utf-8",
    )
    cfg = AppConfig(
        base_url="http://new/v1",
        api_key="secret",
        default_model="model",
        reasoning_effort="high",
        api_type="responses",
        host="127.0.0.1",
        port=9000,
        app_title="QS",
    )

    cfg.save_to_env(env_path)

    content = env_path.read_text(encoding="utf-8")
    assert "# preserved" in content
    assert "BASE_URL=http://new/v1" in content
    assert "CUSTOM=value" in content
    assert "API_KEY=secret" in content
    assert "FE_PORT=9000" in content
    assert not env_path.with_suffix(".env.tmp").exists()
