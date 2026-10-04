from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exmekh_agent.config import config
from exmekh_agent.ui.header import HeaderComponent
from exmekh_agent.ui.settings_drawer import SettingsDrawerComponent
from exmekh_agent.ui.sidebar import SidebarComponent


def test_header_has_no_mode_toggle() -> None:
    header = HeaderComponent(lambda: None)
    assert not hasattr(header, "mode_toggle")


def test_sidebar_optional_callbacks_and_drawer() -> None:
    selected = []
    settings = []
    sidebar = SidebarComponent(
        lambda _: None,
        lambda: None,
        on_sub_agent_selected=selected.append,
        on_project_settings=settings.append,
    )
    sidebar.toggle()
    drawer = MagicMock()
    sidebar.drawer = drawer
    sidebar.toggle()
    drawer.toggle.assert_called_once()
    sidebar.render_sessions = MagicMock()
    sidebar._select_sub_agent("child")
    assert selected == ["child"]
    assert sidebar.selected_sub_agent_id == "child"
    sidebar.clear_sub_agent_selection()
    assert sidebar.selected_sub_agent_id is None
    sidebar._open_project_settings("project")
    assert settings == ["project"]


@pytest.mark.asyncio
async def test_settings_refresh_handles_offline_and_empty_models() -> None:
    drawer = SettingsDrawerComponent(lambda: None)
    drawer.refresh_btn = MagicMock()
    drawer.status_spinner = MagicMock()
    drawer.status_badge = MagicMock()
    drawer.model_select = MagicMock()
    with patch(
        "exmekh_agent.ui.settings_drawer.check_provider_health", new_callable=AsyncMock
    ) as health:
        health.return_value = (False, [])
        await drawer.refresh_models()
    assert drawer.status_badge.text == "Offline"
    drawer.refresh_btn.props.assert_any_call("loading")
    drawer.refresh_btn.props.assert_any_call(remove="loading")
    drawer.status_spinner.set_visibility.assert_any_call(True)
    drawer.status_spinner.set_visibility.assert_any_call(False)
    assert drawer.model_select.options != [config.default_model]


@pytest.mark.asyncio
async def test_settings_save_handles_sync_callback_and_restores_button_on_failure() -> (
    None
):
    saved = []
    drawer = SettingsDrawerComponent(lambda: saved.append(True))
    drawer.input_base_url = MagicMock(value="http://example.test/v1")
    drawer.input_api_key = MagicMock(value="secret")
    drawer.select_api_type = MagicMock(value="responses")
    drawer.select_reasoning = MagicMock(value="low")
    drawer.model_select = MagicMock(value="model")
    drawer.save_button = MagicMock()
    drawer.close = MagicMock()
    with (
        patch("exmekh_agent.ui.settings_drawer.config.update") as update,
        patch("exmekh_agent.ui.settings_drawer.ui.notify"),
    ):
        await drawer._save_settings()
    assert saved == [True]
    update.assert_called_once()
    drawer.close.assert_called_once()
    drawer.save_button.props.assert_any_call(remove="loading")

    drawer.on_settings_saved = MagicMock(side_effect=RuntimeError("callback failed"))
    with patch("exmekh_agent.ui.settings_drawer.config.update"):
        with pytest.raises(RuntimeError, match="callback failed"):
            await drawer._save_settings()
    drawer.save_button.props.assert_any_call(remove="loading")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous_model", "models", "context_length"),
    [
        (
            "Qwen3.8-27B-UD-Q4_K_S",
            [
                "qwen3.8-flash-next",
                "qwen3.8-flash-next-chat",
                "qwen3.8-flash-next-reasoner",
            ],
            16384,
        ),
        ("qwen3.8-flash-next", ["Qwen3.8-27B-UD-Q4_K_S"], 28000),
        (
            "qwen3.8-flash-next-reasoner",
            [
                "qwen3.8-flash-next",
                "qwen3.8-flash-next-chat",
                "qwen3.8-flash-next-reasoner",
            ],
            8192,
        ),
    ],
)
async def test_settings_discovery_tracks_loaded_qwen_and_allocated_context(
    monkeypatch, previous_model, models, context_length
) -> None:
    monkeypatch.setattr(config, "default_model", previous_model)
    monkeypatch.setattr(config, "context_length", 22000)
    monkeypatch.setattr(config, "model_context_lengths", {previous_model: 22000})
    monkeypatch.setattr(config, "model_owners", {})
    monkeypatch.setattr(config, "save_to_env", MagicMock())
    refreshed = []
    drawer = SettingsDrawerComponent(
        lambda: None,
        on_models_refreshed=lambda: refreshed.append(
            (config.default_model, config.get_context_length())
        ),
    )
    drawer.model_select = MagicMock(value=previous_model)
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [
            {"id": model, "owned_by": "ds4.c", "context_length": context_length}
            for model in models
        ]
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response):
        await drawer.refresh_models()
    expected_model = previous_model if previous_model in models else models[0]
    assert config.default_model == expected_model
    assert drawer.model_select.options == models
    assert drawer.model_select.value == expected_model
    assert refreshed == [(expected_model, context_length)]
    assert config.uses_ds4_qwen_prompt(expected_model, config.base_url)
    # Discovery is runtime metadata, not an explicit settings save.
    config.save_to_env.assert_not_called()
