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
