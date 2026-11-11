from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.providers.core_management import CoreClient, CoreProblem
from exmekh_agent.ui.navigation.header import HeaderComponent
from exmekh_agent.ui.navigation.sidebar import SidebarComponent


@pytest.fixture
def client():
    instance = Client(page("/ui-edges"))
    try:
        yield instance
    finally:
        instance.delete()


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
    client = CoreClient()
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [
            {"id": model, "owned_by": "ds4.c", "context_length": context_length}
            for model in models
        ]
    }
    with (
        patch.object(
            client, "status", AsyncMock(side_effect=CoreProblem("legacy provider", 404))
        ),
        patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response),
    ):
        connection = await client.connect()
        client.follow(connection)
    expected_model = previous_model if previous_model in models else models[0]
    assert config.default_model == expected_model
    assert config.get_context_length() == context_length
    assert config.uses_ds4_qwen_prompt(expected_model, config.base_url)
    # Discovery is runtime metadata, not an explicit settings save.
    config.save_to_env.assert_not_called()
