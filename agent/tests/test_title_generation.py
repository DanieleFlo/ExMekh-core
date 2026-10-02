from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from nicegui.client import Client as NiceGUIClient
from nicegui.page import page
from pydantic import ValidationError

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agent_runner import (
    generate_chat_title,
    generate_chat_title_sync,
)
from exmekh_agent.exmekh_bridge.models import ChatTitle
from exmekh_agent.main import index_page
from exmekh_agent.state.session_manager import session_manager
from tests.mock_client import MockProvider


@pytest.fixture
def app_client():
    c = NiceGUIClient(page("/test-title"))
    with c:
        yield c


def test_chat_title_model() -> None:
    """Verifies that ChatTitle validates and serializes correctly."""
    title_obj = ChatTitle(title="Algoritmo di Ordinamento")
    assert title_obj.title == "Algoritmo di Ordinamento"

    # JSON roundtrip
    dumped = title_obj.model_dump_json()
    loaded = ChatTitle.model_validate_json(dumped)
    assert loaded.title == title_obj.title

    # Empty validation requires title
    with pytest.raises(ValidationError):
        ChatTitle.model_validate({})


def test_generate_chat_title_sync_empty_prompt() -> None:
    """Empty or whitespace prompts return default title without calling model."""
    assert generate_chat_title_sync("") == "New Conversation"
    assert generate_chat_title_sync("   \n\t  ") == "New Conversation"


def test_generate_chat_title_sync_with_mock_client() -> None:
    """Verifies that generate_chat_title_sync calls the agentic Client.call_structured."""
    provider = MockProvider()

    with (
        patch(
            "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider",
            return_value=provider,
        ),
        patch(
            "exmekh_lib.core.models.client.Client.call_structured"
        ) as mock_call_structured,
    ):
        mock_response = StructuredProviderResponse(
            StructuredAssistantMessage(ChatTitle(title="Guida Python E2E"))
        )
        mock_call_structured.return_value = mock_response

        title = generate_chat_title_sync("Come posso scrivere un test E2E in Python?")
        assert title == "Guida Python E2E"
        mock_call_structured.assert_called_once()
        args, kwargs = mock_call_structured.call_args
        assert args[1] is ChatTitle
        assert kwargs.get("strict") is True


def test_generate_chat_title_sync_fallback_on_error() -> None:
    """Verifies that when the provider encounters an error, a clean fallback is returned."""
    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider",
        side_effect=ConnectionError("Offline"),
    ):
        title = generate_chat_title_sync(
            "Crea un server HTTP in Rust per microservizi veloci"
        )
        assert len(title) <= 34
        assert title.startswith("Crea un server HTTP")
        assert "..." in title


@pytest.mark.asyncio
async def test_generate_chat_title_async_wrapper() -> None:
    """Verifies that generate_chat_title runs in executor and returns the structured title."""
    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner.generate_chat_title_sync",
        return_value="Titolo Asincrono",
    ):
        result = await generate_chat_title("Qualsiasi prompt")
        assert result == "Titolo Asincrono"


@pytest.mark.asyncio
async def test_handle_send_updates_session_title_structured(
    app_client: NiceGUIClient,
) -> None:
    """Verifies full flow: first message updates session.title using agentic client structured output."""
    handlers = {}

    with (
        patch("exmekh_agent.ui.page_controller.HeaderComponent") as mock_hdr_cls,
        patch("exmekh_agent.ui.page_controller.SidebarComponent") as mock_sidebar_cls,
        patch("exmekh_agent.ui.page_controller.InputBarComponent") as mock_input_cls,
        patch("exmekh_agent.ui.page_controller.ChatViewComponent") as mock_cv_cls,
        patch("exmekh_agent.ui.page_controller.SettingsDrawerComponent") as mock_sd_cls,
    ):

        def capture_sidebar(**kwargs):
            m = MagicMock()
            m.render_sessions = MagicMock()
            return m

        def capture_input(**kwargs):
            handlers["send"] = kwargs["on_send"]
            return MagicMock()

        mock_hdr_cls.return_value = MagicMock()
        mock_sidebar_cls.side_effect = capture_sidebar
        mock_input_cls.side_effect = capture_input
        mock_cv_cls.return_value = MagicMock()
        mock_sd_cls.return_value = MagicMock()

        with app_client:
            index_page()

            # Fake turn success
            async def fake_run_turn(*args, **kwargs):
                return "Risposta"

            async def fake_generate_title(user_prompt: str, model: str | None = None):
                return "Titolo Generato con AI"

            with (
                patch(
                    "exmekh_agent.ui.page_controller.run_agent_turn",
                    side_effect=fake_run_turn,
                ),
                patch(
                    "exmekh_agent.ui.page_controller.generate_chat_title",
                    side_effect=fake_generate_title,
                ) as mock_gen,
            ):
                curr = session_manager.get_current_session()
                curr.messages.clear()
                curr.title = "New Conversation"

                # Send first user message
                handlers["send"]("Spiegami l'architettura dei transformers")
                for _ in range(30):
                    if mock_gen.called:
                        break
                    await asyncio.sleep(0.05)

                mock_gen.assert_called_once()
                assert curr.title == "Titolo Generato con AI"

                # Send second message: should NOT call generate_chat_title again
                mock_gen.reset_mock()
                handlers["send"]("Fai un esempio pratico di attention")
                await asyncio.sleep(0.05)
                mock_gen.assert_not_called()
                assert curr.title == "Titolo Generato con AI"


@pytest.mark.parametrize(
    "api_type,budget_key,reasoning",
    [
        ("responses", "max_output_tokens", {"reasoning": {"effort": "none"}}),
        ("chat_completions", "max_tokens", {"reasoning_effort": "none"}),
    ],
)
def test_title_generation_disables_thinking_and_uses_api_token_budget(
    api_type, budget_key, reasoning
):
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider",
            return_value=MockProvider(),
        ),
        patch("exmekh_lib.core.models.client.Client.call_structured") as call,
    ):
        call.return_value = StructuredProviderResponse(
            StructuredAssistantMessage(ChatTitle(title="Un titolo breve"))
        )
        assert (
            generate_chat_title_sync("Spiega Python", api_type=api_type)
            == "Un titolo breve"
        )
        request = call.call_args.args[0]
        assert request.extra == {"temperature": 0.3, budget_key: 256, **reasoning}
        call.assert_called_once()


def test_title_generation_does_not_repeat_failed_request():
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider",
            return_value=MockProvider(),
        ),
        patch(
            "exmekh_lib.core.models.client.Client.call_structured",
            side_effect=TimeoutError("generation timed out"),
        ) as call,
    ):
        assert generate_chat_title_sync("Spiega Python") == "Spiega Python"
        call.assert_called_once()


@pytest.mark.live
@pytest.mark.parametrize("api_type", ["responses", "chat_completions"])
def test_live_title_generation_returns_structured_title(api_type, caplog):
    try:
        r = httpx.get(f"{config.base_url}/models", timeout=1.0)
        if r.status_code != 200:
            pytest.skip(f"Local provider offline at {config.base_url}.")
        data = r.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        if config.default_model and config.default_model not in models:
            pytest.skip(
                f"Model '{config.default_model}' non caricato sul provider locale."
            )
    except Exception:
        pytest.skip(f"Local provider offline at {config.base_url}.")

    prompt = "Come posso organizzare le fotografie delle mie vacanze in montagna?"
    title = generate_chat_title_sync(prompt, api_type=api_type)
    assert title != prompt[:30] + "...", (
        "Title generation fell back after provider failure"
    )
    assert 2 <= len(title.split()) <= 8
    assert "Error during structured title generation" not in caplog.text
