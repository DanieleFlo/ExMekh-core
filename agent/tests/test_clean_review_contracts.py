from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.user_message import UserMessage

from exmekh_agent.config import AppConfig
from exmekh_agent.exmekh_bridge.converter import convert_to_exmekh_messages
from exmekh_agent.exmekh_bridge.event_handlers import (
    QuickChatEventHandler,
    scoped_reasoning_callback,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.provider_context import ContextTrackingProvider
from exmekh_agent.exmekh_bridge.provider_health import (
    check_provider_health,
    extract_model_context_length,
)
from tests.test_provider_context import StructuredAnswer, StubProvider


@pytest.mark.parametrize(
    "last",
    [
        UserMessage("question"),
        ChatMessage(role="user", content="question"),
        {"role": "USER", "content": "question"},
    ],
)
def test_conversion_deduplicates_the_current_prompt_through_public_message_api(last):
    previous = UserMessage("earlier")
    messages = convert_to_exmekh_messages([previous, last], "question")
    assert messages == [previous]


def test_conversion_ignores_objects_outside_the_history_contract():
    assert convert_to_exmekh_messages([object()], "question") == []


def test_config_owns_runtime_model_selection_without_persisting_settings():
    settings = AppConfig(default_model="old", context_length=22000)
    settings.set_model_context_length("flash", 16384)
    with patch.object(settings, "save_to_env") as save:
        settings.select_available_model([])
        assert settings.default_model == "old"
        settings.select_available_model(["flash"])
        assert settings.default_model == "flash"
        assert settings.get_context_length() == settings.context_length == 16384
        settings.select_available_model(["other", "flash"])
        assert settings.default_model == "flash"
        save.assert_not_called()


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 0.5])
def test_discovery_skips_invalid_context_metadata_and_uses_the_next_field(value):
    assert (
        extract_model_context_length({"context_length": value, "n_ctx": 8192}) == 8192
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], {"data": None}, {"data": {}}, {}])
async def test_health_probe_rejects_malformed_model_lists(body):
    response = MagicMock(status_code=200)
    response.json.return_value = body
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response):
        assert await check_provider_health("http://local/v1", "") == (False, [])


@pytest.mark.asyncio
async def test_health_probe_ignores_bad_entries_and_deduplicates_model_ids():
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [None, {"id": 1}, {"id": ""}, {"id": "flash"}, {"id": "flash"}]
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response):
        assert await check_provider_health("http://local/v1", "") == (True, ["flash"])


def test_quick_chat_supports_the_same_scoped_reasoning_callback_as_deep_agent():
    received = []
    handler = QuickChatEventHandler(
        lambda _: None,
        lambda content, path, depth: received.append((content, path, depth)),
    )
    handler.handle(
        ProviderStreamEvent(type=ProviderStreamEventType.REASONING, content="thought")
    )
    assert received == [("thought", "/", 0)]


def test_reasoning_observer_type_error_propagates_without_duplicate_delivery():
    calls = []

    def observer(*args):
        calls.append(args)
        raise TypeError("observer body failed")

    adapted = scoped_reasoning_callback(observer)
    with pytest.raises(TypeError, match="observer body failed"):
        adapted("thought", "/worker", 1)
    assert calls == [("thought", "/worker", 1)]


def test_unknown_callback_signature_uses_the_scoped_contract():
    callback = MagicMock()
    with patch("inspect.signature", side_effect=ValueError("unknown signature")):
        adapted = scoped_reasoning_callback(callback)
    adapted("thought", "/worker", 1)
    callback.assert_called_once_with("thought", "/worker", 1)


def test_invalid_reasoning_callback_is_rejected_before_inference():
    with pytest.raises(TypeError):
        scoped_reasoning_callback(lambda first, second: None)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("observe_context", [False, True])
def test_structured_requests_share_root_observation_and_child_isolation(
    child, observe_context
):
    provider = StubProvider()
    contexts = []
    requests = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append if observe_context else None,
        context_id="root",
        ds4_qwen=False,
        thinking=False,
        on_request=lambda: requests.append(True),
    )
    request = StructuredProviderRequest(
        [UserMessage("question")],
        StructuredAnswer,
        extra={"agentic": {"agent_context_id": "child" if child else "root"}},
    )
    result = tracker.call_structured(request)
    assert result is provider.structured_result
    assert provider.requests[-1] is request
    assert requests == ([] if child else [True])
    assert len(contexts) == (2 if observe_context and not child else 0)
    if contexts:
        assert contexts[-1].is_final
        assert contexts[-1].prompt_tokens == 4000
        assert contexts[-1].completion_tokens == 100
