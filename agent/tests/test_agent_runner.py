from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.agents.node import DefaultCapabilities
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMX

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agent_runner import (
    check_provider_health,
    execute_agent_turn_sync,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.models import ChatSettings


@pytest.mark.asyncio
async def test_check_provider_health_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"id": "model-beta"}, {"id": "model-alpha"}]
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    with patch(
        "exmekh_agent.exmekh_bridge.provider_health.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        is_alive, models = await check_provider_health(
            "http://localhost:8080/v1", "key123"
        )
        assert is_alive is True
        assert models == ["model-alpha", "model-beta"]


@pytest.mark.asyncio
async def test_check_provider_health_non_200() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    with patch(
        "exmekh_agent.exmekh_bridge.provider_health.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        is_alive, models = await check_provider_health()
        assert is_alive is False
        assert models == []


@pytest.mark.asyncio
async def test_check_provider_health_exception() -> None:
    mock_client = AsyncMock()
    mock_client.get.side_effect = RuntimeError("Network down")

    with patch(
        "exmekh_agent.exmekh_bridge.provider_health.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        is_alive, models = await check_provider_health("http://invalid")
        assert is_alive is False
        assert models == []


def test_execute_agent_turn_sync_quick_chat() -> None:
    tokens = []
    reasoning = []
    tools = []

    mock_response = MagicMock()
    mock_response.get_message.return_value = "Quick chat answer"

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.Client") as mock_client_cls,
    ):
        mock_client_instance = MagicMock()
        mock_client_instance.call.return_value = mock_response
        mock_client_cls.return_value = mock_client_instance

        res = execute_agent_turn_sync(
            user_prompt="Hello",
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="",
            api_type="chat_completions",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda t: tokens.append(t),
            on_reasoning=lambda t, p="/", d=0: reasoning.append(t),
            on_tool_event=lambda ev: tools.append(ev),
        )

        assert res == "Quick chat answer"
        assert tokens == ["Quick chat answer"]


def test_quick_chat_forwards_provider_stream_events() -> None:
    tokens: list[str] = []
    reasoning: list[str] = []
    response = MagicMock()
    response.get_message.return_value = "Hello world"

    def call_with_events(request):
        assert request.streaming is True
        request.on_event.handle(
            ProviderStreamEvent(
                type=ProviderStreamEventType.REASONING, content="Thinking"
            )
        )
        for chunk in ("Hello ", "world"):
            request.on_event.handle(
                ProviderStreamEvent(
                    type=ProviderStreamEventType.RESPONSE, content=chunk
                )
            )
        return response

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.Client") as client_class,
    ):
        client_class.return_value.call.side_effect = call_with_events
        result = execute_agent_turn_sync(
            user_prompt="Hi",
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=tokens.append,
            on_reasoning=reasoning.append,
            on_tool_event=lambda _: None,
        )

    assert result == "Hello world"
    assert tokens == ["Hello ", "world"]
    assert reasoning == ["Thinking"]


def test_execute_agent_turn_sync_deep_agent(tmp_path: Path) -> None:
    tokens = []
    reasoning = []
    tools = []
    skills = []

    cap_dir = tmp_path / "capabilities"
    cap_dir.mkdir()

    mock_response = MagicMock()
    mock_response.get_message.return_value = "Deep agent final output"

    def fake_deep_agent_run(user_message: str):
        # Simulate stream events from DeepAgent
        stream_cb = fake_deep_agent_run.on_stream

        # Event 1: Root reasoning
        pe1 = ProviderStreamEvent(
            type=ProviderStreamEventType.REASONING, content="Root plan"
        )
        ev1 = DeepAgentStreamEvent(provider_event=pe1, depth=0, skill_path="/")
        stream_cb.handle(ev1)

        # Event 2: Sub-skill reasoning at depth 1
        pe2 = ProviderStreamEvent(
            type=ProviderStreamEventType.REASONING, content="Skill reasoning"
        )
        ev2 = DeepAgentStreamEvent(
            provider_event=pe2, depth=1, skill_path="/mock-skill"
        )
        stream_cb.handle(ev2)

        # Event 3: Nested sub-sub-skill at depth 2
        pe3 = ProviderStreamEvent(
            type=ProviderStreamEventType.REASONING, content="Nested reasoning"
        )
        ev3 = DeepAgentStreamEvent(
            provider_event=pe3, depth=2, skill_path="/mock-skill/sub"
        )
        stream_cb.handle(ev3)

        # Event 4: Tool execution inside nested skill
        ev_tool = MagicMock(spec=DeepAgentExecutionEvent)
        ev_tool.type = "tool_execution"
        ev_tool.tool_name = "nested_tool"
        ev_tool.depth = 2
        ev_tool.skill_path = "/mock-skill/sub"
        ev_tool.history = [{"role": "user", "content": "Run task"}]
        ev_tool.history_tree = []
        ev_tool.tool_call_id = "nested-tool-call"
        stream_cb.handle(ev_tool)

        # Event 5: Return to depth 1 (should pop nested skill)
        pe4 = ProviderStreamEvent(
            type=ProviderStreamEventType.REASONING, content="Back to skill 1"
        )
        ev4 = DeepAgentStreamEvent(
            provider_event=pe4, depth=1, skill_path="/mock-skill"
        )
        stream_cb.handle(ev4)

        # A second skill at the same depth closes the first one.
        ev_sibling = DeepAgentStreamEvent(
            provider_event=pe4, depth=1, skill_path="/second-skill"
        )
        stream_cb.handle(ev_sibling)

        # Event 6: Stream response token while still in depth 1.
        pe_tok = ProviderStreamEvent(
            type=ProviderStreamEventType.RESPONSE, content="Final token"
        )
        ev_tok = DeepAgentStreamEvent(
            provider_event=pe_tok, depth=1, skill_path="/second-skill"
        )
        stream_cb.handle(ev_tok)

        return mock_response

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.DeepAgent") as mock_agent_cls,
    ):
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.side_effect = fake_deep_agent_run

        def set_stream_cb(**kwargs):
            fake_deep_agent_run.on_stream = kwargs.get("on_stream")
            return mock_agent_instance

        mock_agent_cls.side_effect = set_stream_cb

        res = execute_agent_turn_sync(
            user_prompt="Run task",
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="secret",
            api_type="responses",
            reasoning_effort_str="medium",
            agent_mode="deep_agent",
            capabilities_path_str=str(cap_dir),
            name_hds="main",
            on_token=lambda t: tokens.append(t),
            on_reasoning=lambda t, p="/", d=0: reasoning.append((t, p, d)),
            on_tool_event=lambda ev: tools.append(ev),
            on_skill_event=lambda node, act: skills.append((node.path, act)),
            on_raw_history=lambda h, t: None,
        )

        assert res == "Deep agent final output"
        assert len(reasoning) >= 4
        assert len(tools) == 1
        assert tools[0].tool_name == "nested_tool"
        assert any(act == "completed" for _, act in skills)
        assert skills.index(("/mock-skill", "completed")) < skills.index(
            ("/second-skill", "opened")
        )


def test_execute_agent_turn_sync_fallback_reasoning_signature() -> None:
    # Test fallback when on_reasoning only accepts single argument
    calls = []

    def single_arg_reasoning(token: str) -> None:
        calls.append(token)

    def fake_deep_agent_run(user_message: str):
        pe = ProviderStreamEvent(
            type=ProviderStreamEventType.REASONING, content="Single arg token"
        )
        ev = DeepAgentStreamEvent(provider_event=pe, depth=0, skill_path="/")
        fake_deep_agent_run.on_stream.handle(ev)
        m = MagicMock()
        m.get_message.return_value = "Done"
        return m

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.DeepAgent") as mock_agent_cls,
    ):
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.side_effect = fake_deep_agent_run
        mock_agent_cls.side_effect = lambda **kwargs: (
            setattr(fake_deep_agent_run, "on_stream", kwargs["on_stream"])
            or mock_agent_instance
        )

        execute_agent_turn_sync(
            user_prompt="Run",
            model="m",
            base_url="http://localhost:8080/v1",
            api_key="k",
            api_type="responses",
            reasoning_effort_str="invalid_effort_causes_none",
            agent_mode="deep_agent",
            capabilities_path_str="nonexistent_cap_path",
            name_hds="",
            on_token=lambda t: None,
            on_reasoning=single_arg_reasoning,
            on_tool_event=lambda ev: None,
        )

        assert calls == ["Single arg token"]


@pytest.mark.asyncio
async def test_run_agent_turn_async() -> None:
    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner.execute_agent_turn_sync",
        return_value="Async turn completed",
    ) as mock_exec:
        res = await run_agent_turn(
            user_prompt="Hello async",
            model="",
            on_token=lambda t: None,
            on_reasoning=lambda t: None,
            on_tool_event=lambda ev: None,
            agent_mode=None,
        )
        assert res == "Async turn completed"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_turn_uses_chat_options_without_overriding_provider() -> None:
    options = ChatSettings(
        name_hds="chat-root",
        capabilities_path="chat-capabilities",
        work_root="chat-work",
        coding_agent_persistent_history=False,
    )
    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner.execute_agent_turn_sync",
        return_value="ok",
    ) as execute:
        await run_agent_turn(
            user_prompt="hello",
            model="model",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
            settings=options,
        )

    call = execute.call_args.kwargs
    assert call["name_hds"] == "chat-root"
    assert call["capabilities_path_str"] == "chat-capabilities"
    assert call["work_root"] == "chat-work"
    assert call["coding_agent_persistent_history"] is False
    assert call["base_url"] == config.base_url
    assert call["api_key"] == config.api_key


@pytest.mark.parametrize(
    ("input_effort", "expected_provider_effort"),
    [
        ("xhigh", ReasoningEffortLMX.XHIGH),
        ("medium", ReasoningEffortLMX.MEDIUM),
        ("low", ReasoningEffortLMX.LOW),
        ("none", ReasoningEffortLMX.OFF),
        ("invalid_causes_none", None),
        ("", None),
    ],
)
def test_execute_agent_turn_sync_reasoning_effort_mapping(
    input_effort: str,
    expected_provider_effort: object,
) -> None:
    mock_response = MagicMock()
    mock_response.get_message.return_value = "Done"

    with (
        patch(
            "exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"
        ) as mock_provider_cls,
        patch("exmekh_agent.exmekh_bridge.agent_runner.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.call.return_value = mock_response
        mock_client_cls.return_value = mock_client

        execute_agent_turn_sync(
            user_prompt="Hi",
            model="Qwen3.8-27B-UD-Q4_K_S",
            base_url="http://localhost:8080/v1",
            api_key="",
            api_type="chat_completions",
            reasoning_effort_str=input_effort,
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda t: None,
            on_reasoning=lambda t: None,
            on_tool_event=lambda ev: None,
        )

        mock_provider_cls.assert_called_once_with(
            model="Qwen3.8-27B-UD-Q4_K_S",
            api="chat_completions",
            reasoning_effort=expected_provider_effort,
            base_url="http://localhost:8080/v1",
            api_key="",
        )


def test_execute_agent_turn_sync_uses_distinct_agent_context_id(
    tmp_path: Path,
) -> None:
    capabilities = tmp_path / "capabilities"
    capabilities.mkdir()
    response = MagicMock()
    response.get_message.return_value = "done"

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.Node") as node_class,
        patch("exmekh_agent.exmekh_bridge.agent_runner.DeepAgent") as agent_class,
    ):
        agent_class.return_value.run.return_value = response
        execute_agent_turn_sync(
            user_prompt="hello",
            model="model",
            base_url="http://localhost:8080/v1",
            api_key="key",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="deep_agent",
            capabilities_path_str=str(capabilities),
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
            agent_context_id="conversation-123",
        )

    node_class.assert_called_once()
    assert node_class.call_args.kwargs["capabilities_paths"] == [capabilities]
    assert node_class.call_args.kwargs["name_hds"] == "main"
    assert node_class.call_args.kwargs["agent_context_id"] == "conversation-123"
    assert [child.name_hds for child in node_class.call_args.kwargs["children"]] == [
        "coding-agent"
    ]


def test_deep_agent_adds_bundled_mock_sub_agent() -> None:
    capabilities = DefaultCapabilities().capabilities_path
    response = MagicMock()
    response.get_message.return_value = "done"

    with (
        patch("exmekh_agent.exmekh_bridge.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agent_runner.Node") as node_class,
        patch("exmekh_agent.exmekh_bridge.agent_runner.DeepAgent") as agent_class,
    ):
        agent_class.return_value.run.return_value = response
        execute_agent_turn_sync(
            user_prompt="hello",
            model="model",
            base_url="http://localhost:8080/v1",
            api_key="key",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="deep_agent",
            capabilities_path_str=str(capabilities),
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
        )

    assert node_class.call_count == 1
    children = node_class.call_args.kwargs["children"]
    assert [child.name_hds for child in children] == [
        "mock-sub-agent",
        "coding-agent",
    ]
    assert children[1].path == "main/coding-agent"


def test_provider_requests_are_serialized_across_worker_threads() -> None:
    active_calls = 0
    peak_calls = 0
    state_lock = threading.Lock()

    def slow_call(**kwargs) -> str:
        nonlocal active_calls, peak_calls
        with state_lock:
            active_calls += 1
            peak_calls = max(peak_calls, active_calls)
        time.sleep(0.03)
        with state_lock:
            active_calls -= 1
        return "done"

    with patch(
        "exmekh_agent.exmekh_bridge.agent_runner._execute_agent_turn_sync_unlocked",
        side_effect=slow_call,
    ):
        common = {
            "model": "model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "key",
            "api_type": "responses",
            "reasoning_effort_str": "none",
            "agent_mode": "quick_chat",
            "capabilities_path_str": "",
            "name_hds": "main",
            "on_token": lambda _: None,
            "on_reasoning": lambda _: None,
            "on_tool_event": lambda _: None,
        }
        threads = [
            threading.Thread(
                target=execute_agent_turn_sync,
                kwargs={"user_prompt": str(index), **common},
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert peak_calls == 1


@pytest.mark.asyncio
async def test_run_agent_turn_does_not_retry_type_error() -> None:
    with (
        patch(
            "exmekh_agent.exmekh_bridge.agent_runner.execute_agent_turn_sync",
            side_effect=TypeError("callback bug"),
        ) as execute,
        pytest.raises(TypeError, match="callback bug"),
    ):
        await run_agent_turn(
            user_prompt="hello",
            model="model",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
        )

    execute.assert_called_once()
