from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage

from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    convert_to_exmekh_messages,
    execute_agent_turn_sync,
    generate_chat_title_sync,
    serialize_skill_execution,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage


def _execution_event(
    kind: str, name: str, path: str = "/", depth: int = 0, call_id: str = ""
):
    event = MagicMock(spec=DeepAgentExecutionEvent)
    event.type = kind
    event.is_tool_running = kind == "tool_start"
    event.failed = False
    event.tool_name = name
    event.skill_path = path
    event.depth = depth
    event.tool_call_id = call_id
    return event


def _stream_event(
    kind: ProviderStreamEventType, content: str | None, path: str = "/", depth: int = 0
):
    return DeepAgentStreamEvent(
        provider_event=ProviderStreamEvent(type=kind, content=content),
        skill_path=path,
        depth=depth,
    )


def _run_deep(
    events: list[object], **overrides: object
) -> tuple[list[str], list[object], list[tuple[str, str]]]:
    tokens: list[str] = []
    tools: list[object] = []
    skills: list[tuple[str, str]] = []
    response = MagicMock()
    response.get_message.return_value = "final"
    response.history = []
    response.skill_executions = []

    def build_agent(**kwargs: object) -> MagicMock:
        agent = MagicMock()

        def run(**_: object) -> MagicMock:
            for event in events:
                if isinstance(event, BaseException):
                    raise event
                kwargs["on_stream"].handle(event)
            return response

        agent.run.side_effect = run
        return agent

    arguments = dict(
        user_prompt="task",
        model="model",
        base_url="http://localhost:8080/v1",
        api_key="key",
        api_type="responses",
        reasoning_effort_str="none",
        agent_mode="deep_agent",
        capabilities_path_str="",
        name_hds="main",
        on_token=tokens.append,
        on_reasoning=lambda *_: None,
        on_tool_event=tools.append,
        on_skill_event=lambda node, action: skills.append((node.path, action)),
    )
    arguments.update(overrides)
    with (
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"),
        patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.DeepAgent",
            side_effect=build_agent,
        ),
    ):
        assert execute_agent_turn_sync(**arguments) == "final"
    return tokens, tools, skills


def test_history_converter_accepts_empty_iterator_and_exmekh_messages() -> None:
    assert convert_to_exmekh_messages(iter(())) == []
    user = UserMessage("question")
    result = convert_to_exmekh_messages(
        [
            SystemMessage("system"),
            user,
            ChatMessage(role="system", content="system"),
            {"role": "system", "content": "system"},
            {"role": "user", "content": "last"},
        ],
        current_user_prompt="last",
    )
    assert result == [user]


def test_history_converter_keeps_nonmatching_last_message() -> None:
    result = convert_to_exmekh_messages(
        [{"role": "user", "content": "previous"}], "current"
    )
    assert len(result) == 1
    assert result[0].get_message() == "previous"


def test_skill_execution_serialization_keeps_checkpoint_metadata() -> None:
    execution = MagicMock()
    execution.history_tree.return_value = {"skill": "read"}
    execution.opening_tool_call_id = "call"
    execution.start_index = 2
    execution.end_index = 5
    assert serialize_skill_execution(execution) == {
        "skill": "read",
        "opening_tool_call_id": "call",
        "start_index": 2,
        "end_index": 5,
        "history": [],
        "children": [],
    }


def test_deep_agent_correlates_tool_events_inside_skill() -> None:
    tokens, tools, skills = _run_deep(
        [
            _execution_event("tool_start", "skill", "/", 0, "skill-call"),
            _stream_event(ProviderStreamEventType.REASONING, "thinking", "/skill", 1),
            _execution_event("tool_start", "read", "/skill", 1, "read-call"),
            _execution_event("tool_end", "read", "/skill", 1, "read-call"),
            _execution_event("tool_end", "orphan", "/skill", 1, "missing"),
            _execution_event("execution", "other", "/skill", 1),
            _stream_event(ProviderStreamEventType.RESPONSE, "answer", "/", 0),
        ]
    )
    assert tokens == ["answer"]
    assert [(event.tool_name, event.status) for event in tools] == [
        ("skill", "completed"),
        ("read", "completed"),
        ("read", "completed"),
        ("orphan", "completed"),
        ("other", "completed"),
        ("skill", "completed"),
    ]
    assert ("/skill", "opened") in skills
    assert ("/skill", "completed") in skills


def test_deep_agent_tolerates_legacy_reasoning_callback_and_raw_history_error() -> None:
    thoughts: list[str] = []

    def old_callback(token: str) -> None:
        thoughts.append(token)

    def bad_history(*_: object) -> None:
        raise RuntimeError("observer failed")

    _run_deep(
        [
            _stream_event(ProviderStreamEventType.REASONING, "thought"),
            _execution_event("tool_start", "read", call_id="read-call"),
            _execution_event("tool_end", "read", call_id="read-call"),
        ],
        on_reasoning=old_callback,
        on_raw_history=bad_history,
    )
    assert thoughts == ["thought"]


def test_history_converter_ignores_unsupported_items() -> None:
    result = convert_to_exmekh_messages([object(), {"role": "user", "content": "kept"}])
    assert result[-1].get_message() == "kept"


def test_deep_agent_runs_without_skill_observer_or_tool_ids() -> None:
    _, tools, _ = _run_deep(
        [
            _execution_event("tool_start", "skill"),
            _stream_event(ProviderStreamEventType.REASONING, "inside", "/skill", 1),
            _execution_event("tool_start", "read", "/skill", 1),
            _execution_event("tool_end", "read", "/skill", 1),
            _execution_event("execution", "other", "/skill", 1),
            _execution_event("tool_end", "orphan"),
            _execution_event("execution", "root"),
            _stream_event(ProviderStreamEventType.END, None),
        ],
        on_skill_event=None,
    )
    assert [event.tool_name for event in tools] == [
        "skill",
        "read",
        "read",
        "other",
        "orphan",
        "root",
    ]


@pytest.mark.parametrize("observer_fails", [False, True])
def test_quick_chat_recovers_from_history_seed_failure_and_captures_raw_history(
    observer_fails: bool,
) -> None:
    response = MagicMock()
    response.get_message.return_value = "answer"
    captured = []

    def capture(raw, tree):
        captured.append((raw, tree))
        if observer_fails:
            raise RuntimeError("observer failed")

    with (
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.Client") as client_class,
    ):
        client = client_class.return_value
        client.replace_history_from.side_effect = ValueError("legacy history")
        client.call.return_value = response
        client.get_history.return_value = [AssistantMessage("answer")]
        result = execute_agent_turn_sync(
            user_prompt="task",
            model="model",
            base_url="http://localhost:8080/v1",
            api_key="key",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
            on_raw_history=capture,
            history=[{"role": "user", "content": "earlier"}],
        )
    assert result == "answer"
    client.replace_history_from.assert_called_once()
    assert captured == [([{"role": "assistant", "content": "answer"}], [])]


def test_empty_structured_title_falls_back_to_user_prompt() -> None:
    response = MagicMock()
    response.get_message.return_value.title = "  "
    with (
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.Client") as client_class,
    ):
        client_class.return_value.call_structured.return_value = response
        title = generate_chat_title_sync("  A short prompt  ")
    assert title == "A short prompt"


def test_deep_agent_ignores_unknown_stream_event() -> None:
    tokens, tools, skills = _run_deep([object()])
    assert tokens == []
    assert tools == []
    assert skills == []


def test_quick_chat_ignores_noncontent_provider_event() -> None:
    response = MagicMock()
    response.get_message.return_value = "answer"

    def call(request):
        request.on_event.handle(ProviderStreamEvent(type=ProviderStreamEventType.END))
        return response

    tokens = []
    with (
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider"),
        patch("exmekh_agent.exmekh_bridge.agents.agent_runner.Client") as client_class,
    ):
        client_class.return_value.call.side_effect = call
        execute_agent_turn_sync(
            user_prompt="task",
            model="model",
            base_url="http://localhost:8080/v1",
            api_key="key",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=tokens.append,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
        )
    assert tokens == ["answer"]


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_deep_agent_failure_finalizes_pending_executions(
    failure: BaseException,
) -> None:
    skills = []
    tools = []
    with pytest.raises(type(failure)):
        _run_deep(
            [
                _stream_event(
                    ProviderStreamEventType.REASONING, "thinking", "/skill", 1
                ),
                _execution_event("tool_start", "read", "/skill", 1, "read-call"),
                failure,
            ],
            on_skill_event=lambda node, action: skills.append((node, action)),
            on_tool_event=tools.append,
        )
    assert skills[-1][1] == "error"
    assert skills[-1][0].status == "error"
    assert tools[-1].status == "error"
    assert tools[-1].duration_seconds is not None
