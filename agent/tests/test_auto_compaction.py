from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest
from exmekh_lib.core.agents.deep_agent import DeepAgentHistoryCompactionEvent
from exmekh_lib.core.agents.node import DefaultCapabilities
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage

from exmekh_agent.config import AppConfig, config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.exmekh_bridge.agents.auto_compaction import AutoCompactionPolicy
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.event_handlers import DeepAgentTurnEventHandler
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEventRouter
from exmekh_agent.exmekh_bridge.history.compaction import (
    HistoryArchive,
    ModelContextCheckpoint,
    replay_model_context,
)
from exmekh_agent.exmekh_bridge.history.converter import (
    convert_to_exmekh_messages,
    exmekh_message_to_raw_dict,
)
from exmekh_agent.exmekh_bridge.history.recorded_output import RecordedToolOutput
from exmekh_agent.exmekh_bridge.history.tool_calls import parse_tool_calls
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ProviderContextUsage,
    QueuedMessage,
    SubAgentConversation,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    build_compacted_context_messages,
    count_text_tokens,
)
from exmekh_agent.exmekh_bridge.providers.ds4_prompt import render_ds4_qwen_prompt
from exmekh_agent.exmekh_bridge.providers.provider_context import (
    ContextTrackingProvider,
)
from exmekh_agent.state.sub_agent_conversations import apply_sub_agent_event
from exmekh_agent.ui.graph.graph_data import build_graph_data_from_session
from exmekh_agent.ui.history.raw_history_view import build_raw_history_from_session
from tests.compaction_probe import CompactionProbeProvider
from tests.image_samples import png_bytes


@pytest.fixture
def probe(monkeypatch):
    provider = CompactionProbeProvider()
    monkeypatch.setattr(agent_runner, "LocalProvider", lambda **kwargs: provider)
    monkeypatch.setattr(config, "server_context_lengths", {})
    monkeypatch.setattr(config, "model_context_lengths", {})
    monkeypatch.setattr(config, "auto_compaction_threshold_percent", 80)
    config.set_model_context_length("compaction-probe", 100000)
    return provider


def run_turn(
    tmp_path,
    session,
    prompt="Start",
    *,
    collector=None,
    sub_events=None,
    image_urls=None,
    **overrides,
):
    collector = collector or TurnStreamCollector()
    options = dict(
        user_prompt=prompt,
        model="compaction-probe",
        base_url=config.base_url,
        api_key="",
        api_type="responses",
        reasoning_effort_str="none",
        agent_mode="deep_agent",
        capabilities_path_str=str(DefaultCapabilities(mocks=True).capabilities_path),
        name_hds="main",
        work_root=str(tmp_path),
        agent_context_id=session.id,
        on_token=collector.append_token,
        on_reasoning=collector.append_reasoning,
        on_tool_event=collector.append_tool,
        on_skill_event=collector.append_skill,
        on_context=collector.set_provider_context,
        on_raw_history=collector.set_raw_history,
        on_model_context=collector.set_model_context,
        history=list(session.messages),
        image_urls=image_urls,
        steering=collector.steering,
        sub_agent_history=session.sub_agent_conversations,
        on_sub_agent_event=lambda event: (
            sub_events.append(event) if sub_events is not None else None,
            apply_sub_agent_event(session, event),
        ),
    )
    options.update(overrides)
    result = agent_runner.execute_agent_turn_sync(**options)
    session.messages.extend(
        [
            ChatMessage(role="user", content=prompt),
            collector.build_message(fallback_content=result),
        ]
    )
    return collector


@pytest.mark.parametrize(
    "used,expected", [(0, False), (799, False), (800, True), (801, True), (1100, True)]
)
def test_policy_boundary_and_no_repeated_attempts(monkeypatch, used, expected):
    monkeypatch.setattr(
        config, "server_context_lengths", {(config.base_url.rstrip("/"), "probe"): 1000}
    )
    monkeypatch.setattr(config, "auto_compaction_threshold_percent", 80)
    tracker = MagicMock()
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=used)
    tracker.last_context_total.return_value = 0
    policy = AutoCompactionPolicy(tracker, model="probe", base_url=config.base_url)
    request = ProviderRequest([UserMessage("task")])
    assert policy.before_inference(request) is expected
    assert policy.before_inference(request) is False
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=100)
    assert not policy.before_inference(request)
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=used)
    assert policy.before_inference(request) is expected


def test_policy_rearms_only_after_effective_compaction(monkeypatch):
    monkeypatch.setattr(
        config, "server_context_lengths", {(config.base_url.rstrip("/"), "probe"): 1000}
    )
    monkeypatch.setattr(config, "auto_compaction_threshold_percent", 80)
    tracker = MagicMock()
    tracker.last_context_total.return_value = 0
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=900)
    policy = AutoCompactionPolicy(tracker, model="probe", base_url=config.base_url)
    request = ProviderRequest([UserMessage("task")])
    assert policy.before_inference(request)
    policy.after_compaction(request)
    assert not policy.before_inference(request)
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=100)
    policy.after_compaction(request)
    tracker.estimate_context.return_value = ProviderContextUsage(prompt_tokens=900)
    assert policy.before_inference(request)


def test_unknown_endpoint_does_not_reuse_another_servers_limit(monkeypatch):
    monkeypatch.setattr(config, "server_context_lengths", {})
    tracker = MagicMock()
    assert not AutoCompactionPolicy(
        tracker, model="unknown", base_url="http://other"
    ).before_inference(ProviderRequest([UserMessage("task")]))
    tracker.estimate_context.assert_not_called()


@pytest.mark.parametrize("threshold", [0, 100, -1, 80.5, True, "80", None])
def test_invalid_threshold_is_rejected_without_partial_updates(threshold):
    with pytest.raises(ValueError):
        AppConfig(auto_compaction_threshold_percent=threshold)
    settings = AppConfig()
    previous = settings.default_model
    with pytest.raises(ValueError):
        settings._apply_updates(
            {"default_model": "changed", "auto_compaction_threshold_percent": threshold}
        )
    assert settings.default_model == previous


def test_settings_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    settings = AppConfig(auto_compaction_threshold_percent=65)
    assert settings.save_to_env(path)
    assert "AUTO_COMPACTION_THRESHOLD_PERCENT=65" in path.read_text()
    monkeypatch.setenv("AUTO_COMPACTION_THRESHOLD_PERCENT", "65")
    assert AppConfig().auto_compaction_threshold_percent == 65


def test_real_client_compacts_and_keeps_raw_graph_and_replay(probe, tmp_path):
    session = ChatSession()
    run_turn(tmp_path, session)
    message = session.messages[-1]
    assert message.model_context is not None
    assert probe.score_requests
    assert "ORIGINAL_1" in str(message.raw_history)
    assert "ORIGINAL_1" not in str(message.model_context.history)
    replay = [
        message.message_dump()
        for message in convert_to_exmekh_messages(session.messages)
    ]
    assert "ORIGINAL_1" not in str(replay)
    assert "compact_history" in str(replay)
    raw, _ = build_raw_history_from_session(session)
    assert "ORIGINAL_1" in str(raw)
    graph = build_graph_data_from_session(session)
    assert "ORIGINAL_1" in str(graph)
    assert "compact_history" in str(graph)
    assert build_compacted_context_messages(session) == message.model_context.history
    restored = ChatSession.model_validate_json(session.model_dump_json())
    assert restored.messages[-1].model_context == message.model_context
    assert "ORIGINAL_1" in str(build_raw_history_from_session(restored))
    run_turn(tmp_path, restored, "Continue")
    assert "ORIGINAL_1" not in str(
        [m.message_dump() for m in probe.requests[-3].get_messages()]
    )
    assert "ORIGINAL_1" in str(build_graph_data_from_session(restored))
    restored_raw, _ = build_raw_history_from_session(restored)
    call_ids = [
        call.id
        for item in restored_raw
        for call in parse_tool_calls(item.get("tool_calls")) or []
        if call.get_name() == "mock-tool"
    ]
    assert len(call_ids) == len(set(call_ids)) == 4


def test_no_compaction_below_threshold(probe, tmp_path):
    probe.usage_tokens = 100
    session = ChatSession()
    run_turn(tmp_path, session)
    assert not probe.score_requests
    assert session.messages[-1].model_context is None


def test_compaction_failure_keeps_original_and_finishes(probe, tmp_path):
    probe.summary_error = RuntimeError("mock summary failed")
    session = ChatSession()
    run_turn(tmp_path, session)
    assert session.messages[-1].content.startswith("COMPACTION_OK")
    assert session.messages[-1].model_context is None
    assert "ORIGINAL_1" in str(session.messages[-1].raw_history)
    assert len(probe.score_requests) == 1


def test_archive_is_independent_and_handles_repeated_compaction():
    marker = {"role": "assistant", "tool_calls": [{"id": "c1"}]}
    response = {"role": "tool", "tool_call_id": "c1", "content": "summary"}
    checkpoint = ModelContextCheckpoint(
        replacements={"c1": [{"role": "assistant", "content": "original"}]}
    )
    checkpoint.replacements["c2"] = [marker, response]
    expanded = checkpoint.expand([{"role": "assistant", "tool_calls": [{"id": "c2"}]}])
    assert expanded[0]["content"] == "original"
    expanded[0]["content"] = "changed"
    assert checkpoint.replacements["c1"][0]["content"] == "original"


def tool_call(name, arguments, call_id):
    return AssistantMessage(
        None, tool_calls=[ToolCall(call_id, FunctionCall(name, arguments))]
    )


def test_steering_received_during_compaction_precedes_resumed_inference(
    probe, tmp_path
):
    collector = TurnStreamCollector()
    probe.script = [
        tool_call("mock-tool", {"text": "original evidence " * 100}, "work"),
        AssistantMessage("DONE WITH GUIDANCE"),
    ]
    probe.on_score = lambda: collector.steering.send(
        QueuedMessage(text="STEER_PRESERVED")
    )
    session = ChatSession()
    run_turn(tmp_path, session, collector=collector)
    assert "STEER_PRESERVED" in str(
        [m.message_dump() for m in probe.requests[-1].get_messages()]
    )
    assert len(probe.requests) == 2
    assert session.messages[-1].steering_messages[0].text == "STEER_PRESERVED"
    restored = ChatSession.model_validate_json(session.model_dump_json())
    assert "STEER_PRESERVED" in str(build_graph_data_from_session(restored))


def test_child_compaction_keeps_separate_history_and_restores_it(probe, tmp_path):
    probe.script = [
        tool_call(
            "call_sub_agent",
            {"sub_agent": "mock-sub-agent", "text": "Child task"},
            "delegate",
        ),
        tool_call(
            "bash",
            {"command": "printf 'CHILD_ORIGINAL " + "evidence " * 150 + "'"},
            "child-work",
        ),
        AssistantMessage("CHILD_DONE"),
        AssistantMessage("ROOT_DONE"),
    ]
    session = ChatSession()
    events = []
    run_turn(tmp_path, session, sub_events=events)
    assert session.messages[-1].content == "ROOT_DONE"
    child = session.sub_agent_conversations[0]
    assert child.messages[-1].model_context is not None
    assert "CHILD_ORIGINAL" in str(child.raw_history)
    assert "CHILD_ORIGINAL" not in str(child.replay_history())
    graph = build_graph_data_from_session(session)
    assert len(graph.histories) == 2
    assert "CHILD_ORIGINAL" in str(graph)
    restored = ChatSession.model_validate_json(session.model_dump_json())
    assert (
        restored.sub_agent_conversations[0].replay_history() == child.replay_history()
    )
    probe.script = [
        tool_call(
            "call_sub_agent",
            {"sub_agent": "mock-sub-agent", "text": "Child again"},
            "delegate2",
        ),
        AssistantMessage("CHILD_AGAIN"),
        AssistantMessage("ROOT_AGAIN"),
    ]
    run_turn(tmp_path, restored, "Continue")
    assert "CHILD_ORIGINAL" not in str(
        [m.message_dump() for m in probe.requests[-2].get_messages()]
    )
    assert "CHILD_ORIGINAL" in str(build_graph_data_from_session(restored))


def test_nested_skills_keep_original_outputs_after_compaction_and_reload(
    probe, tmp_path
):
    probe.script = [
        tool_call("mock-skill", {}, "open"),
        tool_call(
            "mock-sub-tool", {"text": "PARENT_ORIGINAL " + "evidence " * 150}, "parent"
        ),
        tool_call("nested-skill", {}, "nested-open"),
        tool_call(
            "nested-sub-tool",
            {"text": "NESTED_ORIGINAL " + "evidence " * 150},
            "nested-work",
        ),
        tool_call("exit_agent", {"text": "nested done", "items": []}, "nested-exit"),
        tool_call("exit_agent", {"text": "parent done", "items": []}, "parent-exit"),
        AssistantMessage("ROOT_DONE"),
    ]
    session = ChatSession()
    run_turn(tmp_path, session)
    assert session.messages[-1].content == "ROOT_DONE"
    restored = ChatSession.model_validate_json(session.model_dump_json())
    graph = build_graph_data_from_session(restored)
    raw = build_raw_history_from_session(restored)
    assert "PARENT_ORIGINAL" in str(raw)
    assert "NESTED_ORIGINAL" in str(raw)
    assert "NESTED_ORIGINAL" in str(graph)


def test_final_provider_usage_is_restored_before_the_next_turn(probe, tmp_path):
    probe.rounds = 0
    session = ChatSession()
    run_turn(tmp_path, session)
    assert session.messages[-1].model_context is None
    run_turn(tmp_path, session, "Next")
    assert probe.summary_requests
    assert session.messages[-1].model_context is not None


def test_original_output_is_readable_after_serialization_and_new_runtime(
    probe, tmp_path
):
    session = ChatSession()
    run_turn(tmp_path, session)
    first = next(
        item
        for item in session.messages[-1].raw_history
        if item.get("role") == "tool" and item.get("tool_call_id") == "probe-1"
    )
    output_id = RecordedToolOutput.parse(first["content"]).id
    restored = ChatSession.model_validate_json(session.model_dump_json())
    probe.usage_tokens = 0
    probe.script = [
        tool_call("read_tool_output", {"output_id": output_id}, "read-old"),
        AssistantMessage("READ_OK"),
    ]
    run_turn(tmp_path, restored, "Read original")
    assert "ORIGINAL_1" in restored.messages[-1].output_for_call("read-old")
    assert restored.messages[-1].content == "READ_OK"


def test_compaction_uses_request_estimate_without_provider_totals(probe, tmp_path):
    probe.usage_tokens = 0
    estimator = ContextTrackingProvider(
        probe, on_context=None, context_id=None, ds4_qwen=False, thinking=False
    )

    def limit_after_first_request(request):
        if len(probe.requests) == 1:
            config.set_model_context_length(
                "compaction-probe",
                estimator.estimate_context(request).prompt_tokens + 100,
            )

    probe.on_request = limit_after_first_request
    session = ChatSession()
    run_turn(tmp_path, session)
    assert probe.score_requests


def test_threshold_change_is_applied_at_next_boundary(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "auto_compaction_threshold_percent", 80)
    probe.on_request = lambda request: setattr(
        config, "auto_compaction_threshold_percent", 95
    )
    session = ChatSession()
    run_turn(tmp_path, session)
    assert not probe.score_requests


def test_scoring_failure_preserves_the_uncompacted_context(probe, tmp_path):
    def fail():
        raise ValueError("Invalid mock scores")

    probe.on_score = fail
    session = ChatSession()
    run_turn(tmp_path, session)
    assert session.messages[-1].model_context is None
    assert "ORIGINAL_1" in str(session.messages[-1].raw_history)


def test_user_image_survives_compaction_archive_and_replay(probe, tmp_path):
    image_url = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    session = ChatSession()
    run_turn(tmp_path, session, image_urls=[image_url])
    restored = ChatSession.model_validate_json(session.model_dump_json())
    checkpoint = restored.messages[-1].model_context
    assert image_url in str(checkpoint.history)
    assert image_url in str(restored.messages[-1].raw_history)
    replay = convert_to_exmekh_messages(restored.messages)
    assert image_url in str(exmekh_message_to_raw_dict(replay[0]))


def test_checkpoint_without_raw_observer_includes_the_final_answer(probe, tmp_path):
    session = ChatSession()
    run_turn(tmp_path, session, on_raw_history=None)
    assert (
        session.messages[-1].model_context.history[-1]["content"]
        == "COMPACTION_OK: Start"
    )


def test_failing_checkpoint_observer_does_not_lose_the_raw_archive(
    probe, tmp_path, caplog
):
    def fail(context):
        raise RuntimeError("observer failed")

    session = ChatSession()
    run_turn(tmp_path, session, on_model_context=fail)
    assert session.messages[-1].content == "COMPACTION_OK: Start"
    assert "ORIGINAL_1" in str(session.messages[-1].raw_history)
    assert "Unable to publish model context" in caplog.text


def test_explicit_compaction_without_a_tracking_provider(probe, tmp_path, monkeypatch):
    constructor = agent_runner.DeepAgent

    def explicit_policy_agent(**kwargs):
        agent = constructor(**kwargs)
        agent.root_node.set_history_compaction_policy(
            MagicMock(before_inference=lambda request: True)
        )
        return agent

    monkeypatch.setattr(config, "server_context_lengths", {})
    monkeypatch.setattr(agent_runner, "DeepAgent", explicit_policy_agent)
    session = ChatSession()
    run_turn(tmp_path, session, on_context=None, steering=None)
    assert session.messages[-1].model_context is not None
    assert "ORIGINAL_1" in str(session.messages[-1].raw_history)


def test_context_count_includes_new_messages_after_a_saved_checkpoint():
    checkpoint = ModelContextCheckpoint(
        history=[
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "old"},
        ]
    )
    session = ChatSession(
        messages=[
            ChatMessage(role="assistant", model_context=checkpoint),
            ChatMessage(role="user", content="new input"),
            ChatMessage(role="assistant", content="new answer"),
            ChatMessage(role="system", content="display metadata"),
        ]
    )
    result = build_compacted_context_messages(session)
    assert result[:2] == checkpoint.history
    assert result[2]["content"] == "new input"
    assert result[3]["content"] == "new answer"
    assert "display metadata" not in str(result)

    result[0]["content"] = "mutated view"
    assert checkpoint.history[0]["content"] == "instructions"


def test_child_replay_keeps_the_suffix_after_a_saved_checkpoint():
    checkpoint = ModelContextCheckpoint(
        history=[{"role": "user", "content": "earlier compact context"}]
    )
    child = SubAgentConversation(
        path="/child",
        messages=[
            ChatMessage(role="assistant", model_context=checkpoint),
            ChatMessage(role="user", content="new question"),
            ChatMessage(role="assistant", content="new answer"),
        ],
    )
    replay = child.replay_history()
    assert [m.get_message() for m in convert_to_exmekh_messages(replay)] == [
        "earlier compact context",
        "new question",
        "new answer",
    ]
    replay[0]["content"] = "changed"
    assert (
        child.messages[0].model_context.history[0]["content"]
        == "earlier compact context"
    )


def test_archive_owns_replacement_data_and_turn_boundaries():
    archive = HistoryArchive()
    assert archive.checkpoint() is None
    archive.begin_turn([{"role": "user", "content": "earlier question"}])
    assert archive.update([]) == []
    originals = {"summary": [{"role": "assistant", "content": "original body"}]}
    archive.record_compaction(originals)
    originals["summary"][0]["content"] = "external mutation"
    compact = [
        {"role": "user", "content": "earlier question"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "tool_calls": [{"id": "summary"}]},
    ]
    expanded = archive.update(compact)
    assert expanded[2]["content"] == "original body"
    assert archive.checkpoint().turn_start == 2
    compact[1]["content"] = "mutated input"
    snapshot = archive.checkpoint()
    assert snapshot.history[1]["content"] == "new question"
    snapshot.history.clear()
    assert archive.checkpoint().history
    archive.begin_turn()
    assert archive.checkpoint().turn_start is None
    assert replay_model_context([]) is None


def test_collector_owns_the_checkpoint_received_from_the_runtime():
    collector = TurnStreamCollector()
    checkpoint = ModelContextCheckpoint(history=[{"role": "user", "content": "task"}])
    collector.set_model_context(checkpoint)
    checkpoint.history.clear()
    assert collector.snapshot_message().model_context.history[0]["content"] == "task"


@pytest.mark.parametrize("context_id", ["root", "child"])
@pytest.mark.parametrize("observe", [False, True])
def test_ds4_tracks_scope_checkpoints_independently_of_ui(probe, context_id, observe):
    observations = []
    probe.script = [tool_call("some-skill", {}, "open"), AssistantMessage("done")]
    tracker = ContextTrackingProvider(
        probe,
        on_context=observations.append if observe else None,
        context_id="root",
        ds4_qwen=True,
        thinking=False,
    )
    messages = [SystemMessage("instructions"), UserMessage("task")]
    response = tracker.call(
        ProviderRequest(
            messages,
            extra={
                "agentic": {
                    "agent_context_id": context_id,
                    "allowed_skills": ["some-skill"],
                }
            },
        )
    )
    returning = ProviderRequest(
        [*messages, response, ToolMessage("closed", "open", "some-skill")],
        extra={
            "agentic": {
                "agent_context_id": context_id,
                "skill_call_id": "open",
                "operation": "return",
                "allowed_skills": [],
            }
        },
    )
    assert tracker.estimate_context(returning).closure_tokens_ready
    if not observe or context_id == "child":
        assert not observations


def test_legacy_skill_archive_without_transcript_keeps_its_metadata():
    tree = [{"skill": "legacy", "children": [{"skill": "nested", "history": []}]}]
    assert ModelContextCheckpoint().expand_tree(tree) == tree


def test_estimated_saved_usage_does_not_become_authoritative(probe):
    tracker = ContextTrackingProvider(
        probe, on_context=None, context_id="root", ds4_qwen=False, thinking=False
    )
    tracker.restore_context_usage("root", ProviderContextUsage(prompt_tokens=123))
    request = ProviderRequest(
        [UserMessage("task")], extra={"agentic": {"agent_context_id": "root"}}
    )
    assert tracker.last_context_total(request) == 0


def test_compaction_event_is_optional_for_event_observers():
    on_tool = MagicMock()
    handler = DeepAgentTurnEventHandler(
        SubAgentEventRouter(None), lambda _: None, lambda *_: None, on_tool, None, None
    )
    handler.handle(
        DeepAgentHistoryCompactionEvent(
            depth=0, skill_path="/", history=[], history_tree=[], replaced_history={}
        )
    )
    on_tool.assert_not_called()
    assert handler.skill_stack == []


@pytest.mark.parametrize("context_id", ["root", "child"])
@pytest.mark.parametrize("with_user", [True, False])
def test_ds4_preflight_is_stable_and_does_not_change_the_sent_prompt(
    probe, context_id, with_user
):
    observations = []
    tracker = ContextTrackingProvider(
        probe,
        on_context=observations.append,
        context_id="root",
        ds4_qwen=True,
        thinking=False,
    )
    messages = [SystemMessage("instructions")]
    if with_user:
        messages.append(UserMessage("task"))
    request = ProviderRequest(
        messages,
        extra={
            "agentic": {
                "agent_context_id": context_id,
                "allowed_tools": [],
                "allowed_skills": [],
            }
        },
    )
    first = tracker.estimate_context(request).prompt_tokens
    assert first == count_text_tokens(render_ds4_qwen_prompt(
        [message.message_dump() for message in messages], [], thinking=False
    ))
    assert tracker.estimate_context(request).prompt_tokens == first
    if with_user:
        tracker.call(request)
        assert tracker.estimate_context(request).prompt_tokens == first
        if context_id == "root":
            assert observations[0].prompt_tokens == first
        else:
            assert not observations
