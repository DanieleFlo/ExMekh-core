from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.interfaces.provider import Provider
from qs.tokenizers.qwen import QwenTokenizer

from fe.config import config
from fe.qs_bridge.agent_runner import execute_agent_turn_sync
from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.context_usage import (
    compute_session_context_usage,
    count_text_tokens,
    populate_skill_closure_tokens,
)
from fe.qs_bridge.ds4_prompt import apply_ds4_scope, render_ds4_qwen_prompt
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    ProviderContextUsage,
    SkillNode,
    SubAgentConversation,
)
from fe.qs_bridge.provider_context import ContextTrackingProvider


class StubProvider(Provider):
    def __init__(self) -> None:
        super().__init__(QwenTokenizer())
        self.requests: list[ProviderRequest] = []
        self.usage: ModelUsage | None = ModelUsage(4000, 100, 4100)
        self.structured_result = object()
        self.tool_calls = None

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            AssistantMessage("done", tool_calls=self.tool_calls), usage=self.usage
        )

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        return self.structured_result


def search(q: str) -> str:
    return q


def test_request_observer_tracks_root_without_estimating_context_or_child_requests() -> (
    None
):
    provider = StubProvider()
    requests_started: list[bool] = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=None,
        context_id="root",
        ds4_qwen=False,
        thinking=False,
        on_request=lambda: requests_started.append(True),
    )
    root_request = ProviderRequest(
        [UserMessage("Root request")],
        extra={"agentic": {"agent_context_id": "root"}},
    )
    with patch.object(tracker, "_estimate_request", side_effect=AssertionError):
        assert tracker.call(root_request).get_message() == "done"
        assert provider.requests == [root_request]
        assert requests_started == [True]
        child_request = ProviderRequest(
            [UserMessage("Child request")],
            extra={"agentic": {"agent_context_id": "child"}},
        )
        tracker.call(child_request)
        assert requests_started == [True]
        assert provider.requests[-1] is child_request
        tracker.set_request_observer(None)
        tracker.call(root_request)
        assert requests_started == [True]


def test_live_context_deltas_reset_and_provider_totals_survive_save() -> None:
    collector = TurnStreamCollector()
    assert collector.get_provider_context() is None
    session = ChatSession(messages=[ChatMessage(role="user", content="test")])
    collector.append_reasoning("Old reasoning that must not be counted again")
    collector.append_token("Old response")
    collector.set_provider_context(ProviderContextUsage(prompt_tokens=4000))
    assert collector.get_provider_context().prompt_tokens == 4000
    collector.append_reasoning("Current reasoning grows here")
    collector.append_token("Current answer")
    expected = (
        4000
        + count_text_tokens("Current reasoning grows here")
        + count_text_tokens("Current answer")
    )
    assert (
        compute_session_context_usage(session, collector=collector).used_tokens
        == expected
    )
    assert compute_session_context_usage(session, collector=collector).percentage > 0
    snapshot = collector.snapshot_message()
    collector.append_token(" grows")
    assert snapshot.context_usage.streamed_content == "Current answer"
    assert (
        compute_session_context_usage(session, collector=collector).used_tokens
        > expected
    )

    collector.set_provider_context(
        ProviderContextUsage(prompt_tokens=4050, completion_tokens=150, is_final=True)
    )
    assert (
        compute_session_context_usage(session, collector=collector).used_tokens == 4200
    )
    collector.set_provider_context(ProviderContextUsage(prompt_tokens=4300))
    assert (
        compute_session_context_usage(session, collector=collector).used_tokens == 4300
    )
    collector.append_reasoning("New generation")
    assert (
        compute_session_context_usage(session, collector=collector).used_tokens > 4300
    )
    collector.set_provider_context(
        ProviderContextUsage(prompt_tokens=4320, completion_tokens=80, is_final=True)
    )
    session.messages.append(collector.build_message())
    restored = ChatSession.model_validate_json(session.model_dump_json())
    assert compute_session_context_usage(restored).used_tokens == 4400
    sub_agent = SubAgentConversation(path="main/sub", messages=restored.messages)
    assert (
        compute_session_context_usage(restored, sub_agent=sub_agent).used_tokens == 4400
    )


def test_nested_closure_counts_are_exact_and_do_not_follow_final_answer() -> None:
    collector = TurnStreamCollector()
    session = ChatSession(messages=[ChatMessage(role="user", content="test")])
    nested = SkillNode(
        name="nested", path="/skill/nested", depth=2, opening_tool_call_id="nested-call"
    )
    parent = SkillNode(
        name="skill",
        path="/skill",
        depth=1,
        opening_tool_call_id="parent-call",
        sub_skills=[nested],
    )
    collector.append_skill(parent)
    collector.set_provider_context(
        ProviderContextUsage(prompt_tokens=4500, completion_tokens=200, is_final=True)
    )
    nested.status = "completed"
    collector.append_skill(nested.model_copy(deep=True), "completed")
    collector.set_provider_context(
        ProviderContextUsage(prompt_tokens=4100, return_call_id="nested-call")
    )
    collector.append_token("Response after nested skill")
    populate_skill_closure_tokens(session, collector=collector)
    assert collector.skills[0].sub_skills[0].tokens_pending
    assert collector.skills[0].sub_skills[0].tokens_after_close is None
    collector.set_provider_context(
        ProviderContextUsage(
            prompt_tokens=4120,
            completion_tokens=100,
            return_call_id="nested-call",
            is_final=True,
        )
    )
    assert (
        collector.skills[0].sub_skills[0].tokens_before_close,
        collector.skills[0].sub_skills[0].tokens_after_close,
    ) == (4700, 4120)

    parent.status = "completed"
    collector.append_skill(parent.model_copy(deep=True), "completed")
    collector.set_provider_context(
        ProviderContextUsage(prompt_tokens=3900, return_call_id="parent-call")
    )
    collector.append_token("Final answer continues growing")
    assert collector.skills[0].tokens_pending
    collector.set_provider_context(
        ProviderContextUsage(
            prompt_tokens=3950,
            completion_tokens=300,
            return_call_id="parent-call",
            is_final=True,
        )
    )
    tree = [
        {
            "skill": "skill",
            "opening_tool_call_id": "parent-call",
            "output": "done",
            "history": [],
            "children": [
                {
                    "skill": "nested",
                    "opening_tool_call_id": "nested-call",
                    "output": "nested done",
                    "history": [],
                }
            ],
        }
    ]
    collector.set_raw_history([], tree)
    populate_skill_closure_tokens(session, collector=collector)
    assert (
        collector.skills[0].tokens_before_close,
        collector.skills[0].tokens_after_close,
    ) == (4220, 3950)
    collected_tree = collector.snapshot_message().skill_history_tree
    assert collected_tree[0]["tokens_before_close"] == 4220
    assert collected_tree[0]["children"][0]["tokens_after_close"] == 4120
    assert "tokens_before_close" not in tree[0]
    collector.append_token(" More final content")
    populate_skill_closure_tokens(session, collector=collector)
    session.messages.append(collector.build_message())
    populate_skill_closure_tokens(session)
    assert session.messages[-1].skills[0].tokens_after_close == 3950
    assert compute_session_context_usage(session).used_tokens == 4250


def test_raw_history_replacement_retains_badge_identity() -> None:
    collector = TurnStreamCollector()
    node = SkillNode(
        name="skill",
        path="/skill",
        depth=1,
        status="completed",
        opening_tool_call_id="right",
        tokens_before_close=5000,
        tokens_after_close=4000,
    )
    collector.append_skill(node, "completed")
    calls = [
        {"id": "other", "function": {"name": "other"}},
        {"id": "right", "function": {"name": "skill"}},
    ]
    collector.set_raw_history(
        [],
        [
            {
                "skill": "skill",
                "history": [{"tool_calls": json.dumps(calls)}],
                "output": "done",
            },
            {"skill": "missing", "history": [{"tool_calls": "null"}]},
        ],
    )
    assert collector.skill_history_tree[0]["tokens_after_close"] == 4000
    assert "tokens_after_close" not in collector.skill_history_tree[1]
    collector.set_provider_context(
        ProviderContextUsage(
            prompt_tokens=4100, return_call_id="missing", is_final=True
        )
    )
    assert collector.skills[0].tokens_after_close == 4000
    node.await_closure_tokens()
    assert not node.tokens_pending
    snapshot = collector.skill_snapshots()[0]
    snapshot.name = "changed"
    assert collector.skills[0].name == "skill"

    reopened = SkillNode(
        name="skill", path="/skill", depth=1, opening_tool_call_id="second"
    )
    collector.append_skill(reopened)
    assert len(collector.skills) == 2
    assert collector.skill_snapshot("/skill").opening_tool_call_id == "second"
    assert collector.skills[0].tokens_after_close == 4000
    collector.set_raw_history(
        [],
        [
            {
                "skill": "skill",
                "output": None,
                "history": [
                    {"tool_calls": [{"id": "second", "function": {"name": "skill"}}]}
                ],
            }
        ],
    )
    assert collector.skill_history_tree[0]["tokens_pending"] is False
    assert "tokens_after_close" not in collector.skill_history_tree[0]


def test_ds4_render_includes_tools_scope_roles_and_dsml() -> None:
    tools = [{"function": {"name": "search", "parameters": {"type": "object"}}}]
    calls = [
        {
            "function": {
                "name": 'search&"',
                "arguments": json.dumps(
                    {
                        "q": "x </｜DSML｜parameter>",
                        "value": [1, {"text": "</｜DSML｜parameter>"}],
                    }
                ),
            }
        }
    ]
    messages = [
        {"role": "system", "content": " system "},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps(calls)},
        {"role": "tool", "content": "result"},
        {"role": "tool", "content": "second result"},
        {"role": "assistant", "content": "<think>already wrapped</think>answer"},
        {"role": "user", "content": "follow up"},
    ]
    apply_ds4_scope(
        messages,
        {"allowed_tools": ["search"], "allowed_skills": [], "operation": "return"},
    )
    prompt = render_ds4_qwen_prompt(messages, tools)
    assert '"name":"search"' in prompt
    assert "Allowed ordinary tools: search" in prompt
    assert "has already finished and EXITED" in prompt
    assert 'name="search&amp;&quot;"' in prompt
    assert 'string="true">x &lt;/｜DSML｜parameter>' in prompt
    assert "\\u003c/｜DSML｜parameter>" in prompt
    assert prompt.count("<tool_response>") == 2
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert "<think>already wrapped</think>answer" in prompt
    without_thinking = render_ds4_qwen_prompt(messages, [], thinking=False)
    assert without_thinking.endswith("<think>\n\n</think>\n\n")
    assert render_ds4_qwen_prompt([], []) == ""
    assert "old thought" not in render_ds4_qwen_prompt(
        [
            {
                "role": "assistant",
                "content": "earlier",
                "reasoning_content": "old thought",
            },
            {"role": "user", "content": "next"},
        ],
        [],
    )
    assert "new thought" in render_ds4_qwen_prompt(
        [
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "new thought",
            },
        ],
        [],
    )
    scope_only = [{"role": "system", "content": ""}]
    apply_ds4_scope(scope_only, {"allowed_tools": [], "allowed_skills": []})
    assert scope_only[-1]["role"] == "system"
    empty_tool = [{"role": "tool", "content": ""}]
    apply_ds4_scope(empty_tool, {"allowed_tools": [], "allowed_skills": []})
    assert empty_tool[0]["content"].startswith("<agent_runtime>")
    tool_tail = render_ds4_qwen_prompt(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"q": "test"}}}
                ],
            },
            {"role": "tool", "content": "result"},
        ],
        [],
    )
    assert 'string="true">test</｜DSML｜parameter>' in tool_tail
    assert "</tool_response><|im_end|>\n<|im_start|>assistant\n" in tool_tail


def test_tracking_provider_forwards_requests_and_isolates_child_context() -> None:
    provider = StubProvider()
    contexts = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append,
        context_id="root",
        ds4_qwen=True,
        thinking=True,
    )
    request = ProviderRequest(
        [SystemMessage("system"), UserMessage("question")],
        extra={
            "agentic": {
                "agent_context_id": "root",
                "allowed_tools": [],
                "allowed_skills": [],
            }
        },
    )
    response = tracker.call(request)
    assert response.usage.total_tokens == 4100
    assert provider.requests[-1] is request
    assert contexts[-1].prompt_tokens == 4000
    assert contexts[-1].completion_tokens == 100
    assert contexts[-1].is_final
    assert "agent_runtime" not in request.get_messages()[-1].get_message()
    tracker.call(request)
    child_request = ProviderRequest(
        [UserMessage("child")], extra={"agentic": {"agent_context_id": "child"}}
    )
    tracker.call(child_request)
    assert len(contexts) == 4
    changed_request = ProviderRequest([UserMessage("different")])
    tracker.call(changed_request)
    provider.usage = None
    tracker.call(changed_request)
    provider.usage = ModelUsage(completion_tokens=1)
    tracker.call(changed_request)
    assert len(contexts) == 8
    assert tracker.call_structured(object()) is provider.structured_result


def test_tracking_other_provider_counts_tools_and_handles_tool_choice_none() -> None:
    provider = StubProvider()
    contexts = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append,
        context_id=None,
        ds4_qwen=False,
        thinking=False,
    )
    tools = Tools([Tool(search, description="Find a result")])
    request = ProviderRequest([UserMessage("question")], tools=tools)
    tracker.call(request)
    with_tools = contexts[0].prompt_tokens
    tracker.call(
        ProviderRequest([UserMessage("question")], tools=tools, tool_choice="none")
    )
    assert contexts[2].prompt_tokens < with_tools


@pytest.mark.parametrize(
    "base_url", ["http://localhost:8080/v1", "https://inference.example.test/v1"]
)
def test_agent_runner_wires_typed_context_observer(base_url: str) -> None:
    contexts = []
    provider = StubProvider()
    with (
        patch(
            "fe.qs_bridge.agent_runner.LocalProvider", return_value=provider
        ) as provider_factory,
        patch.dict(config.model_owners, {(base_url, "test-qwen"): "ds4.c"}),
    ):
        execute_agent_turn_sync(
            user_prompt="question",
            model="test-qwen",
            base_url=base_url,
            api_key="test",
            api_type="responses",
            reasoning_effort_str="none",
            agent_mode="quick_chat",
            capabilities_path_str="",
            name_hds="main",
            on_token=lambda _: None,
            on_reasoning=lambda _: None,
            on_tool_event=lambda _: None,
            on_context=contexts.append,
        )
    assert contexts[-1].prompt_tokens == 4000
    assert provider_factory.call_args.kwargs["base_url"] == base_url


def test_ds4_checkpoint_publishes_closure_before_the_final_response() -> None:
    provider = StubProvider()
    provider.usage = ModelUsage(4000, 100, 4100)
    call = ToolCall(id="skill-call", function=FunctionCall(name="skill", arguments={}))
    provider.tool_calls = [
        ToolCall(
            id="ordinary-call", function=FunctionCall(name="ordinary", arguments={})
        ),
        call,
    ]
    collector = TurnStreamCollector()
    contexts = []

    def collect(context):
        contexts.append(context)
        collector.set_provider_context(context)

    tracker = ContextTrackingProvider(
        provider, on_context=collect, context_id="root", ds4_qwen=True, thinking=True
    )
    user = UserMessage("question")
    scope = {
        "agent_context_id": "root",
        "allowed_tools": [],
        "allowed_skills": ["skill"],
    }
    tracker.call(ProviderRequest([user], extra={"agentic": scope}))
    collector.append_skill(
        SkillNode(
            name="skill",
            path="/skill",
            depth=1,
            status="completed",
            opening_tool_call_id="skill-call",
        ),
        "completed",
    )
    provider.tool_calls = None
    tracker.call(
        ProviderRequest(
            [
                user,
                AssistantMessage("", tool_calls=[call]),
                ToolMessage("done", "skill-call"),
            ],
            extra={
                "agentic": {
                    **scope,
                    "operation": "return",
                    "skill_call_id": "skill-call",
                }
            },
        )
    )
    start = contexts[2]
    assert start.closure_tokens_ready
    assert start.prompt_tokens > 4101
    assert collector.skills[0].tokens_before_close == 4100
    assert collector.skills[0].tokens_after_close == start.prompt_tokens
    assert not collector.skills[0].tokens_pending
    collector.append_token("The caller continues its final answer")
    collector.set_provider_context(
        ProviderContextUsage(
            prompt_tokens=start.prompt_tokens,
            completion_tokens=300,
            return_call_id="skill-call",
            is_final=True,
        )
    )
    assert collector.skills[0].tokens_after_close == start.prompt_tokens
    tracker.call(
        ProviderRequest(
            [user],
            extra={
                "agentic": {**scope, "operation": "return", "skill_call_id": "missing"}
            },
        )
    )
    assert not contexts[-2].closure_tokens_ready


def test_ds4_checkpoint_requires_completion_usage() -> None:
    provider = StubProvider()
    provider.usage = ModelUsage(prompt_tokens=4000)
    provider.tool_calls = [
        ToolCall(id="skill-call", function=FunctionCall(name="skill", arguments={}))
    ]
    contexts = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append,
        context_id="root",
        ds4_qwen=True,
        thinking=True,
    )
    scope = {
        "agent_context_id": "root",
        "allowed_tools": [],
        "allowed_skills": ["skill"],
    }
    tracker.call(ProviderRequest([UserMessage("question")], extra={"agentic": scope}))
    tracker.call(
        ProviderRequest(
            [
                UserMessage("question"),
                AssistantMessage("", tool_calls=provider.tool_calls),
                ToolMessage("done", "skill-call"),
            ],
            extra={"agentic": {**scope, "skill_call_id": "skill-call"}},
        )
    )
    assert not contexts[2].closure_tokens_ready
