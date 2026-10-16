from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.user_message import UserMessage
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.history.tool_calls import restore_tool_call_ids
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ProviderContextUsage,
    ReasoningBlock,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    ContextUsageSummary,
    compute_session_context_usage,
    count_message_dict_tokens,
    count_text_tokens,
)
from exmekh_agent.exmekh_bridge.providers.provider_context import (
    ContextTrackingProvider,
)
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.history.raw_history_view import format_json_to_highlighted_html
from exmekh_agent.ui.shared.output_format import format_output_html
from tests.test_provider_context import StructuredAnswer, StubProvider


def test_stateless_summaries_do_not_replace_agent_context():
    provider = StubProvider()
    contexts = []
    started = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append,
        context_id="root",
        ds4_qwen=False,
        thinking=True,
        agentic_only=True,
        on_request=lambda: started.append(True),
    )
    tracker.call(
        ProviderRequest(
            [UserMessage("question")], extra={"agentic": {"agent_context_id": "root"}}
        )
    )
    assert len(contexts) == 2
    summary = ProviderRequest(
        [UserMessage("thinking trace")], reasoning_effort_override="none"
    )
    tracker.call(summary)
    structured = StructuredProviderRequest([UserMessage("summary")], StructuredAnswer)
    tracker.call_structured(structured)
    assert provider.requests[-2:] == [summary, structured]
    assert len(contexts) == 2 and started == [True]


@pytest.mark.parametrize("structured", [False, True])
def test_final_recount_without_usage_includes_nonstreamed_content(structured):
    provider = StubProvider()
    provider.usage = None
    provider.tool_calls = [
        ToolCall(
            id="call",
            function=FunctionCall(name="search", arguments={"q": "full arguments"}),
        )
    ]
    provider.structured_result.usage = None
    provider.structured_result.thinking = ModelThinking("Structured reasoning")
    contexts = []
    tracker = ContextTrackingProvider(
        provider,
        on_context=contexts.append,
        context_id=None,
        ds4_qwen=False,
        thinking=True,
    )
    request = (
        StructuredProviderRequest([UserMessage("question")], StructuredAnswer)
        if structured
        else ProviderRequest([UserMessage("question")])
    )
    response = tracker.call_structured(request) if structured else tracker.call(request)
    final = contexts[-1]
    expected = count_message_dict_tokens(response.message_dump())
    if response.thinking:
        expected += count_text_tokens(response.thinking.thoughts)
    assert final.is_final and final.is_estimated
    assert final.total_tokens(count_text_tokens) == final.prompt_tokens + expected
    restored = ProviderContextUsage.model_validate_json(final.model_dump_json())
    assert restored.total_tokens(count_text_tokens) == final.total_tokens(
        count_text_tokens
    )


def test_new_stream_does_not_reuse_previous_turn_total():
    session = ChatSession(
        messages=[
            ChatMessage(
                role="assistant",
                content="Old",
                context_usage=ProviderContextUsage(
                    prompt_tokens=9999, completion_tokens=1, is_final=True
                ),
            ),
            ChatMessage(role="user", content="New"),
        ]
    )
    collector = TurnStreamCollector()
    collector.append_token("Current stream")
    usage = compute_session_context_usage(session, collector)
    assert usage.estimated and usage.used_tokens != 10000


@pytest.mark.parametrize(
    "xml",
    [
        "<result><nested><value>42</value></nested></result>",
        '<?xml version="1.0"?><result><nested><value>42</value></nested></result>',
        "<first><value>42</value></first><second><value>43</value></second>",
    ],
)
def test_xml_documents_and_siblings_are_indented_in_raw_history(xml):
    formatted = format_json_to_highlighted_html({"content": xml})
    assert "json-readable-string" in formatted
    assert "\n  &lt;" in formatted
    assert "&lt;value&gt;42&lt;/value&gt;" in formatted
    assert "<result>" not in formatted


def test_xml_mixed_content_remains_readable_and_escaped():
    text = "<result>before <b>bold</b> after</result>"
    assert "before &lt;b&gt;bold&lt;/b&gt; after" in format_output_html(text)


def test_tool_output_arriving_after_completion_updates_the_open_panel():
    client = Client(page("/live-output"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            view.build()
            view.start_assistant_stream()
            event = ToolEvent(
                tool_name="calculate", tool_call_id="same", status="completed"
            )
            view.add_tool_event(event)
            view._disclosures.set_expanded(f"tool:{event.id}", True)
            view.update_stream_history(
                ChatMessage(
                    role="assistant",
                    raw_history=[
                        {
                            "role": "tool",
                            "tool_call_id": "same",
                            "content": "LIVE RESULT",
                        }
                    ],
                )
            )
            panels = [
                element
                for element in view.current_skills_container.descendants()
                if "tool-output-panel" in element._classes
            ]
            assert panels[0].visible
            assert any(
                "LIVE RESULT" in str(getattr(element, "content", ""))
                for element in panels[0].descendants()
            )
            view.finalize_stream()
            assert view._streaming_history is None
    finally:
        client.delete()


async def test_continuous_recounts_publish_intermediate_results():
    client = Client(page("/context-progress"))
    release = Event()
    begun = asyncio.Event()
    loop = asyncio.get_running_loop()
    session = session_manager.create_session()
    collector = TurnStreamCollector()
    calls = 0
    observed = []
    with client:
        controller = ChatPageController(client)
        controller.input_bar = MagicMock()
        controller.input_bar.update_context_info.side_effect = lambda **kwargs: (
            observed.append(kwargs["used_tokens"])
        )

    def count(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            loop.call_soon_threadsafe(begun.set)
            assert release.wait(5)
        return ContextUsageSummary("model", calls * 100, 10000, True)

    try:
        with patch(
            "exmekh_agent.ui.chat.context_updates.compute_session_context_usage",
            side_effect=count,
        ):
            controller.update_context_usage(session, collector)
            await asyncio.wait_for(begun.wait(), 5)
            # A newer request arrives before the first count has finished.
            controller.update_context_usage(session, collector)
            release.set()
            await asyncio.wait_for(controller._context_updates.wait_idle(), 5)
            assert observed == [100, 200]
    finally:
        release.set()
        client.delete()


async def test_reopened_page_counts_the_active_stream():
    client = Client(page("/context-reconnect"))
    session = session_manager.create_session()
    collector = TurnStreamCollector()
    collector.set_provider_context(ProviderContextUsage(prompt_tokens=1000))
    collector.append_reasoning("Live sub-skill reasoning", "/deep-thinking", 1)
    task = asyncio.create_task(asyncio.Event().wait())
    active_turns.start(session.id, task, collector, started_at=0)
    try:
        with client:
            controller = ChatPageController(client)
            controller.input_bar = MagicMock()
            controller.update_context_usage(session)
        await asyncio.wait_for(controller._context_updates.wait_idle(), 5)
        assert (
            controller.input_bar.update_context_info.call_args.kwargs["used_tokens"]
            > 1000
        )
    finally:
        await active_turns.stop_and_wait(session.id)
        client.delete()


def test_live_dump_ids_match_named_batches_and_preserve_explicit_ids():
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "first"}},
                {"id": "b", "function": {"name": "second"}},
                {"id": "c", "function": {"name": "third"}},
            ],
        },
        {
            "role": "tool",
            "content": "<header><name>second</name><id>out</id></header>Second",
        },
        {"role": "tool", "tool_call_id": "a", "content": "First"},
        {"role": "function", "content": "Third"},
        {"role": "tool", "tool_call_id": "external", "content": "Unmatched"},
        {"role": "tool", "name": "missing", "content": "Unknown"},
    ]
    restore_tool_call_ids(history)
    assert [entry.get("tool_call_id") for entry in history[1:]] == [
        "b",
        "a",
        "c",
        "external",
        None,
    ]
    assert [entry["name"] for entry in history[1:4]] == ["second", "first", "third"]


def test_final_recount_retains_thinking_when_only_streamed():
    provider = StubProvider()
    provider.usage = None
    collector = TurnStreamCollector()
    tracker = ContextTrackingProvider(
        provider,
        on_context=collector.set_provider_context,
        context_id=None,
        ds4_qwen=False,
        thinking=True,
    )

    def stream(request):
        collector.append_reasoning("Thinking supplied only in streaming")
        collector.append_token("Partial answer")
        return ProviderResponse(AssistantMessage("Complete final answer"))

    provider.call = stream
    tracker.call(ProviderRequest([UserMessage("question")]))
    final = collector.get_provider_context()
    assert final.is_final and final.is_estimated
    assert final.total_tokens(count_text_tokens) == (
        final.prompt_tokens
        + count_text_tokens("Complete final answer")
        + count_text_tokens("Thinking supplied only in streaming")
    )


def test_reconnected_reasoning_updates_keep_execution_controls_mounted():
    client = Client(page("/stable-activity"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            view.build()
            view.start_assistant_stream()
            snapshot = ChatMessage(
                role="assistant",
                skills=[
                    SkillNode(
                        name="work",
                        path="/work",
                        depth=1,
                        status="completed",
                        start_time=1,
                        end_time=2,
                        duration_seconds=1,
                    )
                ],
            )
            view.restore_stream_snapshot(snapshot)
            before = tuple(
                element.id for element in view.current_skills_container.descendants()
            )
            updated = snapshot.model_copy(deep=True)
            updated.reasoning_blocks = [ReasoningBlock(content="more root reasoning")]
            view.restore_stream_snapshot(updated)
            after = tuple(
                element.id for element in view.current_skills_container.descendants()
            )
            assert before == after
            changed = updated.model_copy(deep=True)
            changed.skills[0].tokens_before_close = 100
            changed.skills[0].tokens_after_close = 20
            view.restore_stream_snapshot(changed)
            assert (
                tuple(
                    element.id
                    for element in view.current_skills_container.descendants()
                )
                != after
            )
    finally:
        client.delete()


def test_reasoning_expansion_choices_survive_rendering_the_same_blocks_again():
    client = Client(page("/stable-reasoning-expansion"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            block = ReasoningBlock(content="recorded reasoning", end_time=1)
            original = view._render_reasoning_group([block])
            original.set_value(True)
            child = next(
                element
                for element in original.descendants()
                if isinstance(element, ui.expansion)
            )
            child.set_value(True)
            restored = view._render_reasoning_group([block.model_copy(deep=True)])
            assert restored.value is True
            restored_child = next(
                element
                for element in restored.descendants()
                if isinstance(element, ui.expansion)
            )
            assert restored_child.value is True
    finally:
        client.delete()
