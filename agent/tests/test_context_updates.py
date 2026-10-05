from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    SubAgentConversation,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import ContextUsageSummary
from exmekh_agent.ui.chat.context_updates import ContextUsageUpdater


@pytest.fixture
def client():
    client = Client(page("/context-updates"))
    yield client
    if not client.is_deleted:
        client.delete()


async def _wait_started(event: Event) -> None:
    assert await asyncio.to_thread(event.wait, 2)


async def test_count_owns_root_stream_and_subagent_snapshots(client):
    observed = []
    results = []
    session = ChatSession(messages=[ChatMessage(role="user", content="original")])
    collector = TurnStreamCollector()
    collector.append_token("partial")
    subagent = SubAgentConversation(
        path="/worker", messages=[ChatMessage(role="assistant", content="retained")]
    )

    def count(**kwargs):
        observed.append(kwargs)
        return ContextUsageSummary("model", 10, 100, True)

    updater = ContextUsageUpdater(client, lambda *result: results.append(result))
    await updater.wait_idle()
    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage", count
    ):
        updater.request(session, model_name="model", collector=collector)
        session.messages[0].content = "changed"
        collector.append_token(" later")
        await updater.wait_idle()
        assert [m.content for m in observed[0]["session"].messages] == [
            "original",
            "partial",
        ]
        updater.request(
            session, model_name="model", sub_agent=subagent, collector=collector
        )
        subagent.messages[0].content = "changed"
        await updater.wait_idle()
        assert observed[1]["sub_agent"].messages[0].content == "retained"
        assert len(observed[1]["session"].messages) == 1
    assert len(results) == 2


@pytest.mark.parametrize("fail_first", [False, True])
async def test_superseded_targets_never_publish_old_results(client, fail_first):
    started, release = Event(), Event()
    requests, results = [], []
    first, second = ChatSession(), ChatSession()

    def count(**kwargs):
        requests.append(kwargs["session"].id)
        if len(requests) == 1:
            started.set()
            assert release.wait(3)
            if fail_first:
                raise RuntimeError("provider snapshot unavailable")
        return ContextUsageSummary("model", len(requests), 100, True)

    updater = ContextUsageUpdater(client, lambda *result: results.append(result))
    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage", count
    ):
        updater.request(first, model_name="model")
        try:
            await _wait_started(started)
            updater.request(second, model_name="model")
        finally:
            release.set()
        await updater.wait_idle()
    assert requests == [first.id, second.id]
    assert [result[0] for result in results] == [second.id]


async def test_failed_count_can_be_retried(client, caplog):
    results = []
    updater = ContextUsageUpdater(client, lambda *result: results.append(result))
    session = ChatSession()
    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage",
        side_effect=ValueError("invalid history"),
    ):
        updater.request(session, model_name="model")
        await updater.wait_idle()
    assert not results and "Unable to count context tokens" in caplog.text
    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage",
        return_value=ContextUsageSummary("model", 2, 100, True),
    ):
        updater.request(session, model_name="model")
        await updater.wait_idle()
    assert len(results) == 1


async def test_client_deletion_discards_running_and_future_counts(client):
    started, release = Event(), Event()
    results = []
    updater = ContextUsageUpdater(client, lambda *result: results.append(result))

    def count(**kwargs):
        started.set()
        assert release.wait(3)
        return ContextUsageSummary("model", 1, 100, True)

    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage", count
    ):
        updater.request(ChatSession(), model_name="model")
        try:
            await _wait_started(started)
            client.delete()
            updater.request(ChatSession(), model_name="model")
        finally:
            release.set()
        await updater.wait_idle()
    assert not results


async def test_stream_schedule_throttles_and_rechecks_lifecycle(client):
    delivered = []
    updater = ContextUsageUpdater(client, lambda *_: None)
    loop = asyncio.get_running_loop()
    with patch("exmekh_agent.ui.chat.context_updates.time.monotonic", return_value=10):
        updater.schedule(loop, lambda: delivered.append("first"))
        updater.schedule(loop, lambda: delivered.append("throttled"))
        updater.schedule(loop, lambda: delivered.append("forced"), force=True)
        updater.schedule(
            loop,
            lambda: delivered.append("accepted"),
            force=True,
            accepts_events=lambda: True,
        )
        updater.schedule(
            loop,
            lambda: delivered.append("rejected"),
            force=True,
            accepts_events=lambda: False,
        )
        await asyncio.sleep(0)
        updater.schedule(loop, lambda: delivered.append("closed"), force=True)
        updater.close()
        await asyncio.sleep(0)
        updater.schedule(loop, lambda: delivered.append("too late"), force=True)
    assert delivered == ["first", "forced", "accepted"]


async def test_closing_before_worker_starts_never_counts(client):
    updater = ContextUsageUpdater(
        client, lambda *_: pytest.fail("closed view was updated")
    )
    with patch(
        "exmekh_agent.ui.chat.context_updates.compute_session_context_usage"
    ) as count:
        updater.request(ChatSession(), model_name="model")
        updater.close()
        await updater.wait_idle()
    count.assert_not_called()
