from __future__ import annotations

import asyncio

import pytest

from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.state.active_turns import ActiveTurnRegistry


@pytest.mark.asyncio
async def test_active_turn_snapshot_and_queue_handoff() -> None:
    registry = ActiveTurnRegistry()
    collector = TurnStreamCollector()
    task = asyncio.create_task(asyncio.sleep(10))
    other_task = asyncio.create_task(asyncio.sleep(0))
    await other_task

    assert registry.snapshot("chat", after_revision=-1) is None
    assert not registry.is_running("chat")
    assert not registry.is_task("chat", task)
    registry.start("chat", task, collector, started_at=12.0)
    assert registry.is_running("chat")
    assert registry.is_task("chat", task)
    assert not registry.is_task("chat", other_task)

    initial = registry.snapshot("chat", after_revision=-1)
    assert initial is not None
    assert initial.started_at == 12.0
    assert initial.revision == 0
    assert initial.message is not None and initial.message.is_streaming
    assert registry.snapshot("chat", after_revision=0).message is None

    registry.mark_changed("missing", collector)
    registry.mark_changed("chat", TurnStreamCollector())
    assert registry.snapshot("chat", after_revision=0).message is None
    collector.append_token("Hello")
    registry.mark_changed("chat", collector)
    updated = registry.snapshot("chat", after_revision=0)
    assert updated is not None
    assert updated.revision == 1
    assert updated.message is not None and updated.message.content == "Hello"

    registry.finish("chat", other_task, handoff_queue=True)
    assert registry.is_running("chat")
    registry.finish("chat", task, handoff_queue=True)
    assert not registry.is_running("chat")
    assert registry.claim_queue_handoff("chat")
    assert not registry.claim_queue_handoff("chat")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stopping_and_forgetting_active_turns() -> None:
    registry = ActiveTurnRegistry()
    assert not registry.stop("missing")
    registry.cancel_and_forget("missing")

    completed_task = asyncio.create_task(asyncio.sleep(0))
    await completed_task
    registry.start("done", completed_task, TurnStreamCollector(), started_at=1.0)
    assert not registry.stop("done")
    registry.cancel_and_forget("done")

    registry.start("done", completed_task, TurnStreamCollector(), started_at=7.0)
    registry.finish("done", completed_task, handoff_queue=True)
    cancelled_by_cleanup = asyncio.create_task(asyncio.sleep(10))
    registry.start(
        "cleanup", cancelled_by_cleanup, TurnStreamCollector(), started_at=8.0
    )
    registry.cancel_all()
    assert not registry.claim_queue_handoff("done")
    with pytest.raises(asyncio.CancelledError):
        await cancelled_by_cleanup
    assert not registry.is_running("cleanup")
    assert not registry.is_running("done")

    running_task = asyncio.create_task(asyncio.sleep(10))
    registry.start("running", running_task, TurnStreamCollector(), started_at=2.0)
    assert registry.stop("running")
    with pytest.raises(asyncio.CancelledError):
        await running_task
    registry.cancel_and_forget("running")
    assert not registry.is_running("running")

    forgotten_task = asyncio.create_task(asyncio.sleep(10))
    registry.start("forgotten", forgotten_task, TurnStreamCollector(), started_at=3.0)
    registry.cancel_and_forget("forgotten")
    with pytest.raises(asyncio.CancelledError):
        await forgotten_task
    assert not registry.is_running("forgotten")

    registry.start("done", completed_task, TurnStreamCollector(), started_at=4.0)
    registry.finish("done", completed_task)
    assert not registry.claim_queue_handoff("done")

    registry.start("done", completed_task, TurnStreamCollector(), started_at=5.0)
    registry.finish("done", completed_task, handoff_queue=True)
    registry.start("done", completed_task, TurnStreamCollector(), started_at=6.0)
    assert not registry.claim_queue_handoff("done")
    registry.cancel_and_forget("done")
