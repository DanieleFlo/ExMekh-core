import time
from dataclasses import replace

from exmekh_lib.core.agents.deep_agent import DeepAgentCompactionProgressEvent
from exmekh_lib.core.history_compaction.progress import CompactionProgress
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.compaction_progress import CompactionProgressView


def event(progress, scope="/"):
    return DeepAgentCompactionProgressEvent(0, scope, "root", progress)


def test_progress_survives_live_snapshots_but_is_never_saved_in_messages():
    collector = TurnStreamCollector()
    first = CompactionProgress("first", 100, total_outputs=4)
    second = replace(first, operation_id="second")
    collector.set_compaction_progress(event(first))
    collector.set_compaction_progress(event(second, "/child"))
    snapshot = collector.snapshot_message()
    assert len(snapshot.compaction_progress) == 2
    assert "compaction_progress" not in snapshot.model_dump()
    assert not collector.build_message().compaction_progress
    collector.set_compaction_progress(event(replace(first, phase="completed")))
    assert collector.compaction_snapshot() == [event(second, "/child")]
    collector.set_compaction_progress(event(replace(second, phase="failed")))
    assert not collector.compaction_snapshot()
    assert len(snapshot.compaction_progress) == 2


def test_progress_bar_updates_in_place_and_removes_finished_operations():
    client = Client(page("/compaction-progress"))
    try:
        with client:
            view = CompactionProgressView()
            first = CompactionProgress("first", time.monotonic() - 4, total_outputs=5)
            view.show([event(first)])
            assert view.visible
            bar = next(
                item
                for item in view.descendants()
                if isinstance(item, ui.linear_progress)
            )
            assert bar.value == 0
            assert "indeterminate" in bar._props
            label = list(
                item for item in view.descendants() if isinstance(item, ui.label)
            )[-1]
            assert "Evaluating outputs" in label.text
            assert "Elapsed 4s" in label.text
            updated = replace(
                first,
                phase="summarizing",
                completed_outputs=2,
                total_summaries=3,
                completed_summaries=1,
                summary_seconds=4,
            )
            view.show([event(updated)])
            assert bar.value == 0.4
            assert "indeterminate" not in bar._props
            assert "Estimated remaining 8s" in label.text
            view.show([event(replace(updated, completed_outputs=5))])
            assert "Finalizing" in label.text
            view.show([event(replace(updated, phase="failed"))])
            assert not view.visible
            assert not list(view.descendants())
            view.show([])
    finally:
        client.delete()


def test_restored_stream_keeps_progress_until_compaction_or_turn_finishes():
    client = Client(page("/restore-compaction"))
    try:
        with client:
            collector = TurnStreamCollector()
            progress = CompactionProgress("current", time.monotonic(), total_outputs=2)
            collector.set_compaction_progress(event(progress))
            view = ChatViewComponent(lambda _: None)
            view.update_compaction_progress([event(progress)])
            view.build()
            view.start_assistant_stream()
            view.restore_stream_snapshot(collector.snapshot_message())
            bar = view.compaction_progress_view
            assert bar.visible
            view._update_stream_duration()
            collector.set_compaction_progress(
                event(replace(progress, phase="completed"))
            )
            view.restore_stream_snapshot(collector.snapshot_message())
            assert not bar.visible
            view.update_compaction_progress([event(progress)])
            view.finalize_stream()
            assert not bar.visible
    finally:
        client.delete()
