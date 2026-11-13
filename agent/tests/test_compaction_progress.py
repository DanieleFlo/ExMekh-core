import time
from dataclasses import replace

import pytest
from exmekh_lib.core.agents.deep_agent import DeepAgentCompactionProgressEvent
from exmekh_lib.core.history_compaction.progress import CompactionProgress
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.history.compaction import (
    CompactionActivity,
    ModelContextCheckpoint,
    RecordedCompaction,
)
from exmekh_agent.exmekh_bridge.models import ChatMessage, SkillNode, ToolEvent
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.compaction_progress import (
    CompactionProgressView,
    render_applied_compaction,
)


def event(progress, scope="/"):
    return DeepAgentCompactionProgressEvent(0, scope, "root", progress)


def test_progress_survives_live_snapshots_but_is_never_saved_in_messages():
    collector = TurnStreamCollector()
    first = CompactionProgress("first", 100, total_outputs=4)
    second = replace(first, operation_id="second")
    collector.set_compaction_progress(event(first))
    started = collector.compaction_snapshot()[0].timestamp
    collector.set_compaction_progress(event(replace(first, phase="summarizing")))
    assert collector.compaction_snapshot()[0].timestamp == started
    collector.set_compaction_progress(event(second, "/child"))
    snapshot = collector.snapshot_message()
    assert len(snapshot.compaction_progress) == 2
    assert "compaction_progress" not in snapshot.model_dump()
    assert not collector.build_message().compaction_progress
    collector.set_compaction_progress(event(replace(first, phase="completed")))
    assert [item.source for item in collector.compaction_snapshot()] == [
        event(second, "/child")
    ]
    collector.set_compaction_progress(event(replace(second, phase="failed")))
    assert not collector.compaction_snapshot()
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
            elapsed = next(
                item
                for item in view.descendants()
                if "compaction-elapsed" in item._classes
            )
            assert "Elapsed 4s" in elapsed.text
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
            tool = ToolEvent(tool_name="lookup", timestamp=time.time() - 5)
            collector.append_tool(tool)
            before = collector.append_token("Before compaction.")
            progress = CompactionProgress("current", time.monotonic(), total_outputs=2)
            collector.set_compaction_progress(event(progress))
            [active] = collector.compaction_snapshot()
            view = ChatViewComponent(lambda _: None)
            view.update_compaction_progress([active])
            view.build()
            view.start_assistant_stream()
            view.restore_stream_snapshot(collector.snapshot_message())
            bar = view.live_compaction_views[active.id]
            assert bar.visible
            assert view.current_activity_container.default_slot.children[-1] is bar
            view._update_stream_duration()
            view.update_compaction_progress(collector.compaction_snapshot())
            assert view.live_compaction_views[active.id] is bar
            collector.set_compaction_progress(
                event(replace(progress, phase="completed"))
            )
            collector.set_model_context(
                ModelContextCheckpoint(
                    compactions=[
                        RecordedCompaction(
                            id="applied",
                            replacement_call_ids=["block"],
                            timestamp=active.timestamp,
                            duration_seconds=1.5,
                            completed_summaries=1,
                        )
                    ]
                )
            )
            after = collector.append_token("After compaction.")
            view.restore_stream_snapshot(collector.snapshot_message())
            assert bar.is_deleted
            record, text = view.current_activity_container.default_slot.children[-2:]
            assert "compaction-marker" in record._classes
            assert any(
                item.text == "Elapsed 1s · 1 history block · 1 summary"
                for item in record.descendants()
                if isinstance(item, ui.label)
            )
            assert "After compaction." in text.content
            message = ChatMessage.model_validate_json(
                collector.build_message().model_dump_json()
            )
            assert [
                item.id
                for item in message.execution_timeline(
                    include_stream=True
                ).root_activity()
            ] == [
                tool.id,
                before.id,
                "applied",
                after.id,
            ]
            view.update_compaction_progress([active])
            bar = view.live_compaction_views[active.id]
            view.finalize_stream()
            assert not bar.visible
    finally:
        client.delete()


def test_untimed_compaction_uses_its_tool_anchor_inside_the_skill():
    tool = ToolEvent(tool_name="lookup", tool_call_id="anchor", timestamp=10)
    skill = SkillNode(path="/research", depth=1, start_time=5, tool_events=[tool])
    records = [
        RecordedCompaction(
            id="known",
            replacement_call_ids=["block"],
            after_tool_call_id="anchor",
            skill_path=skill.path,
        ),
        RecordedCompaction(
            id="unknown", replacement_call_ids=["prior"], timestamp=None
        ),
    ]
    message = ChatMessage(
        role="assistant",
        skills=[skill],
        model_context=ModelContextCheckpoint(compactions=records),
    )
    timeline = message.execution_timeline(include_stream=True)
    assert [item.id for item in timeline.root_activity()] == ["unknown", skill.id]
    assert [item.id for item in timeline.skill_activity(skill)] == [tool.id, "known"]
    assert isinstance(timeline.root_activity()[0], CompactionActivity)


@pytest.mark.parametrize(
    ("duration", "summaries", "expected"),
    [
        (None, None, "2 history blocks"),
        (0.25, 2, "Elapsed 0.25s · 2 history blocks · 2 summaries"),
    ],
)
def test_compaction_details_use_recorded_measurements(duration, summaries, expected):
    client = Client(page("/compaction-details"))
    try:
        with client, ui.column() as root:
            render_applied_compaction(
                RecordedCompaction(
                    id="operation",
                    replacement_call_ids=["one", "two"],
                    duration_seconds=duration,
                    completed_summaries=summaries,
                )
            )
        details = next(
            item
            for item in root.descendants()
            if "compaction-marker-details" in item._classes
        )
        assert details.text == expected
    finally:
        client.delete()
