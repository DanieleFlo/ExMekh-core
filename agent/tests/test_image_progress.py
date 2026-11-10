"""Invocation feedback and in-place image skeleton rendering."""

from types import SimpleNamespace

from exmekh_lib.interfaces.operation_progress import OperationProgress
from exmekh_lib.interfaces.tool_hook import ToolContext
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.models import ChatMessage, SkillNode, ToolEvent
from exmekh_agent.exmekh_bridge.tools.tool_feedback import ImageArguments, ToolFeedback
from exmekh_agent.ui.chat import image_progress
from exmekh_agent.ui.chat.chat_view import ChatViewComponent


def progress(seq=1, **extra):
    return OperationProgress.model_validate(
        {
            "type": "operation.progress",
            "operation_id": "image-job",
            "request_id": "request",
            "seq": seq,
            "elapsed_seconds": 12,
            "state": "running",
            "phase": "generation",
            "overall_fraction": 0.42,
            "overall_kind": "estimated",
            **extra,
        }
    )


def test_callback_is_scoped_to_tool_and_survives_snapshot_without_prompt_arguments(
    tmp_path,
):
    feedback = ToolFeedback()
    seen = []
    feedback.publish(
        ToolEvent(
            tool_name="generate-image", tool_call_id="one", event_type="tool_start"
        ),
        seen.append,
    )
    decision = ImageArguments(False, feedback).handle(
        ToolContext({}, tmp_path, (), "one")
    )
    assert "on_progress" not in decision.arguments
    decision.on_progress(progress())
    assert seen[-1].image_progress.seq == 1 and seen[-1].is_running
    decision.on_progress(progress())
    assert len([e for e in seen if e.image_progress]) == 1
    collector = TurnStreamCollector()
    collector.append_tool(seen[-1])
    saved = ChatMessage.model_validate_json(
        collector.snapshot_message().model_dump_json()
    )
    assert saved.tool_events[0].image_progress.overall_fraction == 0.42


def test_progress_updates_preserve_invocation_structure_and_results():
    root_tool = ToolEvent(tool_name="root-tool")
    event = ToolEvent(tool_name="generate-image", image_state="completed")
    skill = SkillNode(sub_skills=[SkillNode(path="/child", tool_events=[event])])
    message = ChatMessage(role="assistant", tool_events=[root_tool], skills=[skill])
    changed = message.model_copy(deep=True)
    root, image = changed.iter_tool_events()
    assert root == root_tool and image == event
    image.merge_update(
        event.model_copy(
            update={"image_state": "running", "image_progress": progress()}
        )
    )
    assert image.image_state == "completed"
    assert message.has_same_execution_details(changed)


def test_skeleton_only_shows_percentage_and_time_without_replacing_labels(monkeypatch):
    clock = 100.0
    monkeypatch.setattr(
        image_progress, "time", SimpleNamespace(monotonic=lambda: clock)
    )

    client = Client(page("/image-progress"))
    try:
        with client:
            view = ChatViewComponent(lambda _: None)
            view.build()
            view.start_assistant_stream()
            event = ToolEvent(
                tool_name="generate-image",
                tool_call_id="one",
                event_type="tool_start",
                status="running",
                image_state="running",
                arguments={"width": 256, "height": 256, "n": 2},
            )
            view.add_tool_event(event)
            overlays = view.image_progress_views[event.invocation_key]
            labels = [item.label for item in overlays]
            view.add_tool_event(event.model_copy(update={"image_progress": progress()}))
            assert view.image_progress_views[event.invocation_key] == overlays
            assert overlays[0].label is labels[0]
            assert overlays[0].label.text == "42% · 12s"
            assert overlays[1].label.text == "—% · 12s"
            assert len(list(overlays[0].descendants())) == 1
            overlays[0].show(event)
            assert overlays[0].label.text == "42% · 12s"
            next_item = progress(
                seq=2, item_count=2, item_index=1, overall_fraction=0.75
            )
            view.add_tool_event(event.model_copy(update={"image_progress": next_item}))
            assert view.image_progress_views[event.invocation_key] == overlays
            assert overlays[0].label.text == "99% · 12s"
            assert overlays[1].label.text == "50% · 12s"
            overlays[1].show(event.model_copy(update={"image_progress": progress()}))
            assert overlays[1].label.text == "50% · 12s"
            terminal = progress(
                seq=3,
                type="operation.completed",
                state="completed",
                item_count=2,
                overall_fraction=1,
                overall_kind="work",
                elapsed_seconds=13,
            )
            overlays[1].show(event.model_copy(update={"image_progress": terminal}))
            clock += 10
            overlays[1].refresh()
            assert overlays[1].label.text == "99% · 13s"

            snapshot = event.model_copy(update={"image_progress": progress()})
            failed = image_progress.ImageProgressView(snapshot, 0)
            clock += 3
            failed.show(snapshot.model_copy(update={"image_state": "error"}))
            clock += 10
            failed.refresh()
            assert failed.label.text == "42% · 15s"
    finally:
        client.delete()
