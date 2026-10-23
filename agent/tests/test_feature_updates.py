import asyncio
from dataclasses import replace
from threading import Event, Thread
from unittest.mock import MagicMock, patch

import pytest
from exmekh_lib.core.agents.node import Node
from exmekh_lib.interfaces.tool_hook import ToolContext
from exmekh_lib.utils.tool_errors import ToolInputError
from nicegui import ui
from nicegui.client import Client
from nicegui.page import page
from PIL import Image

from exmekh_agent.config import AppConfig
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.sub_agents import (
    SubAgentEvent,
    SubAgentEventRouter,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    SkillNode,
    ToolEvent,
    record_tool_event,
)
from exmekh_agent.exmekh_bridge.tools.tool_approval import CommandApprovalHook
from exmekh_agent.exmekh_bridge.tools.tool_feedback import (
    ImageArguments,
    ImagePreviews,
    ToolFeedback,
)
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.shared.disclosure import Disclosure
from exmekh_agent.ui.shared.image_preview import show_image_preview


def test_feedback_tracks_each_invocation_in_root_and_child_conversations():
    feedback = ToolFeedback()
    root, child = [], []
    router = SubAgentEventRouter(child.append, feedback=feedback)
    feedback.publish(
        ToolEvent(tool_name="bash", tool_call_id="one", event_type="tool_start"),
        root.append,
    )
    router.publish(
        SubAgentEvent(
            action="tool",
            id="child",
            path="main/child",
            persistent_history=True,
            tool_event=ToolEvent(
                tool_name="bash", tool_call_id="two", event_type="tool_start"
            ),
        ),
    )
    feedback.approval("one", "pending")
    feedback.approval("two", "denied")
    assert root[-1].approval_state == "pending"
    assert child[-1].tool_event.approval_state == "denied"
    router.publish(
        SubAgentEvent(
            action="token", id="child", path="main/child", persistent_history=True
        ),
    )
    feedback.approval("unobserved", "approved")
    feedback.publish(
        ToolEvent(tool_name="bash", tool_call_id="unobserved"), root.append
    )
    assert root[-1].approval_state == "approved"
    feedback.publish(ToolEvent(tool_name="other"), root.append)
    assert root[-1].approval_state is None


@pytest.mark.parametrize("include", [False, True])
def test_image_policy_changes_model_delivery_and_preserves_original_ui_pixels(
    tmp_path, include
):
    feedback = ToolFeedback()
    events = []
    context = ToolContext(
        {"include_image_in_result": not include}, tmp_path, (), "image"
    )
    assert (
        ImageArguments(include, feedback)
        .handle(context)
        .arguments["include_image_in_result"]
        is include
    )
    feedback.publish(
        ToolEvent(
            tool_name="generate-image", tool_call_id="image", event_type="tool_start"
        ),
        events.append,
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (64, 32), "red").save(path)
    result = {"images": [{"path": str(path)}]}
    assert ImagePreviews(feedback).handle(context, result) is result
    assert events[-1].image_urls[0].startswith("data:image/png;base64,")
    stored = ChatMessage(role="assistant", tool_events=[events[-1]])
    restored = ChatMessage.model_validate_json(stored.model_dump_json())
    assert restored.images_for_tool(restored.tool_events[0]) == events[-1].image_urls
    outside = replace(context, work_root=tmp_path / "other")
    assert ImagePreviews(feedback).handle(outside, result) is result
    assert ImagePreviews(feedback).handle(context, "error") == "error"
    assert ImagePreviews(feedback).handle(context, {}) == {}


def test_image_and_approval_updates_merge_without_completing_the_tool():
    events = []
    started = ToolEvent(
        tool_name="generate-image", tool_call_id="image", event_type="tool_start"
    )
    record_tool_event(events, started)
    record_tool_event(events, started.model_copy(update={"approval_state": "pending"}))
    record_tool_event(
        events,
        started.model_copy(
            update={"image_urls": ["image"], "image_state": "completed"}
        ),
    )
    assert events[0].is_running
    record_tool_event(
        events,
        ToolEvent(
            tool_name="generate-image", tool_call_id="image", event_type="tool_end"
        ),
    )
    assert not events[0].is_running
    assert events[0].image_urls == ["image"]
    assert events[0].approval_state == "pending"


@pytest.mark.parametrize("approved", [False, True])
def test_approval_publishes_pending_and_decision_for_the_exact_call(tmp_path, approved):
    states = []
    hook = CommandApprovalHook(
        mode="manual",
        request_manual_approval=lambda _: approved,
        on_state=lambda *state: states.append(state),
    )
    hook.handle(ToolContext({"command": "echo hello"}, tmp_path, (), "command-17"))
    assert states == [
        ("command-17", "pending"),
        ("command-17", "approved" if approved else "denied"),
    ]


def test_generated_images_global_preference_persists_and_restores(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("INCLUDE_GENERATED_IMAGES_IN_CONTEXT", raising=False)
    configuration = AppConfig()
    assert configuration.include_generated_images_in_context
    configuration.include_generated_images_in_context = False
    env = tmp_path / ".env"
    assert configuration.save_to_env(env)
    assert "INCLUDE_GENERATED_IMAGES_IN_CONTEXT=false" in env.read_text()
    monkeypatch.setenv("INCLUDE_GENERATED_IMAGES_IN_CONTEXT", "false")
    assert not AppConfig().include_generated_images_in_context


def test_unplaced_legacy_images_are_preserved_and_tool_images_stay_in_order():
    tools = [
        ToolEvent(
            tool_name="generate-image", tool_call_id="first", image_urls=["first"]
        ),
        ToolEvent(tool_name="generate-image", tool_call_id="last", image_urls=["last"]),
    ]
    message = ChatMessage(
        role="assistant", tool_events=tools, image_urls=["first", "orphan", "last"]
    )
    assert message.image_gallery() == ["first", "last", "orphan"]
    assert message.unplaced_images() == ["orphan"]
    legacy = ToolEvent(tool_name="legacy")
    assert message.images_for_tool(legacy) == []
    assert ChatMessage(role="assistant", image_urls=["old"]).unplaced_images() == [
        "old"
    ]


def test_images_remain_visible_when_their_skill_is_collapsed():
    client = Client(page("/skill-images"))
    disclosures = []

    def record(*args, **kwargs):
        instance = Disclosure(*args, **kwargs)
        disclosures.append(instance)
        return instance

    try:
        with (
            client,
            patch("exmekh_agent.ui.chat.chat_view.Disclosure", side_effect=record),
        ):
            tool = ToolEvent(
                tool_name="generate-image",
                tool_call_id="image",
                skill_path="/skill",
                depth=1,
                image_urls=["preview"],
            )
            message = ChatMessage(
                role="assistant",
                tool_events=[tool],
                skills=[
                    SkillNode(
                        name="skill",
                        path="/skill",
                        depth=1,
                        status="completed",
                        tool_events=[tool],
                    )
                ],
            )
            view = ChatViewComponent(lambda _: None)
            view._render_turn_activity(message)
            gallery = next(
                element
                for element in client.elements.values()
                if "collapsed-skill-images" in element._classes
            )
            assert gallery.visible
            skill = next(
                instance
                for instance in disclosures
                if "skill-compact-header" in instance._header._classes
            )
            skill.toggle()
            assert not gallery.visible
            skill.toggle()
            assert gallery.visible
    finally:
        client.delete()


@pytest.mark.parametrize("with_cancellation", [False, True])
def test_background_images_keep_the_turn_alive_until_pixels_arrive(with_cancellation):
    feedback = ToolFeedback()
    feedback.update("image", image_state="running")
    finished = Event()
    cancellation = Event() if with_cancellation else None

    def wait():
        feedback.wait_for_images(cancellation)
        finished.set()

    thread = Thread(target=wait)
    thread.start()
    try:
        assert not finished.wait(0.02)
        feedback.update("image", image_state="completed")
        assert finished.wait(2)
    finally:
        feedback.update("image", image_state="error")
        thread.join(2)


def test_cancelled_or_failed_image_generation_does_not_leave_a_loading_box():
    feedback = ToolFeedback()
    cancellation = Event()
    cancellation.set()
    feedback.update("image", image_state="running")
    with pytest.raises(asyncio.CancelledError):
        feedback.wait_for_images(cancellation)
    events = []
    feedback.publish(
        ToolEvent(tool_name="generate-image", tool_call_id="image", status="error"),
        events.append,
    )
    assert events[-1].image_state == "error"
    feedback.wait_for_images(None)


def test_removed_generated_original_does_not_fail_a_successful_tool(tmp_path, caplog):
    feedback = ToolFeedback()
    context = ToolContext({}, tmp_path, (), "image")
    result = {"images": [{"path": "removed.png"}]}
    assert ImagePreviews(feedback).handle(context, result) is result
    assert "Unable to retain a generated image preview" in caplog.text


@pytest.mark.parametrize(
    ("copied", "deleted"), [(True, False), (False, False), (False, True)]
)
async def test_gallery_controls_copy_failure_and_download_types(copied, deleted):
    client = Client(page("/gallery-controls"))
    deferred = []

    async def copy(*args, **kwargs):
        if deleted:
            client.delete()
        return copied

    try:
        with (
            client,
            patch("nicegui.ui.download") as download,
            patch("nicegui.ui.run_javascript", side_effect=copy),
            patch("nicegui.ui.notify") as notify,
            patch(
                "nicegui.background_tasks.create_or_defer",
                side_effect=lambda awaitable, **kwargs: deferred.append(awaitable),
            ),
        ):
            show_image_preview(
                "data:image/png;base64,YWJj",
                "Image",
                gallery=["https://example.test/image.png"],
            )
            buttons = {
                element._props.get("aria-label"): element
                for element in client.elements.values()
                if isinstance(element, ui.button)
            }

            def click(name):
                return next(
                    listener.handler
                    for listener in buttons[name]._event_listeners.values()
                    if listener.type == "click"
                )(None)

            click("Download image")
            download.content.assert_called_once_with(b"abc", "image-2.png", "image/png")
            click("Previous image")
            click("Download image")
            download.assert_called_once_with("https://example.test/image.png")
            click("Next image")
            dialog = next(
                element
                for element in client.elements.values()
                if isinstance(element, ui.dialog)
            )
            for listener in dialog._event_listeners.values():
                if listener.type == "keydown":
                    listener.handler()
            click("Copy image")
            await deferred[-1]
            assert notify.called is (not copied and not deleted)
    finally:
        if not client.is_deleted:
            client.delete()


@pytest.mark.parametrize("status", ["failed", "cancelled", "completed"])
def test_background_completion_uses_the_owning_nodes_public_status(status):
    feedback = ToolFeedback()
    feedback.update("image", image_state="running")
    unknown, owner = MagicMock(spec=Node), MagicMock(spec=Node)
    unknown.control_tool_execution.side_effect = ToolInputError(
        "missing", code="unknown_tool_call_id"
    )

    def control(*args, **kwargs):
        if status == "completed":
            feedback.update("image", image_state="completed")
        return {"status": status}

    owner.control_tool_execution.side_effect = control
    feedback.wait_for_images(None, [unknown, owner])
    owner.control_tool_execution.assert_called_once_with("image", wait_seconds=0.1)
    events = []
    feedback.publish(
        ToolEvent(tool_name="generate-image", tool_call_id="image"), events.append
    )
    assert events[-1].image_state == ("completed" if status == "completed" else "error")


def test_unexpected_background_status_errors_are_reported():
    feedback = ToolFeedback()
    feedback.update("image", image_state="running")
    node = MagicMock(spec=Node)
    node.control_tool_execution.side_effect = ToolInputError(
        "invalid control", code="invalid_control"
    )
    with pytest.raises(ToolInputError, match="invalid control"):
        feedback.wait_for_images(None, [node])


def test_stopping_a_turn_finalizes_background_placeholders_and_pending_approvals():
    message = ChatMessage(
        role="assistant",
        tool_events=[
            ToolEvent(
                tool_name="generate-image", event_type="tool_end", image_state="running"
            ),
            ToolEvent(
                tool_name="bash", event_type="tool_start", approval_state="pending"
            ),
        ],
    )
    message.finalize_execution("error")
    assert message.tool_events[0].image_state == "error"
    assert message.tool_events[1].approval_state is None
    assert not message.tool_events[1].is_running


def test_originals_replace_model_previews_without_deduplicating_requested_images():
    collector = TurnStreamCollector()
    collector.append_tool(
        ToolEvent(
            tool_name="generate-image",
            tool_call_id="image",
            image_urls=["original", "original"],
        )
    )
    collector.set_raw_history(
        [
            {
                "role": "tool",
                "tool_call_id": "image",
                "attachments": [{"type": "image_url", "image_url": {"url": "preview"}}],
            }
        ],
        [],
    )
    assert collector.build_message().image_urls == ["original", "original"]


def test_image_launch_is_pending_before_the_background_worker_runs():
    feedback = ToolFeedback()
    observed = []
    launch = ToolEvent(
        tool_name="generate-image", tool_call_id="slow-worker", event_type="tool_start"
    )
    feedback.publish(launch, observed.append)
    assert observed[-1].image_state == "running"
    assert launch.image_state is None
    finished = Event()
    thread = Thread(target=lambda: (feedback.wait_for_images(None), finished.set()))
    thread.start()
    try:
        assert not finished.wait(0.02)
        feedback.update("slow-worker", image_state="completed", image_urls=["original"])
        assert finished.wait(2)
        assert observed[-1].image_urls == ["original"]
    finally:
        feedback.update("slow-worker", image_state="error")
        thread.join(2)


def test_feedback_observers_can_wait_for_an_update_from_another_thread():
    feedback = ToolFeedback()
    received = []
    updated = Event()

    def observe(event):
        received.append(event)
        if event.approval_state is None:
            worker = Thread(
                target=lambda: (feedback.approval("call", "approved"), updated.set())
            )
            worker.start()
            assert updated.wait(2)
            worker.join(2)

    feedback.publish(ToolEvent(tool_name="bash", tool_call_id="call"), observe)
    assert [event.approval_state for event in received] == [None, "approved"]


def test_observer_failure_does_not_block_later_feedback():
    feedback = ToolFeedback()

    def fail(event):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        feedback.publish(ToolEvent(tool_name="bash", tool_call_id="one"), fail)
    events = []
    feedback.publish(ToolEvent(tool_name="bash", tool_call_id="two"), events.append)
    assert len(events) == 1


def test_failed_approval_does_not_remain_pending():
    feedback = ToolFeedback()
    feedback.approval("call", "pending")
    events = []
    feedback.publish(
        ToolEvent(tool_name="bash", tool_call_id="call", status="error"), events.append
    )
    assert events[-1].approval_state is None


def test_unobserved_child_launches_still_retain_image_feedback():
    feedback = ToolFeedback()
    SubAgentEventRouter(None, feedback=feedback).publish(
        SubAgentEvent(
            action="tool",
            id="child",
            path="child",
            persistent_history=True,
            tool_event=ToolEvent(
                tool_name="generate-image",
                tool_call_id="image",
                event_type="tool_start",
            ),
        )
    )
    feedback.update("image", image_state="completed")
    feedback.wait_for_images(None)


def test_invalid_preview_records_do_not_fail_a_successful_tool(tmp_path):
    result = {"images": [None, {"path": None}, {}]}
    assert (
        ImagePreviews(ToolFeedback()).handle(ToolContext({}, tmp_path, ()), result)
        is result
    )


def test_terminal_failure_clears_feedback_when_merging_a_live_invocation():
    message = ChatMessage(role="assistant")
    message.add_tool_event(
        ToolEvent(
            tool_name="generate-image",
            tool_call_id="image",
            event_type="tool_start",
            image_state="running",
            approval_state="pending",
        )
    )
    message.add_tool_event(
        ToolEvent(
            tool_name="generate-image",
            tool_call_id="image",
            event_type="tool_end",
            status="error",
        )
    )
    assert message.tool_events[0].image_state == "error"
    assert message.tool_events[0].approval_state is None


def test_feedback_preserves_existing_event_state_on_first_publication():
    events = []
    ToolFeedback().publish(
        ToolEvent(
            tool_name="generate-image",
            tool_call_id="image",
            image_state="completed",
            image_urls=["original"],
            approval_state="approved",
        ),
        events.append,
    )
    assert events[0].image_state == "completed"
    assert events[0].approval_state == "approved"
    assert events[0].image_urls == ["original"]


def test_image_wait_includes_delivery_to_the_collector():
    feedback = ToolFeedback()
    delivering, release, finished = Event(), Event(), Event()

    def observe(event):
        if event.image_state == "completed":
            delivering.set()
            assert release.wait(2)

    feedback.publish(
        ToolEvent(
            tool_name="generate-image", tool_call_id="image", event_type="tool_start"
        ),
        observe,
    )
    worker = Thread(target=lambda: feedback.update("image", image_state="completed"))
    waiter = Thread(target=lambda: (feedback.wait_for_images(None), finished.set()))
    worker.start()
    try:
        assert delivering.wait(2)
        waiter.start()
        assert not finished.wait(0.02)
        release.set()
        assert finished.wait(2)
    finally:
        release.set()
        worker.join(2)
        if waiter.ident is not None:
            waiter.join(2)
