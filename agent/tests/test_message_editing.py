from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.history.message_edit import prepare_message_edit
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    PreparedUserInput,
    create_staged_attachment_from_file,
    format_user_input_file_xml,
    prepare_user_input,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    QueuedMessage,
    SubAgentConversation,
    ToolEvent,
)
from exmekh_agent.state.active_turns import ActiveTurnRegistry, active_turns
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController


def test_rewrite_persists_only_the_retained_branch_and_child_history(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    first = ChatMessage(role="user", content="First", timestamp=1)
    second = ChatMessage(role="user", content="Old second", timestamp=4)
    session.messages = [
        first,
        ChatMessage(role="assistant", content="First reply", timestamp=2),
        second,
        ChatMessage(role="assistant", content="Obsolete reply", timestamp=5),
    ]
    prefix = [
        {"role": "user", "content": "Earlier child"},
        {"role": "assistant", "content": "Earlier result"},
    ]
    session.sub_agent_conversations = [
        SubAgentConversation(
            id="retained",
            path="/coding",
            messages=[
                ChatMessage(role="user", content="Earlier child", timestamp=2),
                ChatMessage(
                    role="assistant",
                    content="Earlier result",
                    timestamp=3,
                    raw_history=prefix,
                ),
                ChatMessage(role="user", content="Obsolete child", timestamp=5),
                ChatMessage(role="assistant", content="Obsolete result", timestamp=6),
            ],
            raw_history=prefix + [{"role": "user", "content": "Obsolete child"}],
        ),
        SubAgentConversation(
            id="removed",
            path="/coding",
            messages=[ChatMessage(role="user", content="New child", timestamp=5)],
        ),
    ]
    session.add_queued_message("Obsolete queue")
    session.settings.record_command_approval(
        "obsolete command", [{"role": "user", "content": "obsolete"}]
    )
    replacement = second.model_copy(update={"content": "Revised second"})
    assert manager.edit_message(session, second.id, replacement)
    assert [item.content for item in session.messages] == [
        "First",
        "First reply",
        "Revised second",
    ]
    assert len(session.sub_agent_conversations) == 1
    assert session.sub_agent_conversations[0].raw_history == prefix
    assert session.sub_agent_conversations[0].status == "completed"
    assert not session.queued_messages
    assert not session.settings.approved_commands_history
    assert not session.settings.tool_approval_raw_history
    restored = SessionManager(tmp_path / "chats").get_session(session.id)
    assert restored.model_dump() == session.model_dump()


def test_invalid_edits_and_failed_save_do_not_change_the_session(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    user = ChatMessage(role="user", content="Original")
    assistant = ChatMessage(role="assistant", content="Reply")
    session.messages = [user, assistant]
    replacement = user.model_copy(update={"content": "Changed"})
    assert not manager.edit_message(ChatSession(), user.id, replacement)
    assert not manager.edit_message(session, "missing", replacement)
    assert not manager.edit_message(session, assistant.id, replacement)
    before = session.model_dump()
    with patch.object(manager, "save_session", return_value=False):
        assert not manager.edit_message(session, user.id, replacement)
    assert session.model_dump() == before
    child = SubAgentConversation(
        path="/coding",
        messages=[ChatMessage(role="assistant", timestamp=0, error="old failure")],
    )
    session.sub_agent_conversations = [child]
    assert manager.edit_message(session, user.id, replacement)
    assert session.sub_agent_conversations[0].status == "error"


def test_editing_steering_restarts_its_response_and_retains_earlier_input(
    tmp_path: Path,
):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    user = ChatMessage(role="user", content="Task", timestamp=1)
    earlier = QueuedMessage(text="Earlier guidance")
    edited = QueuedMessage(text="Old guidance")
    later = QueuedMessage(text="Obsolete guidance")
    response = ChatMessage(
        role="assistant",
        content="Obsolete response",
        timestamp=2,
        steering_messages=[earlier, edited, later],
    )
    session.messages = [user, response]
    replacement = edited.to_chat_message().model_copy(
        update={"content": "Changed guidance"}
    )
    assert manager.edit_message(session, edited.id, replacement)
    assert [item.content for item in session.messages] == [
        "Task",
        "Earlier guidance",
        "Changed guidance",
    ]


def test_edit_preserves_explicit_uploads_and_saved_images(tmp_path: Path):
    storage = TempStorage(tmp_path)
    _, path = storage.write_file("picture.png", b"saved")
    message = ChatMessage(
        role="user",
        content="Old caption\n\n" + format_user_input_file_xml(path),
        display_content="Old caption",
        image_urls=["data:image/png;base64,saved"],
        image_paths=[path],
    )
    prepared = prepare_message_edit(
        message, "New caption", storage=storage, use_file_references=True
    )
    assert prepared.prompt.startswith("New caption") and path in prepared.prompt
    assert prepared.display_content == "New caption"
    assert prepared.image_urls == message.image_urls
    assert prepared.image_paths == [path]
    message.content += (
        "\n\nPDF file.pdf: pages 1–2 of 2 are attached as images in page order."
    )
    assert (
        "PDF file.pdf"
        in prepare_message_edit(
            message, "PDF caption", storage=storage, use_file_references=False
        ).prompt
    )
    # Legacy file references without display text also remain attached.
    message.display_content = None
    assert (
        path
        in prepare_message_edit(
            message, "Legacy", storage=storage, use_file_references=True
        ).prompt
    )


def test_edit_recreates_automatically_staged_text_from_the_new_content(tmp_path: Path):
    storage = TempStorage(tmp_path)
    old_text = "Original large paragraph " * 400
    new_text = "Revised large paragraph " * 400
    original = prepare_user_input(old_text, storage=storage)
    message = ChatMessage(
        role="user", content=original.prompt, display_content=original.display_content
    )
    prepared = prepare_message_edit(
        message, new_text, storage=storage, use_file_references=True
    )
    assert prepared.display_content == new_text
    assert original.prompt not in prepared.prompt
    assert prepared.prompt != original.prompt
    assert (
        prepare_message_edit(
            message, "Short replacement", storage=storage, use_file_references=True
        ).prompt
        == "Short replacement"
    )


async def test_stop_and_wait_covers_absent_finished_and_not_yet_started_turns():
    registry = ActiveTurnRegistry()
    await registry.stop_and_wait("missing")
    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    registry.start("done", completed, TurnStreamCollector(), started_at=0)
    await registry.stop_and_wait("done")
    assert not registry.is_running("done")
    pending = asyncio.create_task(asyncio.sleep(100))
    registry.start("pending", pending, TurnStreamCollector(), started_at=0)
    await registry.stop_and_wait("pending")
    assert pending.cancelled() and not registry.is_running("pending")


@pytest.mark.parametrize("running", [False, True])
async def test_controller_edits_and_regenerates_without_stale_events(running: bool):
    client = Client(page("/message-edit-test"))
    session = session_manager.get_current_session()
    first = ChatMessage(role="user", content="First")
    session.messages = [first, ChatMessage(role="assistant", content="Earlier reply")]
    calls = []
    started = asyncio.Event()

    async def run(**kwargs):
        calls.append(kwargs)
        if kwargs["user_prompt"] == "Old prompt" and running:
            kwargs["on_token"]("Partial obsolete response")
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                kwargs["on_token"]("LATE obsolete token")
                kwargs["on_command_approved"]("LATE command", [])
                raise
        kwargs["on_token"]("Reply: " + kwargs["user_prompt"])
        return "Reply: " + kwargs["user_prompt"]

    with client:
        controller = ChatPageController(client)
        controller.chat_view = ChatViewComponent(
            lambda _: None, controller.edit_message
        )
        controller.chat_view.build()
        controller.input_bar = InputBarComponent(lambda _: None, controller.stop)
        controller.input_bar.build()
    try:
        with (
            patch(
                "exmekh_agent.ui.chat.page_controller.run_agent_turn", side_effect=run
            ),
            patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
        ):
            controller._send_prepared(
                session, PreparedUserInput("Old prompt", None, "Old prompt")
            )
            if running:
                await started.wait()
            else:
                await controller._active_task
            old = session.messages[2]
            session.add_queued_message("Obsolete queue")
            assert await controller.edit_message(session.id, old.id, "Revised prompt")
            await controller._active_task
            assert [item.content for item in session.messages] == [
                "First",
                "Earlier reply",
                "Revised prompt",
                "Reply: Revised prompt",
            ]
            assert [item.content for item in calls[-1]["history"]] == [
                "First",
                "Earlier reply",
            ]
            assert session.messages[2].id == old.id
            assert not session.queued_messages
            assert not session.settings.approved_commands_history
            assert not active_turns.is_running(session.id)
            assert not controller.input_bar.is_busy
            labels = str(
                [
                    getattr(item, "text", "")
                    for item in controller.chat_view.chat_container.descendants()
                ]
            )
            assert "LATE obsolete" not in labels
            # Editing a preceding message discards both later turns.
            assert await controller.edit_message(session.id, first.id, "New first")
            await controller._active_task
            assert not calls[-1]["history"]
            assert len(session.messages) == 2
    finally:
        client.delete()


async def test_controller_rejects_missing_empty_or_unsavable_edits():
    client = Client(page("/message-edit-errors"))
    with client:
        controller = ChatPageController(client)
    session = session_manager.get_current_session()
    user = ChatMessage(role="user", content="Original")
    session.messages = [user]
    try:
        with patch("exmekh_agent.ui.chat.page_controller.ui.notify"):
            assert not await controller.edit_message("missing", user.id, "Changed")
            assert not await controller.edit_message(session.id, "missing", "Changed")
            assert not await controller.edit_message(session.id, user.id, " ")
            with patch(
                "exmekh_agent.ui.chat.page_controller.prepare_message_edit",
                side_effect=ValueError("Invalid file"),
            ):
                assert not await controller.edit_message(session.id, user.id, "Changed")
            with patch.object(session_manager, "edit_message", return_value=False):
                assert not await controller.edit_message(session.id, user.id, "Changed")
            assert session.messages == [user]
            with patch.object(
                active_turns,
                "stop_and_wait",
                side_effect=lambda _: session_manager.delete_session(session.id),
            ):
                assert not await controller.edit_message(session.id, user.id, "Changed")
    finally:
        client.delete()


async def test_simultaneous_edits_serialize_their_turn_replacement():
    client = Client(page("/simultaneous-edits"))
    with client:
        controller = ChatPageController(client)
    session = session_manager.get_current_session()
    user = ChatMessage(role="user", content="Original")
    session.messages = [user]

    async def run(**kwargs):
        if kwargs["user_prompt"] == "First edit":
            await asyncio.Future()
        kwargs["on_token"]("Reply: " + kwargs["user_prompt"])
        return "Reply: " + kwargs["user_prompt"]

    try:
        with patch(
            "exmekh_agent.ui.chat.page_controller.run_agent_turn", side_effect=run
        ):
            assert await asyncio.gather(
                controller.edit_message(session.id, user.id, "First edit"),
                controller.edit_message(session.id, user.id, "Second edit"),
            ) == [True, True]
            task = controller._active_task
            if task is not None:
                await task
            assert [message.content for message in session.messages] == [
                "Second edit",
                "Reply: Second edit",
            ]
            assert not active_turns.is_running(session.id)
    finally:
        client.delete()


def test_steering_rewrite_removes_child_calls_from_its_owning_turn(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    guidance = QueuedMessage(text="Old guidance")
    session.messages = [
        ChatMessage(role="assistant", content="Earlier response", timestamp=0),
        ChatMessage(role="user", content="Task", timestamp=1),
        ChatMessage(role="assistant", timestamp=3, steering_messages=[guidance]),
    ]
    session.sub_agent_conversations = [
        SubAgentConversation(
            path="/coding",
            messages=[ChatMessage(role="user", content="Discarded child", timestamp=2)],
        )
    ]
    assert manager.edit_message(session, guidance.id, guidance.to_chat_message())
    assert not session.sub_agent_conversations
    # Legacy response without an initiating user still has a safe cutoff.
    session.messages = [ChatMessage(role="assistant", steering_messages=[guidance])]
    assert manager.edit_message(session, guidance.id, guidance.to_chat_message())


def test_explicit_attachment_metadata_overrides_legacy_text_detection(tmp_path: Path):
    storage = TempStorage(tmp_path)
    message = ChatMessage(
        role="user",
        content=format_user_input_file_xml("tmp/automatic.md"),
        attachment_paths=[],
    )
    assert (
        prepare_message_edit(
            message, "Changed", storage=storage, use_file_references=True
        ).prompt
        == "Changed"
    )
    message.attachment_paths = ["tmp/upload.md"]
    edited = prepare_message_edit(
        message, "Changed", storage=storage, use_file_references=True
    )
    assert "tmp/upload.md" in edited.prompt
    assert "tmp/automatic.md" not in edited.prompt
    assert edited.attachment_paths == ["tmp/upload.md"]


@pytest.mark.parametrize("references", [False, True])
def test_edited_text_upload_uses_the_same_input_contract(
    tmp_path: Path, references: bool
):
    storage = TempStorage(tmp_path)
    upload = create_staged_attachment_from_file(
        "code.py", b"print('retained')", storage=storage
    )
    original = prepare_user_input(
        "Old instruction",
        storage=storage,
        attachments=[upload],
        use_file_references=references,
    ).to_chat_message()
    assert original.visible_text == "Old instruction"
    edited = prepare_message_edit(
        original, "New instruction", storage=storage, use_file_references=references
    )
    assert edited.display_content == "New instruction"
    assert edited.attachment_paths == original.attachment_paths
    assert "Old instruction" not in edited.prompt
    if references:
        assert upload.relative_path in edited.prompt
    else:
        assert "print('retained')" in edited.prompt
        assert "<user_input_file" not in edited.prompt
    assert upload.file_path.read_bytes() == b"print('retained')"


def test_revision_and_history_lookups_expose_independent_domain_data(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    guidance = QueuedMessage(text="Guidance", attachment_paths=["tmp/upload.txt"])
    user = ChatMessage(role="user", content="Task", image_urls=["saved"])
    session.messages = [
        user,
        ChatMessage(role="assistant", steering_messages=[guidance]),
    ]
    found = manager.find_user_message(session, user.id)
    found.image_urls.clear()
    assert user.image_urls == ["saved"]
    found_guidance = manager.find_user_message(session, guidance.id)
    found_guidance.attachment_paths.clear()
    assert guidance.attachment_paths == ["tmp/upload.txt"]
    assert manager.find_user_message(ChatSession(), user.id) is None
    assert manager.find_user_message(session, "missing") is None
    assert session.revised_user_message(user.id, ChatMessage(role="assistant")) is None
    with pytest.raises(ValueError, match="another conversation"):
        session.apply_revision(ChatSession())
    revised = session.revised_user_message(user.id, user)
    revised.messages[0].image_urls.clear()
    assert user.image_urls == ["saved"]


async def test_unsaved_steering_can_be_found_through_the_session_manager(
    tmp_path: Path,
):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    collector = TurnStreamCollector()
    guidance = QueuedMessage(text="Live guidance")
    collector.steering.bind(lambda _: None, history_start=0)
    assert collector.steering.send(guidance)
    task = asyncio.create_task(asyncio.sleep(100))
    active_turns.start(session.id, task, collector, started_at=0)
    try:
        assert (
            manager.find_user_message(session, guidance.id).content == "Live guidance"
        )
        assert manager.find_user_message(session, "missing") is None
        with patch.object(active_turns, "snapshot", return_value=None):
            assert manager.find_user_message(session, guidance.id) is None
    finally:
        await active_turns.stop_and_wait(session.id)


def test_failed_subagent_status_and_replay_are_retained_by_revision():
    child = SubAgentConversation(path="/worker")
    child.start_call("Earlier task")
    child.fail_call()
    failed = child.message_history()[-1]
    cutoff = failed.timestamp + 1
    child.start_call("Later task")
    for message in child.messages[-2:]:
        message.timestamp = cutoff + 1
    assert child.discard_calls_from(cutoff)
    assert child.status == "error"
    history = child.replay_history()
    history[0].content = "Changed by caller"
    assert child.message_history()[0].content == "Earlier task"
    raw = [{"role": "user", "content": "Task", "metadata": {"source": "original"}}]
    fresh = SubAgentConversation(path="/worker")
    fresh.start_call("Task")
    fresh.complete_call("Reply", raw)
    raw[0]["metadata"]["source"] = "Changed by publisher"
    fresh.replay_history()[0]["metadata"]["source"] = "Changed by caller"
    assert fresh.raw_history[0]["metadata"]["source"] == "original"
    assert (
        fresh.message_history()[-1].raw_history[0]["metadata"]["source"] == "original"
    )


def test_failed_message_save_keeps_memory_and_disk_consistent(tmp_path: Path):
    manager = SessionManager(tmp_path / "chats")
    session = manager.get_current_session()
    before = session.model_dump()
    with patch.object(manager, "save_session", return_value=False):
        assert not manager.add_message(
            session, ChatMessage(role="user", content="Unsaved")
        )
    assert session.model_dump() == before
    assert not SessionManager(tmp_path / "chats").get_session(session.id).messages


async def test_stop_and_send_are_scoped_to_the_selected_session():
    client = Client(page("/turn-isolation"))
    session_a = session_manager.create_session("A")
    session_b = session_manager.create_session("B")
    task_a = asyncio.create_task(asyncio.sleep(100))
    active_turns.start(session_a.id, task_a, TurnStreamCollector(), started_at=0)
    with client:
        controller = ChatPageController(client)
        controller._active_task = task_a
    try:
        with patch("exmekh_agent.ui.chat.page_controller.ui.notify"):
            controller.stop()
        await asyncio.sleep(0)
        assert not task_a.cancelling()
        with patch.object(controller, "_start_turn") as start:
            controller._send_prepared(
                session_b, PreparedUserInput("B input", None, "B input")
            )
        start.assert_called_once()
        assert session_b.message_history()[-1].content == "B input"
        assert not session_b.queued_messages
        with (
            patch.object(session_manager, "add_message", return_value=False),
            patch.object(controller, "_start_turn") as start,
            patch("exmekh_agent.ui.chat.page_controller.ui.notify"),
        ):
            controller._send_prepared(
                session_b, PreparedUserInput("Unsaved", None, "Unsaved")
            )
        start.assert_not_called()
    finally:
        await active_turns.stop_and_wait(session_a.id)
        client.delete()


@pytest.mark.parametrize("status", ["running", "completed", "error"])
def test_tool_card_uses_the_model_execution_status(status: str):
    client = Client(page("/tool-status"))
    with client:
        view = ChatViewComponent(lambda _: None)
        view.build()
        from_event = ToolEvent(tool_name="bash", status=status)
        with view.chat_container:
            view._render_tool_event(from_event)
    try:
        icons = [
            element.name
            for element in view.chat_container.descendants()
            if "tool-event-status" in element._classes
        ]
        assert (
            icons
            == (
                {"running": [], "completed": ["check_circle"], "error": ["error"]}[
                    status
                ]
            )
        )
    finally:
        client.delete()


async def test_stale_callbacks_cannot_replace_another_chats_pending_approval():
    client = Client(page("/approval-isolation"))
    session_a = session_manager.create_session("A")
    session_b = session_manager.create_session("B")
    with client:
        controller = ChatPageController(client)
        controller.input_bar = InputBarComponent(lambda _: None, controller.stop)
        controller.input_bar.build()
        pending: Future[bool] = Future()
        controller._show_manual_approval(
            "echo current", session_b, pending, lambda: True
        )
        obsolete: Future[bool] = Future()
        controller._show_manual_approval(
            "echo stale", session_a, obsolete, lambda: False
        )
        with pytest.raises(asyncio.CancelledError):
            obsolete.result()
        controller._cancel_pending_approval(
            session_id=session_a.id, interrupt_turn=True
        )
        assert not pending.done()
        assert controller.input_bar.approval_container.visible
        controller._finish_manual_approval(pending, True)
        assert pending.result() is True
    client.delete()


async def test_cancelled_context_callback_cannot_overwrite_regenerated_context():
    client = Client(page("/context-isolation"))
    with client:
        controller = ChatPageController(client)
    accepts_events = True
    with patch.object(controller, "_update_live_context_in_client") as update:
        controller._schedule_live_context_update(
            asyncio.get_running_loop(),
            session_manager.get_current_session(),
            TurnStreamCollector(),
            force=True,
            accepts_events=lambda: accepts_events,
        )
        accepts_events = False
        await asyncio.sleep(0)
        update.assert_not_called()
    client.delete()


@pytest.mark.parametrize("legacy", [False, True])
def test_caption_edit_preserves_repeated_pdf_pages_without_rereading_files(
    tmp_path: Path, legacy: bool
):
    pdf_path = "tmp/removed.pdf"
    message = ChatMessage(
        role="user",
        content="Old caption\n\n" + format_user_input_file_xml(pdf_path),
        display_content="Old caption",
        image_urls=["data:image/png;base64,repeated"] * 2,
        image_paths=[] if legacy else [pdf_path],
        attachment_paths=None if legacy else [pdf_path],
    )
    edited = prepare_message_edit(
        message, "New caption", storage=TempStorage(tmp_path), use_file_references=False
    )
    assert edited.image_urls == message.image_urls
    assert edited.image_paths == [pdf_path]
    assert pdf_path in edited.prompt


@pytest.mark.parametrize("file_references", [False, True])
def test_edit_replaces_attachment_selection_and_rejects_foreign_paths(
    tmp_path: Path, file_references: bool
):
    storage = TempStorage(tmp_path)
    first = create_staged_attachment_from_file(
        "keep.txt", b"Retained text", storage=storage
    )
    removed = create_staged_attachment_from_file(
        "remove.txt", b"Removed text", storage=storage
    )
    added = create_staged_attachment_from_file("new.txt", b"New text", storage=storage)
    message = ChatMessage(
        role="user",
        content="Original",
        attachment_paths=[first.relative_path, removed.relative_path],
    )
    result = prepare_message_edit(
        message,
        "Revised",
        storage=storage,
        use_file_references=file_references,
        attachment_paths=[first.relative_path],
        attachments=[added],
    )
    assert result.attachment_paths == [first.relative_path, added.relative_path]
    assert (
        removed.relative_path not in result.prompt
        and "Removed text" not in result.prompt
    )
    assert ("Retained text" in result.prompt) == (not file_references)
    assert ("New text" in result.prompt) == (not file_references)
    assert storage.resolve_attachment(removed.relative_path).is_file()
    empty = prepare_message_edit(
        message,
        "Text only",
        storage=storage,
        use_file_references=file_references,
        attachment_paths=[],
    )
    assert empty.prompt == "Text only" and empty.attachment_paths == []
    with pytest.raises(ValueError, match="own attachments"):
        prepare_message_edit(
            message,
            "Invalid",
            storage=storage,
            use_file_references=file_references,
            attachment_paths=[added.relative_path],
        )


def test_adding_uploads_without_an_explicit_selection_retains_existing_files(
    tmp_path: Path,
):
    storage = TempStorage(tmp_path)
    retained = create_staged_attachment_from_file(
        "old.txt", b"Existing text", storage=storage
    )
    added = create_staged_attachment_from_file("new.txt", b"New text", storage=storage)
    message = ChatMessage(
        role="user", content="Original", attachment_paths=[retained.relative_path]
    )
    result = prepare_message_edit(
        message,
        "Revised",
        storage=storage,
        use_file_references=False,
        attachments=[added],
    )
    assert result.attachment_paths == [retained.relative_path, added.relative_path]
    assert "Existing text" in result.prompt and "New text" in result.prompt
