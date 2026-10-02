from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from pathlib import Path

from nicegui import ui
from nicegui.client import Client

from fe.config import config
from fe.qs_bridge.agent_runner import generate_chat_title, run_agent_turn
from fe.qs_bridge.collector import TurnStreamCollector
from fe.qs_bridge.context_usage import (
    compute_session_context_usage,
    populate_skill_closure_tokens,
)
from fe.qs_bridge.errors import build_execution_error_message
from fe.qs_bridge.models import (
    ChatMessage,
    ChatSession,
    ProviderContextUsage,
    SkillNode,
    SubAgentConversation,
    ToolEvent,
)
from fe.qs_bridge.pasted_input import (
    PreparedUserInput,
    StagedAttachment,
    prepare_user_input,
)
from fe.qs_bridge.sub_agents import SubAgentEvent
from fe.state.active_turns import active_turns
from fe.state.session_manager import session_manager
from fe.state.sub_agent_conversations import apply_sub_agent_event
from fe.state.temp_storage import TempStorage
from fe.ui.chat_view import ChatViewComponent
from fe.ui.file_icons import format_file_size
from fe.ui.header import HeaderComponent
from fe.ui.input_bar import InputBarComponent
from fe.ui.new_project_dialog import NewProjectDialogComponent
from fe.ui.project_settings_dialog import ProjectSettingsDialogComponent
from fe.ui.settings_drawer import SettingsDrawerComponent
from fe.ui.sidebar import SidebarComponent

logger = logging.getLogger(__name__)

QUEUE_DRAG_JS_PATH: Path = Path(__file__).parent / "js" / "queue_drag.js"
FILE_DROP_JS_PATH: Path = Path(__file__).parent / "js" / "file_drop.js"


def load_queue_drag_js() -> None:
    if QUEUE_DRAG_JS_PATH.exists():
        js_content = QUEUE_DRAG_JS_PATH.read_text(encoding="utf-8")
        ui.add_head_html(f"<script>{js_content}</script>")


def load_file_drop_js() -> None:
    if FILE_DROP_JS_PATH.exists():
        js_content = FILE_DROP_JS_PATH.read_text(encoding="utf-8")
        ui.add_head_html(f"<script>{js_content}</script>")


class ChatPageController:
    """Coordinate one NiceGUI page and its active model turn."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._active_task: asyncio.Task[None] | None = None
        self._followed_session_id: str | None = None
        self._snapshot_session_id: str | None = None
        self._followed_revision = -1
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._last_context_update_ts: float = 0.0
        self._context_update_task: asyncio.Task[None] | None = None
        self._context_update_revision = 0
        self._pending_context_request: (
            tuple[
                ChatSession,
                TurnStreamCollector | None,
                SubAgentConversation | None,
                str,
            ]
            | None
        ) = None
        self.header: HeaderComponent | None = None
        self.sidebar: SidebarComponent | None = None
        self.settings_drawer: SettingsDrawerComponent | None = None
        self.project_settings_dialog: ProjectSettingsDialogComponent | None = None
        self.new_project_dialog: NewProjectDialogComponent | None = None
        self.chat_view: ChatViewComponent | None = None
        self.input_bar: InputBarComponent | None = None
        self._pending_approval: Future[bool] | None = None
        self._client.on_delete(lambda: self._cancel_pending_approval())

    def build(self) -> None:
        self.header = HeaderComponent(
            on_toggle_drawer=self.toggle_settings,
            on_toggle_sidebar=self.toggle_sidebar,
        )
        self.header.build()
        self.sidebar = SidebarComponent(
            on_session_switched=self.session_switched,
            on_new_chat=self.new_chat,
            on_sub_agent_selected=self.sub_agent_selected,
            on_new_project=self.open_new_project,
            on_project_chat=self.new_project_chat,
            on_project_settings=self.open_project_settings,
        )
        self.sidebar.build()
        self.settings_drawer = SettingsDrawerComponent(
            on_settings_saved=self.refresh_models,
            on_models_refreshed=self.update_context_usage,
            temp_storage_getter=self._get_temp_storage,
        )
        self.settings_drawer.build()
        self.project_settings_dialog = ProjectSettingsDialogComponent(
            on_saved=self.update_context_usage
        )
        self.project_settings_dialog.build()
        self.new_project_dialog = NewProjectDialogComponent(
            self.new_project_chat, self.project_settings_dialog
        )
        self.new_project_dialog.build()

        load_queue_drag_js()
        load_file_drop_js()
        ui.on("qs_queue_reorder", self._handle_queue_reorder_event)

        with ui.column().classes("chat-page-layout w-full justify-between"):
            self.chat_view = ChatViewComponent(
                on_suggested_prompt=self.use_suggested_prompt
            )
            self.chat_view.build()
            self.input_bar = InputBarComponent(
                on_send=self.start_send,
                on_send_with_attachments=self.start_send,
                on_stop=self.stop,
                on_edit_queued=self.edit_queued_message,
                on_delete_queued=self.delete_queued_message,
                on_reorder_queued=self.reorder_queued_message,
                on_resume_queue=self.resume_queue,
                on_steer_queued=self.steer_queued_message,
                temp_storage_getter=self._get_temp_storage,
                on_approval_mode_changed=self.approval_mode_changed,
            )
            self.input_bar.build()
            session = session_manager.get_current_session()
            self.input_bar.set_session_id(session.id)
            self.input_bar.set_approval_mode(session.settings.tool_approval_mode)
            self.input_bar.set_agent_mode(session.agent_mode)
        self._render_queue_ui()
        self.update_context_usage()
        self._sync_active_turn()
        ui.timer(0.25, self._sync_active_turn)
        ui.timer(1.0, self._check_temp_storage_alert, once=True)

    @staticmethod
    def _get_temp_storage(session_id: str | None = None) -> TempStorage:
        session = (
            session_manager.get_session(session_id)
            if session_id
            else session_manager.get_current_session()
        )
        if session is None:
            raise ValueError("Upload conversation was not found")
        return session_manager.temp_storage(session)

    async def _check_temp_storage_alert(self) -> None:
        """Check if temporary storage exceeds 1GB and show a gentle alert on page open."""
        if self._client.is_deleted:
            return
        storage = self._get_temp_storage()
        stats = await asyncio.get_running_loop().run_in_executor(None, storage.stats)
        if not self._client.is_deleted and stats.is_over_limit:
            size_str = format_file_size(stats.total_size)
            with self._client:
                ui.notify(
                    f"High temporary storage usage: {size_str} ({stats.total_files} files)",
                    caption="You can free space in global settings.",
                    type="warning",
                    position="bottom-right",
                    close_button=True,
                    timeout=8000,
                )

    def _sync_active_turn(self) -> None:
        """Reconnect this page to a turn started by an earlier page instance."""
        if self._client.is_deleted:
            return
        session = session_manager.get_current_session()
        if self._snapshot_session_id != session.id:
            self._snapshot_session_id = session.id
            self._followed_revision = -1
        snapshot = active_turns.snapshot(
            session.id, after_revision=self._followed_revision
        )
        owns_turn = self._active_task is not None and active_turns.is_task(
            session.id, self._active_task
        )
        with self._client:
            if self.input_bar:
                self._render_queue_ui(session)
            if (
                owns_turn
                and self.chat_view
                and self.chat_view.is_streaming_session(session.id)
                and snapshot
                and snapshot.message is not None
            ):
                self.chat_view.show_steering_messages(
                    snapshot.message.steering_messages
                )
                self.chat_view.show_steering_insertions(snapshot.message)
                self._followed_revision = snapshot.revision
        if snapshot is None:
            if self._followed_session_id == session.id:
                with self._client:
                    if self.chat_view:
                        self.chat_view.finalize_session_stream(session.id)
                    if self.input_bar:
                        self.input_bar.set_busy(False)
                    self._render_queue_ui(session)
                    self.update_context_usage(session=session)
                self._followed_session_id = None
                self._followed_revision = -1
            if active_turns.claim_queue_handoff(session.id):
                with self._client:
                    self._check_and_dispatch_next_queued(session)
            return
        if owns_turn:
            return
        with self._client:
            if self.chat_view and not self.chat_view.is_streaming_session(session.id):
                if self.chat_view.has_active_stream():
                    self.chat_view.finalize_stream()
                self.chat_view.render_current_session()
                self.chat_view.start_assistant_stream(started_at=snapshot.started_at)
                self._followed_revision = -1
                snapshot = active_turns.snapshot(session.id, after_revision=-1)
            if self.input_bar:
                self.input_bar.set_busy(True)
            if self.chat_view and snapshot and snapshot.message is not None:
                self.chat_view.restore_stream_snapshot(snapshot.message)
                self._followed_revision = snapshot.revision
            self._followed_session_id = session.id

    def update_context_usage(
        self,
        session: ChatSession | None = None,
        collector: TurnStreamCollector | None = None,
    ) -> None:
        """Update the bottom-left input bar with the active model and compacted context token count."""
        if not self.input_bar:
            return
        active_session = session or session_manager.get_current_session()
        if not self._is_session_visible(active_session):
            return
        sub_agent = None
        if self.chat_view and self.chat_view.selected_sub_agent_id:
            sub_agent = active_session.find_sub_agent_conversation(
                self.chat_view.selected_sub_agent_id
            )
        self._pending_context_request = (
            active_session,
            collector,
            sub_agent,
            config.default_model or active_session.model,
        )
        self._context_update_revision += 1
        if self._context_update_task is None or self._context_update_task.done():
            self._context_update_task = asyncio.create_task(
                self._refresh_context_usage()
            )

    async def _refresh_context_usage(self) -> None:
        """Count tokens in a worker and apply only the newest result to the UI."""
        loop = asyncio.get_running_loop()
        while self._pending_context_request is not None:
            revision = self._context_update_revision
            session, collector, sub_agent, model_name = self._pending_context_request
            result = await loop.run_in_executor(
                None,
                partial(
                    compute_session_context_usage,
                    session=session,
                    collector=collector,
                    sub_agent=sub_agent,
                    model_override=model_name,
                ),
            )
            if revision != self._context_update_revision:
                continue
            with self._client:
                if self.input_bar and self._is_session_visible(session):
                    self.input_bar.update_context_info(
                        model_name=result.model_name,
                        used_tokens=result.used_tokens,
                        context_length=result.context_length,
                        estimated=result.estimated,
                    )
            self._pending_context_request = None
            return

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def refresh_models(self) -> None:
        if self.settings_drawer:
            await self.settings_drawer.refresh_models()
        self.update_context_usage()

    def toggle_settings(self) -> None:
        if self.project_settings_dialog:
            self.project_settings_dialog.close()
        if self.settings_drawer:
            self.settings_drawer.toggle()

    def open_project_settings(self, project_id: str) -> None:
        if self.settings_drawer:
            self.settings_drawer.close()
        if self.project_settings_dialog:
            self.project_settings_dialog.open(project_id)

    def toggle_sidebar(self) -> None:
        if self.sidebar:
            self.sidebar.toggle()

    def open_new_project(self) -> None:
        if self.new_project_dialog:
            self.new_project_dialog.open()

    def new_project_chat(self, project_id: str) -> None:
        self._prepare_conversation(project_id)

    def approval_mode_changed(self, new_mode: str) -> None:
        session = session_manager.get_current_session()
        if not session_manager.set_tool_approval_mode(session.id, new_mode):
            if self.input_bar:
                self.input_bar.set_approval_mode(session.settings.tool_approval_mode)
            ui.notify("Unable to save approval mode", type="negative")
            return
        ui.notify(f"Approval mode set to: {new_mode}", type="info")

    def session_switched(self, session_id: str) -> None:
        session = session_manager.get_session(session_id)
        if session is None:
            return
        with self._client:
            self._cancel_pending_approval()
            if self.chat_view:
                self.chat_view.show_main()
            if self.input_bar:
                self.input_bar.set_session_id(session.id)
                self.input_bar.set_approval_mode(session.settings.tool_approval_mode)
                self.input_bar.set_agent_mode(session.agent_mode)
            self._render_queue_ui(session)
            self.update_context_usage(session=session)
            self._sync_active_turn()

    def new_chat(self) -> None:
        with self._client:
            self._prepare_conversation()
            ui.notify("Conversation ready", type="positive")

    def _prepare_conversation(self, project_id: str | None = None) -> None:
        # The client context survives the sidebar replacing its button's parent.
        with self._client:
            session = session_manager.get_or_create_empty_session(project_id)
            if self.sidebar:
                self.sidebar.clear_sub_agent_selection()
            self.session_switched(session.id)
            if self.sidebar:
                self.sidebar.render_sessions()

    def use_suggested_prompt(self, prompt: str) -> None:
        if self.input_bar:
            self.input_bar.set_text(prompt)

    def sub_agent_selected(self, conversation_id: str) -> None:
        with self._client:
            if self.chat_view:
                self.chat_view.show_sub_agent(conversation_id)
            self.update_context_usage()

    def start_send(
        self,
        user_text: str,
        attachments: list[StagedAttachment] | None = None,
    ) -> None:
        self._track_task(
            asyncio.create_task(self.send(user_text, attachments=attachments))
        )

    async def send(
        self,
        user_text: str,
        attachments: list[StagedAttachment] | None = None,
    ) -> None:
        with self._client:
            session = session_manager.get_current_session()
            storage = session_manager.temp_storage(session)
        loop = asyncio.get_running_loop()
        try:
            prepared_input = await loop.run_in_executor(
                None,
                partial(
                    prepare_user_input,
                    user_text,
                    storage=storage,
                    attachments=attachments,
                    use_file_references=session.agent_mode == "deep_agent",
                ),
            )
        except (OSError, ValueError) as error:
            if not self._client.is_deleted:
                with self._client:
                    ui.notify(
                        f"Unable to prepare attachments: {error}", type="negative"
                    )
            return
        self._send_prepared(session, prepared_input)

    def _send_prepared(
        self, session: ChatSession, prepared_input: PreparedUserInput
    ) -> None:
        if not session_manager.owns_session(session):
            return
        with self._client:
            if active_turns.is_running(session.id) or (
                self._active_task is not None and not self._active_task.done()
            ):
                session.add_queued_message(
                    prepared_input.prompt,
                    prepared_input.display_content,
                    image_urls=prepared_input.image_urls,
                    image_paths=prepared_input.image_paths,
                )
                session_manager.save_session(session)
                self._render_queue_ui(session)
                ui.notify(
                    "Message added to queue",
                    type="info",
                    position="top",
                )
                return

            session.model = config.default_model or session.model
            if self.sidebar:
                self.sidebar.clear_sub_agent_selection()
            if self.chat_view:
                self.chat_view.show_main()
            prior_history = session.message_history()
            generate_title = not prior_history
            title_text = prepared_input.title_text
            if generate_title:
                session.title = self._fallback_title(title_text)

            session_manager.add_message(
                session,
                ChatMessage(
                    role="user",
                    content=prepared_input.prompt,
                    image_urls=prepared_input.image_urls,
                    image_paths=prepared_input.image_paths,
                    display_content=prepared_input.display_content,
                ),
            )
            self._render_session_lists()
            collector = TurnStreamCollector()
            self.update_context_usage(session=session, collector=collector)

            if self.input_bar:
                self.input_bar.set_busy(True)
            started_at = time.monotonic()
            if self.chat_view:
                self.chat_view.start_assistant_stream(started_at=started_at)

            self._active_task = asyncio.create_task(
                self._run_turn(
                    session=session,
                    user_text=prepared_input.prompt,
                    image_urls=prepared_input.image_urls,
                    prior_history=prior_history,
                    collector=collector,
                    generate_title=generate_title,
                    title_text=title_text,
                )
            )
            active_turns.start(
                session.id,
                self._active_task,
                collector,
                started_at=started_at,
            )

    @staticmethod
    def _fallback_title(user_text: str) -> str:
        clean_text = user_text.strip().replace("\n", " ")
        return clean_text[:30] + ("..." if len(clean_text) > 30 else "")

    def _render_session_lists(self) -> None:
        if self.sidebar:
            self.sidebar.render_sessions()
        if self.chat_view:
            self.chat_view.render_current_session()

    @staticmethod
    def _is_session_visible(session: ChatSession) -> bool:
        return session_manager.is_current_session(session.id)

    def _schedule_live_context_update(
        self,
        loop: asyncio.AbstractEventLoop,
        session: ChatSession,
        collector: TurnStreamCollector,
        *,
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and (now - self._last_context_update_ts) < 0.15:
            return
        self._last_context_update_ts = now
        loop.call_soon_threadsafe(
            self._update_live_context_in_client, session, collector
        )

    def _update_live_context_in_client(
        self,
        session: ChatSession,
        collector: TurnStreamCollector,
    ) -> None:
        if self._client.is_deleted:
            return
        with self._client:
            self.update_context_usage(session=session, collector=collector)

    async def _run_turn(
        self,
        *,
        session: ChatSession,
        user_text: str,
        prior_history: list[ChatMessage],
        collector: TurnStreamCollector,
        generate_title: bool,
        title_text: str | None = None,
        image_urls: list[str] | None = None,
    ) -> None:
        with self._client:
            loop = asyncio.get_running_loop()
            turn_task = asyncio.current_task()
            cancelled = False

            def accepts_events() -> bool:
                return turn_task is not None and not (
                    turn_task.done() or turn_task.cancelling() or cancelled
                )

            def on_sub_agent(event: SubAgentEvent) -> None:
                if accepts_events():
                    self._on_sub_agent(session, event)

            def show_stream_update(action: Callable[[ChatViewComponent], None]) -> None:
                if not self._client.is_deleted:
                    loop.call_soon_threadsafe(
                        self._update_stream_view, session.id, action
                    )

            def mark_stream_changed() -> None:
                active_turns.mark_changed(session.id, collector)

            def on_token(token: str) -> None:
                if not accepts_events():
                    return
                collector.append_token(token)
                mark_stream_changed()
                show_stream_update(lambda view: view.stream_token(token))
                self._schedule_live_context_update(loop, session, collector)

            def on_reasoning(token: str, skill_path: str = "/", depth: int = 0) -> None:
                if not accepts_events():
                    return
                block = collector.append_reasoning(
                    token, skill_path=skill_path, depth=depth
                )
                mark_stream_changed()
                show_stream_update(
                    lambda view: view.stream_reasoning(
                        token,
                        skill_path=skill_path,
                        depth=depth,
                        generation_index=block.generation_index,
                    )
                )
                self._schedule_live_context_update(loop, session, collector)

            def on_tool(event: ToolEvent) -> None:
                if not accepts_events():
                    return
                collector.append_tool(event)
                mark_stream_changed()
                snapshot = event.model_copy(deep=True)
                show_stream_update(lambda view: view.add_tool_event(snapshot))
                self._schedule_live_context_update(loop, session, collector, force=True)

            def on_skill(node: SkillNode, action: str) -> None:
                if not accepts_events():
                    return
                collector.append_skill(node.model_copy(deep=True), action)
                mark_stream_changed()
                populate_skill_closure_tokens(session, collector=collector)
                snapshot = collector.skill_snapshot(node.path) or node.model_copy(
                    deep=True
                )
                show_stream_update(
                    lambda view: view.handle_live_skill_event(snapshot, action)
                )
                self._schedule_live_context_update(loop, session, collector, force=True)

            def on_raw_history(
                raw_history: list[dict[str, object]],
                skill_tree: list[dict[str, object]],
            ) -> None:
                if not accepts_events():
                    return
                collector.set_raw_history(raw_history, skill_tree)
                mark_stream_changed()
                snapshot = collector.snapshot_message()
                show_stream_update(lambda view: view.show_steering_insertions(snapshot))
                populate_skill_closure_tokens(session, collector=collector)
                self._schedule_live_context_update(loop, session, collector, force=True)

            def on_steering_added() -> None:
                if not accepts_events():
                    return
                mark_stream_changed()
                snapshot = collector.snapshot_message()
                show_stream_update(lambda view: view.show_steering_insertions(snapshot))

            def on_context(context: ProviderContextUsage) -> None:
                if not accepts_events():
                    return
                collector.set_provider_context(context)
                mark_stream_changed()
                for snapshot in collector.skill_snapshots():
                    show_stream_update(
                        lambda view, snapshot=snapshot: view.handle_live_skill_event(
                            snapshot, "updated"
                        )
                    )
                self._schedule_live_context_update(loop, session, collector, force=True)

            def on_request_manual_approval(command: str) -> bool:
                decision: Future[bool] = Future()
                loop.call_soon_threadsafe(
                    self._show_manual_approval,
                    command,
                    session,
                    decision,
                    accepts_events,
                )
                return decision.result()

            def on_command_approved(
                command: str, raw_history: list[dict[str, str]]
            ) -> None:
                loop.call_soon_threadsafe(
                    self._record_command_approval, session, command, raw_history
                )

            started_at = time.time()
            message_error = None
            try:
                response_text = await run_agent_turn(
                    user_prompt=user_text,
                    image_urls=image_urls,
                    model=config.default_model,
                    on_token=on_token,
                    on_reasoning=on_reasoning,
                    on_tool_event=on_tool,
                    agent_mode=session.agent_mode,
                    on_skill_event=on_skill,
                    on_raw_history=on_raw_history,
                    history=prior_history,
                    agent_context_id=session.id,
                    on_sub_agent_event=lambda event: loop.call_soon_threadsafe(
                        on_sub_agent, event
                    ),
                    sub_agent_history=list(session.sub_agent_conversations),
                    settings=session_manager.execution_settings(session),
                    on_context=on_context,
                    on_request_manual_approval=on_request_manual_approval,
                    on_command_approved=on_command_approved,
                    steering=collector.steering,
                    on_steering_added=on_steering_added,
                )
            except asyncio.CancelledError:
                logger.info("Agent execution cancelled by user")
                cancelled = True
                session.cleanup_active_sub_agents()
                response_text = "[Generation stopped by user]"
                message_error = "cancelled"
            except Exception as error:  # noqa: BLE001 - UI execution boundary
                session.cleanup_active_sub_agents()
                logger.exception("Agent execution failed")
                response_text = build_execution_error_message(
                    error,
                    base_url=config.base_url,
                    model=config.default_model,
                )
                message_error = str(error)
                if not self._client.is_deleted:
                    ui.notify(
                        f"Error during execution: {error}",
                        type="negative",
                        position="top",
                    )
            finally:
                self._cancel_pending_approval(interrupt_turn=cancelled)

            assistant_message = collector.build_message(
                role="assistant",
                fallback_content=response_text,
                error=message_error,
            )
            assistant_message.duration_seconds = time.time() - started_at

            try:
                if session_manager.add_message(session, assistant_message):
                    if not self._client.is_deleted:
                        self._finalize_turn(session)
                    if generate_title and not cancelled:
                        self._track_task(
                            asyncio.create_task(
                                self._update_title(session, title_text or user_text)
                            )
                        )
            finally:
                self._active_task = None
                if turn_task is not None:
                    active_turns.finish(
                        session.id,
                        turn_task,
                        handoff_queue=(
                            not cancelled
                            and self._client.is_deleted
                            and session.has_queued_messages
                            and session_manager.owns_session(session)
                        ),
                    )
            if (
                not cancelled
                and not self._client.is_deleted
                and session_manager.owns_session(session)
            ):
                self._check_and_dispatch_next_queued(session)

    def _on_sub_agent(self, session: ChatSession, event: SubAgentEvent) -> None:
        apply_sub_agent_event(session, event)
        if event.action in {"started", "completed", "error"}:
            session_manager.save_session(session)
        if self._client.is_deleted:
            return
        with self._client:
            if (
                self.sidebar
                and self._is_session_visible(session)
                and event.action
                in {
                    "started",
                    "completed",
                    "error",
                }
            ):
                self.sidebar.render_sessions()
            if self.chat_view and self._is_session_visible(session):
                if (
                    self.chat_view.is_showing_sub_agent(event.id)
                    or self.chat_view.active_tab == "graph"
                ):
                    self.chat_view.render_current_session()
            if self._is_session_visible(session):
                self.update_context_usage(session=session)

    def _update_stream_view(
        self, session_id: str, action: Callable[[ChatViewComponent], None]
    ) -> None:
        if self._client.is_deleted:
            return
        with self._client:
            if self.chat_view and self.chat_view.is_streaming_session(session_id):
                action(self.chat_view)

    def _finalize_turn(self, session: ChatSession) -> None:
        if self.chat_view:
            self.chat_view.finalize_session_stream(session.id)
        if self.sidebar:
            self.sidebar.render_sessions()
        if self.input_bar:
            self.input_bar.set_busy(False)
        self.update_context_usage(session=session)

    async def _update_title(self, session: ChatSession, user_text: str) -> None:
        with self._client:
            try:
                title = await generate_chat_title(
                    user_prompt=user_text,
                    model=config.default_model,
                )
                if session_manager.update_title(session, title):
                    if self.sidebar and not self._client.is_deleted:
                        self.sidebar.render_sessions()
            except Exception as error:  # noqa: BLE001 - optional background task
                logger.debug("Structured title generation skipped or failed: %s", error)

    def _show_manual_approval(
        self,
        command: str,
        session: ChatSession,
        decision: Future[bool],
        accepts_events: Callable[[], bool],
    ) -> None:
        self._cancel_pending_approval()
        if not accepts_events():
            decision.set_exception(asyncio.CancelledError())
            return
        if (
            self._client.is_deleted
            or self.input_bar is None
            or not self._is_session_visible(session)
        ):
            decision.set_result(False)
            return
        self._pending_approval = decision
        with self._client:
            self.input_bar.show_approval_prompt(
                command,
                on_accept=partial(self._finish_manual_approval, decision, True),
                on_decline=partial(self._finish_manual_approval, decision, False),
            )

    def _finish_manual_approval(self, decision: Future[bool], approved: bool) -> None:
        if not decision.done():
            decision.set_result(approved)
        if self._pending_approval is decision:
            self._pending_approval = None
            if not self._client.is_deleted and self.input_bar:
                with self._client:
                    self.input_bar.hide_approval_prompt()

    def _cancel_pending_approval(self, *, interrupt_turn: bool = False) -> None:
        if self._pending_approval is not None:
            if interrupt_turn and not self._pending_approval.done():
                self._pending_approval.set_exception(asyncio.CancelledError())
            self._finish_manual_approval(self._pending_approval, False)

    @staticmethod
    def _record_command_approval(
        session: ChatSession, command: str, raw_history: list[dict[str, str]]
    ) -> None:
        session_manager.record_command_approval(session, command, raw_history)

    def stop(self) -> None:
        with self._client:
            session = session_manager.get_current_session()
            stopped = active_turns.stop(session.id)
            if not stopped and self._active_task and not self._active_task.done():
                self._active_task.cancel()
                stopped = True
            self._cancel_pending_approval(interrupt_turn=stopped)
            if self.input_bar:
                self.input_bar.hide_approval_prompt()
            if stopped:
                if self.input_bar:
                    self.input_bar.set_busy(False)
                self._render_queue_ui()
                self.update_context_usage()
                ui.notify("Generation stopped", type="warning")
            elif self.chat_view and self.chat_view.has_active_stream():
                self.chat_view.finalize_stream()
                self.chat_view.render_current_session()
                if self.input_bar:
                    self.input_bar.set_busy(False)

    def _render_queue_ui(self, session: ChatSession | None = None) -> None:
        active_session = session or session_manager.get_current_session()
        if self.input_bar and self._is_session_visible(active_session):
            self.input_bar.sync_queue(
                active_session.queued_messages,
                can_steer=active_turns.can_steer(active_session.id),
            )

    def steer_queued_message(self, message_id: str) -> None:
        with self._client:
            session = session_manager.get_current_session()
            message = session.find_queued_message(message_id)
            if message is None:
                return
            if not active_turns.steer(session.id, message):
                self._render_queue_ui(session)
                ui.notify("Steering requires an active agent turn", type="warning")
                return
            session.remove_queued_message(message_id)
            session_manager.save_session(session)
            self._render_queue_ui(session)
            self._sync_active_turn()

    def _check_and_dispatch_next_queued(self, session: ChatSession) -> None:
        next_msg = session.pop_queued_message()
        if next_msg is None:
            return
        session_manager.save_session(session)
        self._render_queue_ui(session)
        prepared_input = PreparedUserInput(
            prompt=next_msg.text,
            display_content=next_msg.display_content,
            title_text=next_msg.visible_text,
            image_urls=next_msg.image_urls,
            image_paths=next_msg.image_paths,
        )
        self._send_prepared(session, prepared_input)

    def edit_queued_message(self, message_id: str) -> None:
        with self._client:
            session = session_manager.get_current_session()
            removed = session.remove_queued_message(message_id)
            if removed:
                session_manager.save_session(session)
                self._render_queue_ui(session)
                if self.input_bar:
                    self.input_bar.set_text(removed.text)
                    self.input_bar.focus()
                ui.notify("Message returned to chat for editing", type="info")

    def delete_queued_message(self, message_id: str) -> None:
        with self._client:
            session = session_manager.get_current_session()
            removed = session.remove_queued_message(message_id)
            if removed:
                session_manager.save_session(session)
                self._render_queue_ui(session)
                ui.notify("Message removed from queue", type="info")

    def reorder_queued_message(
        self, source_id: str, target_id: str, position: str = "before"
    ) -> None:
        with self._client:
            session = session_manager.get_current_session()
            if session.reorder_queued_message(source_id, target_id, position):
                session_manager.save_session(session)
                self._render_queue_ui(session)

    def resume_queue(self) -> None:
        with self._client:
            session = session_manager.get_current_session()
            self._check_and_dispatch_next_queued(session)

    def _handle_queue_reorder_event(self, e: object) -> None:
        with self._client:
            args = getattr(e, "args", {}) if hasattr(e, "args") else {}
            if isinstance(args, dict):
                src_id = str(args.get("source_id", ""))
                target_id = str(args.get("target_id", ""))
                pos = str(args.get("position", "before"))
                if src_id and target_id:
                    self.reorder_queued_message(src_id, target_id, pos)
