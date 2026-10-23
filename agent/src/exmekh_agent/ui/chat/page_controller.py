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

from exmekh_agent.config import config
from exmekh_agent.config.approvals import TOOL_APPROVAL_MANUAL
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.execution import DEFAULTS as EXECUTION_DEFAULTS
from exmekh_agent.exmekh_bridge.agents.agent_runner import (
    generate_chat_title,
    run_agent_turn,
)
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.agents.sub_agents import SubAgentEvent
from exmekh_agent.exmekh_bridge.errors import build_execution_error_message
from exmekh_agent.exmekh_bridge.history.message_edit import prepare_message_edit
from exmekh_agent.exmekh_bridge.history.pasted_input import (
    PreparedUserInput,
    StagedAttachment,
    prepare_user_input,
)
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ProviderContextUsage,
    SkillNode,
    ToolEvent,
)
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    ContextUsageSummary,
    populate_skill_closure_tokens,
)
from exmekh_agent.exmekh_bridge.tools.tool_approval import ApprovalModeChanged
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.state.sub_agent_conversations import apply_sub_agent_event
from exmekh_agent.state.temp_storage import TempStorage
from exmekh_agent.ui.agents.agent_selector import AgentSelector
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.context_updates import ContextUsageUpdater
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from exmekh_agent.ui.navigation.header import HeaderComponent
from exmekh_agent.ui.navigation.new_project_dialog import NewProjectDialogComponent
from exmekh_agent.ui.navigation.sidebar import SidebarComponent
from exmekh_agent.ui.resources import presentation_config, ui_text
from exmekh_agent.ui.settings.project_settings_dialog import (
    ProjectSettingsDialogComponent,
)
from exmekh_agent.ui.settings.settings_drawer import SettingsDrawerComponent
from exmekh_agent.utils.formatting import format_file_size

UI_PRESENTATION = presentation_config("controls/page_controller")

logger = logging.getLogger(__name__)

QUEUE_DRAG_JS_PATH: Path = (
    Path(__file__).parents[1] / "resources" / "js" / "queue_drag.js"
)
FILE_DROP_JS_PATH: Path = (
    Path(__file__).parents[1] / "resources" / "js" / "file_drop.js"
)
CHAT_NAVIGATION_JS_PATH = (
    Path(__file__).parents[1] / "resources" / "js" / "chat_navigation.js"
)


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

    def __init__(self, client: Client, sessions: SessionManager | None = None) -> None:
        self._sessions = sessions
        self._client = client
        self._active_task: asyncio.Task[None] | None = None
        self._followed_session_id: str | None = None
        self._snapshot_session_id: str | None = None
        self._followed_revision = -1
        self._url_session_id: str | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._context_updates = ContextUsageUpdater(client, self._apply_context_usage)
        self.header: HeaderComponent | None = None
        self.sidebar: SidebarComponent | None = None
        self.settings_drawer: SettingsDrawerComponent | None = None
        self.project_settings_dialog: ProjectSettingsDialogComponent | None = None
        self.new_project_dialog: NewProjectDialogComponent | None = None
        self.chat_view: ChatViewComponent | None = None
        self.input_bar: InputBarComponent | None = None
        self._pending_approval: Future[bool] | None = None
        self._pending_approval_session_id: str | None = None
        self._client.on_delete(lambda: self._cancel_pending_approval())

    @property
    def sessions(self) -> SessionManager:
        return self._sessions if self._sessions is not None else session_manager

    def build(self) -> None:
        self.header = HeaderComponent(
            on_toggle_drawer=self.toggle_settings,
            on_toggle_sidebar=self.toggle_sidebar,
        )
        self.header.build()
        self.sidebar = SidebarComponent(
            on_session_switched=self.session_switched,
            sessions=self.sessions,
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
        ui.add_head_html(
            f"<script>{CHAT_NAVIGATION_JS_PATH.read_text(encoding='utf-8')}</script>"
        )
        ui.on("exmekh_queue_reorder", self._handle_queue_reorder_event)
        ui.on("exmekh_chat_navigate", self._handle_chat_navigation)

        with ui.column().classes(
            UI_PRESENTATION["build.classes.chat_page_layout_w_full_justify_between"]
        ):
            self.chat_view = ChatViewComponent(
                on_suggested_prompt=self.use_suggested_prompt,
                sessions=self.sessions,
                on_edit_message=self.edit_message,
                on_split_message=self.split_conversation,
                on_agent_settings=self.open_chat_agent_settings,
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
            session = self.sessions.get_current_session()
            self.input_bar.set_session_id(session.id)
            self.input_bar.set_approval_mode(session.settings.tool_approval_mode)
            self.input_bar.set_agent_mode(session.agent_mode)
        self._render_queue_ui()
        self._sync_chat_url(replace=True)
        self.update_context_usage()
        self._sync_active_turn()
        ui.timer(
            EXECUTION_DEFAULTS["turn_sync_interval_seconds"], self._sync_active_turn
        )
        ui.timer(
            EXECUTION_DEFAULTS["storage_check_delay_seconds"],
            self._check_temp_storage_alert,
            once=True,
        )

    def _get_temp_storage(self, session_id: str | None = None) -> TempStorage:
        session = (
            self.sessions.get_session(session_id)
            if session_id
            else self.sessions.get_current_session()
        )
        if session is None:
            raise ValueError(
                ui_text("chat.page_controller.upload_conversation_was_not_found")
            )
        return self.sessions.temp_storage(session)

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
                    ui_text(
                        "chat.page_controller.high_temporary_storage_usage_files",
                        size_str=size_str,
                        stats_total_files=stats.total_files,
                    ),
                    caption=app_text(
                        "chat.page_controller.you_can_free_space_in_global_settings"
                    ),
                    type="warning",
                    position="bottom-right",
                    close_button=True,
                    timeout=EXECUTION_DEFAULTS["storage_alert_timeout_ms"],
                )

    def _sync_active_turn(self) -> None:
        """Reconnect this page to a turn started by an earlier page instance."""
        if self._client.is_deleted:
            return
        self._sync_pending_approval()
        session = self.sessions.get_current_session()
        if self._url_session_id is not None and self._url_session_id != session.id:
            # A different page may delete the selected chat. Refresh the display,
            # upload scope and URL together before accepting new input.
            self._url_session_id = session.id
            self.session_switched(session.id)
            return
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
                self.chat_view.update_stream_history(snapshot.message)
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
                self.update_context_usage(session=session)
                self._followed_revision = snapshot.revision
            self._followed_session_id = session.id

    def update_context_usage(
        self,
        session: ChatSession | None = None,
        collector: TurnStreamCollector | None = None,
        *,
        include_live: bool = True,
    ) -> None:
        """Update the bottom-left input bar with the active model and compacted context token count."""
        if not self.input_bar:
            return
        active_session = session or self.sessions.get_current_session()
        if not self._is_session_visible(active_session):
            return
        sub_agent = None
        if self.chat_view and self.chat_view.selected_sub_agent_id:
            sub_agent = active_session.find_sub_agent_conversation(
                self.chat_view.selected_sub_agent_id
            )
        if include_live and collector is None and sub_agent is None:
            snapshot = active_turns.snapshot(active_session.id, after_revision=-1)
            if snapshot and snapshot.message is not None:
                active_session = active_session.with_message(snapshot.message)
        self._context_updates.request(
            active_session,
            collector=collector,
            sub_agent=sub_agent,
            model_name=config.default_model or active_session.model,
        )

    def _apply_context_usage(
        self, session_id: str, result: ContextUsageSummary
    ) -> None:
        if self.input_bar and self.sessions.is_current_session(session_id):
            self.input_bar.update_context_info(
                model_name=result.model_name,
                used_tokens=result.used_tokens,
                context_length=result.context_length,
                estimated=result.estimated,
            )

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
        session = self.sessions.get_current_session()
        if not self.sessions.set_tool_approval_mode(session.id, new_mode):
            if self.input_bar:
                self.input_bar.set_approval_mode(session.settings.tool_approval_mode)
            ui.notify(
                ui_text("chat.page_controller.unable_to_save_approval_mode"),
                type="negative",
            )
            return
        self._sync_pending_approval()
        ui.notify(
            ui_text(
                "chat.page_controller.approval_mode_set_to",
                new_mode=session.settings.tool_approval_mode,
            ),
            type="info",
        )

    def session_switched(self, session_id: str) -> None:
        session = self.sessions.get_session(session_id)
        if session is None:
            return
        with self._client:
            self.sessions.switch_session(session.id)
            self._sync_chat_url()
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

    def _sync_chat_url(self, *, replace: bool = False) -> None:
        session = self.sessions.get_current_session()
        path = ChatRoutes(self.sessions).path_for(session)
        if replace or self._url_session_id == session.id:
            ui.navigate.history.replace(path)
        else:
            ui.navigate.history.push(path)
        self._url_session_id = session.id

    def _handle_chat_navigation(self, event: object) -> None:
        args = getattr(event, "args", {})
        session = (
            ChatRoutes(self.sessions).resolve(str(args.get("path", "")))
            if isinstance(args, dict)
            else None
        )
        if session is None:
            self._sync_chat_url(replace=True)
            return
        self.sessions.switch_session(session.id)
        self._url_session_id = session.id
        if self.sidebar:
            self.sidebar.clear_sub_agent_selection()
            if session.project_id:
                self.sidebar.expand_project(session.project_id)
            self.sidebar.render_sessions()
        self.session_switched(session.id)

    def new_chat(self) -> None:
        with self._client:
            self._prepare_conversation()
            ui.notify(
                ui_text("chat.page_controller.conversation_ready"), type="positive"
            )

    def _prepare_conversation(self, project_id: str | None = None) -> None:
        # The client context survives the sidebar replacing its button's parent.
        with self._client:
            session = self.sessions.get_or_create_empty_session(project_id)
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

    def open_chat_agent_settings(self) -> None:
        session = self.sessions.get_current_session()
        if session.agent_mode != "deep_agent":
            return
        settings = self.sessions.execution_settings(session)
        with (
            ui.dialog() as dialog,
            ui.card().classes(
                UI_PRESENTATION[
                    "open_chat_agent_settings.classes.w_440px_max_w_94vw_bg_zinc_900"
                ]
            ),
        ):
            ui.label(ui_text("chat.page_controller.chat_agent_settings")).classes(
                UI_PRESENTATION["open_chat_agent_settings.classes.text_lg"]
            )
            selector = AgentSelector(
                lambda: settings.capability_paths(config.capabilities_path)
            )
            selector.build()
            selector.set_value(settings.agent_profile, settings.agent_design)
            ui.label(
                ui_text("chat.page_controller.applies_to_the_next_response_in_this")
            ).classes(
                UI_PRESENTATION[
                    "open_chat_agent_settings.classes.text_xs_text_zinc_400"
                ]
            )

            async def save_agent() -> None:
                if active_turns.is_running(session.id):
                    ui.notify(
                        ui_text(
                            "chat.page_controller.wait_for_the_current_response_before_changing"
                        ),
                        type="warning",
                    )
                    return
                saved = await asyncio.get_running_loop().run_in_executor(
                    None,
                    partial(
                        self.sessions.save_chat_agent,
                        session.id,
                        selector.profile,
                        selector.design_snapshot(),
                    ),
                )
                with self._client:
                    if saved:
                        dialog.close()
                        ui.notify(
                            ui_text("chat.page_controller.chat_agent_saved"),
                            type="positive",
                        )

            with ui.row().classes(
                UI_PRESENTATION["open_chat_agent_settings.classes.w_full_justify_end"]
            ):
                ui.button(
                    ui_text("chat.page_controller.cancel"), on_click=dialog.close
                ).props(UI_PRESENTATION["open_chat_agent_settings.props.flat"])
                ui.button(
                    ui_text("chat.page_controller.save_chat_agent"), on_click=save_agent
                )
        dialog.open()

    async def edit_message(
        self,
        session_id: str,
        message_id: str,
        text: str,
        attachment_paths: list[str] | None = None,
        attachments: list[StagedAttachment] | None = None,
    ) -> bool:
        async with active_turns.edit_lock(session_id):
            return await self._edit_message(
                session_id, message_id, text, attachment_paths, attachments
            )

    async def split_conversation(self, session_id: str, message_id: str) -> None:
        async with active_turns.edit_lock(session_id):
            try:
                branch = await self.sessions.split_session_async(session_id, message_id)
            except ValueError as error:
                with self._client:
                    ui.notify(str(error), type="warning")
                return
            if branch is None:
                with self._client:
                    ui.notify(ui_text("chat.split.save_failed"), type="negative")
                return
            with self._client:
                self.session_switched(branch.id)
                if self.sidebar:
                    self.sidebar.clear_sub_agent_selection()
                    if branch.project_id:
                        self.sidebar.expand_project(branch.project_id)
                    self.sidebar.render_sessions()

    async def _edit_message(
        self,
        session_id: str,
        message_id: str,
        text: str,
        attachment_paths: list[str] | None = None,
        attachments: list[StagedAttachment] | None = None,
    ) -> bool:
        session = self.sessions.get_session(session_id)
        if session is None:
            return False
        message = self.sessions.find_user_message(session, message_id)
        if message is None:
            return False
        try:
            prepared = await asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    prepare_message_edit,
                    message,
                    text.strip(),
                    storage=self.sessions.temp_storage(session),
                    use_file_references=True,
                    attachment_paths=attachment_paths,
                    attachments=attachments,
                ),
            )
        except (OSError, ValueError) as error:
            with self._client:
                ui.notify(
                    ui_text(
                        "chat.page_controller.unable_to_prepare_message", error=error
                    ),
                    type="negative",
                )
            return False
        if not prepared.prompt and not prepared.image_urls:
            return False
        await active_turns.stop_and_wait(session.id)
        if not self.sessions.owns_session(session):
            return False
        replacement = prepared.to_chat_message()
        with self._client:
            if not self.sessions.edit_message(session, message_id, replacement):
                ui.notify(
                    ui_text("chat.page_controller.unable_to_save_edited_message"),
                    type="negative",
                )
                return False
            if self.chat_view and self.chat_view.is_streaming_session(session.id):
                self.chat_view.finalize_stream()
            self._followed_session_id = None
            self._followed_revision = -1
            self._render_queue_ui(session)
            self._start_turn(
                session,
                prepared,
                prior_history=session.message_history()[:-1],
                generate_title=False,
            )
        return True

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
            session = self.sessions.get_current_session()
            storage = self.sessions.temp_storage(session)
        loop = asyncio.get_running_loop()
        try:
            prepared_input = await loop.run_in_executor(
                None,
                partial(
                    prepare_user_input,
                    user_text,
                    storage=storage,
                    attachments=attachments,
                    use_file_references=True,
                ),
            )
        except (OSError, ValueError) as error:
            if not self._client.is_deleted:
                with self._client:
                    ui.notify(
                        ui_text(
                            "chat.page_controller.unable_to_prepare_attachments",
                            error=error,
                        ),
                        type="negative",
                    )
            return
        self._send_prepared(session, prepared_input)

    def _send_prepared(
        self, session: ChatSession, prepared_input: PreparedUserInput
    ) -> None:
        if not self.sessions.owns_session(session):
            return
        with self._client:
            if active_turns.is_running(session.id):
                session.add_queued_message(
                    prepared_input.prompt,
                    prepared_input.display_content,
                    image_urls=prepared_input.image_urls,
                    image_paths=prepared_input.image_paths,
                    attachment_paths=prepared_input.attachment_paths,
                )
                self.sessions.save_session(session)
                self._render_queue_ui(session)
                ui.notify(
                    ui_text("chat.page_controller.message_added_to_queue"),
                    type="info",
                    position="top",
                )
                return

            session.model = config.default_model or session.model
            prior_history = session.message_history()
            generate_title = not prior_history
            title_text = prepared_input.title_text
            if generate_title:
                session.title = self._fallback_title(title_text)

            if not self.sessions.add_message(session, prepared_input.to_chat_message()):
                ui.notify(
                    ui_text("chat.page_controller.unable_to_save_message"),
                    type="negative",
                )
                return
            self._start_turn(
                session,
                prepared_input,
                prior_history=prior_history,
                generate_title=generate_title,
            )

    def _start_turn(
        self,
        session: ChatSession,
        prepared_input: PreparedUserInput,
        *,
        prior_history: list[ChatMessage],
        generate_title: bool,
    ) -> None:
        with self._client:
            if self._is_session_visible(session):
                if self.sidebar:
                    self.sidebar.clear_sub_agent_selection()
                if self.chat_view:
                    self.chat_view.show_main()
            self._render_session_lists()
            collector = TurnStreamCollector()
            self.update_context_usage(session=session, collector=collector)

            if self.input_bar and self._is_session_visible(session):
                self.input_bar.set_busy(True)
            started_at = time.monotonic()
            if self.chat_view and self._is_session_visible(session):
                self.chat_view.start_assistant_stream(started_at=started_at)

            self._active_task = asyncio.create_task(
                self._run_turn(
                    session=session,
                    user_text=prepared_input.prompt,
                    image_urls=prepared_input.image_urls,
                    prior_history=prior_history,
                    collector=collector,
                    generate_title=generate_title,
                    title_text=prepared_input.title_text,
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
        self._sync_chat_url(replace=True)
        if self.sidebar:
            self.sidebar.render_sessions()
        if self.chat_view:
            self.chat_view.render_current_session()

    def _is_session_visible(self, session: ChatSession) -> bool:
        return self.sessions.is_current_session(session.id)

    def _schedule_live_context_update(
        self,
        loop: asyncio.AbstractEventLoop,
        session: ChatSession,
        collector: TurnStreamCollector,
        *,
        force: bool = False,
        accepts_events: Callable[[], bool] | None = None,
    ) -> None:
        self._context_updates.schedule(
            loop,
            partial(self._update_live_context_in_client, session, collector),
            force=force,
            accepts_events=accepts_events,
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

            def accepts_pending_events() -> bool:
                # Normal completion can precede queued UI callbacks. Cancellation
                # or a replacement turn must still reject those callbacks.
                return (
                    turn_task is not None
                    and not turn_task.cancelling()
                    and not cancelled
                    and (
                        not active_turns.is_running(session.id)
                        or active_turns.is_task(session.id, turn_task)
                    )
                )

            def on_sub_agent(event: SubAgentEvent) -> None:
                if accepts_pending_events():
                    self._on_sub_agent(session, event)

            def show_stream_update(action: Callable[[ChatViewComponent], None]) -> None:
                if not self._client.is_deleted:

                    def update() -> None:
                        if accepts_pending_events():
                            self._update_stream_view(session.id, action)

                    loop.call_soon_threadsafe(update)

            def mark_stream_changed() -> None:
                active_turns.mark_changed(session.id, collector)

            def on_token(token: str) -> None:
                if not accepts_events():
                    return
                collector.append_token(token)
                mark_stream_changed()
                show_stream_update(lambda view: view.stream_token(token))
                self._schedule_live_context_update(
                    loop, session, collector, accepts_events=accepts_pending_events
                )

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
                self._schedule_live_context_update(
                    loop, session, collector, accepts_events=accepts_pending_events
                )

            def on_tool(event: ToolEvent) -> None:
                if not accepts_events():
                    return
                collector.append_tool(event)
                mark_stream_changed()
                snapshot = event.model_copy(deep=True)
                show_stream_update(lambda view: view.add_tool_event(snapshot))
                self._schedule_live_context_update(
                    loop,
                    session,
                    collector,
                    force=True,
                    accepts_events=accepts_pending_events,
                )

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
                self._schedule_live_context_update(
                    loop,
                    session,
                    collector,
                    force=True,
                    accepts_events=accepts_pending_events,
                )

            def on_raw_history(
                raw_history: list[dict[str, object]],
                skill_tree: list[dict[str, object]],
            ) -> None:
                if not accepts_events():
                    return
                collector.set_raw_history(raw_history, skill_tree)
                mark_stream_changed()
                snapshot = collector.snapshot_message()
                show_stream_update(lambda view: view.update_stream_history(snapshot))
                show_stream_update(lambda view: view.show_steering_insertions(snapshot))
                populate_skill_closure_tokens(session, collector=collector)
                self._schedule_live_context_update(
                    loop,
                    session,
                    collector,
                    force=True,
                    accepts_events=accepts_pending_events,
                )

            def on_model_context(context) -> None:
                if accepts_events():
                    collector.set_model_context(context)

            def on_compaction_progress(event) -> None:
                if not accepts_events():
                    return
                collector.set_compaction_progress(event)
                mark_stream_changed()
                snapshot = collector.compaction_snapshot()
                show_stream_update(
                    lambda view: view.update_compaction_progress(snapshot)
                )

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
                self._schedule_live_context_update(
                    loop,
                    session,
                    collector,
                    force=True,
                    accepts_events=accepts_pending_events,
                )

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
                def record() -> None:
                    if accepts_pending_events():
                        self._record_command_approval(session, command, raw_history)

                if accepts_events():
                    loop.call_soon_threadsafe(record)

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
                    on_model_context=on_model_context,
                    on_compaction_progress=on_compaction_progress,
                    history=prior_history,
                    agent_context_id=session.id,
                    on_sub_agent_event=lambda event: loop.call_soon_threadsafe(
                        on_sub_agent, event
                    ),
                    sub_agent_history=list(session.sub_agent_conversations),
                    settings=self.sessions.execution_settings(session),
                    on_context=on_context,
                    on_request_manual_approval=on_request_manual_approval,
                    get_tool_approval_mode=lambda: session.settings.tool_approval_mode,
                    on_command_approved=on_command_approved,
                    steering=collector.steering,
                    on_steering_added=on_steering_added,
                )
            except asyncio.CancelledError:
                logger.info(
                    app_text("chat.page_controller.agent_execution_cancelled_by_user")
                )
                cancelled = True
                session.cleanup_active_sub_agents()
                response_text = ui_text("chat.page_controller.stopped_response_marker")
                message_error = "cancelled"
            except Exception as error:  # noqa: BLE001 - UI execution boundary
                session.cleanup_active_sub_agents()
                logger.exception(
                    app_text("chat.page_controller.agent_execution_failed")
                )
                response_text = build_execution_error_message(
                    error,
                    base_url=config.base_url,
                    model=config.default_model,
                )
                message_error = str(error)
                if not self._client.is_deleted:
                    ui.notify(
                        ui_text(
                            "chat.page_controller.error_during_execution", error=error
                        ),
                        type="negative",
                        position="top",
                    )
            finally:
                self._cancel_pending_approval(
                    session_id=session.id, interrupt_turn=cancelled
                )

            assistant_message = collector.build_message(
                role="assistant",
                fallback_content=response_text,
                error=message_error,
            )
            assistant_message.duration_seconds = time.time() - started_at

            try:
                if self.sessions.add_message(session, assistant_message):
                    if not self._client.is_deleted:
                        self._finalize_turn(session)
                    if generate_title and not cancelled:
                        self._track_task(
                            asyncio.create_task(
                                self._update_title(session, title_text or user_text)
                            )
                        )
            finally:
                if self._active_task is turn_task:
                    self._active_task = None
                if turn_task is not None:
                    active_turns.finish(
                        session.id,
                        turn_task,
                        handoff_queue=(
                            not cancelled
                            and self._client.is_deleted
                            and session.has_queued_messages
                            and self.sessions.owns_session(session)
                        ),
                    )
            if (
                not cancelled
                and not self._client.is_deleted
                and self.sessions.owns_session(session)
            ):
                self._check_and_dispatch_next_queued(session)

    def _on_sub_agent(self, session: ChatSession, event: SubAgentEvent) -> None:
        apply_sub_agent_event(session, event)
        if event.action in {"started", "completed", "error"}:
            self.sessions.save_session(session)
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
        if self.input_bar and self._is_session_visible(session):
            self.input_bar.set_busy(False)
        self.update_context_usage(session=session, include_live=False)

    async def _update_title(self, session: ChatSession, user_text: str) -> None:
        with self._client:
            try:
                title = await generate_chat_title(
                    user_prompt=user_text,
                    model=config.default_model,
                )
                if self.sessions.update_title(session, title):
                    if (
                        self._is_session_visible(session)
                        and not self._client.is_deleted
                    ):
                        self._sync_chat_url(replace=True)
                    if self.sidebar and not self._client.is_deleted:
                        self.sidebar.render_sessions()
            except Exception as error:  # noqa: BLE001 - optional background task
                logger.debug(
                    app_text(
                        "chat.page_controller.structured_title_generation_skipped_or_failed_s"
                    ),
                    error,
                )

    def _show_manual_approval(
        self,
        command: str,
        session: ChatSession,
        decision: Future[bool],
        accepts_events: Callable[[], bool],
    ) -> None:
        if not accepts_events():
            decision.set_exception(asyncio.CancelledError())
            return
        if session.settings.tool_approval_mode != TOOL_APPROVAL_MANUAL:
            decision.set_exception(ApprovalModeChanged())
            return
        if (
            self._client.is_deleted
            or self.input_bar is None
            or not self._is_session_visible(session)
        ):
            decision.set_result(False)
            return
        self._cancel_pending_approval()
        self._pending_approval = decision
        self._pending_approval_session_id = session.id
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
            self._pending_approval_session_id = None
            if not self._client.is_deleted and self.input_bar:
                with self._client:
                    self.input_bar.hide_approval_prompt()

    def _sync_pending_approval(self) -> None:
        if self._pending_approval is None:
            return
        session = self.sessions.get_session(self._pending_approval_session_id or "")
        if (
            session is not None
            and session.settings.tool_approval_mode != TOOL_APPROVAL_MANUAL
        ):
            if not self._pending_approval.done():
                self._pending_approval.set_exception(ApprovalModeChanged())
            self._finish_manual_approval(self._pending_approval, False)

    def _cancel_pending_approval(
        self, *, session_id: str | None = None, interrupt_turn: bool = False
    ) -> None:
        if session_id is not None and self._pending_approval_session_id != session_id:
            return
        if self._pending_approval is not None:
            if interrupt_turn and not self._pending_approval.done():
                self._pending_approval.set_exception(asyncio.CancelledError())
            self._finish_manual_approval(self._pending_approval, False)

    def _record_command_approval(
        self, session: ChatSession, command: str, raw_history: list[dict[str, str]]
    ) -> None:
        self.sessions.record_command_approval(session, command, raw_history)

    def stop(self) -> None:
        with self._client:
            session = self.sessions.get_current_session()
            stopped = active_turns.stop(session.id)
            self._cancel_pending_approval(session_id=session.id, interrupt_turn=stopped)
            if self.input_bar:
                self.input_bar.hide_approval_prompt()
            if stopped:
                if self.input_bar:
                    self.input_bar.set_busy(False)
                self._render_queue_ui()
                self.update_context_usage()
                ui.notify(
                    ui_text("chat.page_controller.generation_stopped"), type="warning"
                )
            elif self.chat_view and self.chat_view.has_active_stream():
                self.chat_view.finalize_stream()
                self.chat_view.render_current_session()
                if self.input_bar:
                    self.input_bar.set_busy(False)

    def _render_queue_ui(self, session: ChatSession | None = None) -> None:
        active_session = session or self.sessions.get_current_session()
        if self.input_bar and self._is_session_visible(active_session):
            self.input_bar.sync_queue(
                active_session.queued_messages,
                can_steer=active_turns.can_steer(active_session.id),
            )

    def steer_queued_message(self, message_id: str) -> None:
        with self._client:
            session = self.sessions.get_current_session()
            message = session.find_queued_message(message_id)
            if message is None:
                return
            if not active_turns.steer(session.id, message):
                self._render_queue_ui(session)
                ui.notify(
                    ui_text(
                        "chat.page_controller.steering_requires_an_active_agent_turn"
                    ),
                    type="warning",
                )
                return
            session.remove_queued_message(message_id)
            self.sessions.save_session(session)
            self._render_queue_ui(session)
            self._sync_active_turn()

    def _check_and_dispatch_next_queued(self, session: ChatSession) -> None:
        next_msg = session.pop_queued_message()
        if next_msg is None:
            return
        self.sessions.save_session(session)
        self._render_queue_ui(session)
        prepared_input = PreparedUserInput(
            prompt=next_msg.text,
            display_content=next_msg.display_content,
            title_text=next_msg.visible_text,
            image_urls=next_msg.image_urls,
            image_paths=next_msg.image_paths,
            attachment_paths=next_msg.attachment_paths or [],
        )
        self._send_prepared(session, prepared_input)

    def edit_queued_message(self, message_id: str) -> None:
        with self._client:
            session = self.sessions.get_current_session()
            removed = session.remove_queued_message(message_id)
            if removed:
                self.sessions.save_session(session)
                self._render_queue_ui(session)
                if self.input_bar:
                    self.input_bar.set_text(removed.text)
                    self.input_bar.focus()
                ui.notify(
                    ui_text(
                        "chat.page_controller.message_returned_to_chat_for_editing"
                    ),
                    type="info",
                )

    def delete_queued_message(self, message_id: str) -> None:
        with self._client:
            session = self.sessions.get_current_session()
            removed = session.remove_queued_message(message_id)
            if removed:
                self.sessions.save_session(session)
                self._render_queue_ui(session)
                ui.notify(
                    ui_text("chat.page_controller.message_removed_from_queue"),
                    type="info",
                )

    def reorder_queued_message(
        self, source_id: str, target_id: str, position: str = "before"
    ) -> None:
        with self._client:
            session = self.sessions.get_current_session()
            if session.reorder_queued_message(source_id, target_id, position):
                self.sessions.save_session(session)
                self._render_queue_ui(session)

    def resume_queue(self) -> None:
        with self._client:
            session = self.sessions.get_current_session()
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
