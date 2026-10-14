from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from pathlib import Path

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.paths import APPLICATION_ROOT
from exmekh_agent.config.storage import DEFAULTS as STORAGE_DEFAULTS
from exmekh_agent.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    Project,
    ProjectSettings,
)
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.agent_design import AgentDesign, AgentProfile
from exmekh_agent.state.json_store import JsonRecordStore
from exmekh_agent.state.temp_storage import TempStorage

logger = logging.getLogger(__name__)
CODE_WORK_ROOT = APPLICATION_ROOT


class SessionManager:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self._default_view: SessionManager | None = None
        self._initialize_storage(
            storage_dir or config.data_dir / STORAGE_DEFAULTS["chats_directory"]
        )

    def for_view(self, session_id: str | None = None) -> SessionManager:
        """Share records and persistence, retaining selection in the caller's page."""
        view = copy(self)
        view._default_view = self._default_view or self
        if session_id is not None:
            view.switch_session(session_id)
        return view

    def _initialize_storage(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.projects_dir = (
            self.storage_dir.parent / STORAGE_DEFAULTS["projects_directory"]
        )
        self._session_store = JsonRecordStore(self.storage_dir, ChatSession)
        self._project_store = JsonRecordStore(self.projects_dir, Project)
        self.projects: dict[str, Project] = {}
        self.sessions: dict[str, ChatSession] = {}
        self.current_session_id: str | None = None
        self.load_all()

        if not self.sessions:
            self.create_session()

    @contextmanager
    def using_storage(self, storage_dir: Path) -> Iterator[None]:
        """Use alternate persistence while retaining the previously loaded state."""
        previous = (
            self.storage_dir,
            self.projects_dir,
            self._session_store,
            self._project_store,
            self.projects,
            self.sessions,
            self.current_session_id,
        )
        try:
            self._initialize_storage(storage_dir)
            yield
        finally:
            (
                self.storage_dir,
                self.projects_dir,
                self._session_store,
                self._project_store,
                self.projects,
                self.sessions,
                self.current_session_id,
            ) = previous

    def create_project(
        self, work_root: Path | str, settings: ProjectSettings | None = None
    ) -> Project:
        root = Path(work_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(
                app_text(
                    "errors_state.session_manager.select_an_existing_project_folder"
                )
            )
        if any(Path(project.work_root) == root for project in self.projects.values()):
            raise ValueError(
                app_text(
                    "errors_state.session_manager.a_project_already_uses_this_folder"
                )
            )
        project = Project(
            work_root=str(root),
            settings=(settings or ProjectSettings()).model_copy(deep=True),
        )
        self.save_project(project)
        self.projects[project.id] = project
        return project

    def get_project(self, project_id: str | None) -> Project | None:
        return self.projects.get(project_id) if project_id else None

    def list_projects(self) -> list[Project]:
        return sorted(
            self.projects.values(), key=lambda project: project.created_at, reverse=True
        )

    def save_project(self, project: Project) -> None:
        self._project_store.save(project.id, project)

    def save_project_settings(self, project_id: str, settings: ProjectSettings) -> bool:
        project = self.get_project(project_id)
        if project is None:
            return False
        updated = project.model_copy(
            update={"settings": settings.model_copy(deep=True)}
        )
        self.save_project(updated)
        self._apply_project_settings(project, updated.settings)
        return True

    async def save_project_settings_async(
        self, project_id: str, settings: ProjectSettings
    ) -> bool:
        """Offload persistence, then resolve chats from their latest state on the caller's loop."""
        project = self.get_project(project_id)
        if project is None:
            return False
        updated = project.model_copy(
            update={"settings": settings.model_copy(deep=True)}
        )
        await asyncio.get_running_loop().run_in_executor(
            None, self.save_project, updated
        )
        project = self.get_project(project_id)
        if project is None:
            # A concurrent deletion must not leave a recreated project record.
            await asyncio.get_running_loop().run_in_executor(
                None, self._project_store.delete, project_id
            )
            return False
        self._apply_project_settings(project, updated.settings)
        return True

    def _apply_project_settings(
        self, project: Project, settings: ProjectSettings
    ) -> None:
        project.settings = settings
        for session in self.list_sessions(project.id):
            session.settings = self.execution_settings(session)

    def save_chat_agent(
        self, session_id: str, profile: AgentProfile, design: AgentDesign | None = None
    ) -> bool:
        session = self.get_session(session_id)
        if session is None:
            return False
        settings = session.settings.with_agent_override(profile, design)
        return self._save_chat_settings(session, settings)

    def execution_settings(self, session: ChatSession) -> ChatSettings:
        """Return an independent execution snapshot without mutating the conversation."""
        project = self.get_project(session.project_id)
        if project:
            return session.settings.with_project(project)
        return session.settings.model_copy(
            deep=True, update={"work_root": str(CODE_WORK_ROOT)}
        )

    def temp_storage(self, session: ChatSession) -> TempStorage:
        """Use the same staging scope for uploads, input processing, previews, and cleanup."""
        return TempStorage(
            self.execution_settings(session).work_root, self.temp_dirname(session)
        )

    def set_tool_approval_mode(self, session_id: str, mode: str) -> bool:
        session = self.get_session(session_id)
        if session is None:
            return False
        settings = session.settings.model_copy(deep=True)
        settings.set_tool_approval_mode(mode)
        return self._save_chat_settings(session, settings)

    def _save_chat_settings(self, session: ChatSession, settings: ChatSettings) -> bool:
        """Publish a chat configuration only after persisting an independent snapshot."""
        updated = session.model_copy(update={"settings": settings})
        if not self.save_session(updated):
            return False
        session.settings = settings
        session.updated_at = updated.updated_at
        return True

    @staticmethod
    def temp_dirname(session: ChatSession) -> str:
        return (
            f"tmp/projects/{session.project_id}"
            if session.project_id
            else "tmp/quickchats"
        )

    def create_session(
        self,
        title: str = app_text("state.session_manager.new_conversation"),
        project_id: str | None = None,
    ) -> ChatSession:
        project = self.get_project(project_id)
        if project_id is not None and project is None:
            raise ValueError(
                app_text("errors_state.session_manager.project_was_not_found")
            )
        session = ChatSession(
            title=title,
            project_id=project_id,
            model=config.default_model,
            agent_mode="deep_agent" if project else "quick_chat",
            settings=ChatSettings(
                **project.settings.model_dump(), work_root=project.work_root
            )
            if project
            else ChatSettings(work_root=str(CODE_WORK_ROOT)),
        )
        self.sessions[session.id] = session
        self.switch_session(session.id)
        self.save_session(session)
        return session

    def get_or_create_empty_session(self, project_id: str | None = None) -> ChatSession:
        """Select the most recent idle, empty chat, or create one if none exists."""
        for session in self.list_sessions_sorted():
            if (
                session.project_id == project_id
                and session.is_empty
                and not active_turns.is_running(session.id)
            ):
                self.switch_session(session.id)
                return session
        return self.create_session(project_id=project_id)

    def get_current_session(self) -> ChatSession:
        if not self.current_session_id or self.current_session_id not in self.sessions:
            self._select_fallback()
        return self.sessions[self.current_session_id]

    def get_session(self, session_id: str) -> ChatSession | None:
        return self.sessions.get(session_id)

    def snapshot_current_session(self) -> ChatSession:
        """Capture persisted messages and the current unfinished response."""
        session = self.get_current_session().model_copy(deep=True)
        turn = active_turns.snapshot(session.id, after_revision=-1)
        if turn is not None and turn.message is not None:
            session.add_message(turn.message)
        return session

    def owns_session(self, session: ChatSession) -> bool:
        """Check that a callback still refers to the managed session instance."""
        return self.get_session(session.id) is session

    def is_current_session(self, session_id: str) -> bool:
        return self.current_session_id == session_id

    def add_message(self, session: ChatSession, message: ChatMessage) -> bool:
        """Append and persist a message while the session is still managed."""
        if not self.owns_session(session):
            return False
        updated = session.model_copy(
            update={"messages": [*session.message_history(), message]}
        )
        if not self.save_session(updated):
            return False
        session.add_message(message)
        session.updated_at = updated.updated_at
        return True

    def find_user_message(
        self, session: ChatSession, message_id: str
    ) -> ChatMessage | None:
        """Find saved input or steering accepted by an unfinished turn."""
        if not self.owns_session(session):
            return None
        message = session.find_user_message(message_id)
        if message is not None:
            return message
        snapshot = active_turns.snapshot(session.id, after_revision=-1)
        return (
            snapshot.message.find_user_message(message_id)
            if snapshot and snapshot.message
            else None
        )

    def edit_message(
        self, session: ChatSession, message_id: str, replacement: ChatMessage
    ) -> bool:
        """Persist a revised branch atomically before exposing it to the UI."""
        if not self.owns_session(session):
            return False
        updated = session.revised_user_message(message_id, replacement)
        if updated is None or not self.save_session(updated):
            return False
        session.apply_revision(updated)
        return True

    def update_title(self, session: ChatSession, title: str) -> bool:
        if not title or not self.owns_session(session):
            return False
        updated = session.model_copy(update={"title": title})
        if not self.save_session(updated):
            return False
        session.title = title
        session.updated_at = updated.updated_at
        return True

    def record_command_approval(
        self, session: ChatSession, command: str, raw_history: list[dict[str, str]]
    ) -> bool:
        if not self.owns_session(session):
            return False
        settings = session.settings.model_copy(deep=True)
        settings.record_command_approval(command, raw_history)
        return self._save_chat_settings(session, settings)

    def switch_session(self, session_id: str) -> ChatSession | None:
        if session_id in self.sessions:
            self.current_session_id = session_id
            if self._default_view is not None:
                # The landing page remembers the last visited chat; existing pages
                # keep their own selection and use the shared records by identity.
                self._default_view.switch_session(session_id)
            return self.sessions[session_id]
        return None

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session is None:
            return
        self._delete_session_record(session)
        if self.current_session_id == session_id:
            self._select_fallback(session.project_id)

    def _delete_session_record(self, session: ChatSession) -> None:
        # A failed disk deletion must leave the chat visible and its turn running.
        self._session_store.delete(session.id)
        active_turns.cancel_and_forget(session.id)
        del self.sessions[session.id]

    def _select_fallback(self, project_id: str | None = None) -> None:
        candidates = self.list_sessions(project_id) or self.list_sessions_sorted()
        if candidates:
            self.switch_session(candidates[0].id)
        else:
            self.create_session()

    def delete_project(self, project_id: str) -> None:
        if self.get_project(project_id) is None:
            return
        try:
            for session in self.list_sessions(project_id):
                self._delete_session_record(session)
            self._project_store.delete(project_id)
            del self.projects[project_id]
        finally:
            if self.current_session_id not in self.sessions:
                self._select_fallback(project_id)

    def save_session(self, session: ChatSession) -> bool:
        updated = session.model_copy(update={"updated_at": time.time()})
        try:
            self._session_store.save(session.id, updated)
        except (OSError, ValueError) as error:
            logger.error(
                app_text("errors_state.session_manager.failed_to_save_session_s_s"),
                session.id,
                error,
            )
            return False
        session.updated_at = updated.updated_at
        return True

    def load_all(self) -> None:
        self.projects.clear()
        self.projects.update(
            (project.id, project) for project in self._project_store.load_all()
        )
        self.sessions.clear()
        self.sessions.update(
            (session.id, session) for session in self._session_store.load_all()
        )
        for store, records in (
            (self._project_store, self.projects.values()),
            (self._session_store, self.sessions.values()),
        ):
            for record in records:
                if record.settings.migrate_legacy_capabilities_path():
                    try:
                        # Keep the original timestamps and conversation history.
                        store.save(record.id, record)
                    except (OSError, ValueError) as error:
                        logger.warning(
                            app_text(
                                "errors_state.session_manager.could_not_persist_capabilities_migration_for_s_s"
                            ),
                            record.id,
                            error,
                        )
        for session in self.list_sessions_sorted():
            needs_project = (
                "project_id" not in session.model_fields_set
                and session.agent_mode == "deep_agent"
            ) or (
                session.project_id is not None
                and self.get_project(session.project_id) is None
            )
            if needs_project:
                self._restore_session_project(session)
            session.settings = self.execution_settings(session)
            session.agent_mode = (
                "deep_agent" if self.get_project(session.project_id) else "quick_chat"
            )
        sessions = self.list_sessions_sorted()
        self.current_session_id = sessions[0].id if sessions else None

    def _restore_session_project(self, session: ChatSession) -> None:
        """Migrate legacy chats and recover missing project metadata from their snapshots."""
        root = str(Path(session.settings.work_root or CODE_WORK_ROOT).resolve())
        project = next(
            (item for item in self.list_projects() if item.work_root == root), None
        )
        if project is None:
            project = Project(
                work_root=root,
                settings=ProjectSettings.model_validate(session.settings.model_dump()),
            )
            if session.project_id:
                project.id = session.project_id
            self.projects[project.id] = project
        session.project_id = project.id
        try:
            self.save_project(project)
            # Migration keeps the original chat timestamp and uses the same atomic store.
            self._session_store.save(session.id, session)
        except (OSError, ValueError) as error:
            logger.warning(
                app_text(
                    "errors_state.session_manager.could_not_persist_project_migration_for_s_s"
                ),
                session.id,
                error,
            )

    def list_sessions(self, project_id: str | None) -> list[ChatSession]:
        """List one project's conversations, or standalone Quick Chats for None."""
        return [
            session
            for session in self.list_sessions_sorted()
            if session.project_id == project_id
        ]

    def list_sessions_sorted(self) -> list[ChatSession]:
        return sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)


session_manager = SessionManager()
