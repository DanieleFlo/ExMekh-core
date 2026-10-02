from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fe.config import config
from fe.exmekh_bridge.models import (
    ChatMessage,
    ChatSession,
    ChatSettings,
    Project,
    ProjectSettings,
)
from fe.state.active_turns import active_turns
from fe.state.json_store import JsonRecordStore
from fe.state.temp_storage import TempStorage

logger = logging.getLogger(__name__)
CODE_WORK_ROOT = Path(__file__).resolve().parents[3]


class SessionManager:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self._initialize_storage(storage_dir or config.data_dir / "chats")

    def _initialize_storage(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.projects_dir = self.storage_dir.parent / "projects"
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
            raise ValueError("Select an existing project folder")
        if any(Path(project.work_root) == root for project in self.projects.values()):
            raise ValueError("A project already uses this folder")
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
        project.settings = updated.settings
        for session in self.list_sessions(project_id):
            session.settings = self.execution_settings(session)
        return True

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
        previous_mode = session.settings.tool_approval_mode
        session.settings.set_tool_approval_mode(mode)
        if self.save_session(session):
            return True
        session.settings.set_tool_approval_mode(previous_mode)
        return False

    @staticmethod
    def temp_dirname(session: ChatSession) -> str:
        return (
            f"tmp/projects/{session.project_id}"
            if session.project_id
            else "tmp/quickchats"
        )

    def create_session(
        self, title: str = "New Conversation", project_id: str | None = None
    ) -> ChatSession:
        project = self.get_project(project_id)
        if project_id is not None and project is None:
            raise ValueError("Project was not found")
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
        self.current_session_id = session.id
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
            if self.sessions:
                self.current_session_id = next(iter(self.sessions))
            else:
                return self.create_session()
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
        session.title = title
        return self.save_session(session)

    def record_command_approval(
        self, session: ChatSession, command: str, raw_history: list[dict[str, str]]
    ) -> bool:
        if not self.owns_session(session):
            return False
        session.settings.record_command_approval(command, raw_history)
        return self.save_session(session)

    def switch_session(self, session_id: str) -> ChatSession | None:
        if session_id in self.sessions:
            self.current_session_id = session_id
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
            logger.error("Failed to save session %s: %s", session.id, error)
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
                "Could not persist project migration for %s: %s", session.id, error
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
