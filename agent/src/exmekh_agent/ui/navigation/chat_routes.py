"""Readable chat URLs with an immutable conversation identity."""

import re
import unicodedata
from urllib.parse import unquote, urlsplit

from exmekh_agent.exmekh_bridge.models import ChatSession
from exmekh_agent.state.session_manager import SessionManager


def _slug(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")[:80] or fallback


class ChatRoutes:
    """Translate URLs using the repository supplied by the calling page."""

    def __init__(self, sessions: SessionManager) -> None:
        self._sessions = sessions

    def path_for(self, session: ChatSession) -> str:
        project = self._sessions.get_project(session.project_id)
        project_name = _slug(project.name, "project") if project else "quick-chats"
        return f"/chat/{project_name}/{_slug(session.title, 'chat')}--{session.id}"

    def resolve(self, path: str) -> ChatSession | None:
        if not path.startswith("/chat/"):
            return None
        parts = unquote(urlsplit(path).path).split("/")
        if len(parts) != 4 or parts[1] != "chat" or not parts[2]:
            return None
        # Names can change and need not be unique; old bookmarks still use the same ID.
        name, separator, session_id = parts[3].rpartition("--")
        return self._sessions.get_session(session_id) if name and separator else None
