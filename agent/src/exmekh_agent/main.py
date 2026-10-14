from __future__ import annotations

import logging

from fastapi import HTTPException
from nicegui import app, ui

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.server import DEFAULTS as SERVER_DEFAULTS
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from exmekh_agent.ui.shared.file_upload import upload_router
from exmekh_agent.ui.styles import load_css

logging.basicConfig(level=logging.INFO, format=SERVER_DEFAULTS["logging_format"])
app.include_router(upload_router)


@ui.page("/")
def index_page() -> None:
    _build_chat_page(session_manager.for_view())


def _build_chat_page(sessions: SessionManager) -> None:
    load_css()
    ui.dark_mode().enable()
    ChatPageController(ui.context.client, sessions=sessions).build()


@ui.page("/chat/{project_name}/{chat_name}")
def chat_page(project_name: str, chat_name: str) -> None:
    session = ChatRoutes(session_manager).resolve(f"/chat/{project_name}/{chat_name}")
    if session is None:
        raise HTTPException(
            status_code=404, detail=app_text("navigation.chat_not_found")
        )
    _build_chat_page(session_manager.for_view(session.id))


def run() -> None:
    ui.run(
        host=config.host,
        port=config.port,
        title=config.app_title,
        dark=True,
        reload=False,
        show=False,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
