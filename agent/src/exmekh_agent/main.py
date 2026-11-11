from __future__ import annotations

import logging

from fastapi import HTTPException
from nicegui import app, ui

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.server import DEFAULTS as SERVER_DEFAULTS
from exmekh_agent.exmekh_bridge.providers.core_management import (
    Connection,
    connect_core,
)
from exmekh_agent.state.session_manager import SessionManager, session_manager
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.navigation.chat_routes import ChatRoutes
from exmekh_agent.ui.resources import render_html
from exmekh_agent.ui.settings.core_navigation import (
    SETTINGS,
    STEPS,
    gate,
    route,
)
from exmekh_agent.ui.settings.setup_pages import setup_page, startup_page
from exmekh_agent.ui.settings.system_pages import system_settings
from exmekh_agent.ui.shared.file_upload import upload_router
from exmekh_agent.ui.styles import load_css

logging.basicConfig(level=logging.INFO, format=SERVER_DEFAULTS["logging_format"])
app.include_router(upload_router)


@ui.page("/")
async def index_page() -> None:
    await _enter_chat()


async def _enter_chat(session_id: str | None = None, destination: str = "/") -> None:
    connection = await connect_core()
    if target := gate(connection, destination):
        ui.navigate.to(target)
        return
    _build_chat_page(session_manager.for_view(session_id), connection)


def _build_chat_page(sessions: SessionManager, connection: Connection) -> None:
    load_css()
    ui.add_body_html(
        render_html(
            "shared/startup_skeleton.html",
            label=app_text("chat.page_controller.loading_conversation"),
        )
    )
    ui.dark_mode().enable()
    client = ui.context.client
    controller = ChatPageController(client, sessions=sessions)
    controller.follow_core(
        connection.status.ready_profile if connection.status else None
    )
    controller.build()

    def remove_startup_skeleton() -> None:
        client.run_javascript(
            "requestAnimationFrame(() => document.getElementById('exmekh-startup')?.remove())"
        )

    client.on_connect(remove_startup_skeleton)
    if connection.status or connection.detail:

        async def refresh_core() -> None:
            session = sessions.get_current_session()
            destination = ChatRoutes(sessions).path_for(session)
            current = await connect_core()
            target = (
                route("/startup", destination, auto=1)
                if connection.status and current.kind == "unavailable"
                else gate(current, destination)
            )
            if target:
                controller.suspend_for_core(target)
                timer.cancel()
                return
            controller.follow_core(
                current.status.ready_profile if current.status else None
            )

        timer = ui.timer(2.0, refresh_core, immediate=False)


@ui.page("/chat/{project_name}/{chat_name}")
async def chat_page(project_name: str, chat_name: str) -> None:
    destination = f"/chat/{project_name}/{chat_name}"
    session = ChatRoutes(session_manager).resolve(destination)
    if session is None:
        raise HTTPException(
            status_code=404, detail=app_text("navigation.chat_not_found")
        )
    await _enter_chat(session.id, destination)


@ui.page("/settings")
def settings_index() -> None:
    ui.navigate.to("/settings/server")


@ui.page("/settings/{section}")
async def settings_page(section: str, return_to: str = "/", setup: int = 0) -> None:
    if section not in SETTINGS:
        raise HTTPException(status_code=404)
    await system_settings(section, return_to, setup)


@ui.page("/setup/{step}")
async def configure_page(
    step: str,
    return_to: str = "/",
    model: str = "",
    mode: str = "",
    tokens: int = 0,
    features: str = "",
) -> None:
    if step not in STEPS:
        raise HTTPException(status_code=404)
    await setup_page(step, return_to, model, mode, tokens, features)


@ui.page("/startup")
async def loading_page(return_to: str = "/", job: str = "", auto: int = 0) -> None:
    await startup_page(return_to, job, auto)


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
