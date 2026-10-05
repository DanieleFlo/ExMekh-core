from __future__ import annotations

import logging

from nicegui import app, ui

from exmekh_agent.config import config
from exmekh_agent.config.server import DEFAULTS as SERVER_DEFAULTS
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.shared.file_upload import upload_router
from exmekh_agent.ui.styles import load_css

logging.basicConfig(level=logging.INFO, format=SERVER_DEFAULTS["logging_format"])
app.include_router(upload_router)


@ui.page("/")
def index_page() -> None:
    load_css()
    ui.dark_mode().enable()
    ChatPageController(ui.context.client).build()


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
