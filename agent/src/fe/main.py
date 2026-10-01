from __future__ import annotations

import logging

from nicegui import app, ui

from fe.config import config
from fe.ui.file_upload import upload_router
from fe.ui.page_controller import ChatPageController
from fe.ui.styles import load_css

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
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
