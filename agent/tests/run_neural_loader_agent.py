from __future__ import annotations

import os

from nicegui import ui

from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.styles import load_css


@ui.page("/")
def index():
    load_css()
    ui.dark_mode().enable()
    with ui.column().classes("w-full max-w-4xl mx-auto"):
        bar = InputBarComponent(
            on_send=lambda _: None, on_stop=lambda: bar.set_busy(False)
        )
        bar.build()
        ui.button("Start generation", on_click=lambda: bar.set_busy(True))
        ui.button("Complete generation", on_click=lambda: bar.set_busy(False))
        ui.button("Remove composer", on_click=bar.root_container.delete)


if __name__ == "__main__":
    ui.run(
        host="127.0.0.1",
        port=int(os.environ["EXMEKH_AGENT_PORT"]),
        reload=False,
        show=False,
        title="Neural loader test",
    )
