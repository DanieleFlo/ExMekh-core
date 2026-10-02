from __future__ import annotations

import asyncio
import os

from nicegui import ui

from fe.exmekh_bridge.collector import TurnStreamCollector
from fe.state.active_turns import active_turns
from fe.state.session_manager import session_manager
from fe.ui.chat_view import ChatViewComponent
from fe.ui.styles import load_css
from tests.interface_sample import interface_session


@ui.page("/")
def index():
    session = interface_session()
    session_manager.sessions[session.id] = session
    session_manager.current_session_id = session.id
    load_css()
    with ui.column().classes("w-full h-screen"):
        view = ChatViewComponent(lambda _: None)
        view.build()
        ui.button("Show child", on_click=lambda: view.show_sub_agent("child"))

        def start_partial_response():
            collector = TurnStreamCollector()
            collector.append_token("Partial response")
            task = asyncio.create_task(asyncio.sleep(60))
            active_turns.start(session.id, task, collector, started_at=0)
            view.start_assistant_stream()
            view.restore_stream_snapshot(collector.snapshot_message())

        ui.button("Start partial response", on_click=start_partial_response)


if __name__ == "__main__":
    ui.run(
        host="127.0.0.1",
        port=int(os.environ["FE_PORT"]),
        reload=False,
        show=False,
        title="Interface test",
    )
