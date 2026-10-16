"""Isolated NiceGUI application with the real node runtime and mock inference."""

from __future__ import annotations

from threading import Event
from typing import Any

from exmekh_lib.core.agents.node import DefaultCapabilities
from nicegui import app

from exmekh_agent.config import application, config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.graph.graph_data import build_graph_data_from_session
from tests.compaction_probe import CompactionProbeProvider

provider = CompactionProbeProvider(rounds=3)
compaction_gate = Event()
compaction_gate.set()
compaction_waiting = Event()


def pause_summary() -> None:
    compaction_waiting.set()
    try:
        if not compaction_gate.wait(20):
            raise TimeoutError("Compaction test did not release the summary")
    finally:
        compaction_waiting.clear()


provider.on_score = pause_summary
agent_runner.LocalProvider = lambda **kwargs: provider
agent_runner.generate_chat_title_sync = lambda *args, **kwargs: "Compaction mock"
config.default_model = "compaction-probe"
config.reasoning_effort = "none"
config.capabilities_path = str(DefaultCapabilities(mocks=True).capabilities_path)
config.work_root = str(config.data_dir / "work")
config.base_url = f"http://{config.host}:{config.port}/provider/v1"
config.auto_compaction_threshold_percent = 80
application.DEFAULT_ENV_PATH = config.data_dir / "settings.env"


@app.get("/provider/v1/models")
def models() -> dict[str, Any]:
    return {"data": [{"id": "compaction-probe", "context_length": 100000}]}


@app.get("/test/state")
def state() -> dict[str, Any]:
    session = session_manager.get_current_session()
    return {
        "sessions": [
            item.model_dump(mode="json") for item in session_manager.sessions.values()
        ],
        "session": session.model_dump(mode="json"),
        "graph": build_graph_data_from_session(session).model_dump(mode="json"),
        "calls": [
            [m.message_dump() for m in request.get_messages()]
            for request in provider.requests
        ],
        "compactions": len(provider.score_requests),
        "threshold": config.auto_compaction_threshold_percent,
        "compaction_waiting": compaction_waiting.is_set(),
    }


@app.post("/test/pause-compaction")
def pause_compaction() -> None:
    compaction_gate.clear()


@app.post("/test/resume-compaction")
def resume_compaction() -> None:
    compaction_gate.set()


@app.post("/test/summary-error/{enabled}")
def summary_error(enabled: bool) -> None:
    provider.summary_error = (
        RuntimeError("Mock compaction unavailable") if enabled else None
    )


if __name__ == "__main__":
    config.data_dir.joinpath("work").mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
