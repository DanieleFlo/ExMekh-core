"""Browser test server using the real ExMekh-Lib runtime and a controlled provider."""

from __future__ import annotations

from pathlib import Path

from exmekh_lib.core.agents.node import DefaultCapabilities
from nicegui import app

from fe.config import config
from fe.exmekh_bridge import agent_runner
from fe.main import main
from fe.state.session_manager import session_manager
from tests.approval_probe import ApprovalProbeProvider

provider = ApprovalProbeProvider()
agent_runner.LocalProvider = lambda *args, **kwargs: provider
config.agent_mode = "deep_agent"
config.default_model = "approval-probe"
config.reasoning_effort = "none"
config.work_root = str(Path(__file__).resolve().parents[1])
config.capabilities_path = str(DefaultCapabilities(mocks=True).capabilities_path)
config.name_hds = "main"


@app.get("/test/approval-stats")
def approval_stats() -> dict[str, int]:
    return {"requests": provider.request_count}


if __name__ == "__main__":
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
