"""Browser server for steering through the real ExMekh-Lib agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from exmekh_lib.core.agents.node import DefaultCapabilities
from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.steering_probe import SteeringProbeProvider

provider = SteeringProbeProvider(
    first_response=os.getenv("EXMEKH_AGENT_STEERING_FIRST_RESPONSE", "tool"),
    block_reply=True,
    recover_steering=os.getenv("EXMEKH_AGENT_STEERING_RECOVER_OUTPUT") == "1",
)
agent_runner.LocalProvider = lambda *args, **kwargs: provider
config.agent_mode = "deep_agent"
config.default_model = "steering-probe"
config.reasoning_effort = "none"
config.work_root = str(Path(__file__).resolve().parents[1])
config.capabilities_path = str(DefaultCapabilities(mocks=True).capabilities_path)
config.name_hds = "main"


@app.get("/test/steering-stats")
def steering_stats() -> dict[str, Any]:
    return {
        "requests": provider.requests,
        "session": session_manager.get_current_session().model_dump(),
    }


@app.post("/test/release-inference")
def release_inference() -> dict[str, bool]:
    provider.release.set()
    return {"released": True}


@app.post("/test/release-reply")
def release_reply() -> dict[str, bool]:
    provider.release_second.set()
    return {"released": True}


if __name__ == "__main__":
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
