"""Two independent requests through the actual serial provider and steering path."""

from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.capability_fixtures import TEST_CAPABILITIES
from tests.steering_probe import SteeringProbeProvider

providers = [
    SteeringProbeProvider(block_reply=True),
    SteeringProbeProvider(first_response="answer"),
]
created = 0


def create_provider(*args, **kwargs):
    global created
    provider = providers[created]
    created += 1
    return provider


agent_runner.LocalProvider = create_provider
agent_runner.generate_chat_title_sync = lambda *args, **kwargs: "Finished request"
config.agent_mode = "deep_agent"
config.default_model = "concurrent-probe"
config.reasoning_effort = "none"
config.capabilities_path = str(TEST_CAPABILITIES)
config.name_hds = "main"


@app.get("/test/concurrent-state")
def state():
    return {
        "started": [provider.started.is_set() for provider in providers],
        "requests": [provider.requests for provider in providers],
        "session": session_manager.get_current_session().model_dump(mode="json"),
    }


@app.post("/test/release/{index}/{stage}")
def release(index: int, stage: str):
    (
        providers[index].release_second
        if stage == "reply"
        else providers[index].release
    ).set()
    return {"released": True}


async def health(*args, **kwargs):
    return True, ["concurrent-probe"]


agent_runner.check_provider_health = health

if __name__ == "__main__":
    work_root = config.data_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(work_root)
    session_manager.create_session(project_id=project.id)
    main()
