"""Run a streaming quick chat ahead of a project chat with a mock tool."""

from __future__ import annotations

import asyncio
from threading import Event

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.exmekh_bridge.providers.context_usage import (
    compute_session_context_usage,
)
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui.chat import context_updates
from exmekh_agent.ui.chat.page_controller import ChatPageController
from exmekh_agent.ui.settings import settings_drawer
from tests.capability_fixtures import TEST_CAPABILITIES
from tests.mock_client import MockProvider

gates = {
    name: Event()
    for name in ("quick-progress", "quick-finish", "project-finish", "project-count")
}
project_started = Event()
project_counting = Event()
quick_progressed = Event()
navigation_ready = asyncio.Event()
navigation_ready.set()
handle_navigation = ChatPageController._handle_chat_navigation


def wait_for_release(name: str) -> None:
    if not gates[name].wait(timeout=30):
        raise TimeoutError(f"Browser did not release {name}")


class OverlapProvider(MockProvider):
    def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = request.get_messages()
        prompt = next(
            msg.get_message()
            for msg in reversed(messages)
            if isinstance(msg, UserMessage)
        )
        if prompt == "Quick question":
            start = "Quick reply started. " + "quick " * 1400
            progress = "Quick reply continued. " + "quick " * 1400
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.RESPONSE, start)
            )
            wait_for_release("quick-progress")
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.RESPONSE, progress)
            )
            quick_progressed.set()
            wait_for_release("quick-finish")
            return ProviderResponse(
                AssistantMessage(start + progress), usage=ModelUsage(4200, 2800, 7000)
            )

        assert prompt == "Project question"
        project_started.set()
        if isinstance(messages[-1], ToolMessage):
            assert messages[-1].get_name() == "mock-tool"
            assert "Mock tool result: project only" in messages[-1].get_message()
            wait_for_release("project-finish")
            return ProviderResponse(
                AssistantMessage("Project tool verified"),
                usage=ModelUsage(2300, 100, 2400),
            )
        return ProviderResponse(
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="project-only-tool",
                        function=FunctionCall("mock-tool", {"text": "project only"}),
                    )
                ],
            )
        )


def count_context(**kwargs):
    if kwargs["session"].project_id is not None:
        project_counting.set()
        wait_for_release("project-count")
    return compute_session_context_usage(**kwargs)


@app.get("/test/overlap")
def overlap():
    return {
        "project_started": project_started.is_set(),
        "project_counting": project_counting.is_set(),
        "quick_progressed": quick_progressed.is_set(),
    }


@app.post("/test/release/{name}")
def release(name: str):
    gates[name].set()
    return {"released": True}


@app.post("/test/navigation/{ready}")
async def set_navigation_ready(ready: bool):
    if ready:
        navigation_ready.set()
    else:
        navigation_ready.clear()
    return {"ready": ready}


async def gated_navigation(controller, event):
    await navigation_ready.wait()
    handle_navigation(controller, event)


async def health(*args, **kwargs):
    return True, ["overlap-probe"]


agent_runner.LocalProvider = lambda *args, **kwargs: OverlapProvider()
ChatPageController._handle_chat_navigation = gated_navigation
agent_runner.generate_chat_title_sync = lambda prompt, *args: prompt
settings_drawer.check_provider_health = health
context_updates.compute_session_context_usage = count_context
config.default_model = "overlap-probe"
config.reasoning_effort = "none"
config.capabilities_path = str(TEST_CAPABILITIES)
config.name_hds = "main"

if __name__ == "__main__":
    quick = session_manager.get_current_session()
    session_manager.update_title(quick, "Quick question")
    work_root = config.data_dir / "Overlap project"
    work_root.mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(work_root)
    session_manager.create_session("Project question", project_id=project.id)
    session_manager.switch_session(quick.id)
    main()
