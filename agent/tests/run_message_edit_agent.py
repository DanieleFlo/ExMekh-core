"""Browser server exercising real ExMekh-Lib clients with a deterministic mock provider."""

from __future__ import annotations

import asyncio
import time
from threading import Event
from typing import Any

from exmekh_lib.core.agents.node import DefaultCapabilities
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge import agent_runner
from exmekh_agent.exmekh_bridge.converter import exmekh_message_to_raw_dict
from exmekh_agent.exmekh_bridge.image_input import content_text
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from exmekh_agent.ui import settings_drawer
from tests.mock_client import MockClient, MockProvider


class MessageEditProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.release = Event()

    def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = [exmekh_message_to_raw_dict(item) for item in request.get_messages()]
        last_user_index = max(
            i for i, item in enumerate(messages) if item["role"] == "user"
        )
        user_content = messages[last_user_index]["content"]
        prompt = content_text(user_content)
        call = {
            "prompt": prompt,
            "history": [
                content_text(item.get("content"))
                for item in messages[1:last_user_index]
            ],
            "messages": messages,
            "image_count": sum(part["type"] == "image_url" for part in user_content)
            if isinstance(user_content, list)
            else 0,
            "cancelled": False,
        }
        self.calls.append(call)
        if prompt == "Await approval" and messages[-1]["role"] == "user":
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            id="edit-approval",
                            function=FunctionCall(
                                name="bash",
                                arguments={
                                    "command": "echo edit_approval",
                                    "wait_seconds": 60,
                                },
                            ),
                        )
                    ],
                )
            )
        reply = "Reply: " + prompt
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, reply)
        )
        if prompt == "Slow original":
            self.release.clear()
            deadline = time.monotonic() + 15
            try:
                while not self.release.wait(0.05):
                    # Exercise cancellation through the provider's public stream API.
                    request.execute_on_stream_event(
                        ProviderStreamEvent(ProviderStreamEventType.RESPONSE, "")
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Browser test did not finish the slow request"
                        )
            except asyncio.CancelledError:
                call["cancelled"] = True
                raise
        return ProviderResponse(AssistantMessage(reply))


provider = MessageEditProvider()
agent_runner.LocalProvider = lambda *args, **kwargs: provider
agent_runner.Client = MockClient
config.default_model = "edit-test-model"
config.reasoning_effort = "none"
config.capabilities_path = str(DefaultCapabilities(mocks=True).capabilities_path)
config.work_root = str(config.data_dir / "work")


async def health(*args: Any, **kwargs: Any) -> tuple[bool, list[str]]:
    return True, [config.default_model]


settings_drawer.check_provider_health = health
agent_runner.check_provider_health = health


@app.get("/test/state")
def state() -> dict[str, Any]:
    return {
        "session": session_manager.get_current_session().model_dump(mode="json"),
        "calls": provider.calls,
    }


@app.post("/test/release-inference")
def release_inference() -> dict[str, bool]:
    provider.release.set()
    return {"released": True}


if __name__ == "__main__":
    config.data_dir.joinpath("work").mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
