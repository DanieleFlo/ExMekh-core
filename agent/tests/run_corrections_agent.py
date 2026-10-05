"""Real library execution with inference gates for live-output regression checks."""

from __future__ import annotations

import time
from threading import Event

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from nicegui import app

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.run_message_edit_agent import MessageEditProvider


class LiveOutputProvider(MessageEditProvider):
    def __init__(self):
        super().__init__()
        self.phase = 0
        self.skill_gate = Event()
        self.final_gate = Event()
        self.summary_calls = 0

    def call(self, request: ProviderRequest) -> ProviderResponse:
        scope = (request.get_extra() or {}).get("agentic")
        if request.get_reasoning_effort_override() == "none" and not scope:
            self.summary_calls += 1
            return ProviderResponse(
                AssistantMessage("THINKING SUMMARY: verify elapsed hours."),
                usage=ModelUsage(50, 10, 60),
            )
        self.phase += 1
        if self.phase == 1:
            return self.tool("deep-thinking", "open")
        if self.phase == 2:
            thought = (
                "Verify the exact elapsed hours before accepting the answer. " * 250
            )
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.REASONING, thought)
            )
            return self.tool(
                "date-time",
                "clock",
                thought=thought,
                operation="difference",
                value="2026-10-01",
                other="2026-10-02",
                timezone="UTC",
                unit="hours",
            )
        if self.phase == 3:
            self.wait_stream(
                request, self.skill_gate, "Inspecting the calculation result. "
            )
            return self.tool(
                "exit_agent", "exit", text="Verified 24 elapsed hours", items=[]
            )
        if self.phase == 4:
            self.wait_stream(
                request, self.final_gate, "Preparing the final explanation. "
            )
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.RESPONSE, "FINAL ANSWER")
            )
            return ProviderResponse(
                AssistantMessage("FINAL ANSWER"), usage=ModelUsage(6000, 100, 6100)
            )
        raise AssertionError(f"Unexpected inference phase {self.phase}")

    def tool(self, name, identifier, thought=None, **arguments):
        return ProviderResponse(
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(id=identifier, function=FunctionCall(name, arguments))
                ],
            ),
            thinking=ModelThinking(thought) if thought else None,
            usage=ModelUsage(3000 + self.phase * 100, 100, 3100 + self.phase * 100),
        )

    def wait_stream(self, request, gate, chunk):
        deadline = time.monotonic() + 30
        while not gate.wait(0.1):
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.REASONING, chunk * 10)
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("Browser did not release inference gate")


provider = LiveOutputProvider()
agent_runner.LocalProvider = lambda *args, **kwargs: provider
config.reasoning_effort = "medium"


@app.get("/test/live")
def live():
    return {"phase": provider.phase, "summaries": provider.summary_calls}


@app.post("/test/release-skill")
def release_skill():
    provider.skill_gate.set()
    return {"released": True}


@app.post("/test/release-final")
def release_final():
    provider.final_gate.set()
    return {"released": True}


if __name__ == "__main__":
    config.data_dir.joinpath("work").mkdir(parents=True, exist_ok=True)
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
