"""Deterministic provider using real clients, node loops, scoring and summaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)

from tests.mock_client import MockProvider


class CompactionProbeProvider(MockProvider):
    def __init__(self, rounds: int = 2, *, usage_tokens: int = 85000) -> None:
        super().__init__()
        self.rounds = rounds
        self.usage_tokens = usage_tokens
        self.requests: list[Any] = []
        self.score_requests: list[Any] = []
        self.summary_requests: list[Any] = []
        self.summary_error: Exception | None = None
        self.on_score: Callable[[], None] | None = None
        self.on_request: Callable[[Any], None] | None = None
        self._rounds_by_user: dict[str, int] = {}
        self.script: list[AssistantMessage] | None = None

    def call(self, request):
        if request.get_reasoning_effort_override() == "none":
            self.summary_requests.append(request)
            if self.summary_error:
                raise self.summary_error
            return ProviderResponse(
                AssistantMessage("Preserved result and pending work.")
            )
        self.requests.append(request)
        if self.on_request:
            self.on_request(request)
        user = next(
            str(item.get_message())
            for item in reversed(request.get_messages())
            if item.get_role_name() == "user"
        )
        round_number = self._rounds_by_user.get(user, 0)
        self._rounds_by_user[user] = round_number + 1
        if self.script is not None:
            assert self.script, "Unexpected extra model inference"
            message = self.script.pop(0)
            if message.get_message():
                request.execute_on_stream_event(
                    ProviderStreamEvent(
                        ProviderStreamEventType.RESPONSE, message.get_message()
                    )
                )
        elif round_number < self.rounds:
            message = AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id=f"probe-{len(self.requests)}",
                        function=FunctionCall(
                            "mock-tool",
                            {
                                "text": f"ORIGINAL_{len(self.requests)} "
                                + "detailed evidence " * 80
                            },
                        ),
                    )
                ],
            )
        else:
            text = "COMPACTION_OK: " + user
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.RESPONSE, text)
            )
            message = AssistantMessage(text)
        return ProviderResponse(
            message,
            usage=ModelUsage(prompt_tokens=self.usage_tokens, completion_tokens=20),
        )

    def call_structured(self, request):
        self.score_requests.append(request)
        if self.on_score:
            self.on_score()
        model = request.get_response_model()
        return StructuredProviderResponse(
            StructuredAssistantMessage(
                model.model_validate({name: 30 for name in model.model_fields})
            )
        )
