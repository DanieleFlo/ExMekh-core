"""Deterministic provider for exercising the real ExMekh-Lib command approval flow."""

from __future__ import annotations

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

from tests.mock_client import MockProvider


class ApprovalProbeProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.request_count = 0

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.request_count += 1
        if self.request_count == 1:
            request.execute_on_stream_event(
                ProviderStreamEvent(
                    ProviderStreamEventType.RESPONSE, "Output before approval"
                )
            )
            return ProviderResponse(
                AssistantMessage(
                    content="Output before approval",
                    tool_calls=[
                        ToolCall(
                            id="approval-probe",
                            function=FunctionCall(
                                name="bash",
                                arguments={
                                    "command": "echo approval_probe",
                                    "wait_seconds": 60,
                                },
                            ),
                        )
                    ],
                )
            )
        reply = "Reply after declining the command"
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, f"\n\n{reply}")
        )
        return ProviderResponse(AssistantMessage(reply))
