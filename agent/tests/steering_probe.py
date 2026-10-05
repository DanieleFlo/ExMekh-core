"""Controlled provider for steering tests against the real ExMekh-Lib runtime."""

from __future__ import annotations

from threading import Event
from typing import Any
from xml.etree import ElementTree as ET

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
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage

from exmekh_agent.exmekh_bridge.history.converter import exmekh_message_to_raw_dict
from tests.mock_client import MockProvider


class SteeringProbeProvider(MockProvider):
    def __init__(
        self,
        *,
        first_response: str = "tool",
        block_second: bool = False,
        block_reply: bool = False,
        recover_steering: bool = False,
    ) -> None:
        super().__init__()
        self.first_response = first_response
        self.block_second = block_second
        self.block_reply = block_reply
        self.recover_steering = recover_steering
        self._initial_turn_complete = False
        self.started = Event()
        self.release = Event()
        self.second_started = Event()
        self.release_second = Event()
        self.requests: list[list[dict[str, Any]]] = []

    @staticmethod
    def _tool_response(
        call_id: str, name: str, arguments: dict[str, Any]
    ) -> ProviderResponse:
        return ProviderResponse(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        function=FunctionCall(name=name, arguments=arguments),
                    ),
                ],
            )
        )

    def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = [exmekh_message_to_raw_dict(item) for item in request.get_messages()]
        self.requests.append(messages)
        request_number = len(self.requests)
        if self.recover_steering and self._initial_turn_complete:
            return self._recover_steering(request, request_number)
        if request_number == 1:
            self.started.set()
            if not self.release.wait(timeout=30):
                raise TimeoutError("Steering test did not release the first inference")
            if self.first_response == "error":
                raise RuntimeError("Mock provider failed during inference")
            if self.first_response == "answer":
                return ProviderResponse(AssistantMessage("Initial answer"))
            opens_skill = self.first_response in ("skill", "nested_skill")
            return self._tool_response(
                "steering-probe",
                "mock-skill" if opens_skill else "bash",
                {} if opens_skill else {"command": "echo steering_probe"},
            )
        if request_number == 2 and (self.block_second or self.block_reply):
            self.second_started.set()
            if not self.release_second.wait(timeout=30):
                raise TimeoutError("Steering test did not release the second inference")
            if self.block_second:
                return ProviderResponse(AssistantMessage("Intermediate answer"))
        if request_number == 2 and self.first_response == "nested_skill":
            return self._tool_response("open-nested-steering-skill", "nested-skill", {})
        if request_number == 3 and self.first_response == "nested_skill":
            return self._tool_response(
                "nested-steering-tool",
                "nested-sub-tool",
                {"text": "Guided nested result"},
            )
        if (request_number == 2 and self.first_response == "skill") or (
            request_number in (4, 5) and self.first_response == "nested_skill"
        ):
            return self._tool_response(
                f"close-steering-skill-{request_number}",
                "exit_agent",
                {"text": "Mock skill output", "items": []},
            )
        prompt = next(
            item["content"] for item in reversed(messages) if item["role"] == "user"
        )
        self._initial_turn_complete = True
        reply = f"Reply following guidance: {prompt}"
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, reply)
        )
        return ProviderResponse(AssistantMessage(reply))

    def _recover_steering(
        self, request: ProviderRequest, request_number: int
    ) -> ProviderResponse:
        messages = request.get_messages()
        last = messages[-1]
        if isinstance(last, UserMessage) and not last.has_attachments():
            return self._tool_response(
                f"inspect-history-{request_number}",
                "get_execution_history",
                {},
            )
        result = next(
            item for item in reversed(messages) if isinstance(item, ToolMessage)
        )
        if result.get_name() == "get_execution_history":
            history = ET.fromstring(result.get_message())
            entries = [
                item
                for item in history.findall("call")
                if item.findtext("name") == "steering_message"
            ]
            assert len(entries) == 1, (
                "Saved steering missing or duplicated in ExMekh Agent history"
            )
            return self._tool_response(
                f"recover-steering-{request_number}",
                "read_tool_output",
                {"output_id": entries[0].findtext("saved-output-id")},
            )
        assert result.get_name() == "read_tool_output"
        attachments = result.get_attachments()
        images = (
            sum(not isinstance(part, str) for part in attachments.iter_content_parts())
            if attachments is not None
            else 0
        )
        reply = f"Recovered steering with {images} images: {result.get_message()}"
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, reply)
        )
        return ProviderResponse(AssistantMessage(reply))
