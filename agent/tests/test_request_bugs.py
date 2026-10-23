"""Regressions for concurrent requests and the complete native image pipeline."""

import asyncio
import base64
import io
import json
import time
from dataclasses import replace
from unittest.mock import patch

import httpx2
import pytest
from exmekh_lib.core.models import tool as tool_model
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.native_suite import NativeSuiteClient
from exmekh_lib.core.models.provider_response import ProviderResponse
from nicegui.client import Client
from nicegui.page import page
from openai import OpenAI
from PIL import Image

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents.agent_runner import execute_agent_turn_sync
from exmekh_agent.exmekh_bridge.agents.collector import TurnStreamCollector
from exmekh_agent.exmekh_bridge.history.pasted_input import PreparedUserInput
from exmekh_agent.exmekh_bridge.models import ChatMessage, ToolEvent
from exmekh_agent.state.active_turns import active_turns
from exmekh_agent.state.session_manager import SessionManager
from exmekh_agent.ui.chat.chat_view import ChatViewComponent
from exmekh_agent.ui.chat.input_bar import InputBarComponent
from exmekh_agent.ui.chat.page_controller import ChatPageController
from tests.mock_client import MockProvider


class ImageProvider(MockProvider):
    def __init__(self, count=1):
        super().__init__()
        self.count = count
        self.generating = False
        self.requests = []
        self.native_requests = []
        self.transport = OpenAI(
            api_key="test",
            base_url="http://native/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.respond)),
        )

    def respond(self, request):
        self.generating = True
        time.sleep(0.05)
        body = json.loads(request.content)
        self.native_requests.append(body)
        width, height = map(int, body["size"].split("x"))
        output = io.BytesIO()
        Image.new("RGB", (width, height), body["prompt"]).save(output, "PNG")
        self.generating = False
        return httpx2.Response(
            200,
            json={
                "created": 1,
                "ds4": {},
                "data": [
                    {
                        "b64_json": base64.b64encode(output.getvalue()).decode(),
                        "ds4": {
                            "file_id": "file-image",
                            "seed": 1,
                            "width": width,
                            "height": height,
                            "recipe": {},
                        },
                    }
                ],
            },
        )

    def native_suite(self):
        return NativeSuiteClient(self.transport)

    def call(self, request):
        assert not self.generating, (
            "Inference started before native service restoration"
        )
        self.requests.append(request.get_messages())
        if len(self.requests) == 1:
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            f"image-{index}",
                            FunctionCall(
                                "generate-image",
                                {
                                    "prompt": ["red", "blue"][index],
                                    "width": 2048 if self.count == 2 else 64,
                                    "height": 1536 if self.count == 2 else 32,
                                },
                            ),
                        )
                        for index in range(self.count)
                    ],
                )
            )
        return ProviderResponse(AssistantMessage("Images ready"))


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_native_generation_saves_and_delivers_each_image(
    tmp_path, monkeypatch, include, count
):
    monkeypatch.setattr(config, "include_generated_images_in_context", include)
    # A native image result must never be replaced by an initial wait receipt.
    monkeypatch.setattr(
        tool_model,
        "RUNTIME",
        replace(tool_model.RUNTIME, default_tool_wait_seconds=0.001),
    )
    provider = ImageProvider(count)
    collector = TurnStreamCollector()
    try:
        with patch(
            "exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider",
            return_value=provider,
        ):
            execute_agent_turn_sync(
                user_prompt="Generate images",
                model="test",
                base_url="http://native/v1",
                api_key="test",
                api_type="responses",
                reasoning_effort_str="none",
                agent_mode="quick_chat",
                capabilities_path_str="",
                name_hds="main",
                work_root=str(tmp_path),
                on_token=collector.append_token,
                on_reasoning=collector.append_reasoning,
                on_tool_event=collector.append_tool,
                on_raw_history=collector.set_raw_history,
            )
        assert len(provider.native_requests) == count
        assert len(list(tmp_path.rglob("*.png"))) == count
        message = collector.build_message()
        assert len(message.image_gallery()) == count, message.model_dump()
        assert all(tool.image_state == "completed" for tool in message.tool_events)
        assert all(tool.status != "error" for tool in message.tool_events)
        assert all(tool.images_in_context is include for tool in message.tool_events)
        tools = [
            message
            for message in provider.requests[-1]
            if message.get_role_name() == "tool"
        ]
        assert len(tools) == count
        assert all(
            "<response-type>final</response-type>" in tool.get_message()
            for tool in tools
        ), [tool.get_message() for tool in tools]
        assert [bool(tool.get_attachments()) for tool in tools] == [include] * count
    finally:
        provider.transport.close()


async def test_switching_between_owned_turns_restores_the_visible_stream(tmp_path):
    manager = SessionManager(tmp_path / "chats")
    first = manager.get_current_session()
    second = manager.create_session("Second")
    client = Client(page("/concurrent-owned-turns"))
    release = asyncio.Event()

    async def run(**kwargs):
        await release.wait()

    try:
        with (
            client,
            patch.object(ChatPageController, "_run_turn", side_effect=run),
            patch.object(ChatPageController, "update_context_usage"),
        ):
            controller = ChatPageController(client, sessions=manager)
            controller.chat_view = ChatViewComponent(lambda _: None, sessions=manager)
            controller.chat_view.build()
            controller.input_bar = InputBarComponent(lambda _: None, lambda: None)
            controller.input_bar.build()
            for session in [first, second]:
                controller.session_switched(session.id)
                assert not controller.input_bar.is_busy
                controller._send_prepared(
                    session, PreparedUserInput(session.title, None, session.title)
                )
                assert controller.chat_view.is_streaming_session(session.id)
                assert controller.chat_view.chat_container.visible
            assert len(controller._active_tasks) == 2
            for session in [first, second, first]:
                controller.session_switched(session.id)
                assert controller.chat_view.is_streaming_session(session.id)
                assert controller.chat_view.chat_container.visible
                assert controller.input_bar.is_busy
            accepted = []
            collector = active_turns.snapshot(first.id, after_revision=-1)
            assert collector is not None
            # Steering stays attached to the first collector after opening the second chat.
            turn_collector = TurnStreamCollector()
            turn_collector.steering.bind(accepted.append, history_start=2)
            active_turns.start(
                first.id,
                controller._active_tasks[first.id],
                turn_collector,
                started_at=0,
            )
            queued = first.add_queued_message("First correction")
            controller.steer_queued_message(queued.id)
            assert accepted == ["First correction"]
            assert not second.queued_messages
    finally:
        release.set()
        if "controller" in locals():
            await asyncio.gather(*controller._active_tasks.values())
        active_turns.cancel_all()
        client.delete()


async def test_attach_generated_image_stages_a_draft_without_sending(tmp_path):
    manager = SessionManager(tmp_path / "chats")
    client = Client(page("/attach-generated"))
    try:
        with client:
            controller = ChatPageController(client, sessions=manager)
            controller.input_bar = InputBarComponent(
                lambda _: None,
                lambda: None,
                temp_storage_getter=controller._get_temp_storage,
            )
            controller.input_bar.build()
            session = manager.get_current_session()
            controller.input_bar.set_session_id(session.id)
            output = io.BytesIO()
            Image.new("RGB", (64, 32), "red").save(output, "PNG")
            url = (
                "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
            )
            await controller.attach_generated_image(url)
            assert len(controller.input_bar.staged_attachments) == 1
            assert controller.input_bar.staged_attachments[0].data == output.getvalue()
            assert not session.messages
            assert not active_turns.is_running(session.id)
            await ChatPageController(client).attach_generated_image(url)
    finally:
        client.delete()


@pytest.mark.parametrize("saved_policy", [None, False, True])
@pytest.mark.parametrize("has_attachments", [False, True])
def test_saved_image_delivery_policy_and_legacy_history(saved_policy, has_attachments):
    tool = ToolEvent(
        tool_name="generate-image",
        tool_call_id="image",
        images_in_context=saved_policy,
        image_urls=["original"],
    )
    message = ChatMessage(
        role="assistant",
        tool_events=[tool],
        raw_history=[
            {
                "role": "tool",
                "tool_call_id": "image",
                "attachments": [{"type": "image_url", "image_url": {"url": "preview"}}]
                if has_attachments
                else None,
            }
        ],
    )
    restored = ChatMessage.model_validate_json(message.model_dump_json())
    assert restored.has_model_images_for_tool(tool) is (
        has_attachments if saved_policy is None else saved_policy
    )
