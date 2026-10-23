"""Deterministic media generation through the real tool, hooks and agent lifecycle."""

import base64
import io
import json
import os
import time
from threading import Event
from uuid import uuid4

import httpx2
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.native_suite import NativeSuiteClient
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.user_message import UserMessage
from nicegui import app
from openai import OpenAI
from PIL import Image

from exmekh_agent.config import application as configuration
from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.mock_client import MockProvider

generated_calls = []
provider_calls = []
user_image_counts = []
generating = False
image_release = Event()


def image_response(request):
    global generating
    generating = True
    body = json.loads(request.content)
    generated_calls.append(config.include_generated_images_in_context)
    try:
        if os.getenv("EXMEKH_AGENT_TEST_HOLD_IMAGES") == "1":
            if not image_release.wait(30):
                raise TimeoutError("Image fixture was not released")
        else:
            time.sleep(3)
        if body["prompt"] == "failure":
            return httpx2.Response(
                500, json={"error": {"message": "Controlled image failure"}}
            )
        width, height = map(int, body["size"].split("x"))
        outputs = []
        for index in range(body["n"]):
            stream = io.BytesIO()
            Image.new("RGB", (width, height), "red" if index == 0 else "blue").save(
                stream, "PNG"
            )
            outputs.append(
                {
                    "b64_json": base64.b64encode(stream.getvalue()).decode(),
                    "ds4": {
                        "file_id": uuid4().hex,
                        "seed": index,
                        "width": width,
                        "height": height,
                        "recipe": {},
                    },
                }
            )
        return httpx2.Response(200, json={"created": 1, "ds4": {}, "data": outputs})
    finally:
        generating = False


class PreviewProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.transport = OpenAI(
            api_key="test",
            base_url="http://native/v1",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(image_response)),
        )

    def native_suite(self):
        return NativeSuiteClient(self.transport)

    def call(self, request):
        assert not generating, (
            "Inference started while the native image service was active"
        )
        messages = request.get_messages()
        provider_calls.append(
            [
                bool(message.get_attachments())
                for message in messages
                if message.get_role_name() == "tool"
            ]
        )
        user_image_counts.append(
            sum(
                isinstance(part, Image.Image)
                for message in messages
                if isinstance(message, UserMessage)
                for part in message.iter_content_parts()
            )
        )
        self._step_counter += 1
        last = next(
            (
                message
                for message in reversed(messages)
                if message.get_role_name() == "user"
            ),
            messages[-1],
        )
        if "Describe attached image" in str(last.get_message()):
            return ProviderResponse(AssistantMessage("Attachment received"))
        if self._step_counter == 1:
            count = 2 if "two" in str(last.get_message()).lower() else 1
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            uuid4().hex,
                            FunctionCall(
                                "generate-image",
                                {
                                    "prompt": "failure"
                                    if "failure" in str(last.get_message()).lower()
                                    else "red",
                                    "n": 1
                                    if "multiple" in str(last.get_message())
                                    else count,
                                    "width": 640,
                                    "height": 320,
                                },
                            ),
                        )
                        for _ in range(
                            2 if "multiple" in str(last.get_message()) else 1
                        )
                    ],
                )
            )
        if self._step_counter == 2:
            return ProviderResponse(
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            uuid4().hex,
                            FunctionCall("inspect-file", {"path": "missing.txt"}),
                        )
                    ],
                )
            )
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.RESPONSE, "Image ready.")
        )
        return ProviderResponse(AssistantMessage("Image ready."))


agent_runner.LocalProvider = lambda *args, **kwargs: PreviewProvider()
agent_runner.generate_chat_title_sync = lambda *args, **kwargs: "Generated images"
config.default_model = "native-preview"
config.agent_mode = "quick_chat"
config.work_root = str(config.data_dir / "work")
configuration.DEFAULT_ENV_PATH = config.data_dir / ".env"


async def health(*args, **kwargs):
    return True, ["native-preview"]


agent_runner.check_provider_health = health


@app.post("/test/release-images")
def release_images():
    image_release.set()
    return {"released": True}


@app.post("/test/legacy-image-policy")
def legacy_image_policy():
    session = session_manager.get_current_session()
    for message in session.messages:
        for tool in message.tool_events:
            tool.images_in_context = None
    session_manager.save_session(session)
    return {"saved": True}


@app.get("/test/media-state")
def media_state():
    return {
        "include_generated_images": config.include_generated_images_in_context,
        "saved_image_preference": configuration.DEFAULT_ENV_PATH.is_file()
        and "INCLUDE_GENERATED_IMAGES_IN_CONTEXT=false"
        in configuration.DEFAULT_ENV_PATH.read_text(),
        "generated_calls": generated_calls,
        "provider_calls": provider_calls,
        "user_image_counts": user_image_counts,
        "session": session_manager.get_current_session().model_dump(mode="json"),
    }


if __name__ == "__main__":
    main()
