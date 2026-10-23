"""Deterministic media generation through the real tool, hooks and agent lifecycle."""

import os
import time
from threading import Event
from uuid import uuid4

from exmekh_lib.core.models import native_media_tools
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from nicegui import app
from PIL import Image

from exmekh_agent.config import application as configuration
from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.mock_client import MockProvider

generated_calls = []
provider_calls = []
image_release = Event()


def generate(
    work_root,
    prompt,
    n=1,
    width=2048,
    height=2048,
    include_image_in_result=True,
    **kwargs,
):
    generated_calls.append(include_image_in_result)
    if os.getenv("EXMEKH_AGENT_TEST_HOLD_IMAGES") == "1":
        if not image_release.wait(30):
            raise TimeoutError("Image fixture was not released")
    else:
        time.sleep(3)
    if prompt == "failure":
        raise RuntimeError("Controlled image generation failure")
    folder = work_root / ".exmekh_lib" / "generated-media" / uuid4().hex
    folder.mkdir(parents=True)
    images, outputs = [], []
    for index in range(n):
        image = Image.new("RGB", (width, height), "red" if index == 0 else "blue")
        path = folder / f"{index}.png"
        image.save(path)
        images.append(image)
        outputs.append({"path": str(path), "width": width, "height": height})
    result = {"images": outputs}
    return (
        AttachedToolResult(result, UserMessage(ImagesUserMessage(images)))
        if include_image_in_result
        else result
    )


class PreviewProvider(MockProvider):
    def call(self, request):
        messages = request.get_messages()
        provider_calls.append(
            [
                bool(message.get_attachments())
                for message in messages
                if message.get_role_name() == "tool"
            ]
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
                                    "n": count,
                                    "width": 640,
                                    "height": 320,
                                    "wait_seconds": 60,
                                    "execution_mode": "background"
                                    if "background" in str(last.get_message()).lower()
                                    else "wait",
                                },
                            ),
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


native_media_tools.generate_images = generate
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


@app.get("/test/media-state")
def media_state():
    return {
        "include_generated_images": config.include_generated_images_in_context,
        "saved_image_preference": configuration.DEFAULT_ENV_PATH.is_file()
        and "INCLUDE_GENERATED_IMAGES_IN_CONTEXT=false"
        in configuration.DEFAULT_ENV_PATH.read_text(),
        "generated_calls": generated_calls,
        "provider_calls": provider_calls,
        "session": session_manager.get_current_session().model_dump(mode="json"),
    }


if __name__ == "__main__":
    main()
