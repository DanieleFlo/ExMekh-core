"""Deterministic tool attachment delivered through the real history/UI path."""

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from PIL import Image

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.main import main
from tests.mock_client import MockProvider


class MediaResponse(ProviderResponse):
    def history_messages(self):
        return [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "image-call", FunctionCall("generate-image", {"prompt": "red"})
                    )
                ],
            ),
            ToolMessage(
                "<result><file_id>file-preview</file_id></result>",
                "image-call",
                "generate-image",
                attachments=UserMessage(
                    ImagesUserMessage([Image.new("RGB", (32, 32), "red")])
                ),
            ),
            AssistantMessage("Image ready."),
        ]


class PreviewProvider(MockProvider):
    def call(self, request):
        return MediaResponse(AssistantMessage("Image ready."))


agent_runner.LocalProvider = lambda *args, **kwargs: PreviewProvider()
config.default_model = "native-preview"
config.agent_mode = "quick_chat"
config.work_root = str(config.data_dir / "work")


async def health(*args, **kwargs):
    return True, ["native-preview"]


agent_runner.check_provider_health = health
if __name__ == "__main__":
    main()
