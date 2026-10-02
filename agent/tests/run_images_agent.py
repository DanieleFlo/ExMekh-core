"""Browser test entry point using the actual ExMekh-Lib execution path."""

from typing import Any

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.user_message import UserMessage
from nicegui import app
from PIL import Image

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge import agent_runner
from exmekh_agent.exmekh_bridge.converter import exmekh_message_to_raw_dict
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.mock_client import MockProvider


class ImageProbeProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(
            [exmekh_message_to_raw_dict(item) for item in request.get_messages()]
        )
        users = [
            item for item in request.get_messages() if isinstance(item, UserMessage)
        ]
        count = sum(
            isinstance(part, Image.Image) for part in users[-1].iter_content_parts()
        )
        return ProviderResponse(AssistantMessage(f"Received {count} images"))


provider = ImageProbeProvider()
agent_runner.LocalProvider = lambda *args, **kwargs: provider
config.default_model = "image-probe"
config.agent_mode = "quick_chat"
config.work_root = str(config.data_dir / "work")


async def health(*args, **kwargs):
    return True, ["image-probe"]


agent_runner.check_provider_health = health


@app.get("/test/image-stats")
def stats() -> dict[str, Any]:
    return {
        "requests": provider.requests,
        "session": session_manager.get_current_session().model_dump(),
    }


if __name__ == "__main__":
    main()
