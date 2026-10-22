"""Real LocalProvider, registered image tool and native service for browser tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from httpx2 import Client as HTTPClient
from nicegui import app
from PIL import Image

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents import agent_runner
from exmekh_agent.exmekh_bridge.models import ProjectSettings
from exmekh_agent.main import main
from exmekh_agent.state.agent_design import AgentDesign, DesignLink, DesignNode
from exmekh_agent.state.session_manager import session_manager

TRACE: dict[str, list[dict[str, Any]]] = {
    "image_requests": [],
    "image_responses": [],
    "provider_calls": [],
}


def record_request(request: Any) -> None:
    if "/images/" in request.url.path:
        TRACE["image_requests"].append(
            {
                "path": request.url.path,
                "body": json.loads(request.content),
                "idempotency_key": request.headers.get("Idempotency-Key"),
            }
        )


def record_response(response: Any) -> None:
    if "/images/" not in response.request.url.path:
        return
    body = json.loads(response.read())
    TRACE["image_responses"].append(
        {
            "status": response.status_code,
            "ds4": body.get("ds4"),
            "error": body.get("error"),
            "images": [
                {
                    **row["ds4"],
                    "png_sha256": hashlib.sha256(
                        base64.b64decode(row["b64_json"])
                    ).hexdigest(),
                }
                for row in body.get("data", [])
            ],
        }
    )


class ObservedLocalProvider(LocalProvider):
    """Observe real requests without replacing model or tool execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            http_client=HTTPClient(
                event_hooks={"request": [record_request], "response": [record_response]}
            ),
            **kwargs,
        )
        self.client.max_retries = 0

    def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = request.get_messages()
        entry = {
            "tool_names": request.get_tools().list_tools()
            if request.get_tools()
            else [],
            "tool_choice": request.get_tool_choice(),
            "extra": request.get_extra(),
            "messages": [
                {"role": message.get_role_name(), "text": message.get_message()}
                for message in messages
                if not isinstance(message, UserMessage)
            ],
            "input_image_sizes": [
                list(part.size)
                for message in messages
                if isinstance(message, UserMessage)
                for part in message.iter_content_parts()
                if isinstance(part, Image.Image)
            ],
            "tool_attachment_counts": [
                sum(
                    isinstance(part, Image.Image)
                    for part in attachment.iter_content_parts()
                )
                for message in messages
                if isinstance(message, ToolMessage)
                and (attachment := message.get_attachments()) is not None
            ],
        }
        TRACE["provider_calls"].append(entry)
        response = super().call(
            request.with_extra(
                {
                    **(request.get_extra() or {}),
                    "max_output_tokens": 1024,
                    "temperature": 0,
                }
            )
        )
        entry["output_text"] = response.get_message()
        entry["tool_calls"] = [
            {"name": call.get_name(), "arguments": call.get_arguments()}
            for call in response.get_tool_calls() or []
        ]
        return response


def image_design() -> AgentDesign:
    root = DesignNode(
        kind="hds",
        name="main",
        content=(
            "Follow the user's exact generate-image arguments. Call generate-image once "
            "per requested generation; wait for its completion, do not regenerate. "
            "After success, answer the user directly with the remembered code. "
            "When asked to recall a code, do not generate an image."
        ),
    )
    tool = DesignNode(
        kind="tool",
        name="generate-image",
        source=str(CAPABILITY_FORMAT.bundled_root / "tools/generate-image/main.py"),
    )
    return AgentDesign(
        nodes=[root, tool],
        links=[DesignLink(parent=root.id, child=tool.id)],
        root_id=root.id,
        skill_connections_explicit=True,
    )


@app.get("/test/native-image-stats")
def stats() -> dict[str, Any]:
    files = []
    for path in sorted(
        Path(config.work_root).glob(".exmekh_lib/generated-media/*/*.png")
    ):
        with Image.open(path) as image:
            files.append(
                {
                    "path": str(path),
                    "size": list(image.size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {
        **TRACE,
        "files": files,
        "session": session_manager.get_current_session().model_dump(mode="json"),
    }


if __name__ == "__main__":
    config.base_url = os.environ["DS4_SUITE_HTTP_URL"]
    config.default_model = os.getenv("DS4_SUITE_LLM_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    config.api_key = "local"
    config.api_type = "responses"
    config.reasoning_effort = "none"
    config.agent_mode = "deep_agent"
    config.work_root = str(config.data_dir / "work")
    Path(config.work_root).mkdir(parents=True, exist_ok=True)
    agent_runner.LocalProvider = ObservedLocalProvider
    agent_runner.generate_chat_title_sync = lambda *args, **kwargs: "Native images E2E"
    project = session_manager.create_project(
        config.work_root,
        ProjectSettings(
            agent_profile="custom",
            agent_design=image_design(),
        ),
    )
    session_manager.create_session(project_id=project.id)
    main()
