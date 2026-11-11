"""Real LocalProvider, registered image tool and native service for browser tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.models.native_suite import ImageBatch, NativeSuiteClient
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from httpx2 import Client as HTTPClient
from nicegui import app
from openai import APIStatusError
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
INFERENCE_EVENTS: list[str] = []


def record_request(request: Any) -> None:
    if "/images/" in request.url.path:
        INFERENCE_EVENTS.append(f"image:{request.url.path.rsplit('/', 1)[-1]}:start")
        TRACE["image_requests"].append(
            {
                "path": request.url.path,
                "body": json.loads(request.content),
                "idempotency_key": request.headers.get("Idempotency-Key"),
            }
        )


class ObservedNativeSuiteClient(NativeSuiteClient):
    def images(self, *args: Any, **kwargs: Any) -> ImageBatch:
        # Wait for the SDK to consume the terminal event without buffering SSE
        # in the HTTP hook. The server sends success after restoring the LLM.
        try:
            batch = super().images(*args, **kwargs)
        except APIStatusError as error:
            TRACE["image_responses"].append(
                {"status": error.status_code, "error": error.body, "images": []}
            )
            raise
        TRACE["image_responses"].append(
            {
                "status": 200,
                "ds4": batch.ds4,
                "error": None,
                "images": [
                    {
                        **item.ds4.model_dump(),
                        "png_sha256": hashlib.sha256(item.png_bytes()).hexdigest(),
                    }
                    for item in batch.data
                ],
            }
        )
        INFERENCE_EVENTS.append("image:completed")
        return batch


class ObservedLocalProvider(LocalProvider):
    """Observe real requests without replacing model or tool execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            http_client=HTTPClient(
                timeout=float(os.getenv("DS4_SUITE_HTTP_TIMEOUT_SECONDS", "600")),
                event_hooks={
                    "request": [record_request],
                },
            ),
            **kwargs,
        )
        self.client.max_retries = 0

    def native_suite(self) -> NativeSuiteClient:
        return ObservedNativeSuiteClient(self.client)

    def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = request.get_messages()
        latest_user = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, UserMessage)
            ),
            None,
        )
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
            "latest_user_image_count": sum(
                isinstance(part, Image.Image)
                for part in latest_user.iter_content_parts()
            )
            if latest_user
            else 0,
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
        INFERENCE_EVENTS.append("llm:start")
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
        INFERENCE_EVENTS.append("llm:completed")
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
def stats(summary: bool = False) -> dict[str, Any]:
    session = session_manager.get_current_session()
    if summary:
        return {
            "inference_events": list(INFERENCE_EVENTS),
            "latest_user_image_counts": [
                entry["latest_user_image_count"] for entry in TRACE["provider_calls"]
            ],
            "image_input_counts": [
                len(entry["body"].get("images", []))
                for entry in TRACE["image_requests"]
            ],
            "last_messages": [
                {
                    "role": message.role,
                    "error": message.error,
                    "tool_errors": [
                        tool.tool_name
                        for tool in message.iter_tool_events()
                        if tool.status == "error"
                    ],
                    "has_response": bool(message.content.strip()),
                }
                for message in session.messages[-2:]
            ],
        }
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
        "session": session.model_dump(mode="json"),
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
