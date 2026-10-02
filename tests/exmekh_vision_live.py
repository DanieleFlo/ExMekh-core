#!/usr/bin/env python3
"""Opt-in vision/memory gate through the ExMekh-Lib wheel installed in FE.

From agent/: uv run --frozen python ../tests/exmekh_vision_live.py --url URL
Requires an idle Qwen27 server with its matching --vision encoder. Real tool
results, two distinct photo inputs, SSE and structured recall are exercised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.providers.local import LocalProvider
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Recall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    secret: str
    first_object: str
    first_receipt: str
    second_object: str
    second_receipt: str


class StreamTrace(ProviderEventHandler):
    def __init__(self) -> None:
        self.types: list[str] = []
        self.text = ""

    def handle(self, event: ProviderStreamEvent) -> None:
        self.types.append(event.type.value)
        if event.type is ProviderStreamEventType.RESPONSE:
            self.text += event.content or ""


class PhotoTools:
    def __init__(self, directory: Path) -> None:
        self.secret = uuid4().hex[:12]
        self.receipts = [uuid4().hex[:12], uuid4().hex[:12]]
        self.note = directory / "note.txt"
        self.note.write_text(self.secret + "\n")
        self.trace: list[dict] = []
        self.audit = directory / "audit.jsonl"

    def read_note(self) -> str:
        """Read the confidential note once. It will be deleted after this read."""
        secret = self.note.read_text().strip()
        self.note.unlink()
        self.trace.append({"tool": "read_note", "secret": secret})
        return f"<note><secret>{secret}</secret></note>"

    def record_picture(self, object_name: str, picture: int) -> str:
        """Record the main object seen in picture 1 or 2 and issue a random receipt."""
        assert picture in (1, 2), picture
        assert (
            "earth" in object_name.casefold() or "planet" in object_name.casefold()
        ), object_name
        assert not any(item.get("picture") == picture for item in self.trace), (
            self.trace
        )
        row = {
            "tool": "record_picture",
            "picture": picture,
            "object": object_name,
            "receipt": self.receipts[picture - 1],
        }
        with self.audit.open("a") as audit:
            audit.write(json.dumps(row) + "\n")
        self.trace.append(row)
        return (
            f"<record><picture>{picture}</picture><object>Earth</object>"
            f"<receipt>{row['receipt']}</receipt></record>"
        )


def request_extra(api: str) -> dict:
    limit = "max_tokens" if api == "chat_completions" else "max_output_tokens"
    return {"temperature": 0, limit: 256, "extra_body": {"thinking": False}}


def finish_tools(client: Client, response, extra: dict, trace: list[dict]):
    for _ in range(4):
        calls = response.get_tool_calls() or []
        if not calls:
            return response
        for call in calls:
            assert client.tools is not None
            tool = client.tools.get_tool(call.get_name())
            assert tool is not None, call.get_name()
            arguments = call.get_arguments()
            result = tool.execute(**arguments)
            trace.append(
                {
                    "id": call.get_id(),
                    "name": call.get_name(),
                    "arguments": arguments,
                    "result": result,
                }
            )
            client.add_tool_response(
                ToolMessage(result, call.get_id(), name=call.get_name())
            )
        response = client.continue_after_tools(extra=extra, tool_choice="none")
    raise AssertionError("Tool loop exceeded its four-step limit")


def photo_memory(provider: LocalProvider, api: str, photo_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="exmekh-photo-memory-") as directory:
        state = PhotoTools(Path(directory))
        tools = Tools(
            [
                Tool(state.read_note, background_enabled=False),
                Tool(state.record_picture, background_enabled=False),
            ]
        )
        client = Client(
            provider,
            SystemMessage(
                "Follow the requested tool and output format exactly. Keep the confidential "
                "note and the issued receipts in memory. Do not print them until the final "
                "recall request. Identify pictures visually; do not infer from filenames."
            ),
            tools,
        )
        extra = request_extra(api)
        calls: list[dict] = []
        response = client.call(
            ClientRequest(
                UserMessage(
                    "Call read_note once to read the confidential note. Then reply READY only."
                ),
                extra=extra,
                tool_choice="required",
            )
        )
        answer = finish_tools(client, response, extra, calls).get_message() or ""
        assert state.trace and state.trace[0]["tool"] == "read_note", calls
        assert state.secret not in answer, answer
        assert not state.note.exists()
        with Image.open(photo_path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        photos = [photo, ImageOps.mirror(photo)]
        fingerprints = [hashlib.sha256(image.tobytes()).hexdigest() for image in photos]
        assert fingerprints[0] != fingerprints[1]
        stream = StreamTrace()
        answers = []
        for picture, image in enumerate(photos, 1):
            # Opposite layouts verify ordered mixed parts and newly appended pixels.
            instruction = (
                f"This is picture {picture}. Identify its main object visually, "
                f"then call record_picture with that object_name and picture={picture}. "
                "After the tool reply, answer RECORDED only. Do not repeat "
                "the confidential note or receipts."
            )
            content = (
                [instruction, ImagesUserMessage([image])]
                if picture == 1
                else [ImagesUserMessage([image]), instruction]
            )
            response = client.call(
                ClientRequest(
                    UserMessage(content),
                    extra=extra,
                    tool_choice="required",
                    streaming=True,
                    on_event=stream,
                )
            )
            answer = finish_tools(client, response, extra, calls).get_message() or ""
            assert state.secret not in answer and all(
                r not in answer for r in state.receipts
            ), answer
            answers.append(answer)
        assert [row["tool"] for row in state.trace] == [
            "read_note",
            "record_picture",
            "record_picture",
        ]
        assert [row.get("picture") for row in state.trace[1:]] == [1, 2]
        assert stream.types.count("end") == 2, stream.types
        response = client.call_structured(
            ClientRequest(
                UserMessage(
                    "Without using any tool, recall the confidential secret read before the images, "
                    "the object seen in the first picture and its receipt, and the object seen "
                    "in the second picture and its receipt. Use the requested JSON schema."
                ),
                extra=extra,
                tool_choice="none",
            ),
            Recall,
        )
        recalled = response.get_message().model_dump()
        assert recalled["secret"] == state.secret, recalled
        assert recalled["first_receipt"] == state.receipts[0], recalled
        assert recalled["second_receipt"] == state.receipts[1], recalled
        assert (
            "earth" in recalled["first_object"].casefold()
            or "planet" in recalled["first_object"].casefold()
        ), recalled
        assert (
            "earth" in recalled["second_object"].casefold()
            or "planet" in recalled["second_object"].casefold()
        ), recalled
        user_messages = [m for m in client.get_history() if isinstance(m, UserMessage)]
        assert all(
            state.secret not in m.message_dump()["content"] for m in user_messages
        )
        images = [
            part
            for message in user_messages
            for part in message.iter_content_parts()
            if isinstance(part, Image.Image)
        ]
        assert [
            hashlib.sha256(image.tobytes()).hexdigest() for image in images
        ] == fingerprints
        return {
            "photo_sha256": hashlib.sha256(photo_path.read_bytes()).hexdigest(),
            "pixel_sha256": fingerprints,
            "tool_calls": calls,
            "answers": answers,
            "recall": recalled,
            "stream_events": stream.types,
            "note_deleted_before_images": True,
            "secret_absent_from_user_prompts": True,
            "history_messages": len(client.get_history()),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_S")
    parser.add_argument(
        "--photo", type=Path, default=ROOT / "tests/vision-fixtures/glm53/earth.jpg"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results: list[dict] = []
    for api in ("chat_completions", "responses"):
        provider = LocalProvider(
            model=args.model,
            api=api,
            base_url=args.url,
            api_key="local-test",
            reasoning_effort=ReasoningEffortLMH.OFF,
        )
        try:
            client = Client(
                provider,
                SystemMessage(
                    "For an image, answer with its dominant color in one word. Answer text questions briefly."
                ),
            )
            for name, color in (("red", (230, 20, 20)), ("blue", (20, 40, 230))):
                reply = client.call(
                    ClientRequest(
                        UserMessage(
                            [
                                "Name the dominant color.",
                                ImagesUserMessage(
                                    [Image.new("RGB", (320, 256), color)]
                                ),
                            ]
                        ),
                        extra=request_extra(api),
                        tool_choice="none",
                    )
                )
                text = (reply.get_message() or "").lower()
                assert name in text, (api, name, text)
                again = client.call(
                    ClientRequest(
                        UserMessage("What was the color of the LAST image? One word."),
                        extra=request_extra(api),
                        tool_choice="none",
                    )
                )
                follow = (again.get_message() or "").lower()
                assert name in follow, (api, name, follow)
                results.append(
                    {
                        "api": api,
                        "case": name,
                        "content": text,
                        "continuation": follow,
                        "status": "PASS",
                    }
                )
                print(api, name, "PASS", flush=True)
            started = time.monotonic()
            detail = photo_memory(provider, api, args.photo)
            results.append(
                {
                    "api": api,
                    "case": "real-photo-tool-memory",
                    "status": "PASS",
                    "seconds": time.monotonic() - started,
                    "detail": detail,
                }
            )
            print(api, "real-photo-tool-memory PASS", flush=True)
        except Exception as error:
            results.append(
                {
                    "api": api,
                    "case": "live-vision",
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(api, results[-1], flush=True)
        finally:
            provider.client.close()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["status"] != "PASS" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
