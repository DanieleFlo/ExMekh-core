#!/usr/bin/env python3
"""Live image gate through the installed QS client, without an agent or new UI.

From agent/: uv run --frozen python ../tests/qs_vision_live.py --url URL
Requires an idle Qwen27 server with its matching --vision encoder.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import ImagesUserMessage, UserMessage
from qs.providers.local import LocalProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_S")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []
    for api in ("chat_completions", "responses"):
        provider = LocalProvider(model=args.model, api=api, base_url=args.url, api_key="local-test")
        client = Client(provider=provider, system_message=SystemMessage(
            "For an image, answer with its dominant color in one word. For a text question, answer briefly."))
        extra = {"temperature": 0, "max_tokens" if api == "chat_completions" else "max_output_tokens": 64,
                 "extra_body": {"thinking": False}}
        for name, color in (("red", (230, 20, 20)), ("blue", (20, 40, 230))):
            reply = client.call(ClientRequest(message=UserMessage(
                ImagesUserMessage(images=[Image.new("RGB", (320, 256), color)])), extra=extra))
            text = (reply.get_message() or "").lower()
            assert name in text, (api, name, text)
            again = client.call(ClientRequest(message=UserMessage("What was the color of the LAST image? One word."),
                                              extra=extra))
            follow = (again.get_message() or "").lower()
            assert name in follow, (api, name, follow)
            results.append(dict(api=api, image=name, content=text, continuation=follow, status="PASS"))
            print(api, name, "PASS", flush=True)
            if args.report:
                args.report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()