"""Opt-in short real-model protocol gates for the 8192/64 Metal profile."""

import json
import os
import unittest
from pathlib import Path
from typing import Literal

from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

URL = os.getenv("DS4_SUITE_HTTP_URL")
MODEL = os.getenv("DS4_SUITE_LLM_MODEL")


class ShortGate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: Literal[True]


@unittest.skipUnless(URL and MODEL, "requires a native qualification server")
class QwenMetalApiLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = OpenAI(
            base_url=URL,
            api_key="local",
            timeout=float(os.getenv("DS4_SUITE_HTTP_TIMEOUT_SECONDS", "300")),
            max_retries=0,
        )
        cls.addClassCleanup(cls.client.close)
        cls.exchanges = []

    @classmethod
    def tearDownClass(cls):
        if destination := os.getenv("DS4_SUITE_E2E_ARTIFACTS"):
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            (output / "short-protocols.json").write_text(
                json.dumps({"model": MODEL, "exchanges": cls.exchanges}, indent=2)
                + "\n"
            )

    def test_chat_repetition_and_sse_match_ordinary_decode(self):
        request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply only OK."}],
            "temperature": 0,
            "reasoning_effort": "none",
            "max_tokens": 16,
        }
        first = self.client.chat.completions.create(**request)
        self.exchanges.append(
            {"request": request, "response": first.model_dump(mode="json")}
        )
        text = first.choices[0].message.content
        self.assertIn("OK", text)
        repeated = self.client.chat.completions.create(**request)
        self.exchanges.append(
            {"request": request, "response": repeated.model_dump(mode="json")}
        )
        self.assertEqual(repeated.choices[0].message.content, text)
        self.assertGreater(repeated.usage.prompt_tokens, 0)
        fragments, finishes = [], []
        with self.client.chat.completions.create(**request, stream=True) as stream:
            for chunk in stream:
                for choice in chunk.choices:
                    if choice.delta.content:
                        fragments.append(choice.delta.content)
                    if choice.finish_reason:
                        finishes.append(choice.finish_reason)
        self.assertEqual("".join(fragments), text)
        self.assertEqual(len(finishes), 1)
        self.exchanges.append(
            {"stream": True, "text": "".join(fragments), "finish_reasons": finishes}
        )

    def test_anthropic_messages_uses_the_matching_family(self):
        response = self.client.post(
            "/messages",
            cast_to=dict,
            body={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply only OK."}],
                "max_tokens": 16,
                "temperature": 0,
                "thinking": {"type": "disabled"},
            },
            options={"headers": {"anthropic-version": "2023-06-01"}},
        )
        self.exchanges.append({"endpoint": "/messages", "response": response})
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["role"], "assistant")
        text = "".join(
            part["text"] for part in response["content"] if part["type"] == "text"
        )
        self.assertIn("OK", text)
        usage = response["usage"]
        self.assertGreater(
            sum(
                usage[name]
                for name in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            ),
            0,
        )

    def test_responses_output_items_keep_live_prefix(self):
        original = {
            "role": "user",
            "content": "Remember PREFIX27. Reply only PREFIX27.",
        }
        arguments = {
            "model": MODEL,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 16,
            "temperature": 0,
        }
        first = self.client.responses.create(input=[original], **arguments)
        self.assertIn("PREFIX27", first.output_text)
        final = self.client.responses.create(
            input=[
                original,
                *[item.model_dump(exclude_unset=True) for item in first.output],
                {"role": "user", "content": "Repeat the remembered code only."},
            ],
            **arguments,
        )
        self.assertIn("PREFIX27", final.output_text)
        self.assertGreater(final.usage.input_tokens_details.cached_tokens, 0)
        self.exchanges.append(
            {
                "prefix_first": first.model_dump(mode="json"),
                "prefix_final": final.model_dump(mode="json"),
            }
        )

    def test_small_strict_json_schema(self):
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Return the JSON object with ok true."}
            ],
            temperature=0,
            reasoning_effort="none",
            max_tokens=32,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "short_gate",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean", "const": True}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        self.exchanges.append(
            {"schema": "short_gate", "response": response.model_dump(mode="json")}
        )
        self.assertEqual(json.loads(response.choices[0].message.content), {"ok": True})
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_typed_client_returns_the_declared_pydantic_model(self):
        provider = LocalProvider(
            model=MODEL, base_url=URL, api="responses", reasoning_effort="none"
        )
        self.addCleanup(provider.close)
        client = Client(
            provider, SystemMessage("Return the requested structured data.")
        )
        response = client.call_structured(
            ClientRequest(
                UserMessage("Return the JSON object with ok true."),
                extra={"max_output_tokens": 32, "temperature": 0},
            ),
            ShortGate,
        )
        value = response.get_message()
        self.assertIsInstance(value, ShortGate)
        self.assertEqual(value.model_dump(), {"ok": True})
        self.exchanges.append(
            {"typed_schema": value.model_dump(), "history": client.dump_history()}
        )
