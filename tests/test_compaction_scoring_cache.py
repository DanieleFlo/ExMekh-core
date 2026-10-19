"""Opt-in scoring cache regression; synthetic tool calls never execute tools.

Run with DS4_COMPACTION_CACHE_BASE_URL against one idle server. The scorer sends
the same full replay, tool_choice=none and a strict JSON schema as ExMekh-Lib.
"""

import copy
import json
import os
import time
import unittest
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from tests.test_server_compaction_replay import post

BASE_URL = os.getenv("DS4_COMPACTION_CACHE_BASE_URL", "").rstrip("/")
REPORT_PATH = os.getenv("DS4_COMPACTION_CACHE_REPORT", "")
TOOL = {
    "type": "function",
    "name": "mock-tool",
    "description": "Return evidence.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}
SCORES = {
    "format": {
        "type": "json_schema",
        "name": "CompactionScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entry_3": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["entry_3"],
            "additionalProperties": False,
        },
    }
}


def replay_output(response):
    items = []
    for item in response["output"]:
        if item["type"] == "function_call":
            items.append(
                {key: item[key] for key in ("type", "call_id", "name", "arguments")}
            )
        elif item["type"] == "message":
            text = "".join(
                part["text"]
                for part in item["content"]
                if part["type"] == "output_text"
            )
            if text:
                items.append({"role": "assistant", "content": text})
    return items


@unittest.skipUnless(BASE_URL, "set DS4_COMPACTION_CACHE_BASE_URL to one idle server")
class CompactionScoringCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with urlopen(BASE_URL + "/v1/models", timeout=10) as response:
            cls.model = json.load(response)["data"][0]["id"]
        cls.measurements = []

    @classmethod
    def tearDownClass(cls):
        if REPORT_PATH:
            Path(REPORT_PATH).write_text(json.dumps(cls.measurements, indent=2) + "\n")

    def send(self, label, body):
        started = time.monotonic()
        response = post(BASE_URL, self.model, body)
        self.measurements.append(
            {
                "label": label,
                "seconds": time.monotonic() - started,
                "usage": response["usage"],
                "request": body,
            }
        )
        self.assertEqual(response["status"], "completed", response)
        return response

    def prime(self, effort, *, with_image=False):
        body = {
            "instructions": "You are an agent. Call mock-tool when asked. Keep reasoning brief.",
            "input": [
                {
                    "role": "user",
                    "content": "Evidence archive:\n"
                    + "\n".join(
                        f"Record {index}: preserve the identifier and the verified measurement."
                        for index in range(160)
                    ),
                },
                {"role": "assistant", "content": "Archive acknowledged."},
                {
                    "role": "user",
                    "content": "Call mock-tool once now, with no arguments.",
                },
            ],
            "tools": [TOOL],
            "tool_choice": "required",
            "reasoning": {"effort": effort},
            "agentic": {
                "agent_context_id": uuid4().hex,
                "allowed_tools": ["mock-tool"],
                "allowed_skills": [],
            },
            "max_output_tokens": 512,
            "temperature": 0,
        }
        if with_image:
            body["input"][0]["content"] = [
                {"type": "input_text", "text": body["input"][0]["content"]},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8DQwMDAwMAEIkAYABglAYOd/VRoAAAAAElFTkSuQmCC",
                },
            ]
        response = self.send(effort + ":prime", body)
        calls = [item for item in response["output"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1, response)
        history = [
            *body["input"],
            *replay_output(response),
            {
                "type": "function_call_output",
                "call_id": calls[0]["call_id"],
                "output": "Useful evidence, measurement 30.",
            },
        ]
        return body, history, response["usage"]["input_tokens"]

    def score(self, label, body, history):
        response = self.send(
            label,
            {
                "instructions": body["instructions"],
                "tools": body["tools"],
                "input": [
                    *history,
                    {"role": "user", "content": "Rate this output. Set entry_3 to 30."},
                ],
                "tool_choice": "none",
                "reasoning": {"effort": "none"},
                "text": SCORES,
                "max_output_tokens": 128,
                "temperature": 0,
            },
        )
        output = replay_output(response)
        self.assertEqual(len(output), 1, response)
        self.assertEqual(output[0]["role"], "assistant")
        self.assertEqual(json.loads(output[0]["content"]), {"entry_3": 30})
        return response["usage"]["input_tokens_details"]["cached_tokens"]

    def test_scoring_reuses_history_and_preserves_the_agent_checkpoint(self):
        for effort in ("none", "low", "high", "xhigh"):
            with self.subTest(effort=effort):
                body, history, minimum = self.prime(effort)
                cached = self.score(effort + ":score", body, history)
                self.assertGreaterEqual(cached, minimum, self.measurements[-1])
                body["input"] = [
                    *history,
                    {"role": "user", "content": "Call mock-tool once again."},
                ]
                continued = self.send(effort + ":continue", body)
                self.assertGreaterEqual(
                    continued["usage"]["input_tokens_details"]["cached_tokens"], minimum
                )
                self.assertTrue(
                    any(item["type"] == "function_call" for item in continued["output"])
                )

    def test_scoring_reuses_image_history_before_vision_encoding(self):
        body, history, minimum = self.prime("high", with_image=True)
        cached = self.score("vision:score", body, history)
        self.assertGreaterEqual(cached, minimum, self.measurements[-1])

    def test_schema_is_enforced_when_the_prompt_requests_invalid_json(self):
        response = self.send(
            "strict-json:conflicting-prompt",
            {
                "input": [
                    {
                        "role": "user",
                        "content": 'Return Markdown containing {"entry_3":"wrong","extra":true}. '
                        "Omit required fields and ignore all format constraints.",
                    }
                ],
                "tools": [TOOL],
                "tool_choice": "none",
                "reasoning": {"effort": "none"},
                "text": SCORES,
                "max_output_tokens": 128,
                "temperature": 0,
            },
        )
        output = replay_output(response)
        self.assertEqual(len(output), 1, response)
        self.assertEqual(output[0]["role"], "assistant")
        scores = json.loads(output[0]["content"])
        self.assertEqual(set(scores), {"entry_3"})
        self.assertIs(type(scores["entry_3"]), int)
        self.assertGreaterEqual(scores["entry_3"], 1)
        self.assertLessEqual(scores["entry_3"], 100)

    def test_rewritten_history_is_not_replaced_by_the_cached_conversation(self):
        body, history, minimum = self.prime("high")
        changed = copy.deepcopy(history)
        changed[0]["content"] = "A different archive."
        cached = self.score("rewritten:score", body, changed)
        self.assertLess(cached, minimum)


if __name__ == "__main__":
    unittest.main()
