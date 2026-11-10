"""Opt-in real scoring, compaction and continuation of a small fixture history."""

import json
import os
import unittest
from pathlib import Path

from exmekh_lib.core.history_compaction import HistoryCompactor
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider

URL = os.getenv("DS4_SUITE_HTTP_URL")
MODEL = os.getenv("DS4_SUITE_LLM_MODEL")


def read_note() -> str:
    """Read the fixed qualification note."""
    return "The required completion code is COMPACT27."


class BoundedLocalProvider(LocalProvider):
    """Use real provider calls with an explicit 96-token development output cap."""

    @staticmethod
    def bounded(request):
        extra = request.get_extra() or {}
        extra["max_output_tokens"] = min(extra.get("max_output_tokens", 96), 96)
        extra["temperature"] = 0
        return request.with_extra(extra)

    def call(self, request):
        return super().call(self.bounded(request))

    def call_structured(self, request):
        return super().call_structured(self.bounded(request))


@unittest.skipUnless(URL and MODEL, "requires a native qualification server")
class QwenMetalCompactionLive(unittest.TestCase):
    def test_real_compaction_keeps_original_and_continues_from_new_client(self):
        with BoundedLocalProvider(
            model=MODEL, base_url=URL, api="responses", reasoning_effort="none"
        ) as provider:
            system = SystemMessage(
                "Use the saved note's completion code. Preserve identifiers."
            )
            user = UserMessage(
                "Read the note and remember its completion code for later."
            )
            tools = Tools([Tool(read_note)])
            original = MessageHistory(
                [
                    system,
                    user,
                    AssistantMessage(
                        None,
                        tool_calls=[
                            ToolCall("fixture-note", FunctionCall("read_note", {}))
                        ],
                    ),
                    ToolMessage(read_note(), "fixture-note", "read_note"),
                    AssistantMessage(
                        "The note has been read and its completion code is ready."
                    ),
                ]
            )
            before = original.dump_history()
            result = HistoryCompactor(provider).compact(original, tools=tools)
            self.assertEqual(original.dump_history(), before)
            self.assertEqual(result.output_count, 1)
            self.assertTrue(result.replaced_history)
            compacted = result.history.dump_history()
            self.assertIn("compact_history", json.dumps(compacted))
            self.assertIs(result.history.get_history()[1], user)
            # A fresh public client consumes the typed replacement history.
            client = Client(provider, system)
            client.replace_history_from(0, result.history.get_history())
            final = client.call(
                ClientRequest(
                    UserMessage("Reply only with the completion code from the note."),
                    extra={"max_output_tokens": 24, "temperature": 0},
                )
            )
            self.assertIn("COMPACT27", final.get_message())
            self.assertFalse(final.get_tool_calls())
            output = Path(os.environ["DS4_SUITE_E2E_ARTIFACTS"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "compaction.json").write_text(
                json.dumps(
                    {
                        "scope": "Constructed short tool history, real native scoring/summary/continuation",
                        "output_cap": 96,
                        "original": before,
                        "compacted": compacted,
                        "archive": result.replaced_history,
                        "continuation": client.dump_history(),
                    },
                    indent=2,
                )
                + "\n"
            )
