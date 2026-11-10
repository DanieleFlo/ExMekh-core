"""Opt-in actual ExMekh Client -> registered image tool -> continuation."""

import json
import os
import unittest
from pathlib import Path

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.tools.deep_tools import DeepTools
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.native_media_tools import generate_images
from exmekh_lib.core.models.native_suite import native_suite_context
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import AttachedToolResult, CalledTool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.xml import stringify_tool_result
from httpx2 import Client as HTTPClient
from PIL import Image

URL = os.getenv("DS4_SUITE_HTTP_URL")


@unittest.skipUnless(URL, "set DS4_SUITE_HTTP_URL for live ExMekh acceptance")
class ExmekhNativeLive(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.captures = []
        self.provider = LocalProvider(
            model=os.getenv("DS4_SUITE_LLM_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            base_url=URL,
            api="responses",
            reasoning_effort="none",
            http_client=HTTPClient(
                timeout=float(os.getenv("DS4_SUITE_HTTP_TIMEOUT_SECONDS", "600")),
                event_hooks={
                    "response": [lambda response: self.captures.append(response.read())]
                }
            ),
        )
        self.addCleanup(self.provider.close)

    def tearDown(self):
        if destination := os.getenv("DS4_SUITE_E2E_ARTIFACTS"):
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            (output / (self.id().rsplit(".", 1)[-1] + ".json")).write_text(
                json.dumps(
                    [json.loads(raw) for raw in self.captures if raw.startswith(b"{")],
                    indent=2,
                )
                + "\n"
            )

    def test_registered_image_tool_retains_conversation(self):
        registry = DeepTools(
            [CAPABILITY_FORMAT.bundled_root / "tools/generate-image"],
            work_root=self.root,
        )
        tool = registry.load()[0]
        self.addCleanup(registry.unload)
        self.assertNotIn("model", tool.get_work_signature().parameters)
        self.assertIn("width", tool.get_work_signature().parameters)
        client = Client(
            self.provider,
            SystemMessage("Use the requested tool exactly. Remember codes."),
            Tools([tool]),
        )
        first = client.call(
            ClientRequest(
                UserMessage(
                    "Remember code BETA18. Call generate-image exactly once with prompt 'A red square on white.', "
                    "n 1, width 256, height 256, seed 18, include_image_in_result false. "
                    "After the tool succeeds reply only BETA18."
                ),
                tool_choice="required",
                extra={"max_output_tokens": 768, "temperature": 0},
            )
        )
        calls = first.get_tool_calls()
        self.assertEqual(len(calls or []), 1)
        call = calls[0]
        self.assertEqual(call.get_arguments().get("prompt"), "A red square on white.")
        self.assertEqual(call.get_arguments().get("n", 1), 1)
        self.assertEqual(call.get_arguments().get("width"), 256)
        self.assertEqual(call.get_arguments().get("height"), 256)
        self.assertFalse(call.get_arguments().get("include_image_in_result", True))
        with native_suite_context(client.native_suite):
            result = CalledTool.from_tool_call(tool, call).execute(work_root=self.root)
        self.assertTrue(Path(result["images"][0]["path"]).is_file())
        client.add_tool_response(
            ToolMessage(stringify_tool_result(result), call.get_id(), call.get_name())
        )
        final = client.continue_after_tools(
            extra={"max_output_tokens": 48, "temperature": 0}
        )
        self.assertIn("BETA18", final.get_message())
        wire = [json.loads(raw) for raw in self.captures if raw.startswith(b"{")]
        response = [row for row in wire if row.get("object") == "response"][-1]
        self.assertGreater(
            response["usage"]["input_tokens_details"]["cached_tokens"], 0
        )

    def test_preview_scales_attachment_and_keeps_original(self):
        # Independent image operation: keep the conversation gate's full
        # checks within its own model-process deadline, without dropping the
        # second native generation or replacing it with a receipt replay.
        client = Client(self.provider, SystemMessage("Use native images."))
        with native_suite_context(client.native_suite):
            preview = generate_images(
                self.root,
                "A blue square",
                width=256,
                height=256,
                seed=19,
                preview_scale=0.5,
            )
        self.assertIsInstance(preview, AttachedToolResult)
        self.assertEqual(
            next(
                part
                for part in preview.attachments.iter_content_parts()
                if isinstance(part, Image.Image)
            ).size,
            (128, 128),
        )
        with Image.open(preview["images"][0]["path"]) as original:
            self.assertEqual(original.size, (256, 256))


if __name__ == "__main__":
    unittest.main(verbosity=2)
