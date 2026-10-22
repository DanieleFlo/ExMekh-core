"""Opt-in registered image tools and native ranking on the real DS4 service."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.native_suite import NATIVE_SUITE
from exmekh_lib.core.agents.tools.deep_tools import DeepTools
from exmekh_lib.core.models.native_suite import native_suite_context
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.web_navigation import ranking
from exmekh_lib.core.web_navigation.image_discovery import DiscoveredImage
from exmekh_lib.providers.local import LocalProvider
from httpx2 import Client as HTTPClient
from PIL import Image

URL = os.getenv("DS4_SUITE_HTTP_URL")


@unittest.skipUnless(URL, "set DS4_SUITE_HTTP_URL for real image tool acceptance")
class ImageToolsE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="ds4-image-tools-")
        cls.root = Path(cls.workspace.name)
        for name, color in (("red", "red"), ("blue", "blue")):
            Image.new("RGB", (256, 256), color).save(cls.root / f"{name}.png")
        cls.provider = LocalProvider(
            model=os.getenv("DS4_SUITE_LLM_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            base_url=URL,
            api_key="local",
            api="responses",
            http_client=HTTPClient(event_hooks={"request": [cls.capture_request]}),
        )
        cls.provider.client.max_retries = 0
        cls.registry = DeepTools(
            [CAPABILITY_FORMAT.bundled_root / "tools/generate-image"],
            work_root=cls.root,
        )
        cls.tool = cls.registry.load()[0]
        cls.requests = []
        cls.observations = []

    @classmethod
    def capture_request(cls, request):
        if request.url.path.endswith(
            ("/images/generations", "/images/edits", "/embeddings")
        ):
            cls.requests.append(
                {"path": request.url.path, "body": json.loads(request.content)}
            )

    @classmethod
    def tearDownClass(cls):
        destination = os.getenv("DS4_SUITE_E2E_ARTIFACTS")
        if destination:
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "tools-and-ranking.json").write_text(
                json.dumps(cls.observations, indent=2)
            )
        cls.registry.unload()
        cls.provider.close()
        cls.workspace.cleanup()

    def invoke(self, **arguments):
        with native_suite_context(self.provider.native_suite):
            return self.tool.execute_with_hooks(
                arguments, work_root=self.root, history=()
            )

    def test_01_registered_tool_input_modes_parameters_and_attachments(self):
        signature = self.tool.get_work_signature()
        self.assertNotIn("model", signature.parameters)
        self.assertNotIn("size", signature.parameters)
        self.assertTrue(NATIVE_SUITE.include_image_in_result)
        self.assertEqual(signature.parameters["include_image_in_result"].default, True)
        cases = [
            (
                "text_only",
                {
                    "prompt": "A red square on white",
                    "n": 2,
                    "background": "opaque",
                    "preview_scale": 0.5,
                },
                [128, 128],
            ),
            (
                "image_only",
                {
                    "images": ["blue.png", "red.png"],
                    "include_image_in_result": False,
                },
                None,
            ),
            (
                "text_and_image",
                {
                    "prompt": "Make the square blue",
                    "images": ["red.png"],
                    "background": "transparent",
                    "preview_scale": 1.0,
                },
                [256, 256],
            ),
        ]
        for index, (name, options, preview_size) in enumerate(cases):
            with self.subTest(mode=name):
                self.requests.clear()
                arguments = {
                    "width": 256,
                    "height": 256,
                    "seed": 201 + index * 10,
                    "output_format": "png",
                    "idempotency_key": str(uuid.uuid4()),
                    **options,
                }
                result = self.invoke(**arguments)
                request = next(r for r in self.requests if "/images/" in r["path"])
                body = request["body"]
                expected_count = options.get("n", 1)
                self.assertEqual(len(result["images"]), expected_count)
                self.assertEqual(body["prompt"], options.get("prompt", ""))
                self.assertEqual(body["size"], "256x256")
                self.assertEqual(body["n"], expected_count)
                self.assertEqual(body["background"], options.get("background", "auto"))
                self.assertEqual(body["output_format"], "png")
                self.assertEqual(
                    body["ds4"],
                    {"version": 1, "seed": arguments["seed"], "sampling": "checkpoint"},
                )
                if "images" in options:
                    self.assertEqual(request["path"], "/v1/images/edits")
                    for file_id, reference in zip(
                        body["images"], options["images"], strict=True
                    ):
                        self.assertEqual(
                            self.provider.native_suite().content(file_id),
                            (self.root / reference).read_bytes(),
                        )
                else:
                    self.assertEqual(request["path"], "/v1/images/generations")
                self.assertEqual(
                    isinstance(result, AttachedToolResult), preview_size is not None
                )
                if preview_size is not None:
                    parts = list(result.attachments.iter_content_parts())
                    self.assertEqual(parts[0], NATIVE_SUITE.image_result_context)
                    previews = [part for part in parts if isinstance(part, Image.Image)]
                    self.assertEqual(
                        [list(image.size) for image in previews],
                        [preview_size] * expected_count,
                    )
                outputs = []
                for offset, row in enumerate(result["images"]):
                    raw = Path(row["path"]).read_bytes()
                    self.assertEqual(
                        raw, self.provider.native_suite().content(row["file_id"])
                    )
                    self.assertEqual(row["seed"], arguments["seed"] + offset)
                    self.assertEqual((row["width"], row["height"]), (256, 256))
                    self.assertEqual(row["recipe"]["steps"], 8)
                    self.assertEqual(row["recipe"]["guidance"], 1)
                    self.assertEqual(row["recipe"]["background"], body["background"])
                    with Image.open(io.BytesIO(raw)) as image:
                        self.assertEqual(image.size, (256, 256))
                        self.assertEqual(image.mode, "RGBA")
                        if body["background"] == "opaque":
                            self.assertEqual(
                                image.getchannel("A").getextrema(), (255, 255)
                            )
                    outputs.append(
                        {**row, "png_sha256": hashlib.sha256(raw).hexdigest()}
                    )
                    destination = os.getenv("DS4_SUITE_E2E_ARTIFACTS")
                    if destination:
                        saved = Path(destination) / f"{name}-{offset}.png"
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        saved.write_bytes(raw)
                        outputs[-1]["artifact"] = str(saved)
                if index == 0:
                    retry = self.invoke(**arguments)
                    self.assertEqual(
                        result["ds4"]["request_id"], retry["ds4"]["request_id"]
                    )
                    self.assertEqual(
                        [r["file_id"] for r in result["images"]],
                        [r["file_id"] for r in retry["images"]],
                    )
                self.observations.append(
                    {
                        "case": name,
                        "request": request,
                        "service": result["ds4"],
                        "outputs": outputs,
                        "preview_size": preview_size,
                    }
                )

    def test_02_invalid_preview_and_paths_do_not_generate(self):
        self.requests.clear()
        for scale in (0, -1, 1.01):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.invoke(prompt="red", preview_scale=scale)
        with self.assertRaises(ValueError):
            self.invoke(prompt="red", images=["../outside.png"])
        self.assertFalse(self.requests)
        self.observations.append({"case": "invalid_arguments", "compute_requests": 0})

    def test_03_text_and_visual_web_ranking_use_native_embeddings(self):
        self.requests.clear()
        imported_before = set(sys.modules)
        candidates = [
            {
                "title": "Red square",
                "url": "https://example.org/red",
                "snippet": "A red square on white",
            },
            {
                "title": "Blue square",
                "url": "https://example.org/blue",
                "snippet": "A blue square",
            },
        ]
        images = [
            DiscoveredImage(
                i, row["url"] + ".png", (self.root / f"{color}.png").read_bytes()
            )
            for i, (row, color) in enumerate(
                zip(candidates, ("red", "blue"), strict=True)
            )
        ]
        with (
            native_suite_context(self.provider.native_suite),
            Image.open(self.root / "red.png") as query,
        ):
            scores = ranking.compute_semantic_similarity("red square", candidates)
            order, urls = ranking.compute_multimodal_ranking(
                "red square", candidates, local_image=query, discovered_images=images
            )
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(sorted(order), [0, 1])
        self.assertEqual(len(urls), 2)
        self.assertIsInstance(
            ranking.get_embedding_backend(), ranking.NativeEmbeddingBackend
        )
        self.assertIsInstance(
            ranking.get_image_embedding_backend(), ranking.NativeEmbeddingBackend
        )
        self.assertGreaterEqual(len(self.requests), 5)
        introduced = {
            name.partition(".")[0] for name in set(sys.modules) - imported_before
        }
        self.assertFalse(introduced & {"torch", "transformers", "openvino"})
        for request in self.requests:
            self.assertEqual(request["path"], "/v1/embeddings")
            self.assertEqual(request["body"]["model"], "google/embeddinggemma-2")
            self.assertEqual(request["body"]["ds4"]["task"], "similarity")
        self.observations.append(
            {
                "case": "native_ranking",
                "scores": scores,
                "order": order,
                "requests": list(self.requests),
            }
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
