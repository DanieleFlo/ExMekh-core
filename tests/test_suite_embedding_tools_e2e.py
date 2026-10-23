"""Opt-in ExMekh consumers -> shared native transport -> real CUDA embeddings."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.native_suite import (
    EmbeddingItem,
    MediaContent,
    TextContent,
    native_suite_context,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import CalledTool, Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from exmekh_lib.core.web_navigation import ranking
from exmekh_lib.core.web_navigation.image_discovery import DiscoveredImage
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.xml import stringify_tool_result
from httpx2 import Client as HTTPClient
from openai import APIStatusError
from PIL import Image

URL = os.getenv("DS4_SUITE_HTTP_URL")
ROOT = Path(__file__).resolve().parents[1]


def rank_local_results(query: str) -> dict:
    """Rank fixed search candidates using the production native embedding consumer."""
    return {
        "scores": ranking.compute_semantic_similarity(
            query,
            [
                {
                    "title": "Alps",
                    "url": "https://example.org/alps",
                    "snippet": "Snowy mountains",
                },
                {
                    "title": "Cooking",
                    "url": "https://example.org/food",
                    "snippet": "Bread recipes",
                },
            ],
        )
    }


@unittest.skipUnless(URL, "set DS4_SUITE_HTTP_URL for live ExMekh embedding acceptance")
class EmbeddingToolsE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="ds4-embedding-tools-")
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.root = Path(cls.workspace.name)
        cls.requests, cls.responses, cls.observations = [], [], []
        cls.provider = LocalProvider(
            model=os.getenv("DS4_SUITE_LLM_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            base_url=URL,
            api_key="local",
            api="responses",
            reasoning_effort="none",
            http_client=HTTPClient(
                event_hooks={
                    "request": [cls.capture_request],
                    "response": [cls.capture_response],
                }
            ),
        )
        cls.addClassCleanup(cls.provider.close)
        cls.provider.client.max_retries = 0
        cls.client = Client(
            cls.provider,
            SystemMessage("Remember the user's code across later messages."),
        )
        cls.suite = cls.client.native_suite()
        cls.image = cls.suite.upload(
            ROOT / "tests/fixtures/native-suite-v1/media/image-0.png", kind="image"
        )
        cls.audio = cls.suite.upload(
            ROOT / "tests/fixtures/native-suite-audio-v2/it.wav", kind="audio"
        )
        source = os.getenv("DS4_SUITE_HTTP_VIDEO")
        clip = (
            Path(source)
            if source
            else ROOT / "tests/fixtures/native-suite-e2e/video-with-audio.mp4"
        )
        if not source:
            manifest = json.loads(clip.with_suffix(".json").read_text())
            if hashlib.sha256(clip.read_bytes()).hexdigest() != manifest["sha256"]:
                raise ValueError("The pinned video fixture checksum does not match")
        cls.video = cls.suite.upload(clip, kind="video")

    @classmethod
    def capture_request(cls, request):
        if request.url.path.endswith("/embeddings"):
            cls.requests.append(json.loads(request.content))

    @classmethod
    def capture_response(cls, response):
        if response.request.url.path.endswith(("/embeddings", "/responses")):
            body = json.loads(response.read())
            cls.responses.append(
                {
                    "path": response.request.url.path,
                    "status": response.status_code,
                    "ds4": body.get("ds4"),
                    "usage": body.get("usage"),
                    "error": body.get("error"),
                }
            )

    @classmethod
    def tearDownClass(cls):
        destination = os.getenv("DS4_SUITE_E2E_ARTIFACTS")
        if destination:
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "embeddings-and-consumers.json").write_text(
                json.dumps(
                    {
                        "cases": cls.observations,
                        "requests": cls.requests,
                        "responses": cls.responses,
                    },
                    indent=2,
                )
            )

    def vectors(self, items, dimensions=128, **options):
        response = self.suite.embeddings(items, dimensions=dimensions, **options)
        self.assertEqual(response["model"], "google/embeddinggemma-2")
        self.assertEqual(
            [row["index"] for row in response["data"]], list(range(len(items)))
        )
        vectors = np.asarray([row["embedding"] for row in response["data"]], np.float32)
        self.assertEqual(vectors.shape, (len(items), dimensions))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        self.assertGreater(response["usage"]["prompt_tokens"], 0)
        return vectors

    def test_01_typed_discovery_dimensions_and_tasks(self):
        models = {model.id: model.ds4 for model in self.suite.models()}
        self.assertEqual(
            set(models), {"Qwen/Qwen-Image-2.1-Turbo", "google/embeddinggemma-2"}
        )
        self.assertEqual(
            set(models["google/embeddinggemma-2"].input_modalities),
            {"text", "image", "video", "audio"},
        )
        self.assertTrue(
            all(model.configured and model.ready for model in models.values())
        )
        for dimensions in (128, 256, 512, 768):
            with self.subTest(dimensions=dimensions):
                self.vectors(
                    ["Una montagna innevata", "A snowy mountain"],
                    dimensions,
                    task="similarity",
                )
        for task in ("raw", "search_query", "code_query", "document"):
            with self.subTest(task=task):
                options = {"title": "Alpine winter"} if task == "document" else {}
                self.vectors(["A snowy mountain"], task=task, **options)
        self.observations.append(
            {
                "case": "typed_discovery_dimensions_tasks",
                "dimensions": [128, 256, 512, 768],
                "tasks": [
                    "raw",
                    "similarity",
                    "search_query",
                    "code_query",
                    "document",
                ],
            }
        )

    def test_02_multimodal_batch_order_and_interleaving(self):
        video = self.video.model_copy(update={"audio": "exclude"})
        parts = [TextContent(text="A red square"), self.image, video, self.audio]
        items = [EmbeddingItem(content=[part]) for part in parts]
        items.append(EmbeddingItem(content=parts))
        first = self.vectors(items)
        second = self.vectors(list(reversed(items)))
        np.testing.assert_allclose(first, second[::-1], atol=1e-6, rtol=1e-6)
        self.assertGreater(float(np.linalg.norm(first[0] - first[1])), 0.01)
        mean = first[:4].mean(axis=0)
        mean /= np.linalg.norm(mean)
        self.assertGreater(float(np.linalg.norm(first[4] - mean)), 0.01)
        reversed_parts = self.vectors([EmbeddingItem(content=list(reversed(parts)))])
        self.assertGreater(float(np.linalg.norm(first[4] - reversed_parts[0])), 1e-5)
        usage = self.responses[-1]["ds4"]["usage"]
        for name in ("image_soft_tokens", "video_soft_tokens", "audio_soft_tokens"):
            self.assertGreater(usage[name], 0)
        self.observations.append(
            {"case": "four_modalities_batch_interleaving", "items": 5, "usage": usage}
        )

    def test_03_video_audio_track_is_explicit(self):
        excluded = EmbeddingItem(
            content=[self.video.model_copy(update={"audio": "exclude"})]
        )
        included = EmbeddingItem(
            content=[self.video.model_copy(update={"audio": "include"})]
        )
        without_audio = self.vectors([excluded])
        self.assertEqual(self.responses[-1]["ds4"]["usage"]["audio_soft_tokens"], 0)
        with_audio = self.vectors([included])
        usage = self.responses[-1]["ds4"]["usage"]
        self.assertGreater(usage["audio_soft_tokens"], 0)
        self.assertGreater(usage["video_soft_tokens"], 0)
        self.assertGreater(float(np.linalg.norm(with_audio - without_audio)), 1e-5)
        self.observations.append(
            {"case": "video_audio_include_exclude", "usage_included": usage}
        )

    def test_04_semantic_ranking_uses_real_vectors_and_explicit_cache(self):
        candidates = [
            {
                "title": "Alps",
                "url": "https://example.org/alps",
                "snippet": "Snowy winter mountains",
            },
            {
                "title": "Cooking",
                "url": "https://example.org/food",
                "snippet": "Bread recipes and ovens",
            },
        ]
        query = ranking.build_embedding_text(candidates[0])
        cache = ranking.EmbeddingLRUCache()
        with native_suite_context(self.client.native_suite):
            scores = ranking.compute_semantic_similarity(query, candidates, cache=cache)
            self.assertGreater(scores[0], scores[1])
            self.assertAlmostEqual(scores[0], 1, places=5)
            before = len(self.requests)
            reordered = ranking.compute_semantic_similarity(
                query, candidates[::-1], cache=cache
            )
            np.testing.assert_allclose(scores, reordered[::-1], atol=1e-6)
            self.assertEqual(len(self.requests), before + 1)
            self.assertEqual(self.requests[-1]["input"], [query])
            ranked = ranking.rank_results(query, candidates)
        self.assertEqual(ranked[0][0], 0)
        self.assertEqual(self.responses[-1]["status"], 200)
        self.assertEqual(self.requests[-1]["ds4"]["task"], "similarity")
        self.observations.append(
            {"case": "semantic_ranking_cache", "scores": scores, "ranking": ranked}
        )

    def test_05_visual_ranking_consumes_native_image_and_text_embeddings(self):
        paths = [
            ROOT / f"tests/fixtures/native-suite-v1/media/image-{i}.png" for i in (0, 2)
        ]
        candidates = [
            {
                "title": color,
                "url": f"https://example.org/{color}",
                "snippet": f"A {color} square on white",
            }
            for color in ("red", "blue")
        ]
        discovered = [
            DiscoveredImage(i, row["url"] + ".png", path.read_bytes())
            for i, (row, path) in enumerate(zip(candidates, paths, strict=True))
        ]
        before = len(self.responses)
        with (
            native_suite_context(self.client.native_suite),
            Image.open(paths[0]) as query,
        ):
            backend = ranking.get_image_embedding_backend()
            vectors = backend.encode_images([query, query])
            np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6)
            order, urls = ranking.compute_multimodal_ranking(
                "A red square on white",
                candidates,
                local_image=query,
                discovered_images=discovered,
            )
        responses = self.responses[before:]
        self.assertGreaterEqual(len(responses), 5)
        self.assertTrue(
            all(
                row["path"] == "/v1/embeddings" and row["status"] == 200
                for row in responses
            )
        )
        self.assertEqual(order[0], 0)
        self.assertEqual(urls, {item.candidate_index: item.url for item in discovered})
        self.observations.append(
            {
                "case": "native_visual_ranking",
                "order": order,
                "requests": len(responses),
            }
        )

    def test_06_invalid_operations_leave_service_usable(self):
        baseline = self.vectors(["Recovery check"])
        cases = [
            ([], {}),
            (["text"], {"dimensions": 127}),
            ([" x" * 9000], {}),
            (
                [
                    EmbeddingItem(
                        content=[MediaContent(type="image", file_id="file-missing")]
                    )
                ],
                {},
            ),
        ]
        for items, options in cases:
            with (
                self.subTest(options=options),
                self.assertRaises(APIStatusError) as error,
            ):
                self.suite.embeddings(items, **options)
            self.assertIn(error.exception.status_code, (400, 404))
        np.testing.assert_array_equal(baseline, self.vectors(["Recovery check"]))
        self.observations.append(
            {"case": "validation_overflow_missing_media_recovery", "errors": len(cases)}
        )

    def test_07_llm_ranking_embedding_llm_retains_conversation(self):
        tool = Tool(rank_local_results)
        client = Client(
            self.provider,
            SystemMessage("Use the requested tool and remember the user's code."),
            Tools([tool]),
        )
        first = client.call(
            ClientRequest(
                UserMessage(
                    "Remember EMBEDDING-E2E-27. Call rank_local_results with query snow. After its result reply only with the remembered code."
                ),
                tool_choice="required",
                extra={"max_output_tokens": 256, "temperature": 0},
            )
        )
        calls = first.get_tool_calls()
        self.assertEqual(len(calls or []), 1)
        call = calls[0]
        self.assertEqual(call.get_name(), "rank_local_results")
        with native_suite_context(client.native_suite):
            ranked = CalledTool.from_tool_call(tool, call).execute(work_root=self.root)
        self.assertTrue(np.isfinite(ranked["scores"]).all())
        item = EmbeddingItem(
            content=[
                TextContent(text="Mountains"),
                self.image,
                self.audio,
                self.video.model_copy(update={"audio": "exclude"}),
            ]
        )
        self.vectors([item])
        client.add_tool_response(
            ToolMessage(stringify_tool_result(ranked), call.get_id(), call.get_name())
        )
        final = client.continue_after_tools(
            extra={"max_output_tokens": 48, "temperature": 0}
        )
        self.assertIn("EMBEDDING-E2E-27", final.get_message())
        cached = self.responses[-1]["usage"]["input_tokens_details"]["cached_tokens"]
        self.assertGreater(cached, 0)
        self.vectors([item])
        # Exercise a fresh vision/prefill working set after every native tower,
        # rather than relying only on the small decode buffers already resident.
        with Image.open(
            ROOT / "tests/fixtures/native-suite-v1/media/image-0.png"
        ) as image:
            last = client.call(
                ClientRequest(
                    UserMessage(
                        [
                            "Reference notes:\n"
                            + "Winter weather in the Alpine valley.\n" * 256
                            + "Ignore the reference notes. Return only the code I asked you to remember.",
                            ImagesUserMessage(images=[image.copy(), image.copy()]),
                        ]
                    ),
                    extra={"max_output_tokens": 48, "temperature": 0},
                )
            )
        self.assertIn("EMBEDDING-E2E-27", last.get_message())
        self.assertGreater(self.responses[-1]["usage"]["input_tokens"], 2048)
        self.observations.append(
            {"case": "llm_embedding_consumers_llm", "cached_tokens": cached}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
