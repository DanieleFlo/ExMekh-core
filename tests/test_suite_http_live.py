"""Opt-in native server acceptance; uses the actual OpenAI SDK and native C service."""

import base64
import concurrent.futures
import io
import json
import os
import socket
import time
import unittest
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from jsonschema import Draft202012Validator
from openai import APIStatusError, OpenAI
from PIL import Image

try:
    import av
except ImportError:
    av = None

URL = os.getenv("DS4_SUITE_HTTP_URL")
IMAGE_MODEL = "Qwen/Qwen-Image-2.1-Turbo"
EMBED_MODEL = "google/embeddinggemma-2"


@unittest.skipUnless(URL, "set DS4_SUITE_HTTP_URL for live native acceptance")
class SuiteHttpLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = OpenAI(base_url=URL, api_key="local", timeout=600, max_retries=0)
        cls.root = Path(__file__).resolve().parents[1]
        path = os.getenv("DS4_SUITE_HTTP_VIDEO")
        if path:
            cls.video = Path(path).read_bytes()
        else:
            if av is None:
                raise RuntimeError(
                    "install development PyAV or set DS4_SUITE_HTTP_VIDEO"
                )
            encoded = io.BytesIO()
            with av.open(encoded, "w", format="mp4") as container:
                stream = container.add_stream("libx264", rate=2)
                stream.width, stream.height, stream.pix_fmt = 48, 48, "yuv420p"
                for i in range(4):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((48, 48, 3), i * 50, np.uint8), format="rgb24"
                    )
                    frame.pts = i
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            cls.video = encoded.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def post(self, path, body, key=None):
        return self.client.post(
            path,
            cast_to=dict,
            body=body,
            options={"headers": {"Idempotency-Key": key}} if key else {},
        )

    def upload(self, path, mime):
        with path.open("rb") as f:
            return self.client.files.create(
                file=(path.name, f, mime), purpose="ds4_media"
            )

    def test_01_sdk_text_tokens_and_identity(self):
        catalog = self.root / "schemas/ds4-suite-catalog-v1.schema.json"
        schema = json.loads(catalog.read_text())
        schema = {"$defs": schema["$defs"], "$ref": "#/$defs/model"}
        entries = self.client.models.list().data
        native = [
            row.model_dump(exclude_unset=True)
            for row in entries
            if row.id in (IMAGE_MODEL, EMBED_MODEL)
        ]
        self.assertEqual(len(native), 2)
        for row in native:
            Draft202012Validator(schema).validate(row)
            self.assertEqual(row["ds4"]["limits"]["max_inflight_requests"], 8)
        response = self.client.embeddings.create(
            model=EMBED_MODEL, input=["ciao", "hello"], dimensions=128
        )
        self.assertEqual(len(response.data), 2)
        for row in response.data:
            self.assertAlmostEqual(float(np.linalg.norm(row.embedding)), 1, places=5)
        plain = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[2, 105, 1],
            dimensions=128,
            encoding_format="float",
        )
        padded = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[2, 105, 1, 0, 0],
            dimensions=128,
            encoding_format="base64",
        )
        encoded = self.post(
            "/embeddings",
            {
                "model": EMBED_MODEL,
                "input": [2, 105, 1],
                "dimensions": 128,
                "encoding_format": "base64",
            },
        )
        values = np.frombuffer(
            base64.b64decode(encoded["data"][0]["embedding"]), dtype="<f4"
        )
        np.testing.assert_array_equal(
            np.asarray(plain.data[0].embedding, dtype=np.float32), values
        )
        padded_values = np.frombuffer(
            base64.b64decode(padded.data[0].embedding), dtype="<f4"
        )
        np.testing.assert_array_equal(
            np.asarray(plain.data[0].embedding, dtype=np.float32), padded_values
        )
        self.assertEqual(
            response.model_extra["ds4"]["revision"],
            "914f7f89142e33e77833254d9c9b90c3cef7303b",
        )

    def test_02_validation_and_receipts(self):
        body = {"model": EMBED_MODEL, "input": "test", "dimensions": 128}
        a = self.post("/embeddings", body, "text-receipt")
        b = self.post("/embeddings", dict(reversed(list(body.items()))), "text-receipt")
        self.assertEqual(a, b)
        with self.assertRaises(APIStatusError) as conflict:
            self.post("/embeddings", {**body, "input": "different"}, "text-receipt")
        self.assertEqual(conflict.exception.status_code, 409)
        for extra in (
            {"dimensions": 64},
            {"ds4": {"version": 2}},
            {"unknown": True},
            {"input": [0, 0]},
            {"input": [258880]},
            {"input": [262144]},
        ):
            with self.subTest(extra=extra), self.assertRaises(APIStatusError) as error:
                self.post("/embeddings", {**body, **extra})
            self.assertEqual(error.exception.status_code, 400)
            self.assertTrue(
                error.exception.response.headers["x-request-id"].startswith("req-")
            )
        with self.assertRaises(APIStatusError) as overflow:
            self.post("/embeddings", {**body, "input": [2] * 8193})
        self.assertEqual(overflow.exception.body["code"], "context_length_exceeded")
        for size in ("4294967552x32", "256x18446744073709551616", "256x+256"):
            with self.subTest(size=size), self.assertRaises(APIStatusError) as invalid:
                self.post(
                    "/images/generations",
                    {"model": IMAGE_MODEL, "prompt": "red", "size": size},
                )
            self.assertEqual(invalid.exception.status_code, 400)

    def test_03_four_modalities_and_composition(self):
        image = self.upload(
            self.root / "tests/fixtures/native-suite-v1/media/image-0.png", "image/png"
        )
        audio = self.upload(
            self.root / "tests/fixtures/native-suite-audio-v2/it.wav", "audio/wav"
        )
        video = self.client.files.create(
            file=("video.mp4", self.video, "video/mp4"), purpose="ds4_media"
        )
        image_part = {"type": "image", "file_id": image.id}
        audio_part = {"type": "audio", "file_id": audio.id}
        video_part = {"type": "video", "file_id": video.id, "audio": "exclude"}
        parts = [
            [image_part],
            [audio_part],
            [video_part],
            [{"type": "text", "text": "ciao"}, image_part, video_part, audio_part],
        ]
        result = self.post(
            "/embeddings",
            {
                "model": EMBED_MODEL,
                "input": [{"content": item} for item in parts],
                "dimensions": 128,
                "ds4": {"version": 1},
            },
        )
        self.assertEqual(len(result["data"]), 4)
        vectors = np.array([item["embedding"] for item in result["data"]])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        self.assertGreater(float(np.linalg.norm(vectors[0] - vectors[1])), 0.01)
        self.assertEqual(
            self.client.files.content(image.id).read(),
            (
                self.root / "tests/fixtures/native-suite-v1/media/image-0.png"
            ).read_bytes(),
        )

    def test_04_generation_retry_edit_and_chat_resume(self):
        tools = [
            {
                "type": "function",
                "name": "generate_image",
                "description": "Generate the requested image.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            }
        ]
        user = {
            "role": "user",
            "content": "Remember code ALFA17. Call generate_image with prompt 'A red square on white.'. After its result, reply with the code.",
        }
        before = self.client.responses.create(
            model="Qwen3.8-27B-UD-Q4_K_S",
            input=[user],
            tools=tools,
            tool_choice="required",
            max_output_tokens=128,
            temperature=0,
            reasoning={"effort": "none"},
        )
        calls = [item for item in before.output if item.type == "function_call"]
        self.assertEqual(len(calls), 1)
        body = {
            "model": IMAGE_MODEL,
            "prompt": "A red square on white.",
            "size": "256x256",
            "n": 1,
            "ds4": {"version": 1, "seed": 18446744073709551614},
        }
        key = "image-" + os.urandom(8).hex()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(
                    lambda _: self.post("/images/generations", body, key), range(2)
                )
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["data"][0]["ds4"]["seed"], 18446744073709551614)
        png = base64.b64decode(results[0]["data"][0]["b64_json"])
        self.assertEqual(Image.open(io.BytesIO(png)).size, (256, 256))
        self.assertEqual(
            self.client.files.content(results[0]["data"][0]["ds4"]["file_id"]).read(),
            png,
        )
        edit = self.client.images.edit(
            model=IMAGE_MODEL,
            prompt="Make it blue",
            size="256x256",
            image=("reference.png", png, "image/png"),
            extra_body={"ds4": {"version": 1, "seed": 17}},
        )
        self.assertEqual(len(edit.data), 1)
        after = self.client.responses.create(
            model="Qwen3.8-27B-UD-Q4_K_S",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": "Image generated successfully. Return the remembered code.",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=32,
            temperature=0,
            reasoning={"effort": "none"},
        )
        self.assertIn("ALFA17", after.output_text)
        self.assertGreater(after.usage.input_tokens_details.cached_tokens, 0)

    def test_05_multiple_outputs_and_image_only(self):
        result = self.client.images.generate(
            model=IMAGE_MODEL,
            prompt="A blue square",
            size="64x96",
            n=2,
            background="opaque",
            extra_body={"ds4": {"version": 1, "seed": 21}},
        ).model_dump(exclude_unset=True)
        self.assertEqual([row["ds4"]["seed"] for row in result["data"]], [21, 22])
        for row in result["data"]:
            image = Image.open(io.BytesIO(base64.b64decode(row["b64_json"])))
            self.assertEqual(image.size, (64, 96))
            self.assertEqual(image.getchannel("A").getextrema(), (255, 255))
        with self.assertRaises(APIStatusError) as unsupported:
            self.post(
                "/images/edits",
                {
                    "model": IMAGE_MODEL,
                    "prompt": "",
                    "images": [result["data"][0]["ds4"]["file_id"]],
                    "size": "64x96",
                    "ds4": {"version": 1},
                },
            )
        self.assertEqual(unsupported.exception.status_code, 400)
        # References retain their supplied geometry: upload an aligned valid
        # Qwen3-VL reference rather than silently resizing it on the server.
        source = Image.open(
            self.root / "tests/fixtures/native-suite-v1/media/image-0.png"
        ).resize((256, 256))
        encoded = io.BytesIO()
        source.save(encoded, format="PNG")
        reference = self.client.files.create(
            file=("aligned.png", encoded.getvalue(), "image/png"), purpose="ds4_media"
        )
        refs = [reference.id, reference.id]
        edited = self.post(
            "/images/edits",
            {
                "model": IMAGE_MODEL,
                "prompt": "",
                "images": refs,
                "size": "64x96",
                "ds4": {"version": 1, "seed": 23},
            },
        )
        self.assertEqual(len(edited["data"]), 1)

    def test_06_disconnect_drains_and_recovers(self):
        target = urlparse(URL)
        body = json.dumps(
            {
                "model": IMAGE_MODEL,
                "prompt": "A green triangle",
                "size": "256x256",
                "ds4": {"version": 1, "seed": 24},
            }
        ).encode()
        with socket.create_connection((target.hostname, target.port or 80)) as peer:
            peer.sendall(
                (
                    f"POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
                    f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n"
                ).encode()
                + body
            )
            time.sleep(1)
        result = self.client.embeddings.create(
            model=EMBED_MODEL, input="after cancellation", dimensions=128
        )
        self.assertEqual(len(result.data[0].embedding), 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
