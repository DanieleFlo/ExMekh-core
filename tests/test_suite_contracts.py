"""Validate the frozen API contracts, without claiming HTTP availability."""

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/ds4-suite-v1.schema.json").read_text())


def validator(name):
    schema = copy.deepcopy(SCHEMA)
    schema["$ref"] = f"#/$defs/{name}"
    return Draft202012Validator(schema)


class SuiteContracts(unittest.TestCase):
    def test_schema(self):
        Draft202012Validator.check_schema(SCHEMA)

    def test_catalog(self):
        schema = json.loads(
            (ROOT / "schemas/ds4-suite-catalog-v1.schema.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        request = {
            "object": "list",
            "data": [
                {
                    "id": "gemma",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "ds4": {
                        "version": 1,
                        "revision": "sha",
                        "profile": "bf16",
                        "operations": ["embed"],
                        "input_modalities": ["text", "image", "video", "audio"],
                        "context_tokens": 8192,
                        "embedding_dimensions": [128, 256, 512, 768],
                        "configured": True,
                        "resident": False,
                        "ready": True,
                        "qualified_backends": ["cuda-sm86"],
                        "components": [
                            {
                                "id": "audio",
                                "optional": True,
                                "input_modalities": ["audio"],
                                "storage_format": "safetensors-bf16",
                                "compute_dtype": "bfloat16",
                                "host_loaded": False,
                                "device_resident": False,
                            }
                        ],
                    },
                }
            ],
        }
        Draft202012Validator(schema).validate(request)
        bad = copy.deepcopy(request)
        bad["data"][0]["ds4"]["qualified_backends"] = []
        self.assertFalse(Draft202012Validator(schema).is_valid(bad))
        bad = copy.deepcopy(request)
        bad["data"][0]["ds4"]["embedding_dimensions"] = []
        self.assertFalse(Draft202012Validator(schema).is_valid(bad))
        request["data"][0]["ds4"]["operations"] = ["transcribe"]
        self.assertFalse(Draft202012Validator(schema).is_valid(request))

    def test_standard_embeddings(self):
        for value in ("ciao", ["ciao", "hello"], [1, 2], [[1], [2, 3]]):
            with self.subTest(value=value):
                validator("embeddings").validate(
                    {"model": "google/embeddinggemma-2", "input": value}
                )

    def test_media_and_interleaving(self):
        for content in (
            [{"type": "image", "file_id": "file-image"}],
            [{"type": "audio", "file_id": "file-audio"}],
            [{"type": "video", "file_id": "file-video", "audio": "include"}],
            [
                {"type": "text", "text": "description"},
                {"type": "image", "file_id": "file-image"},
                {"type": "audio", "file_id": "file-audio"},
            ],
        ):
            for dimension in (128, 256, 512, 768):
                validator("embeddings").validate(
                    {
                        "model": "gemma",
                        "input": [{"content": content}],
                        "dimensions": dimension,
                        "ds4": {"version": 1},
                    }
                )

    def test_embedding_rejections(self):
        baseline = {"model": "gemma", "input": "text"}
        cases = [
            {"input": []},
            {"input": ""},
            {"input": [[]]},
            {"input": [-1]},
            {"input": [2147483648]},
            {"dimensions": 1024},
            {"dimensions": True},
            {"input": [1, 2], "ds4": {"version": 1, "task": "search_query"}},
            {"encoding_format": "int8"},
            {"extra": 1},
            {"ds4": {"version": 2}},
            {"ds4": {"version": 1, "overflow": "truncate"}},
            {"ds4": {"version": 1, "title": "title"}},
            {"input": [{"content": [{"type": "image", "file_id": "file-1"}]}]},
            {"input": [{"content": []}], "ds4": {"version": 1}},
            {
                "input": [
                    {
                        "content": [
                            {"type": "audio", "file_id": "file-1", "audio": "include"}
                        ]
                    }
                ],
                "ds4": {"version": 1},
            },
        ]
        for change in cases:
            with self.subTest(change=change):
                self.assertFalse(
                    validator("embeddings").is_valid(dict(baseline, **change))
                )

    def test_images(self):
        request = {"model": "Qwen/Qwen-Image-2.1-Turbo", "prompt": "A red square"}
        validator("image_generation").validate(request)
        validator("image_edit").validate(
            dict(request, images=["file-1", "file-2"], ds4={"version": 1})
        )
        validator("image_edit").validate(
            dict(request, prompt="", images=["file-1"], ds4={"version": 1})
        )
        for change in (
            {"prompt": ""},
            {"n": 0},
            {"n": 9},
            {"size": "0x1024"},
            {"mask": "file-1"},
            {"output_format": "jpeg", "background": "transparent"},
            {"ds4": {"version": 1, "seed": -1}},
            {"ds4": {"version": 1, "seed": 18446744073709551616}},
        ):
            self.assertFalse(
                validator("image_generation").is_valid(dict(request, **change))
            )

    def test_upload(self):
        request = {
            "purpose": "ds4_media",
            "filename": "clip.mp4",
            "bytes": 1024,
            "content_type": "video/mp4",
        }
        validator("upload_metadata").validate(request)
        self.assertFalse(
            validator("upload_metadata").is_valid(dict(request, bytes=536870913))
        )
        self.assertFalse(
            validator("upload_metadata").is_valid(
                dict(request, content_type="application/octet-stream")
            )
        )


if __name__ == "__main__":
    unittest.main()
