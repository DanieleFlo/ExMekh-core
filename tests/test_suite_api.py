"""Failure outputs must never expose stale ownership to a native API caller."""

import ctypes as C
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


class FailureOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(
            os.environ.get(
                "DS4_SUITE_CPU_LIBRARY",
                str(Path(__file__).resolve().parent / "libds4_suite_cpu.so"),
            )
        )

    def rejected(self, name, args, output_index):
        function = getattr(self.lib, name)
        function.argtypes = [C.c_void_p] * len(args)
        function.restype = C.c_int
        output = C.c_void_p(1)
        args[output_index] = C.byref(output)
        self.assertNotEqual(function(*args), 0)
        self.assertIsNone(output.value, name)

    def test_adapter_constructors_clear_output(self):
        for name, count in (
            ("ds4_embedding_gemma2_open", 5),
            ("ds4_gemma_visual_open", 6),
            ("ds4_gemma_vision_open", 5),
            ("ds4_gemma_audio_open", 4),
            ("ds4_gemma_processor_open", 6),
            ("ds4_media_decode", 7),
            ("ds4_qwen_image_pipeline_open", 5),
            ("ds4_qwen_image_encoder_open", 5),
            ("ds4_qwen_image_dit_open", 4),
            ("ds4_qwen_image_vae_open", 4),
            ("gemma_tokenizer_open", 3),
            ("ds4_qwen_image_tokenizer_open", 3),
        ):
            with self.subTest(api=name):
                self.rejected(name, [None] * count, 0)

    def test_prepare_and_execution_clear_output(self):
        for name, count in (
            ("ds4_gemma_prepare", 4),
            ("ds4_gemma_embed", 5),
            ("ds4_gemma_visual_prepare", 4),
            ("ds4_gemma_visual_embed", 5),
            ("ds4_gemma_vision_encode", 4),
            ("ds4_gemma_audio_encode", 4),
            ("ds4_gemma_multimodal_prepare", 4),
            ("ds4_gemma_media_prepare", 4),
            ("ds4_qwen_image_encode", 4),
            ("ds4_qwen_image_prepare", 5),
            ("ds4_qwen_image_prepare_padded", 6),
            ("ds4_qwen_image_input_create", 7),
            ("ds4_qwen_image_job_create", 6),
            ("ds4_qwen_image_job_take", 2),
        ):
            with self.subTest(api=name):
                # Job creation has an out-first signature; other operations
                # return their new object through the last argument.
                index = 0 if name == "ds4_qwen_image_job_create" else count - 1
                self.rejected(name, [None] * count, index)

    def test_tokenizers_clear_independent_outputs(self):
        for name, count_type, task in (
            ("gemma_tokenize", C.c_uint32, [0]),
            ("ds4_qwen_image_tokenize", C.c_size_t, []),
        ):
            function = getattr(self.lib, name)
            function.argtypes = [C.c_void_p, C.c_char_p]
            if task:
                function.argtypes += [C.c_uint32]
            function.argtypes += [C.POINTER(C.c_void_p), C.POINTER(count_type)]
            for missing in ("neither", "tokens", "count"):
                with self.subTest(api=name, missing=missing):
                    tokens, count = C.c_void_p(1), count_type(99)
                    self.assertNotEqual(
                        function(
                            None,
                            b"test",
                            *task,
                            None if missing == "tokens" else C.byref(tokens),
                            None if missing == "count" else C.byref(count),
                        ),
                        0,
                    )
                    if missing != "tokens":
                        self.assertIsNone(tokens.value)
                    if missing != "count":
                        self.assertEqual(count.value, 0)

    def test_numeric_allocation_clears_output(self):
        create = self.lib.ds4_numeric_cpu
        create.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64]
        alloc = self.lib.ds4_num_alloc
        alloc.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.POINTER(C.c_void_p)]
        for api in ("create", "alloc"):
            output = C.c_void_p(1)
            status = (
                create(C.byref(output), None, 1)
                if api == "create"
                else alloc(None, 1, 30, C.byref(output))
            )
            self.assertNotEqual(status, 0)
            self.assertIsNone(output.value, api)

    def test_borrowed_token_queries_clear_independent_outputs(self):
        for name in ("ds4_gemma_input_tokens", "ds4_gemma_visual_input_tokens"):
            function = getattr(self.lib, name)
            function.argtypes = [
                C.c_void_p,
                C.c_size_t,
                C.POINTER(C.c_void_p),
                C.POINTER(C.c_uint32),
            ]
            for missing in ("neither", "tokens", "count"):
                tokens, count = C.c_void_p(1), C.c_uint32(99)
                self.assertNotEqual(
                    function(
                        None,
                        0,
                        None if missing == "tokens" else C.byref(tokens),
                        None if missing == "count" else C.byref(count),
                    ),
                    0,
                )
                if missing != "tokens":
                    self.assertIsNone(tokens.value, (name, missing))
                if missing != "count":
                    self.assertEqual(count.value, 0, (name, missing))

    def test_encoder_residency_queries_accept_null(self):
        for name in (
            "ds4_embedding_gemma2_resident_weights",
            "ds4_gemma_vision_resident_weights",
            "ds4_gemma_audio_resident_weights",
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = [C.c_void_p], C.c_uint64
            self.assertEqual(function(None), 0, name)

    def test_v1_plan_output_does_not_overwrite_callers_storage(self):
        class Estimate(C.Structure):
            _fields_ = [
                (name, C.c_uint64)
                for name in ("device", "host", "pinned", "checkpoint")
            ] + [("known", C.c_bool)]

        class Plan(C.Structure):
            _fields_ = [
                ("peak", Estimate),
                ("reclaim", C.c_uint64),
                ("modalities", C.c_uint32),
                ("release", C.c_bool),
            ]

        class Guard(C.Structure):
            _fields_ = [("plan", Plan), ("canary", C.c_uint64)]

        output = Guard()
        output.canary = 0x123456789ABCDEF0
        function = self.lib.ds4_gemma_visual_plan
        function.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Plan)]
        self.assertNotEqual(function(None, None, C.byref(output.plan)), 0)
        self.assertEqual(output.canary, 0x123456789ABCDEF0)

    def test_unqualified_profiles_rejected_before_weight_binding(self):
        from tools.suite_bundle import compile_bundle

        lib = self.lib
        lib.ds4_resources_create.argtypes = [C.POINTER(C.c_void_p)]
        lib.ds4_runtime_create.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_void_p,
        ]
        lib.ds4_numeric_cpu.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64]
        lib.ds4_bundle_open.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_char_p,
            C.c_void_p,
            C.c_uint64,
        ]
        for api in (
            "ds4_bundle_close",
            "ds4_numeric_close",
            "ds4_resources_close",
            "ds4_runtime_release",
        ):
            getattr(lib, api).argtypes = [C.c_void_p]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = json.dumps(
                {"fixture": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            ).encode()
            weights = root / "weights.safetensors"
            weights.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            asset = root / "config.json"
            asset.write_text('{"fixture":true}')
            resources, runtime, numeric = C.c_void_p(), C.c_void_p(), C.c_void_p()
            self.assertEqual(lib.ds4_resources_create(C.byref(resources)), 0)
            self.assertEqual(lib.ds4_runtime_create(C.byref(runtime), None, None), 0)
            self.assertEqual(lib.ds4_numeric_cpu(C.byref(numeric), resources, 500), 0)
            for model, revision, profile, api in (
                (
                    "Qwen/Qwen-Image-2.1-Turbo",
                    "fc13ba1986fcb905854dc8de2f52eee395f49b5b",
                    "q8-storage",
                    "ds4_qwen_image_encoder_open",
                ),
                (
                    "Qwen/Qwen-Image-2.1-Turbo",
                    "fc13ba1986fcb905854dc8de2f52eee395f49b5b",
                    "dit-q4-k-m",
                    "ds4_qwen_image_dit_open",
                ),
                (
                    "google/embeddinggemma-2",
                    "914f7f89142e33e77833254d9c9b90c3cef7303b",
                    "q8-bf16",
                    "ds4_embedding_gemma2_open",
                ),
            ):
                with self.subTest(profile=profile):
                    spec = {
                        "version": 1,
                        "model": model,
                        "revision": revision,
                        "profile": profile,
                        "assets": {"config": asset.name},
                        "components": [
                            {
                                "id": "dit",
                                "revision": revision,
                                "for_model_revision": revision,
                                "compute_dtype": "bfloat16",
                                "selectors": [{"file": weights.name, "prefix": ""}],
                            }
                        ],
                        "files": [
                            {
                                "path": weights.name,
                                "revision": revision,
                                "format": "safetensors",
                                "bytes": weights.stat().st_size,
                                "sha256": hashlib.sha256(
                                    weights.read_bytes()
                                ).hexdigest(),
                            },
                            {
                                "path": asset.name,
                                "revision": revision,
                                "format": "asset",
                                "bytes": asset.stat().st_size,
                                "sha256": hashlib.sha256(
                                    asset.read_bytes()
                                ).hexdigest(),
                            },
                        ],
                    }
                    directory = root / profile
                    compile_bundle(spec, root, directory)
                    bundle = C.c_void_p()
                    self.assertEqual(
                        lib.ds4_bundle_open(
                            C.byref(bundle),
                            os.fsencode(directory / "bundle.ds4b"),
                            os.fsencode(root),
                            resources,
                            501,
                        ),
                        0,
                    )
                    function = getattr(lib, api)
                    function.argtypes = [
                        C.POINTER(C.c_void_p),
                        C.c_void_p,
                        C.c_void_p,
                        C.c_void_p,
                    ]
                    args = [C.c_void_p(1), bundle, numeric, runtime]
                    output = args[0]
                    args[0] = C.byref(output)
                    if api != "ds4_qwen_image_dit_open":
                        function.argtypes += [C.c_char_p]
                        args.append(b"unused-asset")
                    self.assertEqual(function(*args), 2)
                    self.assertIsNone(output.value)
                    self.assertEqual(lib.ds4_bundle_close(bundle), 0)
            self.assertEqual(lib.ds4_numeric_close(numeric), 0)
            lib.ds4_runtime_release(runtime)
            self.assertEqual(lib.ds4_resources_close(resources), 0)


if __name__ == "__main__":
    unittest.main()
