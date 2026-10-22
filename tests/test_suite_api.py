"""Failure outputs must never expose stale ownership to a native API caller."""

import ctypes as C
import os
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


if __name__ == "__main__":
    unittest.main()
