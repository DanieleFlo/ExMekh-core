"""Tokenizer/task contracts against the frozen asset, without loading a model.

Set DS4_SUITE_PREPARED to the frozen reference directory. Model-level CUDA
qualification uses probe_native_suite and run_embedding_gemma2_oracle.py.
"""

import ctypes as C
import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compile_embedding_gemma2 import compile_asset


class Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = Path(
            os.environ.get(
                "DS4_SUITE_PREPARED",
                ROOT / "performance-results/native-suite-oracles-20261020",
            )
        )
        cls.model = cls.prepared / "models/google/embeddinggemma-2"
        if not (cls.model / "tokenizer.json").is_file():
            raise unittest.SkipTest("frozen tokenizer assets unavailable")
        sys.path.insert(0, str(cls.prepared / "deps"))
        from tokenizers import Tokenizer as Reference

        cls.reference = Reference.from_file(str(cls.model / "tokenizer.json"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.asset = Path(cls.temp.name) / "gemma.ds4g"
        compile_asset(cls.model, cls.asset)
        cls.hashes = b"".join(
            hashlib.sha256((cls.model / f).read_bytes()).digest()
            for f in (
                "tokenizer.json",
                "config.json",
                "config_sentence_transformers.json",
            )
        )
        cls.lib = C.CDLL(
            os.environ.get(
                "DS4_SUITE_CPU_LIBRARY", str(ROOT / "tests/libds4_suite_cpu.so")
            )
        )
        cls.lib.gemma_tokenizer_open.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_char_p,
        ]
        cls.lib.gemma_tokenizer_open.restype = C.c_int
        cls.lib.gemma_tokenizer_close.argtypes = [C.c_void_p]
        cls.lib.gemma_tokenize.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.c_uint32,
            C.POINTER(C.POINTER(C.c_uint32)),
            C.POINTER(C.c_uint32),
        ]
        cls.lib.gemma_tokenize.restype = C.c_int
        cls.libc = C.CDLL(None)
        cls.libc.free.argtypes = [C.c_void_p]
        cls.tokenizer = C.c_void_p()
        assert (
            cls.lib.gemma_tokenizer_open(
                C.byref(cls.tokenizer), os.fsencode(cls.asset), cls.hashes
            )
            == 0
        )

    @classmethod
    def tearDownClass(cls):
        cls.lib.gemma_tokenizer_close(cls.tokenizer)
        cls.temp.cleanup()

    def encode(self, text, task=0):
        ids = C.POINTER(C.c_uint32)()
        count = C.c_uint32()
        status = self.lib.gemma_tokenize(
            self.tokenizer, text.encode(), task, C.byref(ids), C.byref(count)
        )
        self.assertEqual(status, 0)
        result = list(ids[: count.value])
        self.libc.free(ids)
        return result

    def test_character_bpe_unicode_whitespace_and_specials(self):
        cases = [
            "",
            " ",
            "   ",
            "\n\n\t",
            "Caffe\u0300 e caffè",
            "中文 日本語 العربية Ελληνικά",
            "👨‍👩‍👧‍👦 🌈 \U0001fae8",
            "def f(x):\n\treturn x**2\n",
            "<bos><eos><|image|>",
            "<unused999> A<pad>B",
            "a" * 2000,
            " " * 512,
            "\U00010fff\U0001fffe",
        ]
        rng = random.Random(20261021)
        alphabet = list(" abcdef01234\n\téè中日本語") + [
            "🌈",
            "😀",
            "\u0301",
            "\U00010fff",
        ]
        cases += [
            "".join(rng.choices(alphabet, k=rng.randrange(1, 200))) for _ in range(128)
        ]
        for text in cases:
            with self.subTest(text=text[:40]):
                self.assertEqual(self.encode(text), self.reference.encode(text).ids)

    def test_task_prefixes_and_formatted_title(self):
        prompts = json.loads(
            (self.model / "config_sentence_transformers.json").read_text()
        )["prompts"]
        for task, name in enumerate(
            ("query", "document", "CodeRetrieval", "SentenceSimilarity"), 1
        ):
            text = "Trova il codice per una ricerca."
            self.assertEqual(
                self.encode(text, task), self.reference.encode(prompts[name] + text).ids
            )
            self.assertEqual(
                self.encode(prompts[name] + text, task),
                self.reference.encode(prompts[name] + text).ids,
            )
            formatted = "title: Nome originale | text: Corpo del documento"
            self.assertEqual(
                self.encode(formatted, task), self.reference.encode(formatted).ids
            )

    def test_corruption_and_identity_rejected(self):
        data = self.asset.read_bytes()
        bad = self.asset.parent / "bad.ds4g"
        for changed in (
            data[:104] + bytes([data[104] ^ 1]) + data[105:],
            data[:-1],
            data[:8] + bytes(96) + data[104:],
        ):
            bad.write_bytes(changed)
            handle = C.c_void_p()
            self.assertNotEqual(
                self.lib.gemma_tokenizer_open(
                    C.byref(handle), os.fsencode(bad), self.hashes
                ),
                0,
            )
            self.assertFalse(handle.value)

    def test_context_and_invalid_utf8_rejected_without_output(self):
        for text, task in [(b"word " * 9000, 0), (b"\xff", 0), (b"abc", 99)]:
            ids = C.POINTER(C.c_uint32)()
            count = C.c_uint32()
            self.assertNotEqual(
                self.lib.gemma_tokenize(
                    self.tokenizer, text, task, C.byref(ids), C.byref(count)
                ),
                0,
            )
            self.assertFalse(ids)
            self.assertEqual(count.value, 0)


if __name__ == "__main__":
    unittest.main()
