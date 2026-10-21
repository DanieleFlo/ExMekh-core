import ctypes as C
import hashlib
import json
import os
import re
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ImageTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(str(ROOT / "tests/libds4_qwen_image_tokenizer.so"))
        cls.lib.ds4_qwen_image_tokenizer_open.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_void_p,
        ]
        cls.lib.ds4_qwen_image_tokenizer_open.restype = C.c_int
        cls.lib.ds4_qwen_image_tokenizer_close.argtypes = [C.c_void_p]
        cls.lib.ds4_qwen_image_tokenize.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.POINTER(C.POINTER(C.c_uint32)),
            C.POINTER(C.c_size_t),
        ]
        cls.lib.ds4_qwen_image_tokenize.restype = C.c_int
        cls.lib.ds4_qwen_image_prompt.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.POINTER(C.POINTER(C.c_uint32)),
            C.POINTER(C.c_size_t),
            C.POINTER(C.c_size_t),
        ]
        cls.lib.ds4_qwen_image_prompt.restype = C.c_int
        cls.libc = C.CDLL(None)
        cls.libc.free.argtypes = [C.c_void_p]
        cls.lib.ds4_unicode_nfc.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        cls.lib.ds4_unicode_nfc.restype = C.c_int

    def read(self, ctx, text):
        ids = C.POINTER(C.c_uint32)()
        count = C.c_size_t()
        self.assertEqual(
            self.lib.ds4_qwen_image_tokenize(
                ctx, text.encode(), C.byref(ids), C.byref(count)
            ),
            0,
        )
        result = list(ids[: count.value])
        self.libc.free(ids)
        return result

    def test_native_asset_validation_and_byte_bpe(self):
        sha = hashlib.sha256(b"fixture").digest()
        payload = (
            struct.pack("<5I", 1, 0, 0, 0, 0)
            + struct.pack("<256I", *range(256))
            + struct.pack("<3I", 97, 98, 256)
        )
        content = b"DS4QTK1\0" + sha + hashlib.sha256(payload).digest() + payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.ds4q"
            path.write_bytes(content)
            ctx = C.c_void_p()
            self.assertEqual(
                self.lib.ds4_qwen_image_tokenizer_open(
                    C.byref(ctx), os.fsencode(path), sha
                ),
                0,
            )
            self.assertEqual(self.read(ctx, "abab ac"), [256, 256, 32, 97, 99])
            self.lib.ds4_qwen_image_tokenizer_close(ctx)
            self.assertEqual(
                self.lib.ds4_qwen_image_tokenizer_open(
                    C.byref(ctx), os.fsencode(path), bytes(32)
                ),
                1,
            )
            for length in (0, 100, len(content) - 1):
                path.write_bytes(content[:length])
                self.assertEqual(
                    self.lib.ds4_qwen_image_tokenizer_open(
                        C.byref(ctx), os.fsencode(path), sha
                    ),
                    1,
                )
            corrupt = bytearray(content)
            corrupt[-1] ^= 1
            path.write_bytes(corrupt)
            self.assertEqual(
                self.lib.ds4_qwen_image_tokenizer_open(
                    C.byref(ctx), os.fsencode(path), sha
                ),
                1,
            )

    def test_pinned_processor_unicode_templates_and_reference_order(self):
        source = os.environ.get("DS4_SUITE_TOKENIZER_JSON")
        if not source:
            self.skipTest(
                "set DS4_SUITE_TOKENIZER_JSON to the verified processor asset"
            )
        from tokenizers import Tokenizer

        from tools.compile_qwen_image_tokenizer import compile_tokenizer

        source = Path(source)
        oracle = Tokenizer.from_file(str(source))
        recipe = json.loads((ROOT / "config/qwen-image-encoder.json").read_text())
        examples = [
            "",
            "Un cerchio blu, perché?",
            "1234567890 １２３",
            "e\u0301 é Ελληνικά العربية 中文 日本語",
            "We're I'LL ain't\t\n  ",
            "<|im_start|>user\n<image2><|vision_start|><|image_pad|><|vision_end|>",
            "👩🏽‍💻 🏳️‍🌈 \r\n",
            "x    y\n\n \t\n",
            "// C++\nint main() { return 42; }",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.ds4q"
            compile_tokenizer(source, path)
            ctx = C.c_void_p()
            sha = hashlib.sha256(source.read_bytes()).digest()
            self.assertEqual(
                self.lib.ds4_qwen_image_tokenizer_open(
                    C.byref(ctx), os.fsencode(path), sha
                ),
                0,
            )
            try:
                for text in examples:
                    self.assertEqual(
                        self.read(ctx, text), oracle.encode(text).ids, text
                    )
                # Exercise both sides of every frozen character-class range.
                # The legacy tokenizer uses newer Unicode classes and differs
                # on thousands of scalars, although ordinary prose matches.
                table = (ROOT / "third_party/unicode/image_classes.inc").read_text()
                points = {0x88F, 0xC5C, 0xCDC, 0x11DE0, 0x16FF4}
                for lo, hi in re.findall(r"\{(0x[0-9a-f]+), (0x[0-9a-f]+)\}", table):
                    lo, hi = int(lo, 16), int(hi, 16)
                    points.update((lo - 1, lo, hi, hi + 1))
                for cp in sorted(points):
                    if cp <= 0 or cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
                        continue
                    text = "a" + chr(cp) + "b 1" + chr(cp) + "2\n"
                    self.assertEqual(
                        self.read(ctx, text), oracle.encode(text).ids, hex(cp)
                    )
                for text in examples[:5]:
                    for counts in ([], [4], [4, 9]):
                        images = (C.c_uint32 * len(counts))(*counts)
                        ids = C.POINTER(C.c_uint32)()
                        count = C.c_size_t()
                        drop = C.c_size_t()
                        self.assertEqual(
                            self.lib.ds4_qwen_image_prompt(
                                ctx,
                                text.encode(),
                                images,
                                len(counts),
                                C.byref(ids),
                                C.byref(count),
                                C.byref(drop),
                            ),
                            0,
                        )
                        tags = " ".join(
                            f"<image{i + 1}><|vision_start|>"
                            + "<|image_pad|>" * n
                            + "<|vision_end|>"
                            for i, n in enumerate(counts)
                        )
                        expected = oracle.encode(
                            recipe["prefix"].format(**recipe)
                            + tags
                            + (text or recipe["empty_prompt"])
                            + recipe["suffix"]
                        ).ids
                        self.assertEqual(list(ids[: count.value]), expected)
                        self.assertEqual(drop.value, 14)
                        self.libc.free(ids)
            finally:
                self.lib.ds4_qwen_image_tokenizer_close(ctx)
        # Exhaustively compare scalar values, plus canonical ordering,
        # exclusions, astral compositions and algorithmic Hangul recomposition.
        for first in range(0, 0x110000, 32768):
            text = "".join(
                chr(cp)
                for cp in range(first, min(first + 32768, 0x110000))
                if cp and not 0xD800 <= cp <= 0xDFFF
            )
            result = C.c_void_p()
            self.assertEqual(
                self.lib.ds4_unicode_nfc(text.encode(), C.byref(result)), 0
            )
            self.assertEqual(
                C.string_at(result).decode(), oracle.normalizer.normalize_str(text)
            )
            self.libc.free(result)
        for text in [
            "e\u0301",
            "A\u0315\u0300\u05ae\u0301",
            "\u1100\u1161\u11a8",
            "\U00011099\U000110ba",
            "\u0301\u0327\u0345",
        ]:
            result = C.c_void_p()
            self.assertEqual(
                self.lib.ds4_unicode_nfc(text.encode(), C.byref(result)), 0
            )
            self.assertEqual(
                C.string_at(result).decode(), oracle.normalizer.normalize_str(text)
            )
            self.libc.free(result)


if __name__ == "__main__":
    unittest.main()
