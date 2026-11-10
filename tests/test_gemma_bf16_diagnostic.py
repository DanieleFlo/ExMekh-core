"""The diagnostic must preserve BF16 ties and blame neither backend by fiat."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.diagnose_gemma_metal_bf16 import (
    PLETrace,
    bf16_bits,
    bf16_values,
    compare_bits,
    dot_witnesses,
    read_boundary,
)


class GemmaBF16Diagnostic(unittest.TestCase):
    def test_trace_owns_input_compatibility_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            native, oracle = Path(directory) / "native", Path(directory) / "oracle"
            for folder in (native, oracle):
                folder.mkdir()
                np.zeros((1, 512), np.uint16).tofile(
                    folder / "item-0-embeddings-0.bf16"
                )
                np.zeros((1, 12288), np.uint16).tofile(folder / "item-0-ple-0.bf16")
            trace = PLETrace.load(native, oracle, 0)
            self.assertEqual(trace.embeddings.shape, (1, 512))
            self.assertEqual(len(trace.files), 4)
            np.ones((1, 512), np.uint16).tofile(oracle / "item-0-embeddings-0.bf16")
            with self.assertRaisesRegex(ValueError, "identical embeddings"):
                PLETrace.load(native, oracle, 0)
            np.zeros((1, 512), np.uint16).tofile(oracle / "item-0-embeddings-0.bf16")
            np.zeros((2, 12288), np.uint16).tofile(native / "item-0-ple-0.bf16")
            with self.assertRaisesRegex(ValueError, "token counts"):
                PLETrace.load(native, oracle, 0)

    def test_rounding_ties_and_nonfinite_storage(self):
        # Midpoints adjacent to even and odd BF16 significands, both signs.
        values = np.array(
            [
                1 + 1 / 256,
                1 + 3 / 256,
                -1 - 1 / 256,
                -1 - 3 / 256,
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
            ],
            np.float32,
        )
        bits = bf16_bits(values)
        np.testing.assert_array_equal(
            bits[:8], [0x3F80, 0x3F82, 0xBF80, 0xBF82, 0, 0x8000, 0x7F80, 0xFF80]
        )
        self.assertTrue(np.isnan(bf16_values(bits[-1])))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_bits(bits, bits)

    def test_fp64_witness_retains_cancellation_lost_by_float32(self):
        x = np.array([[2**24, 1.0, -(2**24)], [1.0, 2.0, 3.0]], np.float32)
        weights = np.ones((1, 3), np.float32)
        reference, candidate = bf16_bits([[0.0], [6.0]]), bf16_bits([[1.0], [6.0]])
        witness = dot_witnesses(x, weights, reference, candidate)
        self.assertEqual(len(witness), 1)
        self.assertEqual(witness[0]["fp64_sum"], 1.0)
        self.assertEqual(witness[0]["cpu_absolute_error"], 1.0)
        self.assertEqual(witness[0]["metal_absolute_error"], 0.0)
        self.assertEqual(dot_witnesses(x, weights, reference, candidate, limit=0), [])

    def test_oversized_truncated_and_mismatched_boundaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.bf16"
            for size in (0, 3, 128 * 512 * 2 + 1024):
                path.write_bytes(bytes(size))
                with self.assertRaises(ValueError):
                    read_boundary(path, 512, 128)
            path.write_bytes(bytes(512 * 2))
            self.assertEqual(read_boundary(path, 512, 128).shape, (1, 512))
        with self.assertRaises(ValueError):
            compare_bits(np.zeros(1, np.uint16), np.zeros(2, np.uint16))


if __name__ == "__main__":
    unittest.main()
