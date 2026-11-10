"""Metric contract checks; no model checkpoint or GPU required."""

import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.qwen_vision_reference import compare, read_embedding_dump


class VisionComparisonTests(unittest.TestCase):
    def test_empty_and_inconsistent_embeddings_fail_without_reducing_empty_arrays(self):
        for values, grid in [
            (np.empty((0, 4)), (0, 1)),
            (np.ones((2, 4)), (1, 3)),
            (np.ones((2, 4, 1)), (1, 2)),
        ]:
            with self.subTest(shape=values.shape):
                self.assertFalse(compare(values, values, grid, grid, 0.99)["pass"])

    def test_invalid_threshold_cannot_qualify(self):
        values = np.ones((2, 4))
        for threshold in [0, 1.1, float("nan")]:
            self.assertFalse(compare(values, values, (1, 2), (1, 2), threshold)["pass"])

    def test_dump_rejects_malformed_layout_truncation_and_nonfinite_values(self):
        good = struct.pack("<4I", 2, 4, 1, 2) + np.ones((2, 4), dtype="<f4").tobytes()
        bad = [
            b"short",
            good[:-1],
            good + b"x",
            struct.pack("<4I", 2, 4, 1, 3) + good[16:],
            struct.pack("<4I", 0, 4, 0, 2),
            good[:16] + np.full((2, 4), float("nan"), dtype="<f4").tobytes(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.bin"
            path.write_bytes(good)
            values, grid = read_embedding_dump(path)
            self.assertEqual(grid, (1, 2))
            np.testing.assert_array_equal(values, np.ones((2, 4)))
            for payload in bad:
                path.write_bytes(payload)
                with self.subTest(payload=payload[:16]), self.assertRaises(ValueError):
                    read_embedding_dump(path)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "tests/test_qwen4_vision").is_file(),
        "native vision probe is not built",
    )
    def test_native_cli_rejects_invalid_budgets_before_opening_files(self):
        binary = Path(__file__).resolve().parents[1] / "tests/test_qwen4_vision"
        for limits in [
            ("bad",),
            ("64x",),
            ("-1",),
            ("0",),
            ("64", "32"),
            ("64", "16384"),
            ("64", "128", "extra"),
        ]:
            with (
                self.subTest(limits=limits),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = subprocess.run(
                    [str(binary), "--patches", "missing.png", "out.bin", *limits],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("token limits", result.stderr)
                self.assertFalse((Path(directory) / "out.bin").exists())

    def test_grid_mismatch_with_identical_shape_fails(self):
        values = np.ones((6, 4))
        self.assertFalse(compare(values, values, (2, 3), (3, 2), 0.99)["pass"])

    def test_nonfinite_reference_or_actual_fails(self):
        good = np.ones((2, 4))
        for value in [float("nan"), float("inf")]:
            bad = good.copy()
            bad[0, 0] = value
            for actual, reference in [(bad, good), (good, bad)]:
                self.assertFalse(
                    compare(actual, reference, (1, 2), (1, 2), 0.99)["pass"]
                )

    def test_one_bad_token_fails_despite_high_mean(self):
        reference = np.tile([1.0, 0.0], (1000, 1))
        actual = reference.copy()
        actual[73] = [0.0, 1.0]
        result = compare(actual, reference, (20, 50), (20, 50), 0.99)
        self.assertGreater(result["mean_cos"], 0.99)
        self.assertFalse(result["pass"])
        self.assertEqual(result["worst_token"], 73)

    def test_scaled_vectors_pass_cosine_but_report_absolute_error(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = compare(reference * 2, reference, (1, 2), (1, 2), 0.99)
        self.assertTrue(result["pass"])
        self.assertEqual(result["max_abs"], 4.0)


if __name__ == "__main__":
    unittest.main()
