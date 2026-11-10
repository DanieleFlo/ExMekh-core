"""Regression for encoder failures hidden by acceptable final image metrics."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest("requires NumPy") from error

from tools.qualify_metal_suite import analyze_existing, compare_boundaries
from tools.qualify_qwen_image_pipeline import main


class PipelineQualificationTests(unittest.TestCase):
    def test_full_corpus_admission_uses_marked_shape_fixtures_without_loading_oracle_noise(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plans"

            def plan(command, **kwargs):
                native = Path(command[4])
                for index in range(len(command) - 5):
                    case = native / f"case-{index}"
                    case.mkdir(parents=True)
                    (case / "plan.json").write_text('{"status":0}')
                return subprocess.CompletedProcess(command, 0)

            arguments = [
                "qualify",
                "--prepared",
                directory,
                "--bundle",
                directory,
                "--tokenizer",
                directory,
                "--output",
                str(output),
                "--full-corpus",
                "--plan-only",
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "tools.qualify_qwen_image_pipeline.subprocess.run", side_effect=plan
                ),
                patch("numpy.load", side_effect=AssertionError("oracle noise opened")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            cases = json.loads((output / "cases.json").read_text())
            corpus = json.loads(
                Path("tests/fixtures/native-suite-v1/corpus.json").read_text()
            )
            self.assertEqual(
                sum("split" in case for case in cases), 2 * len(corpus["qwen_recipes"])
            )
            self.assertEqual(
                len(json.loads((output / "admission.json").read_text())), len(cases)
            )
            self.assertFalse((output / "verification.json").exists())
            for case in cases:
                self.assertEqual(case["noise_kind"], "admission_zero")
                noise = np.fromfile(case["noise"], np.float32)
                self.assertEqual(
                    noise.size, (case["height"] // 16) * (case["width"] // 16) * 64
                )
                self.assertFalse(np.any(noise))

    def test_conditioning_failure_rejects_an_identical_final_image(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            native = output / "native/case-0"
            oracle = output / "oracle/text"
            native.mkdir(parents=True)
            oracle.mkdir(parents=True)
            np.full(4096, 0x3F80, np.uint16).tofile(native / "conditioning.bf16")
            np.full(4096, 0x4000, np.uint16).tofile(oracle / "conditioning.bf16")
            for folder in (native, oracle):
                for step in range(8):
                    np.zeros(16 * 16 * 64, np.float32).tofile(
                        folder / f"latent-{step}.f32"
                    )
                np.zeros((256, 256, 4), np.float32).tofile(folder / "rgba.f32")
            (native / "report.json").write_text("{}")
            arguments = [
                "qualify",
                "--prepared",
                directory,
                "--bundle",
                directory,
                "--tokenizer",
                directory,
                "--output",
                directory,
                "--case-ids",
                "text",
                "--skip-native",
                "--skip-oracle",
                "--oracle-device",
                "mps",
                "--oracle-encoder-device",
                "cpu",
                "--oracle-vae-device",
                "mps",
            ]
            with (
                patch.object(sys, "argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as rejected,
            ):
                main()
            self.assertEqual(rejected.exception.code, 1)
            report = json.loads((output / "verification.json").read_text())
            self.assertEqual(report["verdict"], "FAIL")
            self.assertEqual(
                report["oracle_devices"],
                {"encoder": "cpu", "encode": "mps", "dit": "mps", "decode": "mps"},
            )
            case = report["cases"][0]
            self.assertEqual(case["conditioning"]["verdict"], "FAIL")
            self.assertTrue(all(step["verdict"] == "PASS" for step in case["latents"]))
            self.assertEqual(case["rgba"]["rmse"], 0)


class EmbeddingQualificationTests(unittest.TestCase):
    def test_internal_failure_rejects_identical_final_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            oracle = output / "oracle"
            oracle.mkdir()
            expected = np.zeros((1, 768), np.float32)
            expected[0, 0] = 1
            np.save(oracle / "vectors.npy", expected)
            for dimensions in (128, 256, 512, 768):
                native = output / f"native-{dimensions}"
                native.mkdir()
                expected[:, :dimensions].tofile(native / "vectors.f32")
            np.array([0x3F80], np.uint16).tofile(oracle / "item-0-layer-0.bf16")
            np.array([0x3F81], np.uint16).tofile(
                output / "native-128/item-0-layer-0.bf16"
            )
            report = analyze_existing(output, "text")
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(
                all(
                    row["status"] == "PASS" for row in report["output_vectors"].values()
                )
            )
            self.assertEqual(report["internal_boundaries"]["exact"], 0)
            self.assertEqual(report, json.loads((output / "analysis.json").read_text()))

    def test_strict_boundaries_preserve_signed_zero_and_reject_bad_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            native, oracle = Path(directory) / "native", Path(directory) / "oracle"
            native.mkdir()
            oracle.mkdir()
            with self.assertRaisesRegex(ValueError, "no internal boundaries"):
                compare_boundaries(native, oracle)
            np.array([0.0], np.float32).tofile(oracle / "audio-probabilities-0.f32")
            np.array([-0.0], np.float32).tofile(native / "audio-probabilities-0.f32")
            self.assertEqual(compare_boundaries(native, oracle)["status"], "FAIL")
            for path in (native, oracle):
                np.array([0x7FC0], np.uint16).tofile(path / "layer-0.bf16")
            with self.assertRaisesRegex(ValueError, "non-finite boundary"):
                compare_boundaries(native, oracle)
            (native / "layer-0.bf16").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "truncated boundary"):
                compare_boundaries(native, oracle)


if __name__ == "__main__":
    unittest.main()
