"""Missing, truncated or non-finite output must never qualify a model run."""

import array
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.suite_artifacts import frontier_artifacts, probe_artifacts, read_json

ROOT = Path(__file__).resolve().parents[1]


class SuiteArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_probe(self):
        (self.directory / "result.json").write_text(
            json.dumps({"vocab": 2, "prefix_tokens": 1, "tokens": [0, 1]})
        )
        (self.directory / "logits.f32").write_bytes(
            array.array("f", [1.0, -1.0] * 3).tobytes()
        )
        (self.directory / "session.payload").write_bytes(b"checkpoint")

    def test_probe_requires_complete_finite_shape(self):
        self.write_probe()
        self.assertTrue(
            next(
                r for r in probe_artifacts(self.directory) if r["path"] == "logits.f32"
            )["finite"]
        )
        for values in ([], [1.0], [float("nan")] * 6, [float("inf")] * 6):
            with self.subTest(values=values):
                (self.directory / "logits.f32").write_bytes(
                    array.array("f", values).tobytes()
                )
                with self.assertRaises(ValueError):
                    probe_artifacts(self.directory)

    def test_probe_requires_payload_and_metadata(self):
        self.write_probe()
        (self.directory / "session.payload").write_bytes(b"")
        with self.assertRaises(ValueError):
            probe_artifacts(self.directory)
        (self.directory / "result.json").unlink()
        with self.assertRaises(OSError):
            probe_artifacts(self.directory)

    def test_json_rejects_non_finite_numbers(self):
        path = self.directory / "value.json"
        for number in ("NaN", "Infinity", "-Infinity", "1e999"):
            path.write_text('{"logits":[' + number + "]}")
            with self.subTest(number=number), self.assertRaises(ValueError):
                read_json(path)

    def test_frontiers_require_paired_full_vocabulary_rows(self):
        logits = self.directory / "frontier_000032.logits.json"
        decode = self.directory / "frontier_000032.decode.json"
        logits.write_text(json.dumps({"vocab": 2, "logits": [1.0, 2.0]}))
        with self.assertRaises(ValueError):
            frontier_artifacts(self.directory)
        decode.write_text(
            json.dumps(
                {
                    "vocab": 2,
                    "logits": [2.0, 1.0],
                    "generated_count": 1,
                    "generated_tokens": [1],
                }
            )
        )
        self.assertEqual(len(frontier_artifacts(self.directory)), 2)
        logits.write_text(json.dumps({"vocab": 2, "logits": [1.0]}))
        with self.assertRaises(ValueError):
            frontier_artifacts(self.directory)

    @unittest.skipUnless(os.name == "posix", "development probes execute on Linux/WSL")
    def test_successful_process_without_outputs_is_a_failed_baseline(self):
        tools = self.directory / "tools"
        tools.mkdir()
        probe = tools / "probe_suite_baseline"
        probe.write_text("#!/bin/sh\nexit 0\n")
        probe.chmod(0o755)
        output = self.directory / "experiment"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/run_suite_baseline.py"),
                "--output",
                str(output),
            ],
            cwd=self.directory,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        report = json.loads((output / "model-runs.json").read_text())
        self.assertEqual(report[0]["exit_code"], 0)
        self.assertIn("validation_error", report[0])
        flash_output = self.directory / "flash-experiment"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/run_suite_flash_baseline.py"),
                "--baseline",
                str(probe),
                "--candidate",
                str(probe),
                "--output",
                str(flash_output),
            ],
            cwd=self.directory,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        report = json.loads((flash_output / "flash.json").read_text())
        self.assertEqual(report["runs"][0]["exit_code"], 0)
        self.assertIn("validation_error", report["runs"][0])


if __name__ == "__main__":
    unittest.main()
