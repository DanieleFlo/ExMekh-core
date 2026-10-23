import argparse
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tools.evaluate_suite_profiles import GATES, evaluate
from tools.package_native_suite import MANDATORY, package, release_ready
from tools.qualify_suite_release import timing


class ProfileQualification(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "id": "fixture",
            "objective": "latency",
            "offer_when_qualified": True,
        }
        self.measurement = {
            "native_binding": "PASS",
            "artifact_sha256": "a" * 64,
            "sources": ["frozen-source"],
            "heldout_split": "held-out",
            "warmup": True,
            "platform": "fixture-sm86",
            "gates": dict.fromkeys(GATES, "PASS"),
            "baseline_seconds": [2] * 5,
            "candidate_seconds": [1] * 5,
        }

    def test_missing_or_unstable_measurements_cannot_publish(self):
        for missing in (
            "native_binding",
            "artifact_sha256",
            "heldout_split",
            "baseline_seconds",
        ):
            with self.subTest(missing=missing):
                measurement = {
                    k: v for k, v in self.measurement.items() if k != missing
                }
                result = evaluate(self.candidate, measurement)
                self.assertEqual(result["verdict"], "NEED_MORE_DATA")
                self.assertFalse(result["public"])
        self.measurement["candidate_seconds"] = [0.5, 1, 1.5, 1, 1]
        self.assertEqual(
            evaluate(self.candidate, self.measurement)["verdict"], "NEED_MORE_DATA"
        )

    def test_all_gates_required_individually(self):
        self.assertEqual(evaluate(self.candidate, self.measurement)["verdict"], "KEEP")
        for gate in GATES:
            with self.subTest(gate=gate):
                self.measurement["gates"][gate] = "FAIL"
                result = evaluate(self.candidate, self.measurement)
                self.assertEqual(result["verdict"], "REJECT")
                self.assertFalse(result["public"])
                self.measurement["gates"][gate] = "PASS"

    def test_storage_ratio_or_unsupported_binding_never_promotes(self):
        self.assertEqual(
            evaluate(self.candidate, {"storage_ratio": 0.25})["verdict"],
            "NEED_MORE_DATA",
        )
        self.measurement["native_binding"] = "UNSUPPORTED"
        self.assertEqual(
            evaluate(self.candidate, self.measurement)["verdict"], "REJECT"
        )

    def test_performance_requires_gain_above_noise(self):
        self.measurement["candidate_seconds"] = [2] * 5
        self.assertEqual(
            evaluate(self.candidate, self.measurement)["verdict"], "NEED_MORE_DATA"
        )
        self.measurement["candidate_seconds"] = [3] * 5
        self.assertEqual(
            evaluate(self.candidate, self.measurement)["verdict"], "REJECT"
        )

    def test_nan_and_undersampled_timings(self):
        with self.assertRaises(ValueError):
            timing([float("nan")])
        self.assertEqual(timing([1, 1])["verdict"], "NEED_MORE_DATA")

    def test_package_cannot_hide_a_missing_release_gate(self):
        self.assertFalse(release_ready({}))
        gates = dict.fromkeys(MANDATORY, "PASS")
        self.assertTrue(release_ready({"gates": gates}))
        for gate in MANDATORY:
            gates[gate] = "NOT_VERIFIED"
            self.assertFalse(release_ready({"gates": gates}))
            gates[gate] = "PASS"


class NativePackageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch("tools.package_native_suite.ROOT", self.root))
        self.dependencies = self.enterContext(
            patch(
                "tools.package_native_suite.subprocess.run",
                return_value=Mock(stdout="libc.so.6 => /lib/libc.so.6\n"),
            )
        )
        for name, content in {
            "server": b"native binary fixture",
            "asset.json": b"{}",
            "suite.json": json.dumps(
                {"models": [{"index": "asset.json", "root": "/private/models"}]}
            ).encode(),
            "qualification.json": b'{"gates": {}}',
            "candidates.json": b"{}",
            "ds4.c": b"/* source fixture */",
            "ds4.o": b"compiled object must not be packaged",
        }.items():
            (self.root / name).write_bytes(content)
        self.args = argparse.Namespace(
            binary=self.root / "server",
            config=self.root / "suite.json",
            qualification=self.root / "qualification.json",
            candidates=self.root / "candidates.json",
            media_prefix=self.root / "codecs",
            development=True,
            output=self.root / "package",
        )

    def run_package(self):
        with redirect_stdout(io.StringIO()):
            package(self.args)

    def test_failed_copy_removes_staging_and_allows_retry(self):
        asset = self.root / "asset.json"
        asset.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_package()
        self.assertFalse(self.args.output.exists())
        self.assertFalse(list(self.root.glob(".package-*")))
        asset.write_text("{}")
        self.run_package()
        output = self.args.output
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "development")
        self.assertFalse(manifest["release_qualified"])
        for entry in manifest["files"]:
            path = output / entry["path"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"]
            )
        self.assertEqual(
            (output / "source/ds4.c").read_bytes(), (self.root / "ds4.c").read_bytes()
        )
        self.assertFalse((output / "source/ds4.o").exists())
        config = json.loads((output / "config/native-suite.json").read_text())
        self.assertEqual(config["models"][0]["root"], "models")
        self.assertTrue((output / config["models"][0]["index"]).is_file())
        self.assertTrue((output / "run-suite").stat().st_mode & 0o111)
        with self.assertRaises(FileExistsError):
            self.run_package()
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)

    def test_release_and_native_dependency_failures_leave_no_output(self):
        self.args.development = False
        with self.assertRaisesRegex(ValueError, "release blocked"):
            self.run_package()
        self.assertFalse(self.args.output.exists())
        self.args.development = True
        self.dependencies.return_value.stdout = "libavcodec.so => not found\n"
        with self.assertRaisesRegex(ValueError, "dependency is unresolved"):
            self.run_package()
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
