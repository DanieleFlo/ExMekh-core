"""Qualification failures must remain evidence, without requiring a Metal device."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.qualify_metal_foundations import (
    CommandRunner,
    QualificationReport,
    artifact_identity,
    build_tensor_inventory,
    symbol_verdict,
    tensor_operations,
)


def snapshot_fixture():
    return {
        "gguf": {"tensor_count": 2, "data_offset_bytes": 32},
        "metadata": {"qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1},
        "artifact": {"size_bytes": 1024},
        "tensor_type_counts": {"Q3_K": 1, "F32": 1},
        "tensors": [
            {
                "name": "token_embd.weight",
                "shape": [256, 2],
                "type": "Q3_K",
                "offset": 0,
            },
            {"name": "blk.64.norm.weight", "shape": [16], "type": "F32", "offset": 256},
        ],
    }


class TensorInventoryTests(unittest.TestCase):
    def test_inventory_preserves_source_and_assigns_actual_roles(self):
        snapshot = snapshot_fixture()
        original = copy.deepcopy(snapshot)
        inventory = build_tensor_inventory(snapshot)
        self.assertEqual(snapshot, original)
        self.assertEqual(inventory[0]["bytes"], 220)
        self.assertEqual(inventory[0]["operations"], ["embedding", "gather"])
        self.assertEqual(inventory[0]["stage"], "target")
        self.assertEqual(inventory[1]["stage"], "mtp")
        self.assertEqual(
            tensor_operations({"name": "blk.0.ssm_conv1d.weight", "shape": [4, 10240]}),
            ["causal-convolution"],
        )
        self.assertEqual(
            tensor_operations({"name": "blk.0.projection.weight", "shape": [256, 256]}),
            ["decode-matvec", "prefill-matmul"],
        )

    def test_invalid_inventory_is_rejected_before_emitting_evidence(self):
        changes = [
            lambda s: s["gguf"].update(tensor_count=3),
            lambda s: s["tensor_type_counts"].update(Q3_K=2),
            lambda s: s["tensors"][0].update(shape=[128, 4]),
            lambda s: s["tensors"][0].update(shape=[256, -2]),
            lambda s: s["tensors"][0].update(shape=[True, 2]),
            lambda s: s["tensors"][0].update(offset=-1),
            lambda s: s["artifact"].update(size_bytes=100),
            lambda s: s["tensors"][1].update(name="token_embd.weight"),
        ]
        for change in changes:
            with self.subTest(change=change):
                snapshot = snapshot_fixture()
                change(snapshot)
                with self.assertRaises(ValueError):
                    build_tensor_inventory(snapshot)

    def test_empty_or_unresolved_symbol_inventory_cannot_pass(self):
        header = "int ds4_gpu_qwen35_example(void);"
        self.assertEqual(
            symbol_verdict(
                header, {"exit_code": 0, "stdout": "T _ds4_gpu_qwen35_example"}
            )["status"],
            "PASS",
        )
        for source, symbols in [
            ("", "T _ds4_gpu_qwen35_example"),
            (header, "U _ds4_gpu_qwen35_example"),
            (header, ""),
        ]:
            with self.subTest(source=source, symbols=symbols):
                self.assertEqual(
                    symbol_verdict(source, {"exit_code": 0, "stdout": symbols})[
                        "status"
                    ],
                    "FAIL",
                )


class QualificationExecutionTests(unittest.TestCase):
    def test_report_records_the_configured_reserve_including_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            for reserve in (0, 2**31):
                output = Path(directory) / str(reserve)
                with patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES=str(reserve)):
                    report = QualificationReport(output)
                report.record_section("review", "configuration snapshot")
                data = json.loads((output / "verification.json").read_text())
                self.assertEqual(data["host_reserve_bytes"], reserve)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "report"
        self.report = QualificationReport(self.output)
        self.runner = CommandRunner(self.root, self.output, self.report)

    def read_report(self):
        return json.loads((self.output / "verification.json").read_text())

    def test_feature_scope_is_recorded_even_before_a_build_can_run(self):
        output = self.root / "features"
        report = QualificationReport(output, features=True)
        self.assertFalse(report.finish())
        document = json.loads((output / "verification.json").read_text())
        self.assertEqual(document["schema"], "exmekh-metal-features-v1")
        self.assertIn("T04/T05", document["scope"])
        self.assertFalse(document["all_executed_checks_pass"])

    def test_command_failure_is_recorded_and_success_does_not_hide_it(self):
        self.assertFalse(
            self.runner.run("failure", [sys.executable, "-c", "raise SystemExit(7)"])
        )
        self.assertTrue(
            self.runner.run("success", [sys.executable, "-c", "print('ok')"])
        )
        self.assertFalse(self.report.finish())
        document = self.read_report()
        self.assertEqual(document["checks"][0]["exit_code"], 7)
        self.assertFalse(document["all_executed_checks_pass"])
        self.assertEqual((self.output / "success.log").read_text(), "ok\n")

    def test_missing_executable_and_timeout_are_distinct(self):
        self.assertFalse(
            self.runner.run("missing", [str(self.root / "missing-program")])
        )
        self.assertFalse(
            self.runner.run(
                "timeout",
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=0.05,
            )
        )
        checks = self.read_report()["checks"]
        self.assertEqual(checks[0]["status"], "FAIL")
        self.assertIsNone(checks[0]["exit_code"])
        self.assertEqual(checks[1]["status"], "NOT_VERIFIED")
        self.assertNotEqual(checks[1]["exit_code"], 0)
        self.assertFalse(self.report.finish())

    def test_failed_validation_prevents_pass_and_output_is_never_reused(self):
        self.assertFalse(self.report.finish())
        self.assertTrue(self.runner.run("ok", [sys.executable, "-c", "pass"]))
        self.report.record_validation("model", {"status": "FAIL"})
        self.assertFalse(self.report.finish())
        before = (self.output / "verification.json").read_bytes()
        with self.assertRaises(FileExistsError):
            QualificationReport(self.output)
        self.assertEqual((self.output / "verification.json").read_bytes(), before)
        self.assertFalse(list(self.output.glob("*.tmp")))

    def test_runner_clears_inherited_device_flags_and_applies_explicit_ones(self):
        with patch.dict("os.environ", {"DS4_TEST_FLAG": "inherited"}):
            runner = CommandRunner(self.root, self.output, self.report)
        command = [
            sys.executable,
            "-c",
            "import os; print(os.getenv('DS4_TEST_FLAG', 'absent'))",
        ]
        self.assertTrue(runner.run("clean", command))
        self.assertTrue(
            runner.run("explicit", command, extra={"DS4_TEST_FLAG": "explicit"})
        )
        self.assertEqual((self.output / "clean.log").read_text(), "absent\n")
        self.assertEqual((self.output / "explicit.log").read_text(), "explicit\n")
        self.assertTrue(self.report.finish())

    def test_artifact_mismatch_is_an_explicit_failed_validation(self):
        model = self.root / "fixture.gguf"
        model.write_bytes(b"fixture")
        result = artifact_identity(model, {"size_bytes": 7, "sha256": "0" * 64})
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["matches_pinned_qwen27"])


if __name__ == "__main__":
    unittest.main()
