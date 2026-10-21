"""Artifact corruption must fail the gate and overwrite a stale PASS report."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]


class GraphComparison(unittest.TestCase):
    def run_gate(self, root):
        output = root / "result.json"
        output.write_text('{"passed":true}')
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/compare_qwen_image_graph.py"),
                str(root / "oracle"),
                str(root / "native"),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        result = json.loads(output.read_text())
        self.assertNotIn("Traceback", run.stderr)
        self.assertEqual(run.returncode, 0 if result["passed"] else 1)
        return result

    def fixture(self, root):
        oracle, native = root / "oracle", root / "native"
        oracle.mkdir()
        native.mkdir()
        payload = b"fixed input"
        (oracle / "input.ds4g").write_bytes(payload)
        (oracle / "oracle.json").write_text(
            json.dumps(
                dict(steps=8, cfg=1, fixture_sha256=hashlib.sha256(payload).hexdigest())
            )
        )
        return oracle, native

    def test_empty_inventory_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = self.run_gate(root)
            self.assertFalse(result["passed"])

    def test_missing_metadata_overwrites_old_success(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_gate(Path(temp))
            self.assertFalse(result["passed"])
            self.assertIn("error", result)

    def test_non_finite_boundary_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oracle, native = self.fixture(root)
            for directory in (oracle, native):
                (directory / "step-0-block-0.bf16").write_bytes(b"\xc0\x7f")
            result = self.run_gate(root)
            self.assertFalse(result["passed"])
            self.assertIn("non-finite", result["error"])

    def test_changed_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oracle, _ = self.fixture(root)
            (oracle / "input.ds4g").write_bytes(b"changed")
            result = self.run_gate(root)
            self.assertFalse(result["passed"])
            self.assertIn("identity mismatch", result["error"])

    def test_complete_inventory_passes_and_missing_boundary_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oracle, native = self.fixture(root)
            for step in range(8):
                names = [f"step-{step}-block-{layer}.bf16" for layer in range(32)]
                names += [
                    f"step-{step}-{stage}-0.bf16"
                    for stage in ("joint", "modulation", "velocity")
                ]
                names += [f"step-{step}-latent-{step}.f32"]
                for name in names:
                    value = b"\0" * (4 if name.endswith("f32") else 2)
                    for directory in (oracle, native):
                        (directory / name).write_bytes(value)
                if step:
                    (native / f"uncached-same-input-velocity-{step}.bf16").write_bytes(
                        b"\0\0"
                    )
            self.assertTrue(self.run_gate(root)["passed"])
            (oracle / "step-7-block-31.bf16").unlink()
            (native / "step-7-block-31.bf16").unlink()
            self.assertFalse(self.run_gate(root)["passed"])

    def test_corrupt_png_and_wrong_dimensions_fail(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oracle, native = root / "oracle", root / "native"
            oracle.mkdir()
            native.mkdir()
            payload = b"fixed image"
            (oracle / "input.ds4v").write_bytes(payload)
            (oracle / "oracle.json").write_text(
                json.dumps(
                    dict(
                        width=16,
                        height=16,
                        fixture_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )
            )
            for directory in (oracle, native):
                (directory / "encoded-0.bf16").write_bytes(b"\0\0")
                (directory / "pixels.f32").write_bytes(b"\0" * (16 * 16 * 4 * 4))
            (native / "result.png").write_bytes(b"invalid PNG")
            self.assertFalse(self.run_gate(root)["passed"])
            Image.new("RGBA", (8, 32)).save(native / "result.png")
            self.assertFalse(self.run_gate(root)["passed"])
            Image.new("RGBA", (16, 16)).save(native / "result.png")
            self.assertTrue(self.run_gate(root)["passed"])


if __name__ == "__main__":
    unittest.main()
