"""Vision adapter/report contracts; no GPU or pretrained model is loaded."""

import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tools import qualify_qwen_metal_vision as qualification


def metadata(projection=2560):
    values = {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        **{
            f"clip.vision.{name}": value
            for name, value in {
                "projection_dim": projection,
                "block_count": 27,
                "embedding_length": 1152,
                "feed_forward_length": 4304,
                "attention.head_count": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
            }.items()
        },
    }
    return SimpleNamespace(
        fields={
            name: SimpleNamespace(contents=lambda v=value: v)
            for name, value in values.items()
        },
        tensors=[SimpleNamespace(name="v.position_embd.weight", shape=[1152, 2304])],
    )


class PublicConfig:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            raise AssertionError("adapter must use the model's public attention setter")
        super().__setattr__(name, value)


class VisionReferenceTests(unittest.TestCase):
    def test_family_selection_and_lazy_loading_use_public_model_methods(self):
        for width, model_name, config_name in [
            (2560, "Qwen4ExpVisionModel", "Qwen4ExpVisionConfig"),
            (5120, "Qwen3_5VisionModel", "Qwen3_5VisionConfig"),
        ]:
            with self.subTest(width=width):
                model = MagicMock()
                model.float.return_value = model.eval.return_value = model
                model.load_state_dict.return_value = ([], [])
                model.return_value = SimpleNamespace(pooler_output=torch.ones(1, width))
                with (
                    patch.object(
                        qualification, "GGUFReader", return_value=metadata(width)
                    ),
                    patch.object(qualification, config_name, PublicConfig),
                    patch.object(
                        qualification, model_name, return_value=model
                    ) as factory,
                    patch.object(qualification, "gguf_state", return_value={}),
                ):
                    reference = qualification.VisionReference(Path("model.gguf"), width)
                    factory.assert_not_called()
                    inputs = (torch.ones(4, 1536), torch.tensor([[1, 2, 2]]))
                    first, grid = reference.encode(inputs)
                    second, _ = reference.encode(inputs)
                    self.assertEqual(first.shape, (1, width))
                    self.assertEqual(grid, (1, 1))
                    np.testing.assert_array_equal(first, second)
                    factory.assert_called_once()
                    model.set_attn_implementation.assert_called_once_with("eager")
                    model.load_state_dict.assert_called_once()

    def test_incompatible_projectors_and_architecture_are_rejected_before_loading(self):
        for width in (3584, 5120):
            with (
                patch.object(qualification, "GGUFReader", return_value=metadata(width)),
                self.assertRaisesRegex(ValueError, "projector width"),
            ):
                qualification.VisionReference(Path("model.gguf"), 2560)
        wrong = metadata()
        wrong.fields["general.architecture"] = SimpleNamespace(
            contents=lambda: "unsupported"
        )
        with (
            patch.object(qualification, "GGUFReader", return_value=wrong),
            self.assertRaisesRegex(ValueError, "architecture"),
        ):
            qualification.VisionReference(Path("model.gguf"), 2560)

    def test_native_patch_layout_repeats_temporal_frames_without_mixing_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patches.bin"
            patches = np.repeat(np.array([1, 2, 3], dtype="<f4"), 256)
            path.write_bytes(
                struct.pack("<4I", 4, 768, 2, 2) + np.tile(patches, 4).tobytes()
            )
            with patch.object(qualification.subprocess, "run") as run:
                pixels, grid = qualification.native_inputs(
                    "native", Path("image.png"), Path(directory), 64
                )
            self.assertEqual(run.call_args.kwargs["timeout"], 120)
            self.assertEqual(grid.tolist(), [[1, 2, 2]])
            channels = pixels.reshape(4, 3, 2, 256)
            for channel in range(3):
                self.assertTrue(torch.all(channels[:, channel] == channel + 1))
            path.write_bytes(
                struct.pack("<4I", 6, 768, 2, 3)
                + np.ones((6, 768), dtype="<f4").tobytes()
            )
            with (
                patch.object(qualification.subprocess, "run"),
                self.assertRaisesRegex(ValueError, "patch layout"),
            ):
                qualification.native_inputs(
                    "native", Path("image.png"), Path(directory), 64
                )


class QualificationReportTests(unittest.TestCase):
    def invoke(self, outcome):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            args = [
                "qualify",
                "--mmproj",
                "model.gguf",
                "--image",
                "image.png",
                "--projection",
                "2560",
                "--report",
                str(path),
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(qualification, "qualify", side_effect=outcome),
                redirect_stdout(io.StringIO()),
            ):
                code = qualification.main()
            return code, json.loads(path.read_text())

    def test_errors_timeouts_and_interruptions_always_leave_evidence(self):
        cases = [
            (ValueError("bad layout"), "FAIL", 1),
            (subprocess.CalledProcessError(7, ["native"]), "FAIL", 1),
            (subprocess.TimeoutExpired(["native"], 120), "NOT_VERIFIED", 1),
            (KeyboardInterrupt(), "NOT_VERIFIED", 130),
        ]
        for error, status, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                code, report = self.invoke(error)
                self.assertEqual(code, expected_code)
                self.assertEqual(report["status"], status)
                self.assertIn("error", report)
                self.assertEqual(report["llm_multimodal_gate"], "NOT_VERIFIED")

    def test_a_single_failed_comparison_cannot_qualify_the_encoder(self):
        def result(args, report):
            report.update(implementation={"pass": True}, image_pipeline={"pass": False})

        code, report = self.invoke(result)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")
