"""Small real PyTorch checks for the independent oracle's lazy transport."""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file

    from tools.suite_offloaded_oracle import SafetensorsWeights, load_offloaded_model
except ModuleNotFoundError as error:
    raise unittest.SkipTest(
        "requires the prepared PyTorch oracle environment"
    ) from error


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    @classmethod
    def load_config(cls, directory):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()

    def forward(self, value):
        return self.linear(value)


class OffloadedOracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = TinyModel().to(dtype=torch.bfloat16)
        save_file(self.model.state_dict(), self.directory / "model.safetensors")

    def test_lazy_loading_preserves_values_and_does_not_cache_weights(self):
        weights = SafetensorsWeights(self.directory, self.model)
        self.assertEqual(weights.reads, 0)
        self.assertEqual(set(weights), set(self.model.state_dict()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(weights[name], value))
        self.assertEqual(weights.reads, 2)
        self.assertEqual(weights.bytes, 20)

    def test_offloaded_forward_matches_resident_and_returns_parameters_to_meta(self):
        original_dtype = torch.get_default_dtype()
        offloaded, weights = load_offloaded_model(TinyModel, self.directory, "cpu")
        self.assertEqual(torch.get_default_dtype(), original_dtype)
        value = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.model(value), offloaded(value)))
        self.assertTrue(all(p.device.type == "meta" for p in offloaded.parameters()))
        self.assertEqual(weights.reads, 2)

    def test_missing_and_wrong_shapes_are_rejected_before_execution(self):
        checkpoint = self.directory / "model.safetensors"
        save_file({"linear.weight": torch.zeros(2, 4)}, checkpoint)
        with self.assertRaisesRegex(ValueError, "Missing oracle weight"):
            SafetensorsWeights(self.directory, self.model)
        save_file({"linear.weight": torch.zeros(4, 2)}, checkpoint)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            SafetensorsWeights(self.directory, self.model)

    def test_duplicate_sharded_weights_are_rejected_instead_of_silently_replaced(self):
        save_file(self.model.state_dict(), self.directory / "second.safetensors")
        with self.assertRaisesRegex(ValueError, "Duplicate oracle weight"):
            SafetensorsWeights(self.directory, self.model)

    def test_prefixed_checkpoint_and_model_buffers_use_the_public_mapping_contract(
        self,
    ):
        self.model.register_buffer("constant", torch.ones(1))
        save_file(
            {"model." + name: value for name, value in self.model.named_parameters()},
            self.directory / "model.safetensors",
        )
        weights = SafetensorsWeights(self.directory, self.model)
        self.assertEqual(len(weights), 3)
        self.assertEqual(weights["constant"].data_ptr(), self.model.constant.data_ptr())
        self.assertEqual(weights.reads, 0)
        self.assertTrue(torch.equal(weights["linear.weight"], self.model.linear.weight))


if __name__ == "__main__":
    unittest.main()
