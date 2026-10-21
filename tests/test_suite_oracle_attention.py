"""Large-oracle query tiling must preserve masks and the full softmax axis."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from tools.run_qwen_image_pipeline_oracle import bounded_math_sdpa


@unittest.skipIf(torch is None, "PyTorch development oracle unavailable")
class OracleAttention(unittest.TestCase):
    def test_row_masks_and_broadcast_key_masks(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(20261021)
        q = torch.randn((1, 2, 513, 32), generator=generator)
        k = torch.randn((1, 2, 79, 32), generator=generator)
        v = torch.randn((1, 2, 79, 24), generator=generator)
        original = torch.nn.functional.scaled_dot_product_attention
        tiled = bounded_math_sdpa(torch, original, budget=1)
        mask = torch.rand((513, 79), generator=generator) > 0.2
        for selected in (None, mask, mask[:1]):
            expected = original(q, k, v, attn_mask=selected)
            actual = tiled(q, k, v, attn_mask=selected)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_causal_offset_is_global_across_tiles(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(20261020)
        q, k, v = [torch.randn((1, 2, 513, 32), generator=generator) for _ in range(3)]
        original = torch.nn.functional.scaled_dot_product_attention
        tiled = bounded_math_sdpa(torch, original, budget=1)
        torch.testing.assert_close(
            tiled(q, k, v, is_causal=True),
            original(q, k, v, is_causal=True),
            rtol=1e-5,
            atol=1e-6,
        )
        with self.assertRaises(ValueError):
            tiled(q, k, v, dropout_p=0.1)


if __name__ == "__main__":
    unittest.main()
