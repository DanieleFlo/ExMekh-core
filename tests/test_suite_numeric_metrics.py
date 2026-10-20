import unittest

import numpy as np

from tools.suite_numeric_metrics import compare, embedding_metrics, rgba_metrics


class NumericMetrics(unittest.TestCase):
    def test_identity_and_drift(self):
        values = np.arange(64, dtype=np.float32)
        self.assertTrue(compare(values, values)["bit_exact"])
        self.assertAlmostEqual(compare(values, values + 1)["rmse"], 1.0)
        self.assertFalse(compare(np.array([0.0]), np.array([-0.0]))["bit_exact"])

    def test_invalid_values(self):
        for b in ([float("nan")], [float("inf")], [], [1, 2]):
            with self.assertRaises(ValueError):
                compare([1], b)

    def test_normalization_and_direction(self):
        a = np.zeros((1, 128))
        a[0, 0] = 1
        b = a.copy()
        b[0, 0] = 0
        b[0, 1] = 1
        self.assertEqual(embedding_metrics(a, a)["cosine_min"], 1.0)
        self.assertFalse(embedding_metrics(a, a.astype(np.float32))["bit_exact"])
        self.assertEqual(embedding_metrics(a, b)["cosine_min"], 0.0)
        self.assertEqual(embedding_metrics(a, a * 2)["unit_norm_error_max"], 1.0)

    def test_alpha_does_not_hide_in_rgb(self):
        a = np.ones((16, 16, 4))
        b = a.copy()
        b[..., 3] = 0
        self.assertFalse(rgba_metrics(a, a.astype(np.float32))["bit_exact"])
        result = rgba_metrics(a, b)
        self.assertEqual(result["ssim_rgb"], 1.0)
        self.assertEqual(result["alpha_mae"], 1.0)
        self.assertEqual(
            rgba_metrics(b.astype(np.uint8), a.astype(np.uint8))["alpha_mae"], 1.0
        )

    def test_ssim_detects_changed_structure(self):
        a = np.ones((16, 16, 4))
        b = a.copy()
        b[4:12, 4:12, :3] = 0
        self.assertLess(rgba_metrics(a, b)["ssim_rgb"], 0.99)


if __name__ == "__main__":
    unittest.main()
