"""Native wrapper ownership and the public interface used by diagnostics."""

import ctypes as C
import os
import unittest
from unittest.mock import Mock, patch

import torch

from tests.suite_numeric_api import ROOT, NativeError, Numeric, Status


class NumericOwner(unittest.TestCase):
    def test_public_operations_and_context_teardown(self):
        with Numeric(library_path=ROOT / "tests/libds4_numeric_cpu.so") as numeric:
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
            weights = torch.tensor([[2.0, 0.0]], dtype=torch.bfloat16)
            result = numeric.tensor(torch.zeros((2, 1), dtype=torch.bfloat16))
            numeric.linear(result, numeric.tensor(x), numeric.tensor(weights), 2, 2, 1)
            numeric.scale(result, result, 0.5)
            numeric.fence()
            torch.testing.assert_close(
                numeric.read(result, (2, 1)), torch.tensor([[1.0], [3.0]])
            )
            self.assertGreater(numeric.memory_usage().live_bytes, 0)
            numeric.clear()
            self.assertEqual(numeric.memory_usage().live_bytes, 0)
        numeric.close()  # Teardown is safe after a context manager too.

    def test_native_failure_does_not_lose_a_buffer_handle(self):
        with Numeric() as numeric:
            numeric.tensor(torch.ones(4))
            before = numeric.memory_usage()
            with (
                patch.object(numeric.lib, "ds4_num_free", return_value=5),
                self.assertRaisesRegex(AssertionError, "native status 5"),
            ):
                numeric.clear()
            self.assertEqual(numeric.memory_usage(), before)
            numeric.clear()
            self.assertEqual(numeric.memory_usage().live_bytes, 0)

    def test_slices_borrow_storage_and_check_native_bounds(self):
        with Numeric() as numeric:
            backing = numeric.tensor(torch.arange(6).float())
            before = numeric.memory_usage()
            middle = numeric.slice(backing, 1, 3)
            numeric.scale(middle, middle, 2)
            torch.testing.assert_close(
                numeric.read(backing, (6,)),
                torch.tensor([0.0, 2.0, 4.0, 6.0, 4.0, 5.0]),
            )
            self.assertEqual(numeric.memory_usage(), before)
            for offset, count in (
                (-1, 1),
                (0, 0),
                (6, 1),
                (0, 7),
                (2**64, 1),
                (0, 2**64),
                (0.0, 1),
                (True, 1),
            ):
                with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                    numeric.slice(backing, offset, count)

    def test_context_creation_failure_closes_the_resource_registry(self):
        library = Mock()

        def create_resources(output):
            C.cast(output, C.POINTER(C.c_void_p))[0] = C.c_void_p(123)
            return 0

        library.ds4_resources_create.side_effect = create_resources
        library.ds4_numeric_cpu.return_value = Status.NOMEM
        library.ds4_resources_close.return_value = 0
        with (
            patch.object(Numeric, "_load_library", return_value=library),
            self.assertRaises(NativeError) as failed,
        ):
            Numeric()
        self.assertEqual(failed.exception.status, Status.NOMEM)
        library.ds4_resources_close.assert_called_once()
        library.ds4_numeric_close.assert_not_called()

    def test_context_close_failure_can_be_retried(self):
        numeric = Numeric()
        try:
            with (
                patch.object(numeric.lib, "ds4_numeric_close", return_value=5),
                self.assertRaisesRegex(AssertionError, "native status 5"),
            ):
                numeric.close()
        finally:
            numeric.close()
        numeric.close()

    def test_invalid_backend_and_dtype_fail_before_native_allocation(self):
        with patch.object(Numeric, "_load_library") as load:
            for backend in (True, False, "unknown"):
                with self.assertRaises(ValueError):
                    Numeric(backend)
            load.assert_not_called()
        with Numeric() as numeric:
            for dtype in (torch.float16, torch.float64, torch.int32):
                with self.assertRaises(ValueError):
                    numeric.tensor(torch.ones(4, dtype=dtype))
            self.assertEqual(numeric.memory_usage().live_bytes, 0)
            with self.assertRaisesRegex(ValueError, "Metal context"):
                numeric.configure_metal_budget(reserve_bytes=0, storage_limit_bytes=64)

    def test_file_tensors_validate_public_dtypes_and_ranges_before_allocation(self):
        with (
            Numeric() as numeric,
            patch.object(numeric.lib, "ds4_num_alloc") as allocate,
        ):
            for dtype in (0, 30, False, torch.float16, torch.float64):
                with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                    numeric.file_tensor("missing-file", 1, dtype)
            for count, offset in (
                (0, 0),
                (-1, 0),
                (True, 0),
                (1.0, 0),
                (1, -1),
                (1, False),
                (1, 2**63),
                (2**63, 0),
            ):
                with (
                    self.subTest(count=count, offset=offset),
                    self.assertRaises(ValueError),
                ):
                    numeric.file_tensor(
                        "missing-file", count, torch.float32, offset=offset
                    )
            with self.assertRaises(FileNotFoundError):
                numeric.file_tensor("missing-file", 1, torch.float32)
            allocate.assert_not_called()

    @unittest.skipUnless(
        os.environ.get("DS4_SUITE_NUMERIC_METAL"), "Metal budget fixture"
    )
    def test_metal_budget_caps_storage_and_allows_recovery(self):
        with Numeric("metal") as numeric:
            numeric.configure_metal_budget(
                reserve_bytes=2 * 1024**3, storage_limit_bytes=1024**2
            )
            with self.assertRaises(NativeError) as failed:
                numeric.tensor(torch.ones(300000))
            self.assertEqual(failed.exception.status, Status.NOMEM)
            numeric.tensor(torch.ones(16))
            numeric.clear()
            self.assertEqual(numeric.memory_usage().live_bytes, 0)


if __name__ == "__main__":
    unittest.main()
