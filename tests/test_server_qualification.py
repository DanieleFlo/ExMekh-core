"""A qualification report must preserve failing subtests and skipped checks."""

import io
import unittest

from tests.run_server_qualification import QualificationResult


class QualificationReportTests(unittest.TestCase):
    def test_subtest_failures_and_errors_are_never_reported_as_success(self):
        class Cases(unittest.TestCase):
            def test_operations(self):
                with self.subTest(operation="delete"):
                    self.assertEqual("rewrite", "delete")
                with self.subTest(operation="decode"):
                    raise ValueError("invalid payload")

            @unittest.skip("backend unavailable")
            def test_backend(self):
                pass

        result = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=QualificationResult
        ).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(
            (result.testsRun, len(result.failures), len(result.errors)), (2, 1, 1)
        )
        self.assertEqual(
            [record["status"] for record in result.records],
            ["skipped", "failed", "error"],
        )
        self.assertIn("operation='delete'", result.records[1]["test"])
        self.assertIn("invalid payload", result.records[2]["traceback"])
        self.assertEqual(result.records[0]["reason"], "backend unavailable")


if __name__ == "__main__":
    unittest.main()
