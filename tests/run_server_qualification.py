"""Run server Python suites and retain failures, subtests and skips as JSON.

Live suites use their existing opt-in environment variables. No server process
is started here: target and MTP instances must be run sequentially by the caller.
"""

import argparse
import json
import time
import traceback
import unittest
from pathlib import Path


class QualificationResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.started = time.monotonic()

    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)

    def record(self, test, status, error=None, reason=None):
        value = {
            "test": test.id(),
            "status": status,
            "seconds": round(time.monotonic() - self.started, 3),
        }
        if error:
            value["traceback"] = "".join(traceback.format_exception(*error))
        if reason:
            value["reason"] = reason
        self.records.append(value)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "passed")

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.record(test, "failed", error)

    def addError(self, test, error):
        super().addError(test, error)
        self.record(test, "error", error)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, "skipped", reason=reason)

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error:
            self.record(
                subtest,
                "failed" if issubclass(error[0], test.failureException) else "error",
                error,
            )

    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self.record(test, "expected_failure", error)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, "unexpected_success")


def run(names, report):
    suite = (
        unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromName(name) for name in names
        )
        if names
        else unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    )
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=QualificationResult).run(
        suite
    )
    record = {
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "seconds": round(time.monotonic() - started, 3),
        "tests": result.records,
    }
    Path(report).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suites",
        nargs="*",
        help="unittest module/class/test names; default: all Python fixtures",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.suites, args.report))
