"""Qualification evidence belongs to the client invocation, without inference."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.qualification_client import QualificationClient


class QualificationClientTests(unittest.TestCase):
    def test_suite_cli_rejects_abbreviations_that_bypass_owned_url_validation(self):
        from tools import qualify_suite_release

        with (
            patch(
                "sys.argv",
                [
                    "qualify_suite_release.py",
                    "--bounded",
                    "--cycles",
                    "1",
                    "--output",
                    "unused",
                    "--url",
                    "http://owned/v1",
                    "--ur",
                    "http://other/v1",
                ],
            ),
            patch("sys.stderr", io.StringIO()),
            patch.object(qualify_suite_release, "run") as run,
            self.assertRaises(SystemExit) as failed,
        ):
            qualify_suite_release.main()
        self.assertEqual(failed.exception.code, 2)
        run.assert_not_called()

    def test_entrypoint_detection_ignores_option_values_and_interpreter_flags(self):
        for command, expected in (
            (["python", "-u", "tools/qualify_suite_release.py"], True),
            (["python3.13", "-X", "dev", "-m", "tools.qualify_suite_release"], True),
            (["./tools/qualify_suite_release.py"], True),
            (["python", "other.py", "--output", "qualify_suite_release.py"], False),
            (["python", "-c", "print('x')", "qualify_suite_release.py"], False),
            (["python", "other.py", "--output", "pytest"], False),
        ):
            with self.subTest(command=command):
                client = QualificationClient.prepare(
                    command, Path.cwd(), Path("unused.xml")
                )
                self.assertEqual(client.is_suite_release, expected)
                self.assertIsNone(client.junit)

    def test_suite_options_match_argparse_duplicates_and_validate_the_owned_url(self):
        client = QualificationClient.prepare(
            [
                "python",
                "tools/qualify_suite_release.py",
                "--output",
                "first",
                "--output=last",
                "--url",
                "http://local/v1",
                "--url=http://other/v1",
            ],
            Path.cwd(),
            Path("unused.xml"),
        )
        self.assertEqual(client.suite_output, Path.cwd() / "last")
        with self.assertRaisesRegex(ValueError, "owned qualification server"):
            client.for_suite_server("http://local/v1", True)
        pinned = client.for_suite_server("http://other/v1", True)
        self.assertEqual(pinned.option("--url"), "http://other/v1")
        self.assertNotIn("--resume", client.command)
        self.assertEqual(pinned.command.count("--resume"), 1)
        self.assertEqual(pinned.for_suite_server("http://other/v1/", True), pinned)
        self.assertEqual(pinned.identity_command(), list(client.command))
        for command in (
            ["python", "tools/qualify_suite_release.py", "--output"],
            ["python", "tools/qualify_suite_release.py", "--output", "--bounded"],
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                _ = QualificationClient.prepare(
                    command, Path.cwd(), Path("unused.xml")
                ).suite_output

    def test_segmented_success_is_demoted_by_failed_or_missing_junit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.xml"
            client = QualificationClient.prepare(["pytest"], output.parent, output)
            report = {"status": "PASS_SEGMENTED"}
            client.record_results(report)
            self.assertEqual(report["status"], "NOT_VERIFIED")
            output.write_text(
                '<testsuite><testcase><failure message="bad"/></testcase></testsuite>'
            )
            report = {"status": "PASS_SEGMENTED"}
            client.record_results(report)
            self.assertEqual(report["status"], "FAIL")

    def test_partial_client_xml_keeps_timeout_evidence_and_never_becomes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted.xml"
            output.write_text('<testsuites><testsuite><testcase name="unfinished"')
            for status in ("NOT_VERIFIED", "SKIP_RESOURCE_LIMIT", "PASS"):
                with self.subTest(status=status):
                    report = {"status": status, "elapsed_seconds": 123}
                    QualificationClient.prepare(
                        ["pytest"], output.parent, output
                    ).record_results(report)
                    self.assertEqual(
                        report["status"], "FAIL" if status == "PASS" else status
                    )
                    self.assertEqual(report["elapsed_seconds"], 123)
                    self.assertIn("test_results_error", report)

    def test_case_results_keep_model_skips_distinct_from_executed_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "cases.xml"
            report.write_text(
                "<testsuites><testsuite>"
                '<testcase name="passed" time="1.2"/>'
                '<testcase name="limited"><skipped message="MODEL_LIMITATION"/></testcase>'
                '<testcase name="broken"><failure message="bad output"/></testcase>'
                "</testsuite></testsuites>"
            )
            client = QualificationClient.prepare(["pytest"], report.parent, report)
            cases = client.read_test_results()
            self.assertEqual(
                [case["status"] for case in cases], ["PASS", "NOT_VERIFIED", "FAIL"]
            )
            self.assertEqual(cases[1]["detail"], "MODEL_LIMITATION")
            self.assertEqual(cases[0]["seconds"], 1.2)
            recorded = {"status": "PASS"}
            client.record_results(recorded)
            self.assertEqual(recorded["status"], "FAIL")
            self.assertFalse(recorded["all_selected_tests_passed"])

    def test_a_selection_is_pass_only_when_every_selected_case_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.xml"
            passed = '<testcase name="passed"/>'
            skipped = '<testcase name="limited"><skipped message="MODEL_LIMITATION"/></testcase>'
            for cases, expected in (
                (passed, "PASS"),
                (skipped, "NOT_VERIFIED"),
                (passed + skipped, "NOT_VERIFIED"),
            ):
                with self.subTest(cases=cases):
                    output.write_text(f"<testsuite>{cases}</testsuite>")
                    report = {"status": "PASS"}
                    QualificationClient.prepare(
                        ["pytest"], output.parent, output
                    ).record_results(report)
                    self.assertEqual(report["status"], expected)
                    self.assertEqual(
                        report["all_selected_tests_passed"], expected == "PASS"
                    )

    def test_missing_or_empty_junit_does_not_qualify_a_successful_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.xml"
            client = QualificationClient.prepare(["pytest"], output.parent, output)
            for content in (None, "<testsuites/>"):
                if content is not None:
                    output.write_text(content)
                for initial in ("PASS", "FAIL", "NOT_VERIFIED", "SKIP_RESOURCE_LIMIT"):
                    with self.subTest(content=content, initial=initial):
                        report = {"status": initial}
                        client.record_results(report)
                        self.assertEqual(
                            report["status"],
                            "NOT_VERIFIED" if initial == "PASS" else initial,
                        )
                        self.assertFalse(report["all_selected_tests_passed"])
                        self.assertIn("test_results_error", report)

    def test_explicit_junit_aliases_are_resolved_from_the_client_working_directory(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            for option in ("--junitxml", "--junit-xml"):
                for arguments in ([option, "custom.xml"], [option + "=custom.xml"]):
                    with self.subTest(arguments=arguments):
                        command = ["python", "-m", "pytest", *arguments]
                        client = QualificationClient.prepare(
                            command, cwd, cwd / "unused.xml"
                        )
                        self.assertEqual(client.junit, cwd / "custom.xml")
                        self.assertEqual(list(client.command), command)
                        client.junit.write_text(
                            '<testsuite><testcase name="ok"/></testsuite>'
                        )
                        report = {"status": "PASS"}
                        client.record_results(report)
                        self.assertTrue(report["all_selected_tests_passed"])

    def test_source_hashes_follow_the_client_cwd_and_preserve_node_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            (cwd / "tests").mkdir()
            source = cwd / "tests/test_client.py"
            source.write_text("client-specific fixture")
            client = QualificationClient.prepare(
                ["pytest", "tests/test_client.py::test_one"], cwd, cwd / "cases.xml"
            )
            self.assertEqual(
                client.source_hashes(),
                {
                    "tests/test_client.py": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                },
            )
            self.assertEqual(client.command[-1], f"--junitxml={cwd / 'cases.xml'}")

    def test_non_pytest_commands_do_not_require_a_junit_report(self):
        client = QualificationClient.prepare(
            ["python", "script.py"], Path.cwd(), Path("unused.xml")
        )
        report = {"status": "PASS"}
        client.record_results(report)
        self.assertIsNone(client.junit)
        self.assertEqual(report, {"status": "PASS", "test_results": []})

    def test_default_junit_is_relative_to_client_and_precedes_positional_separator(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            client = QualificationClient.prepare(
                ["python", "-m", "pytest", "--", "tests"], cwd, Path("cases.xml")
            )
            self.assertEqual(client.junit, cwd / "cases.xml")
            self.assertEqual(
                client.command[-3:], (f"--junitxml={cwd / 'cases.xml'}", "--", "tests")
            )

    def test_invalid_junit_options_and_durations_are_rejected(self):
        for arguments in (["--junitxml"], ["--junitxml="], ["--junit-xml", "-q"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                QualificationClient.prepare(
                    ["pytest", *arguments], Path.cwd(), Path("unused.xml")
                )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.xml"
            client = QualificationClient.prepare(["pytest"], output.parent, output)
            for seconds in ("nan", "inf", "-1"):
                output.write_text(
                    f'<testsuite><testcase time="{seconds}"/></testsuite>'
                )
                report = {"status": "PASS"}
                client.record_results(report)
                self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
