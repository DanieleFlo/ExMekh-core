"""Timeout/resource failures keep evidence and cannot become successful gates."""

import contextlib
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import qualify_metal_server

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tools.metal_command", ROOT / "tools/qualify_metal_command.py"
)
COMMAND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMAND)


class BoundedCommandTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES="2147483648")
        )

    def test_resolved_reserve_reaches_command_report_and_monitor(self):
        for value, reserve in (("0", 0), ("invalid", 2**31)):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES=value),
            ):
                code = (
                    "import os; "
                    f"assert os.environ['DS4_METAL_HOST_RESERVE_BYTES'] == '{reserve}'"
                )
                result = self.run_command(code, host_available=reserve + 1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["memory_policy"]["host_reserve_bytes"], reserve)
            self.assertEqual(
                result["environment"]["DS4_METAL_HOST_RESERVE_BYTES"], str(reserve)
            )

    def run_command(
        self,
        code,
        *,
        footprint=1024,
        timeout=3,
        host_available=10 * 2**30,
        pressure=1,
        footprint_limit=1,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            measured = {
                "pressure": pressure,
                "swap_bytes": 10,
                "footprint_bytes": footprint,
                "resident_bytes": footprint,
                "host_available_bytes": host_available,
            }
            with patch.object(COMMAND, "sample", return_value=measured):
                result = COMMAND.run(
                    [sys.executable, "-c", code], output, timeout, footprint_limit
                )
            recorded = json.loads((output / "report.json").read_text())
            self.assertEqual(result, recorded)
            self.assertTrue((output / "command.log").is_file())
            self.assertEqual(len(result["log_sha256"]), 64)
            self.assertEqual(result["reason"], "initial bounded qualification")
            self.assertIn(sys.executable, result["executables_sha256"])
            return result

    def test_nonzero_exit_is_failure(self):
        result = self.run_command("raise SystemExit(7)")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["exit_code"], 7)

    def test_success_requires_zero_exit(self):
        self.assertEqual(self.run_command('print("complete")')["status"], "PASS")

    def test_local_timeout_multiplier_scales_and_records_without_changing_default(self):
        for multiplier in (1, 2, 4):
            with (
                self.subTest(multiplier=multiplier),
                patch.dict(os.environ, DS4_METAL_TIMEOUT_MULTIPLIER=str(multiplier)),
            ):
                report = self.run_command("print('complete')", timeout=2)
                self.assertEqual(report["base_timeout_seconds"], 2)
                self.assertEqual(report["timeout_multiplier"], multiplier)
                self.assertEqual(report["timeout_seconds"], 2 * multiplier)

    def test_sigterm_cleans_the_child_and_persists_interrupted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            ready = Path(directory) / "ready"
            child = (
                "import os,time; from pathlib import Path; "
                f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
            )
            code = (
                "import sys; from pathlib import Path; "
                "from tools import qualify_metal_command as m; "
                "m.sample=lambda pid: dict(pressure=1,swap_bytes=0,"
                "host_available_bytes=10*2**30,footprint_bytes=1024); "
                "\nwith m.interrupt_on_termination():\n"
                f" m.run([sys.executable,'-c',{child!r}],Path({str(output)!r}),30,0)\n"
            )
            monitor = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and monitor.poll() is None:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                child_pid = int(ready.read_text())
                monitor.terminate()
                monitor.wait(timeout=5)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "NOT_VERIFIED")
                self.assertIn("termination requested", report["error"])
                self.assertFalse(report["cleanup_forced"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if monitor.poll() is None:
                    monitor.terminate()
                    monitor.wait(timeout=5)

    def test_timeout_keeps_unverified_evidence(self):
        result = self.run_command("import time; time.sleep(30)", timeout=0.01)
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertLess(result["elapsed_seconds"], 5)
        self.assertNotEqual(result["exit_code"], 0)

    def test_host_reserve_stops_before_pressure_or_swap(self):
        result = self.run_command("import time; time.sleep(30)", host_available=2**30)
        self.assertEqual(result["status"], "SKIP_RESOURCE_LIMIT")

    def test_resource_stop_is_distinct_from_failure(self):
        result = self.run_command("import time; time.sleep(30)", footprint=2**31)
        self.assertEqual(result["status"], "SKIP_RESOURCE_LIMIT")
        self.assertNotEqual(result["exit_code"], 0)

    def test_available_ram_policy_allows_warning_pressure_and_large_footprint(self):
        result = self.run_command(
            "print('complete')", footprint=12 * 2**30, footprint_limit=0, pressure=2
        )
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["footprint_limit_gib"])

    def test_critical_pressure_stops_with_evidence(self):
        result = self.run_command("import time; time.sleep(30)", pressure=4)
        self.assertEqual(result["status"], "SKIP_RESOURCE_LIMIT")

    def test_cleanup_failure_keeps_report_and_cannot_promote_a_success(self):
        with patch.object(
            COMMAND, "stop", side_effect=PermissionError("signal denied")
        ):
            result = self.run_command("print('complete')")
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("signal denied", result["cleanup_error"])

    def test_monitor_failure_preserves_its_cause_and_drains_the_child(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            with (
                patch.object(
                    COMMAND, "sample", side_effect=OSError("telemetry denied")
                ),
                self.assertRaisesRegex(OSError, "telemetry denied"),
            ):
                COMMAND.run(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    output,
                    3,
                    0,
                )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "NOT_VERIFIED")
            self.assertIn("telemetry denied", report["error"])
            self.assertIsNotNone(report["exit_code"])
            self.assertLess(report["elapsed_seconds"], 5)

    def test_forced_cleanup_cannot_promote_a_success(self):
        with patch.object(COMMAND, "stop", return_value=False):
            result = self.run_command("print('complete')")
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertTrue(result["cleanup_forced"])

    def test_group_cleanup_waits_for_workers_and_only_kills_after_deadline(self):
        process = MagicMock(pid=42)
        process.poll.return_value = 0
        with (
            patch.object(COMMAND.os, "killpg") as send,
            patch.object(COMMAND, "process_group_finished", side_effect=[False, True]),
            patch.object(COMMAND.time, "sleep"),
        ):
            self.assertTrue(COMMAND.stop(process))
        send.assert_called_once_with(42, signal.SIGTERM)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 0), None]
        with patch.object(COMMAND.os, "killpg") as send:
            self.assertFalse(COMMAND.stop(process, grace_seconds=0))
        self.assertEqual(
            [call.args for call in send.call_args_list],
            [(42, signal.SIGTERM), (42, signal.SIGKILL)],
        )

    def test_signal_permission_race_is_ignored_only_for_a_finished_group(self):
        process = MagicMock(pid=42)
        process.poll.return_value = 0
        with (
            patch.object(COMMAND.os, "killpg", side_effect=PermissionError),
            patch.object(COMMAND, "process_group_finished", return_value=True),
        ):
            COMMAND.stop(process)
        with (
            patch.object(COMMAND.os, "killpg", side_effect=PermissionError),
            patch.object(COMMAND, "process_group_finished", return_value=False),
            self.assertRaises(PermissionError),
        ):
            COMMAND.stop(process)

    def test_zombies_are_finished_but_live_workers_are_not(self):
        listing = MagicMock(stdout="42 Z\n17 S\n42 Z+\n")
        with patch.object(COMMAND.subprocess, "run", return_value=listing):
            self.assertTrue(COMMAND.process_group_finished(42))
            listing.stdout += "42 S+\n"
            self.assertFalse(COMMAND.process_group_finished(42))


class BoundedServerArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES="2147483648")
        )

    def test_no_reserve_profile_reaches_server_and_monitor(self):
        with patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES="0"):
            report, command = self.interrupt_server("--automatic-cache")
        self.assertEqual(report["host_reserve_bytes"], 0)
        self.assertNotIn("--ssd-streaming-cache-experts", command)

    def test_suite_operation_deadline_scales_without_modifying_source_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.json"
            original = '{"version":1,"timeout_seconds":300,"models":[]}\n'
            source.write_text(original)
            for factor in (1, 4):
                output = root / str(factor)
                output.mkdir()
                prepared, base = qualify_metal_server.prepare_suite_configuration(
                    source, output, factor
                )
                configuration = json.loads(prepared.read_text())
                self.assertEqual(base, 300)
                self.assertEqual(configuration["timeout_seconds"], 300 * factor)
                self.assertEqual(
                    configuration["media_directory"], str(output / "media")
                )
                self.assertEqual(source.read_text(), original)

    def test_invalid_scaled_service_deadline_is_rejected_before_preparing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "invalid.json"
            for timeout in (0, "300", True, 86400):
                source.write_text(json.dumps({"timeout_seconds": timeout}))
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    qualify_metal_server.prepare_suite_configuration(source, output, 4)
                self.assertFalse((output / "suite-config.json").exists())
                self.assertFalse((output / "media").exists())

    def test_server_interruption_drains_children_saves_evidence_and_reaches_owner(self):
        report, command = self.interrupt_server()
        self.assertIn("--ssd-streaming-cache-experts", command)
        self.assertEqual(report["qualification_profile"]["ssd_cache_bytes"], 2**31)
        self.assertEqual(report["qualification_profile"]["ssd_cache_mode"], "explicit")

    def test_extended_automatic_profile_keeps_memory_guards_and_records_actual_command(
        self,
    ):
        report, command = self.interrupt_server(
            "--automatic-cache", "--timeout", "3600", "--timeout-multiplier", "4"
        )
        self.assertNotIn("--ssd-streaming-cache-experts", command)
        self.assertIsNone(report["qualification_profile"]["ssd_cache_bytes"])
        self.assertEqual(report["qualification_profile"]["ssd_cache_mode"], "automatic")
        self.assertEqual(report["host_reserve_bytes"], 2**31)
        self.assertEqual(report["critical_pressure"], 4)
        self.assertEqual(report["timeout_seconds"], 14400)

    def interrupt_server(self, *options):
        _, invoke = self.invoke(*options)
        arguments = invoke.call_args.args[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments.output = root / "run"
            (root / "ds4-server").write_bytes(b"qualification server fixture")
            server = MagicMock(pid=42, returncode=0)
            with (
                patch.object(qualify_metal_server, "ROOT", root),
                patch.object(qualify_metal_server.socket, "socket"),
                patch.object(
                    qualify_metal_server.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, stdout="fixture-head"
                    ),
                ),
                patch.object(
                    qualify_metal_server.subprocess, "Popen", return_value=server
                ) as spawn,
                patch.object(
                    qualify_metal_server,
                    "sample",
                    side_effect=KeyboardInterrupt("stop"),
                ),
                patch.object(qualify_metal_server, "stop", return_value=True) as stop,
                self.assertRaises(KeyboardInterrupt),
            ):
                qualify_metal_server.run(arguments)
            stop.assert_called_once_with(server, grace_seconds=30)
            report = json.loads((arguments.output / "report.json").read_text())
            self.assertEqual(report["status"], "NOT_VERIFIED")
            self.assertEqual(report["error"], "stop")
            self.assertEqual(report["cleanup_forced"], [])
            self.assertEqual(
                spawn.call_args.kwargs["env"]["DS4_METAL_HOST_RESERVE_BYTES"],
                str(report["host_reserve_bytes"]),
            )
            return report, spawn.call_args.args[0]

    def test_occupied_port_never_starts_or_qualifies_an_unowned_server(self):
        _, invoke = self.invoke()
        arguments = invoke.call_args.args[0]
        with socket.socket() as listener, tempfile.TemporaryDirectory() as directory:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            arguments.port = listener.getsockname()[1]
            root = Path(directory)
            arguments.output = root / "run"
            (root / "ds4-server").write_bytes(b"qualification server fixture")
            with (
                patch.object(qualify_metal_server, "ROOT", root),
                patch.object(
                    qualify_metal_server.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, stdout="fixture-head"
                    ),
                ),
                patch.object(qualify_metal_server.subprocess, "Popen") as spawn,
            ):
                report = qualify_metal_server.run(arguments)
            spawn.assert_not_called()
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("Address already in use", report["error"])
            self.assertEqual(report["cleanup_forced"], [])

    def test_cleanup_requires_graceful_groups_and_zero_server_exit(self):
        client, server = MagicMock(pid=41), MagicMock(pid=42)
        for initial in ("PASS", "FAIL"):
            for forced, code in ((False, 0), (True, 0), (False, -15)):
                with self.subTest(initial=initial, forced=forced, code=code):
                    report = {"status": initial}
                    server.returncode = code
                    with patch.object(
                        qualify_metal_server, "stop", side_effect=[not forced, True]
                    ) as stop:
                        qualify_metal_server.cleanup_processes(report, client, server)
                    expected = initial
                    if initial == "PASS" and (forced or code != 0):
                        expected = "NOT_VERIFIED"
                    self.assertEqual(report["status"], expected)
                    self.assertEqual(report["cleanup_forced"], [41] if forced else [])
                    self.assertEqual(report["server_exit_code"], code)
                    self.assertEqual(
                        [call.kwargs for call in stop.call_args_list],
                        [{"grace_seconds": 5}, {"grace_seconds": 30}],
                    )

    def test_cleanup_error_keeps_its_cause_and_attempts_remaining_processes(self):
        client, server = MagicMock(pid=41), MagicMock(pid=42, returncode=0)
        report = {"status": "PASS"}
        with patch.object(
            qualify_metal_server, "stop", side_effect=[PermissionError("denied"), True]
        ) as stop:
            qualify_metal_server.cleanup_processes(report, client, server)
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(report["status"], "NOT_VERIFIED")
        self.assertEqual(report["cleanup_errors"][0]["pid"], 41)
        self.assertIn("denied", report["cleanup_errors"][0]["error"])

    def invoke(self, *arguments):
        command = [
            "qualify_metal_server.py",
            "--family",
            "qwen27",
            "--output",
            "unused",
            *arguments,
        ]
        with (
            patch.object(sys, "argv", command),
            patch.object(
                qualify_metal_server,
                "run",
                return_value={
                    "status": "PASS",
                    "elapsed_seconds": 0,
                },
            ) as run,
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as stopped,
        ):
            qualify_metal_server.main()
        return stopped.exception.code, run

    def test_explicit_extended_browser_budget_is_recorded_before_execution(self):
        code, run = self.invoke("--timeout", "1200", "--tool-start-timeout", "720")
        self.assertEqual(code, 0)
        profile = run.call_args.args[0]
        self.assertEqual(profile.timeout, 1200)
        self.assertEqual(profile.tool_start_timeout, 720)
        self.assertEqual(profile.max_footprint_gib, 0)

    def test_invalid_environment_multiplier_is_rejected_before_starting_a_server(self):
        for value in ("3", "0", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, DS4_METAL_TIMEOUT_MULTIPLIER=value),
            ):
                code, run = self.invoke()
                self.assertEqual(code, 2)
                run.assert_not_called()

    def test_timeout_multiplier_is_explicit_and_can_override_the_local_environment(
        self,
    ):
        with patch.dict(os.environ, DS4_METAL_TIMEOUT_MULTIPLIER="4"):
            code, run = self.invoke("--timeout", "1200")
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.args[0].timeout_multiplier, 4)
            self.assertEqual(run.call_args.args[0].timeout, 1200)
            code, run = self.invoke("--timeout-multiplier", "1")
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.args[0].timeout_multiplier, 1)
        for value in ("0", "3", "nan"):
            code, run = self.invoke("--timeout-multiplier", value)
            self.assertEqual(code, 2)
            run.assert_not_called()

    def test_invalid_deadlines_never_start_a_model(self):
        for arguments in (
            ("--timeout", "0"),
            ("--timeout", str(qualify_metal_server.MAX_QUALIFICATION_SECONDS + 1)),
            ("--timeout", "nan"),
            ("--timeout", "inf"),
            ("--tool-start-timeout", "0"),
            ("--tool-start-timeout", "nan"),
            ("--timeout", "120", "--tool-start-timeout", "121"),
        ):
            with self.subTest(arguments=arguments):
                code, run = self.invoke(*arguments)
                self.assertEqual(code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
