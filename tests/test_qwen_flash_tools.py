"""Numerical gates and benchmark lifetimes without a model or CUDA device."""
import json
import math
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from qwen_flash_launcher import launcher_environment
from qwen_flash_metrics import compare_teacher_logits, teacher_quality_passes

try:
    import bench_qwen_flash_agent as benchmark
    from exmekh_lib.core.models.user_message import UserMessage
except ModuleNotFoundError as error:
    if not error.name.startswith(("exmekh_agent", "exmekh_lib")):
        raise
    benchmark = None


class TeacherMetricsTests(unittest.TestCase):
    def test_known_distributions_and_teacher_loss(self):
        reference = np.log([[.5, .25, .25]])
        candidate = np.log([[.25, .5, .25]])
        metrics = compare_teacher_logits(reference, candidate, [0])
        self.assertAlmostEqual(metrics["kl_max"], .25 * math.log(2))
        self.assertAlmostEqual(metrics["loss_reference"], math.log(2))
        self.assertAlmostEqual(metrics["loss_candidate"], math.log(4))
        self.assertFalse(teacher_quality_passes(metrics))

    def test_same_argmax_does_not_hide_distribution_drift(self):
        metrics = compare_teacher_logits([[3., 2., 1.]], [[3., 1., 1.]], [0])
        self.assertEqual(metrics["argmax_rows_equal"], 1)
        self.assertFalse(teacher_quality_passes(metrics))

    def test_exact_rows_and_uniform_zero_logits(self):
        for logits in ([[1., 2., 3.], [-1., 4., 2.]], [[0., 0., 0.], [0., 0., 0.]]):
            metrics = compare_teacher_logits(logits, logits, [1, 2])
            self.assertTrue(metrics["bit_exact"])
            self.assertEqual(metrics["loss_delta"], 0)
            self.assertEqual(metrics["kl_max"], 0)
            self.assertTrue(teacher_quality_passes(metrics))

    def test_malformed_or_incomplete_dumps_fail_closed(self):
        cases = [([], [], []), ([1., 2.], [1., 2.], [0]),
                 ([[1., 2.]], [[1.]], [0]), ([[1., 2.]], [[1., 2.]], []),
                 ([[np.nan, 2.]], [[1., 2.]], [0]), ([[1., 2.]], [[np.inf, 2.]], [0]),
                 ([[1., 2.]], [[1., 2.]], [-1]), ([[1., 2.]], [[1., 2.]], [2]),
                 ([[1., 2.]], [[1., 2.]], [0.5])]
        for reference, candidate, targets in cases:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                compare_teacher_logits(reference, candidate, targets)


class LauncherEnvironmentTests(unittest.TestCase):
    def test_actual_defaults_ignore_inherited_diagnostic_overrides(self):
        with patch.dict("os.environ", {"DS4_CUDA_QWEN4_KV_STREAMING": "0", "DS4_CTX": "1",
                                       "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "2"}):
            env = launcher_environment()
        self.assertEqual(env["DS4_CUDA_QWEN4_KV_STREAMING"], "1")
        self.assertEqual(env["DS4_CUDA_WEIGHT_CACHE_LIMIT_GB"], "auto")
        self.assertEqual(env["DS4_CUDA_QWEN4_HYBRID_PREFILL"], "0")
        self.assertEqual(set(launcher_environment("legacy")), {
            "DS4_CUDA_QWEN4_TIERED_CACHE", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "DS4_CUDA_QWEN4_RAM_CACHE_MB"})


@unittest.skipIf(benchmark is None, "requires the agent virtual environment")
class AgentBenchmarkTests(unittest.TestCase):
    def test_failed_spawn_closes_log_and_remains_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            server = benchmark.OwnedServer(directory, {}, 0)
            with (patch.object(benchmark.subprocess, "Popen", side_effect=OSError("spawn failed")) as spawn,
                  self.assertRaisesRegex(OSError, "spawn failed")):
                server.__enter__()
            self.assertTrue(spawn.call_args.kwargs["stdout"].closed)
            server.close()
            with self.assertRaisesRegex(RuntimeError, "already used"), \
                    patch.object(benchmark.subprocess, "Popen") as retry:
                server.__enter__()
            retry.assert_not_called()
            self.assertEqual(json.loads((directory / "gpu-memory.json").read_text()), [])

    def test_malformed_startup_drains_only_the_owned_process(self):
        process = Mock(pid=424242)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(benchmark.subprocess, "Popen", return_value=process) as spawn, \
                patch.object(benchmark.urllib.request, "urlopen") as request, \
                patch.object(benchmark.os, "killpg") as kill:
            request.return_value.__enter__.return_value.read.return_value = b'{"data": []}'
            server = benchmark.OwnedServer(Path(temporary), {}, 0)
            with self.assertRaises(IndexError):
                server.__enter__()
            server.close()
            kill.assert_called_once_with(process.pid, benchmark.signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=30)
            self.assertTrue(spawn.call_args.kwargs["stdout"].closed)

    def test_occupied_port_never_launches_or_stops_another_server(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as temporary:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = benchmark.OwnedServer(Path(temporary), {}, listener.getsockname()[1])
            with patch.object(benchmark.subprocess, "Popen") as spawn:
                with self.assertRaises(OSError):
                    server.__enter__()
                spawn.assert_not_called()

    def test_exit_race_during_forced_shutdown_still_closes_log(self):
        process = Mock(pid=424242)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned server", 30), None]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(benchmark.subprocess, "Popen", return_value=process) as spawn, \
                patch.object(benchmark.urllib.request, "urlopen") as request, \
                patch.object(benchmark.os, "killpg", side_effect=[None, ProcessLookupError()]) as kill:
            request.return_value.__enter__.return_value.read.return_value = b'{"data": []}'
            server = benchmark.OwnedServer(Path(temporary), {}, 0)
            with self.assertRaises(IndexError):
                server.__enter__()
            self.assertEqual(kill.call_count, 2)
            self.assertEqual(process.wait.call_count, 2)
            self.assertTrue(spawn.call_args.kwargs["stdout"].closed)
            server.close()

    def test_provider_configuration_is_local_to_each_instance(self):
        # Both instances use the public ProviderRequest API; neither call may
        # redirect another instance's captured request or mutate its input.
        request = benchmark.ProviderRequest([UserMessage("benchmark")],
                                            extra={"sentinel": "preserve"}, stream=True)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(benchmark.LocalProvider, "__init__", return_value=None), \
                patch.object(benchmark.LocalProvider, "call") as call:
            directories = [Path(temporary) / name for name in ("first", "second")]
            for directory in directories:
                directory.mkdir()
            first = benchmark.GreedyProvider(capture_directory=directories[0], thinking=False)
            second = benchmark.GreedyProvider(capture_directory=directories[1], thinking=True)
            second.call(request)
            first.call(request)
            self.assertTrue(all((directory / "request-0.json").is_file() for directory in directories))
            self.assertEqual(call.call_args.args[0].get_extra(), {"sentinel": "preserve", "temperature": 0, "seed": 424242})
            self.assertEqual(request.get_extra(), {"sentinel": "preserve"})

    def test_three_turns_and_mismatched_pairs(self):
        initial = {"prefill_chunks": [{"total": 100}], "prompt_done_seconds": [10],
                   "first_visible_token_seconds": 11, "answer": "READY"}
        followup = {"wall_seconds": 2, "answer": "DONE"}
        records = [{"sample": 0, "warmup": False, "profile": "first", "turns": [initial, followup, {"answer": "extra"}]}]
        self.assertEqual(benchmark.comparison(records)["profiles"]["first"]["prefill_full_tps"]["median"], 10)
        records.append({"sample": 0, "warmup": False, "profile": "second", "turns": [initial, followup]})
        with self.assertRaisesRegex(ValueError, "different turn counts"):
            benchmark.comparison(records)


if __name__ == "__main__":
    unittest.main()
