"""Inspect the real shell launchers without loading a model or allocating VRAM."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FlashLauncherTests(unittest.TestCase):
    def command(self, profile="standard", **overrides):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("DS4_", "CUDA_ARCH"))
        }
        env.update(overrides)
        if profile is not None:
            env["DS4_FLASH_PROFILE"] = profile
        result = subprocess.run(
            ["bash", str(ROOT / "agent/start-ds4-server-flash.sh"), "--print-command"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = shlex.split(result.stdout)
        return args, dict(
            item.split("=", 1) for item in args[1 : args.index("./ds4-server")]
        )

    def test_standard_complete_gpu_profile(self):
        args, env = self.command()
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "2048")
        self.assertEqual(args[args.index("--ctx") + 1], "65536")
        self.assertIn("--mtp", args)
        self.assertEqual(env["DS4_CUDA_QWEN4_PCIE_FRAC"], "1")
        self.assertEqual(env["DS4_CUDA_QWEN4_HYBRID_PREFILL"], "0")
        self.assertEqual(env["DS4_CUDA_QWEN4_KV_RESIDENT"], "32768")
        for key in (
            "KV_STREAMING",
            "PREFILL_RING",
            "DECODE_GRAPHS",
            "DECODE_RANKS",
            "KV_DECODE_BATCH",
            "QSA_REGISTER_KEYS",
        ):
            self.assertEqual(env["DS4_CUDA_QWEN4_" + key], "1")
        self.assertNotIn("DS4_CUDA_QWEN4_MTP_RESIDENT", env)
        self.assertNotIn("DS4_CUDA_QWEN4_PREFILL_AHEAD", env)

    def test_default_is_the_qualified_profile(self):
        self.assertEqual(self.command(None), self.command("standard"))

    def test_explicit_overrides(self):
        args, env = self.command(
            DS4_PREFILL_CHUNK="4096",
            DS4_CTX="65536",
            DS4_CUDA_QWEN4_KV_STREAMING="0",
            DS4_CUDA_QWEN4_PCIE_FRAC="0.75",
        )
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "4096")
        self.assertEqual(args[args.index("--ctx") + 1], "65536")
        self.assertEqual(env["DS4_CUDA_QWEN4_KV_STREAMING"], "0")
        self.assertEqual(env["DS4_CUDA_QWEN4_PCIE_FRAC"], "0.75")

    def test_legacy_reproduces_original(self):
        args, env = self.command("legacy")
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "256")
        self.assertEqual(
            set(env),
            {
                "DS4_CUDA_QWEN4_TIERED_CACHE",
                "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB",
                "DS4_CUDA_QWEN4_RAM_CACHE_MB",
            },
        )

    def test_27b_launcher_ignores_flash_profile(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
        env["DS4_FLASH_PROFILE"] = "standard"
        result = subprocess.run(
            ["bash", str(ROOT / "agent/start-ds4-server.sh"), "--print-command"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = shlex.split(result.stdout)
        self.assertEqual(args[args.index("--ctx") + 1], "26000")
        self.assertIn("./ds4-server-suite", args)
        self.assertEqual(
            args[args.index("--suite-config") + 1], "config/native-suite.example.json"
        )
        self.assertIn("--mtp", args)
        self.assertIn("--vision", args)
        self.assertNotIn("--prefill-chunk", args)
        self.assertFalse(any("DS4_CUDA_QWEN4" in value for value in args))

    def test_native_agent_configuration_preserves_explicit_capacity_and_flags(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
        env.update(
            DS4_SUITE_CONFIG="/custom path/suite.json",
            DS4_CTX="28000",
            DS4_SERVER_PORT="8090",
        )
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "agent/start-ds4-server.sh"),
                "--print-command",
                "--no-mtp",
            ],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = shlex.split(result.stdout)
        self.assertEqual(
            args[args.index("--suite-config") + 1], "/custom path/suite.json"
        )
        self.assertEqual(args[args.index("--ctx") + 1], "28000")
        self.assertEqual(args[args.index("--port") + 1], "8090")
        self.assertEqual(args[-1], "--no-mtp")

    def test_root_launcher_retains_llm_default_and_accepts_native_config(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
        command = ["bash", str(ROOT / "start-ds4-server.sh"), "--print-command"]
        legacy = shlex.split(
            subprocess.run(
                command, env=env, text=True, capture_output=True, check=True
            ).stdout
        )
        self.assertIn("./ds4-server", legacy)
        self.assertNotIn("--suite-config", legacy)
        env["DS4_SUITE_CONFIG"] = "config/native-suite.example.json"
        native = shlex.split(
            subprocess.run(
                command, env=env, text=True, capture_output=True, check=True
            ).stdout
        )
        self.assertIn("./ds4-server-suite", native)
        self.assertEqual(
            native[native.index("--ctx") + 1], legacy[legacy.index("--ctx") + 1]
        )

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.command("invalid")


if __name__ == "__main__":
    unittest.main()
