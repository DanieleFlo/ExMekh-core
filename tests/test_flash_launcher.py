"""Inspect the real shell launchers without loading a model or allocating VRAM."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launcher_environment(**overrides):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DS4_", "CUDA_ARCH"))
    }
    return environment | overrides


class ServerLauncherTests(unittest.TestCase):
    def flash_command(self, profile="standard", **overrides):
        env = launcher_environment(DS4_BACKEND="cuda")
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
        environment = {}
        for argument in args[1:]:
            if "=" not in argument:
                break
            key, value = argument.split("=", 1)
            environment[key] = value
        return args, environment

    def test_standard_complete_gpu_profile(self):
        args, env = self.flash_command()
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
        self.assertEqual(self.flash_command(None), self.flash_command("standard"))

    def test_explicit_overrides(self):
        args, env = self.flash_command(
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
        args, env = self.flash_command("legacy")
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
        env = launcher_environment()
        env["DS4_BACKEND"] = "cuda"
        env["DS4_VISION_MODEL"] = __file__
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
        env = launcher_environment()
        env.update(
            DS4_BACKEND="cuda",
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
        env = launcher_environment()
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

    def test_streamed_profiles_align_server_and_client(self):
        for family, wrapper in (("flash", "start-ds4-server-flash.sh"),
                                ("qwen27", "start-ds4-server.sh")):
            with self.subTest(family=family):
                result = subprocess.run(
                    ["bash", "-c", 'set -a; source "$1"; shift; exec "$@"',
                     "profile", str(ROOT / f"config/{family}-metal-streamed.env.example"),
                     "bash", str(ROOT / "agent" / wrapper), "--print-command"],
                    cwd="/tmp", env=launcher_environment(), text=True,
                    capture_output=True, check=True,
                )
                args = shlex.split(result.stdout)
                self.assertEqual(args[args.index("--ctx") + 1], "8192")
                self.assertEqual(args[args.index("--prefill-chunk") + 1], "64")
                self.assertNotIn("--ssd-streaming-cache-experts", args)
                self.assertIn("--ssd-streaming", args)
                self.assertIn("--no-mtp", args)
                self.assertNotIn("--vision", args)
                self.assertNotIn("--suite-config", args)
                profile = (ROOT / f"config/{family}-metal-streamed.env.example").read_text()
                self.assertIn("CONTEXT_LENGTH=8192", profile)
                self.assertIn("REASONING_EFFORT=none", profile)
                model = "qwen3.8-flash-next" if family == "flash" else "Qwen3.8-27B-UD-Q4_K_S"
                self.assertIn(f"DEFAULT_MODEL={model}", profile)

    def test_metal_flash_uses_small_context_and_no_cuda_profile(self):
        args, env = self.flash_command(DS4_BACKEND="metal")
        self.assertIn("--metal", args)
        self.assertIn("--no-mtp", args)
        self.assertEqual(args[args.index("--ctx") + 1], "512")
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "64")
        self.assertFalse(any(key.startswith("DS4_CUDA_") for key in env))

    def test_streamed_cache_uses_metal_admission_and_keeps_explicit_ceiling(self):
        args, _ = self.flash_command(DS4_BACKEND="metal", DS4_SSD_STREAMING="1")
        self.assertNotIn("--ssd-streaming-cache-experts", args)
        args, _ = self.flash_command(DS4_BACKEND="metal", DS4_SSD_STREAMING="1",
                                     DS4_SSD_CACHE_GB="1.5")
        self.assertEqual(args[args.index("--ssd-streaming-cache-experts") + 1], "1.5GB")
        args, _ = self.flash_command(DS4_BACKEND="cuda", DS4_SSD_STREAMING="1")
        self.assertEqual(args[args.index("--ssd-streaming-cache-experts") + 1], "2GB")

    def test_metal_quick_chat_profile_has_explicit_agent_capacity(self):
        env = launcher_environment()
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -a; source "$1"; shift; exec "$@"',
                "profile",
                str(ROOT / "config/qwen27-metal-quick.env.example"),
                "bash",
                str(ROOT / "agent/start-ds4-server.sh"),
                "--print-command",
                "--host",
                "127.0.0.1",
            ],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = shlex.split(result.stdout)
        self.assertIn("./ds4-server", args)
        self.assertIn("--metal", args)
        self.assertIn("--no-mtp", args)
        self.assertEqual(args[args.index("--ctx") + 1], "4096")
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "64")
        self.assertNotIn("--suite-config", args)
        self.assertNotIn("--vision", args)
        self.assertEqual(args[-2:], ["--host", "127.0.0.1"])

    def test_metal_agent_honors_explicit_native_suite(self):
        env = launcher_environment()
        env.update(
            DS4_BACKEND="metal", DS4_SUITE_CONFIG="config/native-suite.example.json"
        )
        result = subprocess.run(
            ["bash", str(ROOT / "agent/start-ds4-server.sh"), "--print-command"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = shlex.split(result.stdout)
        self.assertIn("./ds4-server-suite", args)
        self.assertIn("--metal", args)
        self.assertIn("--no-mtp", args)
        self.assertEqual(args[args.index("--ctx") + 1], "512")
        self.assertEqual(args[args.index("--prefill-chunk") + 1], "64")

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.flash_command("invalid")

    def test_invalid_backend_and_mtp_are_rejected(self):
        for setting in ({"DS4_BACKEND": "invalid"}, {"DS4_MTP": "invalid"}):
            with (
                self.subTest(setting=setting),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                self.flash_command(**setting)

    def test_explicit_missing_vision_remains_visible_in_dry_run(self):
        args, _ = self.flash_command(
            DS4_BACKEND="metal", DS4_VISION_MODEL="/missing vision/model.gguf"
        )
        self.assertEqual(args[args.index("--vision") + 1], "/missing vision/model.gguf")

    def test_explicit_missing_vision_fails_before_building(self):
        env = launcher_environment()
        env.update(
            DS4_BACKEND="metal",
            DS4_MODEL=str(Path(__file__).resolve()),
            DS4_VISION_MODEL="/missing vision/model.gguf",
        )
        result = subprocess.run(
            ["bash", str(ROOT / "start-ds4-server.sh")],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Vision model not found:", result.stderr)
        self.assertNotIn("Checking", result.stdout)


if __name__ == "__main__":
    unittest.main()
