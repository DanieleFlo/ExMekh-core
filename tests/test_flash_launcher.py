"""Inspect the real shell launchers without loading a model or allocating VRAM."""
import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlashLauncherTests(unittest.TestCase):
    def command(self, profile="standard", **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4_", "CUDA_ARCH"))}
        env.update(overrides)
        if profile is not None:
            env["DS4_FLASH_PROFILE"] = profile
        result = subprocess.run(["bash", str(ROOT / "agent/start-ds4-server-flash.sh"), "--print-command"],
                                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        args = shlex.split(result.stdout)
        return args, dict(item.split("=",1) for item in args[1:args.index("./ds4-server")])

    def test_standard_complete_gpu_profile(self):
        args, env = self.command()
        self.assertEqual(args[args.index("--prefill-chunk")+1], "2048")
        self.assertEqual(args[args.index("--ctx")+1], "65536")
        self.assertIn("--mtp", args)
        self.assertEqual(env["DS4_CUDA_QWEN4_PCIE_FRAC"], "1")
        self.assertEqual(env["DS4_CUDA_QWEN4_HYBRID_PREFILL"], "0")
        self.assertEqual(env["DS4_CUDA_QWEN4_KV_RESIDENT"], "32768")
        for key in ("KV_STREAMING", "PREFILL_RING", "DECODE_GRAPHS", "DECODE_RANKS", "KV_DECODE_BATCH", "QSA_REGISTER_KEYS"):
            self.assertEqual(env["DS4_CUDA_QWEN4_"+key], "1")
        self.assertNotIn("DS4_CUDA_QWEN4_MTP_RESIDENT", env)
        self.assertNotIn("DS4_CUDA_QWEN4_PREFILL_AHEAD", env)

    def test_default_is_the_qualified_profile(self):
        self.assertEqual(self.command(None), self.command("standard"))

    def test_explicit_overrides(self):
        args, env = self.command(DS4_PREFILL_CHUNK="4096", DS4_CTX="65536",
                                 DS4_CUDA_QWEN4_KV_STREAMING="0", DS4_CUDA_QWEN4_PCIE_FRAC="0.75")
        self.assertEqual(args[args.index("--prefill-chunk")+1], "4096")
        self.assertEqual(args[args.index("--ctx")+1], "65536")
        self.assertEqual(env["DS4_CUDA_QWEN4_KV_STREAMING"], "0")
        self.assertEqual(env["DS4_CUDA_QWEN4_PCIE_FRAC"], "0.75")

    def test_legacy_reproduces_original(self):
        args, env = self.command("legacy")
        self.assertEqual(args[args.index("--prefill-chunk")+1], "256")
        self.assertEqual(set(env), {"DS4_CUDA_QWEN4_TIERED_CACHE", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "DS4_CUDA_QWEN4_RAM_CACHE_MB"})

    def test_27b_launcher_ignores_flash_profile(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
        env["DS4_FLASH_PROFILE"] = "standard"
        result = subprocess.run(["bash", str(ROOT / "agent/start-ds4-server.sh"), "--print-command"],
                                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        args = shlex.split(result.stdout)
        self.assertEqual(args[args.index("--ctx")+1], "26000")
        self.assertNotIn("--prefill-chunk", args)
        self.assertFalse(any("DS4_CUDA_QWEN4" in value for value in args))

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.command("invalid")


if __name__ == "__main__":
    unittest.main()
