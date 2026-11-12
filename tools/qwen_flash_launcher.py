"""Share Flash CUDA placement defaults with the shell launcher and benchmarks."""

import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASH_CUDA_STANDARD = dict(
    line.split("=", 1)
    for line in (ROOT / "config/flash-cuda-standard.env")
    .read_text(encoding="utf-8")
    .splitlines()
)


def launcher_environment(profile="standard", *, backend="auto"):
    """Use the launcher's public dry-run interface without loading a model."""
    # These inherited controls could hide launcher defaults. The benchmark's
    # hardware-independent baseline budget must not override automatic sizing.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DS4_", "CUDA_ARCH"))}
    env["DS4_FLASH_PROFILE"] = profile
    env["DS4_BACKEND"] = backend
    env["DS4_MODEL_PROFILE"] = "flash"
    output = subprocess.check_output(
        ["bash", str(ROOT / "start-ds4-server.sh"), "--manual", "--print-command"],
        cwd=ROOT, env=env, text=True)
    command = shlex.split(output)
    if not command or command[0] != "env" or "./ds4-server" not in command:
        raise ValueError("unexpected Flash launcher dry-run command")
    return dict(item.split("=", 1) for item in command[1:command.index("./ds4-server")])
