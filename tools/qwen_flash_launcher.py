"""Read Flash placement defaults from the production shell launcher."""

import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launcher_environment(profile="standard"):
    """Use the launcher's public dry-run interface without loading a model."""
    # These inherited controls could hide launcher defaults. The benchmark's
    # hardware-independent baseline budget must not override automatic sizing.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DS4_", "CUDA_ARCH"))}
    env["DS4_FLASH_PROFILE"] = profile
    output = subprocess.check_output(
        ["bash", str(ROOT / "agent/start-ds4-server-flash.sh"), "--print-command"],
        cwd=ROOT, env=env, text=True)
    command = shlex.split(output)
    if not command or command[0] != "env" or "./ds4-server" not in command:
        raise ValueError("unexpected Flash launcher dry-run command")
    return dict(item.split("=", 1) for item in command[1:command.index("./ds4-server")])
