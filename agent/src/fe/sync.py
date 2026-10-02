from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def sync_wheel() -> int:
    agent_dir = Path(__file__).resolve().parents[2]
    exmekh_dir = agent_dir.parent / "exmekh_lib"
    wheels_dir = agent_dir / "wheels"
    uv = (
        os.getenv("UV_BIN")
        or shutil.which("uv")
        or str(Path.home() / ".local" / "bin" / "uv")
    )

    if not (exmekh_dir / "pyproject.toml").is_file():
        print(f"[ERROR] ExMekh-Lib source project not found: {exmekh_dir}")
        return 1

    try:
        print(f"Building ExMekh-Lib from {exmekh_dir}...", flush=True)
        with TemporaryDirectory(prefix="exmekh-wheel-") as build_dir:
            subprocess.run(
                [uv, "build", "--wheel", "--out-dir", build_dir],
                cwd=exmekh_dir,
                check=True,
            )
            wheels = list(Path(build_dir).glob("exmekh_lib-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(
                    "ExMekh-Lib build must produce exactly one exmekh_lib wheel"
                )

            wheels_dir.mkdir(parents=True, exist_ok=True)
            target = wheels_dir / wheels[0].name
            shutil.copy2(wheels[0], target)
            print(f"Installing {target.name} in {agent_dir}...", flush=True)
            subprocess.run(
                [
                    uv,
                    "add",
                    "--no-sync",
                    target.relative_to(agent_dir).as_posix(),
                ],
                cwd=agent_dir,
                check=True,
            )
            # Re-resolve ExMekh-Lib even when a rebuilt wheel keeps its version and path.
            subprocess.run(
                [
                    uv,
                    "lock",
                    "--upgrade-package",
                    "exmekh-lib",
                    "--refresh-package",
                    "exmekh-lib",
                ],
                cwd=agent_dir,
                check=True,
            )
            subprocess.run(
                [uv, "sync", "--locked", "--reinstall-package", "exmekh-lib"],
                cwd=agent_dir,
                check=True,
            )

            # Keep the wheel used by the updated project and lockfile.
            for pattern in ("exmekh_lib-*.whl", "qs-*.whl"):
                for old_wheel in wheels_dir.glob(pattern):
                    if old_wheel != target:
                        old_wheel.unlink()
        print("ExMekh-Lib synchronized and installed successfully.")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] ExMekh-Lib synchronization failed: {error}")
        return 1


def main() -> None:
    sys.exit(sync_wheel())


if __name__ == "__main__":
    main()
