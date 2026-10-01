from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def sync_wheel() -> int:
    agent_dir = Path(__file__).resolve().parents[2]
    qs_dir = agent_dir.parent / "qs"
    wheels_dir = agent_dir / "wheels"
    uv = (
        os.getenv("UV_BIN")
        or shutil.which("uv")
        or str(Path.home() / ".local" / "bin" / "uv")
    )

    if not (qs_dir / "pyproject.toml").is_file():
        print(f"[ERROR] QS source project not found: {qs_dir}")
        return 1

    try:
        print(f"Building QS from {qs_dir}...", flush=True)
        with TemporaryDirectory(prefix="qs-wheel-") as build_dir:
            subprocess.run(
                [uv, "build", "--wheel", "--out-dir", build_dir],
                cwd=qs_dir,
                check=True,
            )
            wheels = list(Path(build_dir).glob("qs-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("QS build must produce exactly one qs wheel")

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
            # Re-resolve QS even when a rebuilt wheel keeps its version and path.
            subprocess.run(
                [uv, "lock", "--upgrade-package", "qs", "--refresh-package", "qs"],
                cwd=agent_dir,
                check=True,
            )
            subprocess.run(
                [uv, "sync", "--locked", "--reinstall-package", "qs"],
                cwd=agent_dir,
                check=True,
            )

            # Keep the wheel used by the updated project and lockfile.
            for old_wheel in wheels_dir.glob("qs-*.whl"):
                if old_wheel != target:
                    old_wheel.unlink()
        print("QS synchronized and installed successfully.")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] QS synchronization failed: {error}")
        return 1


def main() -> None:
    sys.exit(sync_wheel())


if __name__ == "__main__":
    main()
