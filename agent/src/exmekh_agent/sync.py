from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from exmekh_agent.package_paths import APPLICATION_ROOT
from exmekh_agent.sync_config import WHEEL_SYNC, app_text


def _build_wheel(source: Path, output: Path, uv: str) -> Path:
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(output)], cwd=source, check=True
    )
    wheels = list(output.glob(WHEEL_SYNC.wheel_pattern))
    if len(wheels) != 1:
        raise RuntimeError(
            app_text(
                "errors_exmekh_agent.sync.exmekh_lib_build_must_produce_exactly_one_exmekh_lib_wheel"
            )
        )
    return wheels[0]


def _install_wheel(agent_dir: Path, wheel: Path, uv: str) -> None:
    subprocess.run(
        [uv, "add", "--no-sync", wheel.relative_to(agent_dir).as_posix()],
        cwd=agent_dir,
        check=True,
    )
    # A rebuilt wheel can retain its version and path; resolve and reinstall it.
    subprocess.run(
        [
            uv,
            "lock",
            "--upgrade-package",
            WHEEL_SYNC.package_name,
            "--refresh-package",
            WHEEL_SYNC.package_name,
        ],
        cwd=agent_dir,
        check=True,
    )
    subprocess.run(
        [uv, "sync", "--locked", "--reinstall-package", WHEEL_SYNC.package_name],
        cwd=agent_dir,
        check=True,
    )


def sync_wheel() -> int:
    agent_dir = APPLICATION_ROOT
    exmekh_dir = agent_dir.parent / WHEEL_SYNC.library_directory
    wheels_dir = agent_dir / WHEEL_SYNC.wheels_directory
    uv = (
        os.getenv(WHEEL_SYNC.uv_environment)
        or shutil.which(WHEEL_SYNC.uv_executable)
        or str(Path.home().joinpath(*WHEEL_SYNC.uv_home_path))
    )
    if not (exmekh_dir / WHEEL_SYNC.project_metadata).is_file():
        print(
            app_text(
                "errors_exmekh_agent.sync.source_project_missing", exmekh_dir=exmekh_dir
            )
        )
        return 1
    try:
        print(
            app_text(
                "configuration.sync.building_exmekh_lib_from_exmekh_dir",
                exmekh_dir=exmekh_dir,
            ),
            flush=True,
        )
        with TemporaryDirectory(prefix=WHEEL_SYNC.temporary_prefix) as build_dir:
            wheel = _build_wheel(exmekh_dir, Path(build_dir), uv)
            wheels_dir.mkdir(parents=True, exist_ok=True)
            target = wheels_dir / wheel.name
            shutil.copy2(wheel, target)
            print(
                app_text(
                    "configuration.sync.installing_name_in_agent_dir",
                    name=target.name,
                    agent_dir=agent_dir,
                ),
                flush=True,
            )
            _install_wheel(agent_dir, target, uv)
            for old_wheel in wheels_dir.glob(WHEEL_SYNC.wheel_pattern):
                if old_wheel != target:
                    old_wheel.unlink()
        print(app_text("configuration.sync.completed"))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(app_text("errors_exmekh_agent.sync.failed", error=error))
        return 1


def main() -> None:
    sys.exit(sync_wheel())


if __name__ == "__main__":
    main()
