"""Workspace wheel build and installation policy."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WheelSyncSettings:
    library_directory: str = "exmekh_lib"
    wheels_directory: str = "wheels"
    package_name: str = "exmekh-lib"
    wheel_pattern: str = "exmekh_lib-*.whl"
    temporary_prefix: str = "exmekh-wheel-"
    project_metadata: str = "pyproject.toml"
    uv_environment: str = "UV_BIN"
    uv_executable: str = "uv"
    uv_home_path: tuple[str, ...] = (".local", "bin", "uv")


WHEEL_SYNC = WheelSyncSettings()
