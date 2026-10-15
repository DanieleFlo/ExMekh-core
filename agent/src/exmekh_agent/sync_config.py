"""Wheel sync configuration, available before application dependencies exist."""

from dataclasses import dataclass

from exmekh_agent.package_paths import CONFIG_DIR
from exmekh_agent.utils.resources import catalog_message


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


def app_text(key: str, **values: object) -> str:
    return catalog_message(CONFIG_DIR, key, **values)
