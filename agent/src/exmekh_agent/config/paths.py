"""Installed-package and workspace paths, resolved in one place."""

from exmekh_agent.config.storage import DEFAULTS as STORAGE_DEFAULTS
from exmekh_agent.package_paths import APPLICATION_ROOT, PACKAGE_ROOT

DEFAULT_DATA_DIR = APPLICATION_ROOT / STORAGE_DEFAULTS["data_directory"]
DEFAULT_ENV_PATH = APPLICATION_ROOT / STORAGE_DEFAULTS["env_file"]
BUNDLED_CAPABILITIES_DIR = PACKAGE_ROOT / "capabilities"
