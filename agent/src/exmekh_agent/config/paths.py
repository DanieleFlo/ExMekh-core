"""Installed-package and workspace paths, resolved in one place."""

from pathlib import Path

from exmekh_agent.config.storage import DEFAULTS as STORAGE_DEFAULTS

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_DATA_DIR = APPLICATION_ROOT / STORAGE_DEFAULTS["data_directory"]
DEFAULT_ENV_PATH = APPLICATION_ROOT / STORAGE_DEFAULTS["env_file"]
BUNDLED_CAPABILITIES_DIR = PACKAGE_ROOT / "capabilities"
