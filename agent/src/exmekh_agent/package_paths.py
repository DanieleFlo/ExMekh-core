"""Package locations usable before application dependencies are installed."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
APPLICATION_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
