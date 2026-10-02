"""Isolate persistence before pytest imports the application's session manager."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from exmekh_agent.config import config

# Collection imports the global session manager before any fixtures can run.
# Browser server entry points also import this package, but use their own storage.
if "pytest" in sys.modules:
    _test_root = Path(__file__).resolve().parent.parent / ".test-tmp"
    _test_root.mkdir(exist_ok=True)
    _collection_storage = TemporaryDirectory(
        prefix="pytest-state-", dir=_test_root, ignore_cleanup_errors=True
    )
    config.data_dir = Path(_collection_storage.name)
