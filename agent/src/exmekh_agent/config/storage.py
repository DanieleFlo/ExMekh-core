"""Defaults and policy for storage."""

from exmekh_agent.config.catalogs import settings

DEFAULTS = settings("storage")

TEMP_STORAGE_ALERT_THRESHOLD_BYTES = DEFAULTS["alert_threshold_bytes"]
