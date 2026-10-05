"""Read provider model metadata without running inference or saving preferences."""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx

from exmekh_agent.config import config
from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS

logger = logging.getLogger(__name__)


def extract_model_context_length(model_data: dict[str, Any]) -> int | None:
    """Extract the maximum context length reported by the inference engine for a model."""
    candidates: list[Any] = [
        model_data.get("context_length"),
        model_data.get("top_provider", {}).get("context_length")
        if isinstance(model_data.get("top_provider"), dict)
        else None,
        model_data.get("max_model_len"),
        model_data.get("n_ctx"),
        model_data.get("context_window"),
    ]
    for val in candidates:
        if (
            isinstance(val, (int, float))
            and not isinstance(val, bool)
            and (not isinstance(val, float) or math.isfinite(val))
            and val >= 1
        ):
            return int(val)
    return None


async def check_provider_health(
    base_url: str | None = None, api_key: str | None = None
) -> tuple[bool, list[str]]:
    """Check the configured provider endpoint and list its models."""
    configured_endpoint = (config.base_url, config.api_key)
    url = (config.base_url if base_url is None else base_url).rstrip("/")
    key = config.api_key if api_key is None else api_key

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    models_url = f"{url}/models"

    try:
        async with httpx.AsyncClient(
            timeout=PROVIDER_DEFAULTS["health_timeout_seconds"]
        ) as client:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("data") if isinstance(data, dict) else None
                if not isinstance(entries, list):
                    return False, []
                models: list[str] = []
                for m in entries:
                    if not isinstance(m, dict):
                        continue
                    m_id = m.get("id")
                    if isinstance(m_id, str) and m_id:
                        models.append(m_id)
                        config.set_model_owner(
                            m_id, str(m.get("owned_by") or ""), base_url=url
                        )
                        ctx_len = extract_model_context_length(m)
                        if ctx_len is not None and configured_endpoint == (
                            config.base_url,
                            config.api_key,
                        ):
                            config.set_model_context_length(m_id, ctx_len)
                return True, sorted(set(models))
            return False, []
    except Exception as e:  # noqa: BLE001 - health probe converts failures to status
        logger.debug(
            app_text("errors_providers.provider_health.provider_health_check_failed_s"),
            e,
        )
        return False, []
