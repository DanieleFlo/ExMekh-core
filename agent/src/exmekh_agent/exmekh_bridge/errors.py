from __future__ import annotations

from exmekh_agent.config.catalogs import message as app_text
from exmekh_agent.config.provider import DEFAULTS as PROVIDER_DEFAULTS
from exmekh_agent.utils.exceptions import (
    format_exception_chain as format_exception_chain,
)
from exmekh_agent.utils.exceptions import iter_exception_chain as iter_exception_chain


def build_execution_error_message(
    error: BaseException,
    *,
    base_url: str,
    model: str,
) -> str:
    """Build actionable UI copy for provider and application failures."""
    diagnostics = format_exception_chain(error)
    connection_error_names = PROVIDER_DEFAULTS["connection_error_names"]
    is_connection_error = any(
        type(item).__name__ in connection_error_names
        for item in iter_exception_chain(error)
    )

    guidance = (
        app_text(
            "errors_provider.execution.connection_guidance",
            base_url=base_url,
            model=model,
        )
        if is_connection_error
        else app_text("errors_provider.execution.general_guidance")
    )
    return app_text(
        "errors_provider.execution.failure", diagnostics=diagnostics, guidance=guidance
    )
