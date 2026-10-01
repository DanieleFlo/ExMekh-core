from __future__ import annotations

from collections.abc import Iterator


def iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit or implicit causes once each."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def format_exception_chain(error: BaseException) -> str:
    """Return concise diagnostics without discarding the provider's root cause."""
    parts = [f"{type(item).__name__}: {item}" for item in iter_exception_chain(error)]
    return " → ".join(dict.fromkeys(parts))


def build_execution_error_message(
    error: BaseException,
    *,
    base_url: str,
    model: str,
) -> str:
    """Build actionable UI copy for provider and application failures."""
    diagnostics = format_exception_chain(error)
    connection_error_names = {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "ReadTimeout",
        "TimeoutError",
    }
    is_connection_error = any(
        type(item).__name__ in connection_error_names
        for item in iter_exception_chain(error)
    )

    if is_connection_error:
        guidance = (
            f"Verify that the provider responds at `{base_url}` and that the model "
            f"`{model}` is available. If the server is running, check the logs "
            "to see if the connection was closed during a long request."
        )
    else:
        guidance = (
            "The server may still be running: check the technical cause and request logs "
            "before restarting it."
        )

    return f"⚠️ **Error during model execution**:\n\n`{diagnostics}`\n\n{guidance}"
