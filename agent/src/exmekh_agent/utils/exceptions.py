"""Diagnostics for nested exceptions, preserving causes and detecting cycles."""

from __future__ import annotations

from collections.abc import Iterator

from exmekh_agent.config.formatting import DEFAULTS as FORMAT_DEFAULTS


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
    return FORMAT_DEFAULTS["exception_separator"].join(dict.fromkeys(parts))
