from __future__ import annotations

from fe.exmekh_bridge.errors import (
    build_execution_error_message,
    format_exception_chain,
)


def test_format_exception_chain_preserves_root_cause() -> None:
    root = ConnectionError("socket closed")
    wrapped = RuntimeError("Local provider request failed")
    wrapped.__cause__ = root

    assert format_exception_chain(wrapped) == (
        "RuntimeError: Local provider request failed → ConnectionError: socket closed"
    )


def test_execution_error_does_not_claim_the_server_is_offline() -> None:
    message = build_execution_error_message(
        ValueError("invalid agent context"),
        base_url="http://localhost:8080/v1",
        model="model",
    )

    assert "ValueError: invalid agent context" in message
    assert "The server may still be running" in message
