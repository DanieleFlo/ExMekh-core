"""Shared classification for provider opt-in and exclusive scheduling."""

import pytest


def is_live_provider_test(item: pytest.Item) -> bool:
    return (
        item.get_closest_marker("live") is not None
        and item.get_closest_marker("mock") is None
    )
