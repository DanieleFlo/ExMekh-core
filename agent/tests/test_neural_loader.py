from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nicegui.client import Client
from nicegui.page import page

from exmekh_agent.exmekh_bridge.providers.loader_tokens import sample_loader_tokens
from exmekh_agent.ui.shared.neural_loader import NeuralLoader


def test_loader_samples_complete_library_tokens_and_reuses_pool():
    vocabulary = ("a", "in", "out", "token", "longword", "42")
    tokenizer = SimpleNamespace(get_id_vocabulary=lambda: vocabulary)
    sample_loader_tokens.cache_clear()
    try:
        with patch(
            "exmekh_agent.exmekh_bridge.providers.loader_tokens.get_qwen_tokenizer",
            return_value=tokenizer,
        ) as get_tokenizer:
            pool = sample_loader_tokens()
            assert set(pool) == {"in", "out", "token", "42"}
            assert sample_loader_tokens() is pool
            get_tokenizer.assert_called_once()
        tokenizer.get_id_vocabulary = lambda: ()
        sample_loader_tokens.cache_clear()
        with patch(
            "exmekh_agent.exmekh_bridge.providers.loader_tokens.get_qwen_tokenizer",
            return_value=tokenizer,
        ):
            assert sample_loader_tokens() == ()
    finally:
        sample_loader_tokens.cache_clear()


@pytest.mark.parametrize("deleted", [False, True])
async def test_loader_delivers_worker_result_only_to_its_own_live_client(deleted):
    client = Client(page("/neural-loader-test"))
    try:
        with client, patch("exmekh_agent.ui.shared.neural_loader.ui.timer"):
            loader = NeuralLoader()
            loader.set_active(True)
            assert loader._props["active"] is True
            if deleted:
                loader.delete()
        with patch(
            "exmekh_agent.ui.shared.neural_loader.sample_loader_tokens",
            return_value=("in", "out"),
        ):
            await loader._load_tokens()
        assert loader._props["tokens"] == ([] if deleted else ["in", "out"])
    finally:
        client.delete()


async def test_loader_vocabulary_failure_keeps_the_indicator_usable():
    client = Client(page("/neural-loader-offline"))
    try:
        with client, patch("exmekh_agent.ui.shared.neural_loader.ui.timer"):
            loader = NeuralLoader()
        with patch(
            "exmekh_agent.ui.shared.neural_loader.sample_loader_tokens",
            side_effect=OSError("Tokenizer cache unavailable"),
        ):
            await loader._load_tokens()
        assert loader._props["tokens"] == []
        loader.set_active(True)
        assert loader._props["active"] is True
    finally:
        client.delete()
