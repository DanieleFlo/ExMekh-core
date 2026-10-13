"""Sample the library's ID vocabulary for the decorative loading indicator."""

from __future__ import annotations

from functools import lru_cache
from random import sample

from exmekh_agent.exmekh_bridge.providers.context_usage import get_qwen_tokenizer


@lru_cache(maxsize=1)
def sample_loader_tokens() -> tuple[str, ...]:
    """Share a small pool of complete, readable tokens across page instances."""
    vocabulary = get_qwen_tokenizer().get_id_vocabulary()
    tokens = tuple(token for token in vocabulary if 2 <= len(token) <= 6)
    return tuple(sample(tokens, min(len(tokens), 256)))
