"""Normalize ordered filesystem references without sharing mutable lists."""

from collections.abc import Iterable
from pathlib import Path


def normalize_paths(paths: Iterable[str | Path]) -> list[str]:
    return list(
        dict.fromkeys(
            str(Path(path).expanduser().resolve())
            for path in paths
            if str(path).strip()
        )
    )
