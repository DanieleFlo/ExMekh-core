"""Explicit, temporary timeout scaling for local Metal qualification."""

import argparse
import os


def add_timeout_multiplier(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout-multiplier",
        type=parse_timeout_multiplier,
        choices=(1, 2, 4),
        default=os.getenv("DS4_METAL_TIMEOUT_MULTIPLIER", "1"),
        help="Temporary local factor; defaults to DS4_METAL_TIMEOUT_MULTIPLIER or 1",
    )


def parse_timeout_multiplier(value: str) -> int:
    if value not in ("1", "2", "4"):
        raise ValueError("DS4_METAL_TIMEOUT_MULTIPLIER must be 1, 2 or 4")
    return int(value)


def local_timeout_multiplier() -> int:
    return parse_timeout_multiplier(os.getenv("DS4_METAL_TIMEOUT_MULTIPLIER", "1"))
