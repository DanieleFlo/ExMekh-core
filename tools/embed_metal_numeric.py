#!/usr/bin/env python3
"""Embed the suite's Metal source and local includes for installed libraries."""

import json
import sys
from pathlib import Path


def embed(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "metal/numeric.metal").read_text()
    includes = {
        "../ds4_numeric_metal_args.h": root / "ds4_numeric_metal_args.h",
        "numeric_ordered.metal": root / "metal/numeric_ordered.metal",
    }
    for name, path in includes.items():
        source = source.replace(f'#include "{name}"', path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "static const char numeric_metal_source[] = " + json.dumps(source) + ";\n"
    )


if __name__ == "__main__":
    embed(Path(sys.argv[1]))
