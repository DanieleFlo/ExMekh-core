#!/usr/bin/env python3
"""Materialize held-out Turbo initial latents before kernel comparisons."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(args.corpus.read_text())
    records = []
    seen = set()
    for recipe in corpus["qwen_recipes"]:
        for seed in recipe["seeds"]:
            key = (recipe["height"], recipe["width"], seed)
            if key in seen:
                continue
            seen.add(key)
            height, width = recipe["height"] // 16, recipe["width"] // 16
            generator = torch.Generator(device="cpu").manual_seed(seed)
            # Exact prepare_latents layout at the pinned Diffusers revision:
            # [B,1,C,H/16,W/16] -> [B,H/16*W/16,C], unpatched latent tokens.
            value = torch.randn(
                (1, 1, 64, height, width), dtype=torch.bfloat16, generator=generator
            )
            packed = value.view(1, 64, height * width).transpose(1, 2).contiguous()
            path = args.output / f"{recipe['width']}x{recipe['height']}-seed-{seed}.npz"
            np.savez(path, latents=packed.float().numpy())
            records.append(
                {
                    "width": recipe["width"],
                    "height": recipe["height"],
                    "seed": seed,
                    "path": path.name,
                    "shape": list(packed.shape),
                    "stored_dtype": "float32-lossless-bf16",
                    "bf16_sha256": hashlib.sha256(
                        packed.view(torch.uint16).numpy().tobytes()
                    ).hexdigest(),
                    "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    report = {
        "torch": torch.__version__,
        "generator": "torch.Generator cpu, randn bf16",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "latents": records,
    }
    (args.output / "noise.json").write_text(json.dumps(report, indent=2) + "\n")
    print("frozen latent arrays", len(records))


if __name__ == "__main__":
    main()
