#!/usr/bin/env python3
"""Extract real storage geometries and their native operation bindings."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    directory = ROOT / "docs/performance/native-suite-t00-20261020"
    source = json.loads((directory / "tensors.json").read_text())
    records = []
    for model in source["models"]:
        for artifact in model["files"]:
            for name, tensor in artifact["tensors"].items():
                shape = tensor["shape"]
                kind = "parameter"
                if len(shape) == 2:
                    kind = (
                        "embedding_gather"
                        if any(
                            s in name
                            for s in (
                                "embed_tokens",
                                "pos_embed",
                                "position_embedding_table",
                            )
                        )
                        else "gemm"
                    )
                elif len(shape) in (3, 4, 5) and any(
                    s in name for s in ("conv", "patch_embed.proj")
                ):
                    kind = f"conv{len(shape) - 2}d"
                records.append(
                    {
                        "model": model["repository"],
                        "artifact": artifact["path"],
                        "tensor": name,
                        "shape": shape,
                        "dtype": tensor["dtype"],
                        "operation": kind,
                    }
                )
    return {
        "version": 1,
        "source_sha256": hashlib.sha256(
            (directory / "tensors.json").read_bytes()
        ).hexdigest(),
        "tensors": records,
        "profile": {
            "storage": ["BF16", "F32"],
            "accumulation": "FP32",
            "tf32": False,
            "float16": "REJECT",
            "fast_math": False,
        },
        "bindings": {
            "gemm": "cuBLAS C API; CPU scalar reference; encode/DiT GEMM, no LLM decode GEMV substitution",
            "convolution": "cuDNN C API; deterministic FP32 accumulation; <=64 MiB admitted workspace; CPU independent direct convolution",
            "attention": "FP32 online/segmented or BF16 256-query tiled QK-softmax-PV; GQA/MQA; explicit bounds/scale/softcap/relative keys",
            "rms": "direct scale for actual Qwen/Gemma; scale-free and zero-centered tested independently; Qwen casts normalized values before scale",
            "layer_norm": "vision/audio affine or scale-free; explicit epsilon",
            "vae_norm": "FP32 L2 clamp denominator, cast, sqrt(channels) scale, gamma, optional bias; channel layout via reorder",
            "activations": "SiLU, sigmoid, GELU exact/tanh, ReLU, softplus, tanh, min/max clipping, add/multiply/scale/cast",
            "rope": "split-half Qwen/Gemma, adjacent-pair DiT; cosine/sine dtype and product rounding explicit",
            "layout": "positive-stride reorder, nearest-exact resample, indexed spatial pooling with explicit divisor/padding",
            "storage_conversion": "bounded BF16/F32/Q8_0/Q4_0 slices; other quantized candidates explicitly unsupported pending T20",
        },
        "attention_shapes": {
            "qwen_text": [32, 8, 128],
            "qwen_vision": [16, 16, 72],
            "qwen_dit": [32, 32, 128],
            "gemma_local": [4, 2, 256],
            "gemma_full": [4, 1, 512],
            "gemma_vision": [12, 12, 64],
            "gemma_audio": [8, 8, 128],
        },
        "ownership": "Mapped storage stays authoritative; groups/outputs/workspace have separate lifetimes; no global LLM kernel modified",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory(), indent=2) + "\n")
