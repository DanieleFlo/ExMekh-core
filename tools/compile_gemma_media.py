#!/usr/bin/env python3
"""Compile the frozen Gemma image/video/audio processor data, offline only."""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def compile_processor(prepared: Path, output: Path):
    lock = json.loads((prepared / "prepared.json").read_text())
    source = next(
        s for s in lock["sources"] if s["repository"] == "huggingface/transformers"
    )
    sys.path.insert(
        0,
        str(
            (
                prepared / "sources" / f"transformers-{source['revision']}" / "src"
            ).resolve()
        ),
    )
    sys.path.insert(0, str((prepared / "deps").resolve()))
    import numpy as np
    import torch
    from transformers import AutoProcessor

    model = prepared / "models/google/embeddinggemma-2"
    names = ("config.json", "processor_config.json", "preprocessor_config.json")
    raw = [(model / n).read_bytes() for n in names]
    config = json.loads(raw[0])
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    image, video, audio = (
        processor.image_processor,
        processor.video_processor,
        processor.feature_extractor,
    )
    if (
        image.patch_size,
        image.pooling_kernel_size,
        image.max_soft_tokens,
        video.patch_size,
        video.pooling_kernel_size,
        video.max_soft_tokens,
        video.fps,
        video.max_frames,
        video.overflow_strategy,
        video.add_timestamps,
    ) != (16, 3, 280, 16, 3, 140, 1, 32, "uniform", False):
        raise ValueError("unsupported visual recipe")
    if (
        audio.sampling_rate,
        audio.frame_length,
        audio.hop_length,
        audio.fft_length,
        audio.feature_size,
        audio.dither,
        audio.preemphasis,
        audio.input_scale_factor,
    ) != (16000, 320, 160, 512, 128, 0, 0, 1):
        raise ValueError("unsupported audio recipe")
    vision = config["vision_config"]
    if (
        vision["hidden_size"],
        vision["num_hidden_layers"],
        vision["num_attention_heads"],
        vision["head_dim"],
        vision["pooling_kernel_size"],
        vision["standardize"],
        vision["use_clipped_linears"],
        vision["rope_parameters"],
    ) != (
        768,
        16,
        12,
        64,
        3,
        False,
        False,
        {"rope_theta": 100.0, "rope_type": "axial"},
    ):
        raise ValueError("unsupported vision graph")
    payload = struct.pack(
        "<13I4xd",
        16,
        3,
        280,
        140,
        32,
        16000,
        8192,
        320,
        160,
        512,
        128,
        128,
        480000,
        0.001,
    )
    payload += np.asarray(audio.window, dtype="<f4").tobytes()
    payload += np.asarray(audio.mel_filters, dtype="<f8").tobytes()
    frequencies = 1.0 / (100.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    payload += np.asarray(frequencies.tolist(), dtype="<f4").tobytes()
    twiddles = [
        v
        for i in range(256)
        for v in (math.cos(2 * math.pi * i / 512), -math.sin(2 * math.pi * i / 512))
    ]
    payload += struct.pack("<512d", *twiddles)
    result = b"DS4GMP1\0" + b"".join(hashlib.sha256(r).digest() for r in raw)
    result += hashlib.sha256(payload).digest() + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)
    return {
        "bytes": len(result),
        "sha256": hashlib.sha256(result).hexdigest(),
        "sources": {n: hashlib.sha256(r).hexdigest() for n, r in zip(names, raw)},
        "transformers": source["revision"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_processor(args.prepared, args.output), indent=2))
