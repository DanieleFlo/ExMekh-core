#!/usr/bin/env python3
"""Compile the pinned image processor's byte BPE into a checked native asset.

Offline conversion only. The runtime checks source SHA against its open bundle.
No Python/tokenizers dependency is required for inference.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def compile_tokenizer(source, output):
    raw = source.read_bytes()
    data = json.loads(raw)
    model = data["model"]
    if (
        data["normalizer"] != {"type": "NFC"}
        or data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] != PATTERN
        or model["type"] != "BPE"
        or model["dropout"] is not None
        or model["ignore_merges"]
    ):
        raise ValueError("unsupported tokenizer semantics")
    # GPT-2's reversible byte alphabet; token strings stay in the offline tool.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = dict(zip(bs, map(chr, bs)))
    for b in range(256):
        if b not in alphabet:
            alphabet[b] = chr(256 + sum(i not in bs for i in range(b)))
    vocab = model["vocab"]
    merges = [
        pair.split(" ") if isinstance(pair, str) else pair for pair in model["merges"]
    ]
    specials = data["added_tokens"]
    if any(
        t["single_word"] or t["lstrip"] or t["rstrip"] or t["normalized"]
        for t in specials
    ):
        raise ValueError("unsupported added-token normalization")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(raw.decode())
    recipe = json.loads((ROOT / "config/qwen-image-encoder.json").read_text())
    prefix = tokenizer.encode(recipe["prefix"].format(**recipe)).ids
    suffix = tokenizer.encode(recipe["suffix"]).ids
    drop = len(
        tokenizer.encode(
            f"<|im_start|>system\n{recipe['system_prompt']}<|im_end|>\n"
        ).ids
    )
    payload = struct.pack(
        "<5I", len(merges), len(specials), len(prefix), len(suffix), drop
    )
    payload += struct.pack("<256I", *(vocab[alphabet[b]] for b in range(256)))
    payload += struct.pack(f"<{len(prefix) + len(suffix)}I", *(prefix + suffix))
    for token in specials:
        value = token["content"].encode()
        payload += struct.pack("<II", token["id"], len(value)) + value
    for left, right in merges:
        payload += struct.pack("<III", vocab[left], vocab[right], vocab[left + right])
    content = (
        b"DS4QTK1\0"
        + hashlib.sha256(raw).digest()
        + hashlib.sha256(payload).digest()
        + payload
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "drop": drop,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(compile_tokenizer(a.source, a.output), indent=2))
