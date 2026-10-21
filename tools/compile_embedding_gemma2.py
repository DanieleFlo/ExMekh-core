#!/usr/bin/env python3
"""Compile pinned Gemma configuration and character BPE for native C inference.

Offline conversion only. The native reader checks all three source hashes
against the open bundle; Python is never used by the embedding adapter.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def compile_asset(source, output):
    # Offline only: preserve the reference's FP32 pow/divide boundary. Python
    # double pow rounded once is a different table at 42/128 local frequencies.
    import torch

    names = ("tokenizer.json", "config.json", "config_sentence_transformers.json")
    raw = [(source / name).read_bytes() for name in names]
    tok, config, tasks = map(json.loads, raw)
    m = tok["model"]
    if (
        m["type"] != "BPE"
        or m["dropout"] is not None
        or not m["byte_fallback"]
        or m["ignore_merges"]
        or m["continuing_subword_prefix"]
        or m["end_of_word_suffix"]
        or not m["fuse_unk"]
        or tok["normalizer"]
        != {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        or tok["pre_tokenizer"]
        != {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": False,
        }
    ):
        raise ValueError("unsupported tokenizer")
    if tok["post_processor"]["single"] != [
        {"SpecialToken": {"id": "<bos>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<eos>", "type_id": 0}},
    ]:
        raise ValueError("unsupported token template")
    c = config["text_config"]
    if (
        c["hidden_size"],
        c["num_hidden_layers"],
        c["vocab_size"],
        c["embedding_dim"],
        c["hidden_size_per_layer_input"],
        c["intermediate_size"],
    ) != (512, 24, 262144, 768, 512, 2048):
        raise ValueError("unsupported backbone")
    if c["hidden_activation"] != "gelu_pytorch_tanh" or c["attention_bias"]:
        raise ValueError("unsupported graph")
    vocab = m["vocab"]
    chars = sorted((ord(s), i) for s, i in vocab.items() if len(s) == 1)
    specials = tok["added_tokens"]
    if any(
        t["single_word"] or t["lstrip"] or t["rstrip"] or t["normalized"]
        for t in specials
    ):
        raise ValueError("unsupported special tokens")
    payload = bytearray(
        struct.pack(
            "<4I", len(chars), len(m["merges"]), len(specials), c["sliding_window"]
        )
    )
    payload += struct.pack("<f", c["rms_norm_eps"])
    for i, kind in enumerate(c["layer_types"]):
        pc = c["per_layer_config"].get(f"{i:02}", {})
        dim, kv = (
            pc.get("head_dim", c["head_dim"]),
            pc.get("num_key_value_heads", c["num_key_value_heads"]),
        )
        theta = c["rope_parameters"][kind]["rope_theta"]
        payload += struct.pack(
            "<4I", dim, kv, kind == "sliding_attention", c["num_attention_heads"]
        )
        # Round inverse frequencies once to FP32, as in the pinned reference.
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        payload += struct.pack(f"<{dim // 2}f", *frequencies.tolist())
    prefixes = [
        tasks["prompts"][k]
        for k in ("query", "document", "CodeRetrieval", "SentenceSimilarity")
    ]
    for value in prefixes:
        b = value.encode()
        payload += struct.pack("<I", len(b)) + b
    payload += struct.pack("<256I", *(vocab[f"<0x{b:02X}>"] for b in range(256)))
    for cp, i in chars:
        payload += struct.pack("<2I", cp, i)
    for t in specials:
        b = t["content"].encode()
        payload += struct.pack("<2I", t["id"], len(b)) + b
    for a, b in m["merges"]:
        payload += struct.pack("<3I", vocab[a], vocab[b], vocab[a + b])
    result = b"DS4EGM1\0" + b"".join(hashlib.sha256(x).digest() for x in raw)
    result += hashlib.sha256(payload).digest() + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)
    return {
        "bytes": len(result),
        "sha256": hashlib.sha256(result).hexdigest(),
        "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in zip(names, raw)},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(compile_asset(a.source, a.output), indent=2))
