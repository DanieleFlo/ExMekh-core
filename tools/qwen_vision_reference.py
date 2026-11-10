"""Shared Qwen vision artifacts, metrics and GGUF-to-reference weight mapping.

Metrics and dump validation require only NumPy. Torch and GGUF are optional
until a caller requests the weight mapping for an independent reference model.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_embedding_dump(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError("truncated vision embedding header")
        count, width, height, cols = struct.unpack("<4I", header)
        if not count or not width or not height or not cols or count != height * cols:
            raise ValueError("invalid vision embedding layout")
        if path.stat().st_size != 16 + count * width * 4:
            raise ValueError("vision embedding size does not match its header")
        data = np.frombuffer(stream.read(), dtype="<f4").reshape(count, width)
    if not np.isfinite(data).all():
        raise ValueError("non-finite vision embeddings")
    return data, (height, cols)


def ds4_embeddings(binary, mmproj, image_path, out_path, min_tokens, max_tokens):
    subprocess.run(
        [
            str(binary),
            str(mmproj),
            str(image_path),
            str(out_path),
            str(min_tokens),
            str(max_tokens),
        ],
        check=True,
        timeout=120,
    )
    return read_embedding_dump(out_path)


def gguf_state(model, path):
    """Load every GGUF tensor into the HF tower, including split temporal Conv3D."""
    import torch

    from gguf import GGUFReader, dequantize

    reader = GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    used = set()

    def tensor(name):
        used.add(name)
        t = tensors[name]
        shape = tuple(reversed(t.shape.tolist()))
        return torch.from_numpy(
            dequantize(t.data, t.tensor_type).reshape(shape).copy()
        ).float()

    state = {}
    for key in model.state_dict():
        if key == "patch_embed.proj.weight":
            state[key] = torch.stack(
                [tensor("v.patch_embd.weight"), tensor("v.patch_embd.weight.1")], dim=2
            )
            continue
        if key.startswith("blocks."):
            _, layer, rest = key.split(".", 2)
            for old, new in [
                ("attn.proj.", "attn_out."),
                ("attn.qkv.", "attn_qkv."),
                ("mlp.linear_fc1.", "ffn_up."),
                ("mlp.linear_fc2.", "ffn_down."),
                ("norm1.", "ln1."),
                ("norm2.", "ln2."),
            ]:
                if rest.startswith(old):
                    name = f"v.blk.{layer}." + rest.replace(old, new, 1)
                    break
            else:
                raise ValueError(f"unmapped HF vision tensor: {key}")
        elif key.startswith("merger."):
            name = (
                key.replace("merger.linear_fc1.", "mm.0.")
                .replace("merger.linear_fc2.", "mm.2.")
                .replace("merger.norm.", "v.post_ln.")
            )
        elif key == "patch_embed.proj.bias":
            name = "v.patch_embd.bias"
        elif key == "pos_embed.weight":
            name = "v.position_embd.weight"
        else:
            raise ValueError(f"unmapped HF vision tensor: {key}")
        state[key] = tensor(name)
    if used != set(tensors):
        raise ValueError(f"unused GGUF tensors: {sorted(set(tensors) - used)}")
    return state


def compare(actual, reference, actual_grid, reference_grid, threshold):
    if not 0 < threshold <= 1:
        return {"pass": False, "error": "invalid cosine threshold"}
    if (
        actual.ndim != 2
        or reference.ndim != 2
        or not all(actual.shape)
        or actual.shape != reference.shape
        or actual_grid != reference_grid
        or len(actual_grid) != 2
        or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in actual_grid)
        or actual.shape[0] != actual_grid[0] * actual_grid[1]
    ):
        return {"pass": False, "error": "token layout mismatch"}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"pass": False, "error": "non-finite embeddings"}
    # Use float64 for the metric so numerical noise does not yield cosine > 1.
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    cos = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )
    diff = np.abs(a - b)
    return {
        "pass": bool(cos.min() >= threshold),
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "worst_token": int(cos.argmin()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }
