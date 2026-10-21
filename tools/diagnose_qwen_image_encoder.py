#!/usr/bin/env python3
"""Isolate native primitive error on real Qwen encoder layer operands."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ctypes as C

from tests.test_suite_numeric import Attention, Numeric, View


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--native", type=Path)
    args = p.parse_args()
    from safetensors import safe_open

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    n = Numeric(True)

    def load(name):
        return (
            torch.from_numpy(np.fromfile(args.oracle / f"{name}.bf16", dtype=np.uint16))
            .view(torch.bfloat16)
            .reshape(1, -1, 4096)
            .cuda()
        )

    x = load("embeddings-0" if args.layer == 0 else f"text-{args.layer - 1}")[0]
    prefix = f"model.language_model.layers.{args.layer}."
    with safe_open(args.model, framework="pt", device="cpu") as f:
        weights = {
            key[len(prefix) :]: f.get_tensor(key).cuda()
            for key in f.keys()  # noqa: SIM118 -- safe_open is not iterable.
            if key.startswith(prefix)
        }

    def compare(name, actual, expected):
        actual = actual.cuda().reshape_as(expected)
        d = actual.float() - expected.float()
        print(
            json.dumps(
                {
                    "name": name,
                    "max_abs": d.abs().max().item(),
                    "rmse": d.square().mean().sqrt().item(),
                    "different": int((d != 0).sum()),
                    "elements": d.numel(),
                }
            ),
            flush=True,
        )
        n.clear()

    def native(name, expected):
        if args.native and args.layer == 0:
            data = torch.from_numpy(
                np.fromfile(args.native / f"{name}-0.bf16", dtype=np.uint16)
            ).view(torch.bfloat16)
            compare("native-" + name, data, expected)

    def norm(x, name):
        return weights[name] * (
            (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            ).to(torch.bfloat16)
        )

    def check_norm(x, name):
        w = weights[name]
        y = n.tensor(torch.zeros_like(x))
        n.call(
            "norm", y, n.tensor(x), n.tensor(w), View(), x.shape[-1], 1e-6, 0, 1, True
        )
        z = norm(x, name)
        compare(name, n.read(y, z.shape), z)
        return z

    def linear(x, name):
        w = weights[name]
        y = n.tensor(torch.zeros((*x.shape[:-1], w.shape[0]), dtype=x.dtype))
        n.call(
            "linear",
            y,
            n.tensor(x),
            n.tensor(w),
            View(),
            x.numel() // x.shape[-1],
            x.shape[-1],
            w.shape[0],
        )
        z = F.linear(x, w)
        compare(name, n.read(y, z.shape), z)
        return z

    z = check_norm(x, "input_layernorm.weight")
    native("input_norm", z)
    q = linear(z, "self_attn.q_proj.weight").reshape(-1, 32, 128)
    k = linear(z, "self_attn.k_proj.weight").reshape(-1, 8, 128)
    v = linear(z, "self_attn.v_proj.weight").reshape(-1, 8, 128)
    q = check_norm(q, "self_attn.q_norm.weight")
    k = check_norm(k, "self_attn.k_norm.weight")
    native("q_norm", q)
    native("k_norm", k)
    native("v_proj", v)
    positions = (
        torch.from_numpy(
            np.fromfile(args.oracle / "positions.u32", dtype=np.uint32)
            .reshape(-1, 3)
            .copy()
        )
        .cuda()
        .float()
    )
    inv = (1 / (5000000 ** (torch.arange(0, 128, 2).float() / 128))).cuda()
    freq = positions[:, 0, None] * inv[None, :]
    freq[:, 1:60:3] = (positions[:, 1, None] * inv[None, :])[:, 1:60:3]
    freq[:, 2:60:3] = (positions[:, 2, None] * inv[None, :])[:, 2:60:3]
    co, si = (
        freq.cos().to(torch.bfloat16)[:, None],
        freq.sin().to(torch.bfloat16)[:, None],
    )
    native("rope_cos", co)
    native("rope_sin", si)

    def rope(x):
        return x * torch.cat((co, co), -1) + torch.cat(
            (-x[..., 64:], x[..., :64]), -1
        ) * torch.cat((si, si), -1)

    for name, z in (("q", q), ("k", k)):
        y = n.tensor(torch.zeros_like(z))
        n.call(
            "rope",
            y,
            n.tensor(z),
            n.tensor(co),
            n.tensor(si),
            z.shape[1],
            128,
            False,
            True,
        )
        expected = rope(z)
        compare("rope-" + name, n.read(y, z.shape), expected)
    q, k = rope(q), rope(k)
    seq = x.shape[0]
    native("q_rope", q)
    native("k_rope", k)
    raw = q.transpose(0, 1) @ k.repeat_interleave(4, dim=1).transpose(0, 1).transpose(
        -1, -2
    )
    for h in range(32):
        y = n.tensor(torch.zeros(seq, seq, dtype=torch.bfloat16))
        n.call(
            "linear",
            y,
            n.tensor(q[:, h]),
            n.tensor(k[:, h // 4]),
            View(),
            seq,
            128,
            seq,
        )
        actual = n.read(y, (seq, seq)).cuda()
        d = actual - raw[h]
        if (d != 0).any():
            compare(f"qk-{h}", actual, raw[h])
        n.clear()
    scores = raw * (128**-0.5)
    scores.masked_fill_(
        torch.ones(seq, seq, device="cuda", dtype=torch.bool).triu(1), -float("inf")
    )
    probs = scores.float().softmax(-1).to(torch.bfloat16)
    for h in range(32):
        bounds = np.array([[0, i + 1] for i in range(seq)], dtype=np.uint32)
        y = n.tensor(torch.zeros(seq, seq, dtype=torch.bfloat16))
        n.call(
            "softmax",
            y,
            n.tensor(raw[h]),
            bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
            seq,
            seq,
            128**-0.5,
        )
        actual = n.read(y, (seq, seq)).cuda()
        d = actual - probs[h]
        if (d != 0).any():
            compare(f"softmax-{h}", actual, probs[h])
        n.clear()
    expected = (
        scores.float().softmax(-1).to(torch.bfloat16)
        @ v.repeat_interleave(4, dim=1).transpose(0, 1)
    ).transpose(0, 1)
    bounds = np.array([[0, i + 1] for i in range(seq)], dtype=np.uint32)
    a = Attention(1, seq, seq, 32, 8, 128, 128**-0.5, 0, True, True)
    y = n.tensor(torch.zeros_like(q))
    n.call(
        "attend",
        y,
        n.tensor(q),
        n.tensor(k),
        n.tensor(v),
        bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
        C.byref(a),
    )
    compare("attention", n.read(y, q.shape), expected)
    native("attention", expected)
    residual = x + linear(expected.reshape(seq, 4096), "self_attn.o_proj.weight")
    native("attention_residual", residual)
    z = check_norm(residual, "post_attention_layernorm.weight")
    g = linear(z, "mlp.gate_proj.weight")
    u = linear(z, "mlp.up_proj.weight")
    native("mlp_mid", F.silu(g) * u)
    y = n.tensor(torch.zeros_like(g))
    n.call("point", y, n.tensor(g), View(), 2, 0)
    compare("silu", n.read(y, g.shape), F.silu(g))
    final = residual + linear(F.silu(g) * u, "mlp.down_proj.weight")
    if args.layer < 3 and (args.oracle / f"deepstack-{args.layer}.bf16").exists():
        mask = np.load(args.oracle / "inputs.npz")["input_ids"][0] == 151655
        deep = (
            torch.from_numpy(
                np.fromfile(
                    args.oracle / f"deepstack-{args.layer}.bf16", dtype=np.uint16
                )
            )
            .view(torch.bfloat16)
            .reshape(-1, 4096)
            .cuda()
        )
        final[torch.from_numpy(mask).cuda()] += deep
    compare("manual-vs-oracle", final, load(f"text-{args.layer}")[0])
    n.close()


if __name__ == "__main__":
    main()
