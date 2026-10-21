#!/usr/bin/env python3
"""Independent real-patch comparison of Conv3d/GEMM casting boundaries."""

import argparse
import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors import safe_open

from tests.test_suite_numeric import Attention, Conv, Numeric, View


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument(
        "--native", type=Path, help="Native trace directory for frequency controls"
    )
    args = p.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def compare(name, a, b):
        d = a.float() - b.float()
        print(
            json.dumps(
                {
                    "name": name,
                    "max_abs": d.abs().max().item(),
                    "rmse": d.square().mean().sqrt().item(),
                    "different": int((d != 0).sum()),
                }
            ),
            flush=True,
        )

    inputs = np.load(args.oracle / "inputs.npz")
    x = torch.from_numpy(inputs["pixel_values"]).cuda().bfloat16()
    with safe_open(args.model, framework="pt", device="cpu") as f:
        w = f.get_tensor("model.visual.patch_embed.proj.weight").cuda()
        b = f.get_tensor("model.visual.patch_embed.proj.bias").cuda()
        pos_table = f.get_tensor("model.visual.pos_embed.weight").cuda()
        prefix = f"model.visual.blocks.{args.layer}."
        weights = {
            key[len(prefix) :]: f.get_tensor(key).cuda()
            for key in f.keys()  # noqa: SIM118 -- safe_open is not iterable.
            if key.startswith(prefix)
        }
    ref = F.conv3d(x.reshape(-1, 3, 2, 16, 16), w, b, stride=(2, 16, 16)).flatten(1)
    compare("linear-fused-bias", F.linear(x, w.flatten(1), b), ref)
    compare("linear-separate-bias", F.linear(x, w.flatten(1)) + b, ref)
    compare(
        "conv-separate-bias",
        F.conv3d(x.reshape(-1, 3, 2, 16, 16), w, None, stride=(2, 16, 16)).flatten(1)
        + b,
        ref,
    )
    n = Numeric(True)
    descriptor = Conv(
        x.shape[0],
        3,
        1152,
        1,
        (2, 16, 16),
        (2, 16, 16),
        (2, 16, 16),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
    )
    output = n.tensor(torch.zeros_like(ref))
    n.call("conv", output, n.tensor(x), n.tensor(w), n.tensor(b), C.byref(descriptor))
    compare("native-cudnn", n.read(output, ref.shape).cuda(), ref)
    n.clear()
    prepared = args.model.parents[4]
    sys.path.insert(0, str(prepared / "deps"))
    sys.path.insert(0, str(next((prepared / "sources").glob("transformers-*")) / "src"))
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    grid = torch.from_numpy(inputs["image_grid_thw"]).cuda()
    indices, iw = get_vision_interpolation_indices_and_weights(
        grid, 48, mode="bilinear", align_corners=True, spatial_merge_size=2
    )
    pos = (F.embedding(indices, pos_table) * iw[:, :, None]).sum(1).bfloat16()
    vision_x = ref + pos
    if args.layer:
        vision_x = (
            torch.from_numpy(
                np.fromfile(
                    args.oracle / f"vision-{args.layer - 1}.bf16", dtype=np.uint16
                )
            )
            .view(torch.bfloat16)
            .reshape(-1, 1152)
            .cuda()
        )

    def norm(z, name):
        expected = F.layer_norm(
            z, (1152,), weights[name + ".weight"], weights[name + ".bias"], 1e-6
        )
        y = n.tensor(torch.zeros_like(z))
        n.call(
            "norm",
            y,
            n.tensor(z),
            n.tensor(weights[name + ".weight"]),
            n.tensor(weights[name + ".bias"]),
            1152,
            1e-6,
            1,
            1,
            False,
        )
        compare("norm-" + name, n.read(y, expected.shape).cuda(), expected)
        n.clear()
        return expected

    def linear(z, name):
        w, b = weights[name + ".weight"], weights[name + ".bias"]
        expected = F.linear(z, w, b)
        compare("separate-bias-" + name, F.linear(z, w) + b, expected)
        compare(
            "f32-fused-bias-" + name,
            F.linear(z.float(), w.float(), b.float()).bfloat16(),
            expected,
        )
        y = n.tensor(torch.zeros_like(expected))
        n.call(
            "linear",
            y,
            n.tensor(z),
            n.tensor(w),
            n.tensor(b),
            z.shape[0],
            z.shape[-1],
            w.shape[0],
        )
        compare("linear-" + name, n.read(y, expected.shape).cuda(), expected)
        n.clear()
        return expected

    q, k, v = (
        linear(norm(vision_x, "norm1"), "attn.qkv").reshape(-1, 3, 16, 72).unbind(1)
    )
    positions = get_vision_position_ids(grid, 2)
    inv = 1 / (10000 ** (torch.arange(0, 36, 2).float() / 36))
    freq = positions[..., None].float() * inv.cuda()
    co = torch.cat((freq.cos()[:, 0], freq.cos()[:, 1]), -1)
    si = torch.cat((freq.sin()[:, 0], freq.sin()[:, 1]), -1)
    if args.native:
        native_co = (
            torch.from_numpy(np.fromfile(args.native / "vision_cos-0.bf16", np.float32))
            .reshape(co.shape)
            .cuda()
        )
        native_si = (
            torch.from_numpy(np.fromfile(args.native / "vision_sin-0.bf16", np.float32))
            .reshape(si.shape)
            .cuda()
        )
        compare("production-cos", native_co, co)
        compare("production-sin", native_si, si)

    def rope(z):
        z = z.float()
        expected = (
            z * torch.cat((co, co), -1)[:, None]
            + torch.cat((-z[..., 36:], z[..., :36]), -1)
            * torch.cat((si, si), -1)[:, None]
        ).bfloat16()
        y = n.tensor(torch.zeros_like(expected))
        n.call(
            "rope",
            y,
            n.tensor(z.bfloat16()),
            n.tensor(co),
            n.tensor(si),
            16,
            72,
            False,
            False,
        )
        compare("rope", n.read(y, expected.shape).cuda(), expected)
        n.clear()
        if args.native:
            y = n.tensor(torch.zeros_like(expected))
            n.call(
                "rope",
                y,
                n.tensor(z.bfloat16()),
                n.tensor(native_co),
                n.tensor(native_si),
                16,
                72,
                False,
                False,
            )
            compare("production-rope", n.read(y, expected.shape).cuda(), expected)
            n.clear()
        return expected

    q, k = rope(q), rope(k)
    seq = q.shape[0]
    scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) * (72**-0.5)
    expected = (scores.float().softmax(-1).bfloat16() @ v.transpose(0, 1)).transpose(
        0, 1
    )
    bounds = np.array([[0, seq] for _ in range(seq)], dtype=np.uint32)
    a = Attention(1, seq, seq, 16, 16, 72, 72**-0.5, 0, True, True)
    y = n.tensor(torch.zeros_like(expected))
    n.call(
        "attend",
        y,
        n.tensor(q),
        n.tensor(k),
        n.tensor(v),
        bounds.ctypes.data_as(C.POINTER(C.c_uint32)),
        C.byref(a),
    )
    compare("attention", n.read(y, expected.shape).cuda(), expected)
    n.clear()
    residual = vision_x + linear(expected.reshape(seq, 1152), "attn.proj")
    hidden = linear(norm(residual, "norm2"), "mlp.linear_fc1")
    y = n.tensor(torch.zeros_like(hidden))
    n.call("point", y, n.tensor(hidden), View(), 4, 0)
    compare(
        "gelu-tanh", n.read(y, hidden.shape).cuda(), F.gelu(hidden, approximate="tanh")
    )
    n.clear()
    final = residual + linear(F.gelu(hidden, approximate="tanh"), "mlp.linear_fc2")
    target = (
        torch.from_numpy(
            np.fromfile(args.oracle / f"vision-{args.layer}.bf16", dtype=np.uint16)
        )
        .view(torch.bfloat16)
        .reshape(-1, 1152)
        .cuda()
    )
    compare("manual-vs-oracle", final, target)
    n.close()


if __name__ == "__main__":
    main()
