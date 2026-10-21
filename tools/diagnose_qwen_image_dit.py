#!/usr/bin/env python3
"""Diagnose the first divergent block with identical native input/parameters."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    spec = json.loads((a.prepared / "prepared.json").read_text())
    for s in spec["sources"]:
        sys.path.insert(
            0,
            str(
                a.prepared
                / "sources"
                / f"{s['repository'].split('/')[-1]}-{s['revision']}"
                / "src"
            ),
        )
    sys.path.insert(0, str(a.prepared / "deps"))
    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21TransformerBlock,
        QwenImage21Rope,
        QwenImage21Transformer2DModel,
        apply_rotary_emb_qwen,
    )
    from suite_numeric_metrics import compare

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def read(name):
        return (
            torch.from_numpy(
                np.fromfile(a.native / f"step-0-{name}.bf16", dtype=np.uint16)
            )
            .view(torch.bfloat16)
            .to("cuda")
        )

    x = read("joint-0").reshape(1, -1, 4096)
    mod = read("modulation-0").reshape(2, -1)
    n = x.shape[1]
    prefix = n - 96
    shapes = [(1, 8, 12)]
    image_pad = torch.zeros(n, dtype=torch.bool, device="cuda")
    image_pad[prefix:] = True
    if a.fixture:
        import struct

        fixture = a.fixture.read_bytes()
        hh, ww, text, ns = struct.unpack_from("<4I", fixture)
        shapes = [
            (1, *struct.unpack_from("<2I", fixture, 16 + 8 * i)) for i in range(ns)
        ]
        original = np.frombuffer(
            fixture, dtype=np.uint8, count=text, offset=16 + ns * 8
        ).astype(bool)
        image_pad = torch.tensor(
            np.concatenate(
                [
                    np.repeat(original, np.where(original, 4, 1)),
                    np.ones(hh * ww, dtype=bool),
                ]
            ),
            device="cuda",
        )
        prefix = n - hh * ww
    block = QwenImage21TransformerBlock(4096, 32, 128).to(torch.bfloat16)
    weights = {}
    for path in (a.prepared / "models/Qwen/Qwen-Image-2.1-Turbo/transformer").glob(
        "*.safetensors"
    ):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("transformer_blocks.0."):
                    weights[key.removeprefix("transformer_blocks.0.")] = f.get_tensor(
                        key
                    )
    block.load_state_dict(weights, strict=True)
    block.eval().to("cuda")

    def save(name, t):
        values = t.detach().cpu().contiguous()
        values.view(torch.uint16).numpy().tofile(a.output / f"{name}.bf16")
        actual = read(name).float().cpu().numpy()
        expected = values.float().numpy().reshape(-1)
        print(name, json.dumps(compare(expected, actual)), flush=True)

    ids, mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad, shapes)
    rope = QwenImage21Rope(10000, [16, 56, 56])(shapes, image_pad, torch.device("cuda"))
    # Distinguish phase construction from complex multiplication rounding.
    import ctypes as C

    lib = C.CDLL("libm.so.6")
    for fn in ["powf", "cosf", "sinf", "fmaf"]:
        f = getattr(lib, fn)
        f.restype = C.c_float
        f.argtypes = [C.c_float] * (3 if fn == "fmaf" else 2 if fn == "powf" else 1)
    pos = np.array(
        [[i, i, i] for i in range(prefix)]
        + [[prefix, yy - 4, xx - 6] for yy in range(8) for xx in range(12)]
    )
    for convention in [] if a.fixture else ["divide", "multiply"]:
        phase = np.empty((n, 64), np.complex64)
        for i in range(n):
            at = 0
            for axis, dim in enumerate([16, 56, 56]):
                for j in range(dim // 2):
                    power = lib.powf(10000, float(np.float32(2 * j) / dim))
                    angle = (
                        np.float32(pos[i, axis] / power)
                        if convention == "divide"
                        else np.float32(pos[i, axis] * np.float32(1 / power))
                    )
                    phase[i, at] = complex(lib.cosf(angle), lib.sinf(angle))
                    at += 1
        print(
            "phase",
            convention,
            compare(rope.cpu().view(torch.float32).numpy(), phase.view(np.float32)),
            flush=True,
        )
        if convention == "multiply":
            target = rope.cpu().view(torch.float32).numpy()
            indices = np.argwhere(target != phase.view(np.float32))
            print(
                "phase differences",
                len(indices),
                [
                    (
                        i.tolist(),
                        float(target[tuple(i)]),
                        float(phase.view(np.float32)[tuple(i)]),
                    )
                    for i in indices[:12]
                ],
                flush=True,
            )
        for name in ["q", "k"]:
            norm = read(f"{name}norm-0").float().reshape(1, n, 32, 64, 2)
            real, imag = norm[..., 0], norm[..., 1]
            co = torch.tensor(phase.real, device="cuda")[None, :, None]
            si = torch.tensor(phase.imag, device="cuda")[None, :, None]
            separate = (
                torch.stack([real * co - imag * si, imag * co + real * si], -1)
                .flatten(-2)
                .to(torch.bfloat16)
            )
            cmplx = apply_rotary_emb_qwen(
                norm.flatten(-2).to(torch.bfloat16),
                torch.tensor(phase, device="cuda"),
                use_real=False,
            )
            native = read(f"rope_{name}-0").reshape(cmplx.shape)
            print(
                "rotation",
                convention,
                name,
                "separate",
                compare(separate.cpu().float().numpy(), native.cpu().float().numpy()),
                "complex",
                compare(cmplx.cpu().float().numpy(), native.cpu().float().numpy()),
                flush=True,
            )
            if convention == "multiply":
                for rf in [0, 1]:
                    for inf in [0, 1]:
                        re = (
                            (real.double() * co.double() - (imag * si).double()).float()
                            if not rf
                            else (
                                (real * co).double() - imag.double() * si.double()
                            ).float()
                        )
                        im = (
                            (imag.double() * co.double() + (real * si).double()).float()
                            if not inf
                            else (
                                (imag * co).double() + real.double() * si.double()
                            ).float()
                        )
                        fused = torch.stack([re, im], -1).flatten(-2).to(torch.bfloat16)
                        print(
                            "fma",
                            name,
                            rf,
                            inf,
                            "diff_count",
                            int((fused != cmplx).sum()),
                            flush=True,
                        )
    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        for phase in range(2):
            norm = block.img_norm1(x) if not phase else block.img_norm2(x)
            save(f"norm-{phase}", norm)
            m = mod[:, phase * 8192 : (phase + 1) * 8192]
            norm, gate = block._modulate(norm, m, mask)
            save(f"modulated-{phase}", norm)
            if not phase:
                q, k, v = (
                    block.attn.to_q(norm),
                    block.attn.to_k(norm),
                    block.attn.to_v(norm),
                )
                for name, t in [("qproj", q), ("kproj", k), ("vproj", v)]:
                    save(f"{name}-0", t)
                q = block.attn.norm_q(q.unflatten(-1, (32, 128)))
                k = block.attn.norm_k(k.unflatten(-1, (32, 128)))
                save("qnorm-0", q)
                save("knorm-0", k)
                q = apply_rotary_emb_qwen(q, rope, use_real=False)
                k = apply_rotary_emb_qwen(k, rope, use_real=False)
                save("rope_q-0", q)
                save("rope_k-0", k)
                valid = (
                    torch.arange(n, device="cuda")[None]
                    <= torch.arange(n, device="cuda")[:, None]
                )
                valid |= (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
                runs = []
                start = 0
                while start < prefix:
                    end = start + 1
                    while end < prefix and ids[end] == ids[start]:
                        end += 1
                    runs.append((start, end))
                    start = end
                runs.append((prefix, n))

                def sdpa(qq, kk, vv):
                    parts = []
                    for start, end in runs:
                        keys = end if start < prefix else n
                        parts.append(
                            torch.nn.functional.scaled_dot_product_attention(
                                qq[:, :, start:end],
                                kk[:, :, :keys],
                                vv[:, :, :keys],
                                attn_mask=valid[start:end, :keys],
                            )
                        )
                    return torch.cat(parts, dim=2).transpose(1, 2).flatten(2)

                attended = sdpa(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.unflatten(-1, (32, 128)).transpose(1, 2),
                )
                save("attended-0", attended)
                native_q, native_k, native_v = (
                    read(f"{key}-0").reshape(1, n, 32, 128).transpose(1, 2)
                    for key in ["rope_q", "rope_k", "vproj"]
                )
                native_input_result = sdpa(native_q, native_k, native_v)
                print(
                    "attention identical inputs",
                    compare(
                        native_input_result.cpu().float().numpy().ravel(),
                        read("attended-0").cpu().float().numpy(),
                    ),
                    flush=True,
                )
                output = block.attn.to_out[0](attended)
                save("attn_projected-0", output)
            else:
                save("gate-1", block.img_mlp.gate_layer(norm))
                save("up-1", block.img_mlp.proj(norm))
                output = block.img_mlp(norm)
                save("mlp-1", output)
            x = x + gate.tanh() * output
            save(f"residual-{phase}", x)


if __name__ == "__main__":
    main()
