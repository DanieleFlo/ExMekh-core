#!/usr/bin/env python3
"""Development-only frozen DiT/VAE oracle. Native inference never imports this.

Captures every block/step in full. Run one component process at a time.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--conditioning", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--component", choices=["dit", "vae"], required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument(
        "--check-uncached",
        action="store_true",
        help="also recompute full DiT on each identical cached input",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    for s in prepared["sources"]:
        sys.path.insert(
            0,
            str(
                args.prepared
                / "sources"
                / f"{s['repository'].split('/')[-1]}-{s['revision']}"
                / "src"
            ),
        )
    sys.path.insert(0, str(args.prepared / "deps"))
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import (
        QwenImage21Transformer2DModel,
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_path = args.prepared / "models/Qwen/Qwen-Image-2.1-Turbo"
    active = {}

    def dump(name, value):
        if isinstance(value, tuple):
            value = value[0]
        value = value.detach().cpu().contiguous()
        ext = "bf16" if value.dtype == torch.bfloat16 else "f32"
        value.view(torch.uint16).numpy().tofile(
            active["dir"] / f"{active.get('prefix', '')}{name}.{ext}"
        ) if ext == "bf16" else value.float().numpy().tofile(
            active["dir"] / f"{active.get('prefix', '')}{name}.{ext}"
        )

    if args.component == "dit":
        model = (
            QwenImage21Transformer2DModel.from_pretrained(
                model_path / "transformer",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .to("cuda")
        )
        model.modulation.register_forward_hook(lambda m, a, o: dump("modulation-0", o))
        for i, block in enumerate(model.transformer_blocks):
            block.register_forward_pre_hook(
                lambda m, a, kw, i=i: dump("joint-0", kw["hidden_states"])
                if i == 0
                else None,
                with_kwargs=True,
            )
            block.register_forward_hook(lambda m, a, o, i=i: dump(f"block-{i}", o))
        for case in args.cases or ["text", "reference", "multi", "rgba"]:
            source = args.conditioning / case
            inputs = np.load(source / "inputs.npz")
            report = json.loads((source / "oracle.json").read_text())
            hidden = (
                torch.from_numpy(
                    np.fromfile(source / "conditioning-0.bf16", dtype=np.uint16)
                )
                .view(torch.bfloat16)
                .reshape(1, -1, 4096)
                .to("cuda")
            )
            image_mask = inputs["input_ids"][0, report["drop"] :] == 151655
            th, tw = (8, 12) if case == "text" else (16, 16)
            refs = int(report["references"])
            shapes = [(16, 16)] * refs + [(th, tw)]
            assert image_mask.sum() * 4 == refs * 256
            rng = np.random.default_rng(20261020)
            reference = rng.standard_normal((1, refs * 256, 64), dtype=np.float32)
            noise = rng.standard_normal((1, th * tw, 64), dtype=np.float32)
            directory = args.output / f"dit-{case}"
            directory.mkdir(exist_ok=False)
            active.update(dir=directory)
            ref = torch.tensor(reference, device="cuda", dtype=torch.bfloat16)
            mask = torch.tensor(
                np.concatenate([image_mask, np.ones(th * tw // 4, dtype=bool)]),
                device="cuda",
            )[None]
            fixture = struct.pack(
                "<4I", th, tw, hidden.shape[1], len(shapes)
            ) + b"".join(struct.pack("<2I", *s) for s in shapes)
            fixture += (
                image_mask.astype(np.uint8).tobytes()
                + hidden.cpu().view(torch.uint16).numpy().tobytes()
                + ref.cpu().view(torch.uint16).numpy().tobytes()
                + noise.tobytes()
            )
            (directory / "input.ds4g").write_bytes(fixture)
            sigmas = json.loads((model_path / "model_index.json").read_text())[
                "sample_sigmas"
            ]
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path / "scheduler", local_files_only=True
            )
            scheduler.set_timesteps(sigmas=sigmas, device="cuda", mu=0.5)
            (directory / "sigmas.f32").write_bytes(
                scheduler.sigmas.cpu().numpy().tobytes()
            )
            cache = QwenImage21KVCache(32)
            latent = torch.tensor(noise, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for i, t in enumerate(scheduler.timesteps):
                    active["prefix"] = f"step-{i}-"
                    full = torch.cat([ref, latent.to(torch.bfloat16)], dim=1)
                    velocity = model(
                        full,
                        hidden,
                        (t / 1000).expand(1),
                        [[(1, *s) for s in shapes]],
                        mask,
                        kv_cache=cache,
                        kv_cache_mode="extract" if i == 0 else "cached",
                    ).sample[:, -th * tw :]
                    dump("velocity-0", velocity)
                    if args.check_uncached and i:
                        previous_dir, previous_prefix = active["dir"], active["prefix"]
                        active["dir"] = directory / "uncached" / str(i)
                        active["dir"].mkdir(parents=True)
                        active["prefix"] = ""
                        recomputed = model(
                            full,
                            hidden,
                            (t / 1000).expand(1),
                            [[(1, *s) for s in shapes]],
                            mask,
                        ).sample[:, -th * tw :]
                        active.update(dir=previous_dir, prefix=previous_prefix)
                        recomputed.cpu().contiguous().view(torch.uint16).numpy().tofile(
                            directory / f"uncached-same-input-velocity-{i}.bf16"
                        )
                    latent = scheduler.step(velocity, t, latent, return_dict=False)[0]
                    dump(f"latent-{i}", latent.float())
            (directory / "oracle.json").write_text(
                json.dumps(
                    {
                        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
                        "shapes": shapes,
                        "steps": 8,
                        "cfg": 1,
                        "attention": "math SDPA FP32 intermediate",
                        "peak": torch.cuda.max_memory_allocated(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(directory, flush=True)
    else:
        model = (
            AutoencoderKLQwenImage21.from_pretrained(
                model_path / "vae", torch_dtype=torch.bfloat16, local_files_only=True
            )
            .eval()
            .to("cuda")
        )
        for side in ["encoder", "decoder"]:
            obj = getattr(model, side)
            blocks = obj.down_blocks if side == "encoder" else obj.up_blocks
            for i, block in enumerate(blocks):
                block.register_forward_hook(
                    lambda m, a, o, i=i, side=side: dump(f"{side}-{i}", o)
                )
            obj.mid_block.register_forward_hook(
                lambda m, a, o, side=side: dump(f"{side}-5", o)
            )
        mean = torch.tensor(
            model.config.latents_mean, device="cuda", dtype=torch.bfloat16
        ).view(1, 64, 1, 1, 1)
        std = torch.tensor(
            model.config.latents_std, device="cuda", dtype=torch.bfloat16
        ).view(1, 64, 1, 1, 1)
        for case in args.cases or ["rgba", "transparent", "rgb", "wide"]:
            h, w = (64, 96) if case != "wide" else (256, 768)
            yy, xx = np.mgrid[:h, :w]
            pixels = np.stack(
                [
                    (3 * xx + yy) % 256,
                    (5 * yy) % 256,
                    (xx + 2 * yy) % 256,
                    (xx + yy) % 256,
                ],
                -1,
            ).astype(np.uint8)
            if case == "transparent":
                pixels[..., 3] = 0
            if case == "rgb":
                pixels = pixels[..., :3].copy()
            rgba = (
                pixels
                if pixels.shape[-1] == 4
                else np.concatenate([pixels, np.full((h, w, 1), 255, np.uint8)], -1)
            )
            directory = args.output / f"vae-{case}"
            directory.mkdir(exist_ok=False)
            active.update(dir=directory, prefix="")
            fixture = struct.pack("<3I", w, h, pixels.shape[-1]) + pixels.tobytes()
            (directory / "input.ds4v").write_bytes(fixture)
            image = (
                torch.tensor(rgba, device="cuda")
                .float()
                .permute(2, 0, 1)[None, :, None]
                / 255
                * 2
                - 1
            )
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                raw = model.encode(image.to(torch.bfloat16)).latent_dist.mode()
                dump("posterior-0", raw)
                latent = (raw - mean) / std
                packed = latent.permute(0, 2, 3, 4, 1).reshape(-1, 64)
                dump("encoded-0", packed)
                packed.float().cpu().numpy().tofile(directory / "decode.f32")
                decoded = model.decode(latent * std + mean).sample
                pixels = (
                    (decoded.float() * 0.5 + 0.5)[0, :, 0].permute(1, 2, 0).contiguous()
                )
                pixels.cpu().numpy().tofile(directory / "pixels.f32")
            (directory / "oracle.json").write_text(
                json.dumps(
                    {
                        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
                        "width": w,
                        "height": h,
                        "peak": torch.cuda.max_memory_allocated(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(directory, flush=True)


if __name__ == "__main__":
    main()
