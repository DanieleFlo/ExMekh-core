#!/usr/bin/env python3
"""Sequential frozen image-pipeline oracle, one model component per process.

Cases specify id,prompt,width,height,noise and references (.ds4i). Explicit
reference dimensions match native recipe v1; no implicit resize is introduced.
Run stages encoder, encode, dit, decode in that order into the same directory.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def bounded_math_sdpa(torch, original, budget=512 * 1024 * 1024):
    """Tile independent query rows; retain every key and the original mask.

    Development oracle only: eager math SDPA otherwise materializes several
    multi-GiB score matrices at corpus sizes. Each tile still calls PyTorch's
    math backend with the complete softmax axis, no approximation or key pruning.
    """

    def attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *,
        scale=None,
        enable_gqa=False,
    ):
        rows, keys = query.shape[-2], key.shape[-2]
        score_bytes = query.numel() // query.shape[-1] * keys * 4
        options = {"dropout_p": dropout_p, "scale": scale, "enable_gqa": enable_gqa}
        if score_bytes <= budget:
            return original(
                query, key, value, attn_mask=attn_mask, is_causal=is_causal, **options
            )
        if dropout_p:
            raise ValueError("oracle query tiling requires dropout zero")
        output = torch.empty(
            (*query.shape[:-1], value.shape[-1]), device=query.device, dtype=query.dtype
        )
        for first in range(0, rows, 256):
            last = min(first + 256, rows)
            mask = attn_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., first:last, :]
            if is_causal:
                if mask is not None:
                    raise ValueError("causal oracle cannot also supply a mask")
                mask = (
                    torch.arange(keys, device=query.device)[None, :]
                    <= torch.arange(first, last, device=query.device)[:, None]
                )
            output[..., first:last, :] = original(
                query[..., first:last, :],
                key,
                value,
                attn_mask=mask,
                is_causal=False,
                **options,
            )
        return output

    return attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--stage", choices=["encoder", "encode", "dit", "decode"], required=True
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((a.prepared / "prepared.json").read_text())
    for s in lock["sources"]:
        sys.path.insert(
            0,
            str(
                (
                    a.prepared
                    / "sources"
                    / f"{s['repository'].split('/')[-1]}-{s['revision']}"
                    / "src"
                ).resolve()
            ),
        )
    sys.path.insert(0, str((a.prepared / "deps").resolve()))
    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Transformer2DModel,
    )
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoProcessor, Qwen3VLModel

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Same math kernel and full-key softmax; bound only query-row scheduling.
    score_budget = int(os.environ.get("DS4_SUITE_ORACLE_SCORE_BYTES", "536870912"))
    torch.nn.functional.scaled_dot_product_attention = bounded_math_sdpa(
        torch, torch.nn.functional.scaled_dot_product_attention, score_budget
    )
    modelpath = a.prepared / "models/Qwen/Qwen-Image-2.1-Turbo"
    recipe = json.loads(Path("config/qwen-image-encoder.json").read_text())
    cases = json.loads(a.cases.read_text())

    def read_reference(path):
        raw = Path(path).read_bytes()
        w, h, c = struct.unpack("<3I", raw[:12])
        return np.frombuffer(raw[12:], np.uint8).reshape(h, w, c)

    def read_bf16(path, shape):
        return (
            torch.from_numpy(np.fromfile(path, np.uint16))
            .view(torch.bfloat16)
            .reshape(shape)
            .to("cuda")
        )

    def dump(path, value):
        value.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(path)

    if a.stage == "encoder":
        processor = AutoProcessor.from_pretrained(
            modelpath / "processor", local_files_only=True
        )
        model = (
            Qwen3VLModel.from_pretrained(
                modelpath / "text_encoder",
                dtype=torch.bfloat16,
                attn_implementation="eager",
                local_files_only=True,
            )
            .eval()
            .to("cuda")
        )
        active = {}
        model.language_model.norm.register_forward_pre_hook(
            lambda m, args: active.update(hidden=args[0].detach())
        )
    elif a.stage in ("encode", "decode"):
        model = (
            AutoencoderKLQwenImage21.from_pretrained(
                modelpath / "vae", torch_dtype=torch.bfloat16, local_files_only=True
            )
            .eval()
            .to("cuda")
        )
        mean = torch.tensor(
            model.config.latents_mean, device="cuda", dtype=torch.bfloat16
        ).view(1, 64, 1, 1, 1)
        std = torch.tensor(
            model.config.latents_std, device="cuda", dtype=torch.bfloat16
        ).view(1, 64, 1, 1, 1)
    else:
        model = (
            QwenImage21Transformer2DModel.from_pretrained(
                modelpath / "transformer",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .to("cuda")
        )
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for case in cases:
            out = a.output / case["id"]
            out.mkdir(exist_ok=True)
            refs = [read_reference(x) for x in case.get("references", [])]
            resolution = case.get("reference_resolution", 0)
            if resolution:
                processed = []
                for ref in refs:
                    ratio = ref.shape[1] / ref.shape[0]
                    width = (resolution * resolution * ratio) ** 0.5
                    height = width / ratio
                    size = (round(width / 32) * 32, round(height / 32) * 32)
                    processed.append(
                        np.array(
                            Image.fromarray(ref)
                            .convert("RGBA")
                            .resize(size, Image.Resampling.LANCZOS)
                        )
                    )
                refs = processed
            h, w = case["height"] // 16, case["width"] // 16
            if a.stage == "encoder":
                images = []
                for ref in refs:
                    image = Image.fromarray(ref)
                    if ref.shape[-1] == 4:
                        white = Image.new("RGB", image.size, (255, 255, 255))
                        white.paste(image, mask=image.getchannel("A"))
                        image = white
                    images.append(image)
                tags = " ".join(
                    f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(len(refs))
                )
                text = (
                    recipe["prefix"].format(**recipe)
                    + tags
                    + case["prompt"]
                    + recipe["suffix"]
                )
                ins = processor(text=[text], images=images or None, return_tensors="pt")
                drop = len(
                    processor.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": recipe["system_prompt"]}
                                ],
                            }
                        ],
                        tokenize=True,
                        return_dict=False,
                    )[0]
                )
                model(**ins.to("cuda"), use_cache=False)
                dump(out / "conditioning.bf16", active["hidden"][:, drop:])
                mask = ins["input_ids"][0, drop:] == 151655
                mask.cpu().numpy().astype(np.uint8).tofile(out / "image-mask.u8")
                (out / "conditioning.json").write_text(
                    json.dumps({"count": len(mask), "drop": drop})
                )
            elif a.stage == "encode":
                encoded = []
                for ref in refs:
                    if ref.shape[-1] == 3:
                        ref = np.concatenate(
                            [ref, np.full((*ref.shape[:2], 1), 255, np.uint8)], -1
                        )
                    image = (
                        torch.tensor(ref, device="cuda")
                        .float()
                        .permute(2, 0, 1)[None, :, None]
                        / 255
                        * 2
                        - 1
                    )
                    latent = (
                        model.encode(image.to(torch.bfloat16)).latent_dist.mode() - mean
                    ) / std
                    encoded.append(latent.permute(0, 2, 3, 4, 1).reshape(1, -1, 64))
                if encoded:
                    dump(out / "references.bf16", torch.cat(encoded, 1))
            elif a.stage == "dit":
                conditioning = read_bf16(out / "conditioning.bf16", (1, -1, 4096))
                slots = np.fromfile(out / "image-mask.u8", np.uint8)
                reference = (
                    read_bf16(out / "references.bf16", (1, -1, 64))
                    if refs
                    else torch.empty((1, 0, 64), device="cuda", dtype=torch.bfloat16)
                )
                noise = np.fromfile(case["noise"], np.float32).reshape(1, h * w, 64)
                latent = torch.tensor(noise, device="cuda", dtype=torch.bfloat16)
                mask = torch.tensor(
                    np.concatenate([slots, np.ones(h * w // 4, np.uint8)]),
                    device="cuda",
                    dtype=torch.bool,
                )[None]
                shapes = [(1, r.shape[0] // 16, r.shape[1] // 16) for r in refs] + [
                    (1, h, w)
                ]
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    modelpath / "scheduler", local_files_only=True
                )
                scheduler.set_timesteps(
                    sigmas=json.loads((modelpath / "model_index.json").read_text())[
                        "sample_sigmas"
                    ],
                    device="cuda",
                    mu=0.5,
                )
                cache = QwenImage21KVCache(32)
                for i, t in enumerate(scheduler.timesteps):
                    velocity = model(
                        torch.cat([reference, latent.to(torch.bfloat16)], 1),
                        conditioning,
                        (t / 1000).expand(1),
                        [shapes],
                        mask,
                        kv_cache=cache,
                        kv_cache_mode="extract" if i == 0 else "cached",
                    ).sample[:, -h * w :]
                    latent = scheduler.step(velocity, t, latent, return_dict=False)[0]
                    latent.float().cpu().numpy().tofile(out / f"latent-{i}.f32")
                del cache
            else:
                latent = (
                    torch.from_numpy(np.fromfile(out / "latent-7.f32", np.float32))
                    .reshape(1, h, w, 64)
                    .permute(0, 3, 1, 2)[:, :, None]
                    .to(device="cuda", dtype=torch.bfloat16)
                )
                image = (
                    model.decode(latent * std + mean)
                    .sample.float()[0, :, 0]
                    .permute(1, 2, 0)
                )
                rgba = (image * 0.5 + 0.5).clamp(0, 1).contiguous().cpu().numpy()
                rgba.tofile(out / "rgba.f32")
                Image.fromarray(np.rint(rgba * 255).astype(np.uint8)).save(
                    out / "result.png"
                )
            print(a.stage, case["id"], flush=True)
    (a.output / f"{a.stage}-execution.json").write_text(
        json.dumps(
            {
                "backend": "PyTorch MATH",
                "query_tile": 256,
                "tile_if_score_bytes_exceeds": score_budget,
                "full_key_axis": True,
                "cases": [case["id"] for case in cases],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
