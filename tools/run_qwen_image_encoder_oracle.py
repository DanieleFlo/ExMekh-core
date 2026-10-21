#!/usr/bin/env python3
"""Development-only Qwen3-VL boundary oracle at the frozen T00 revisions.

Loads only the image conditioning encoder. Native DS4 never launches this.
Every BF16 boundary is saved in full, including pre/post final-norm controls.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", default=["text", "reference", "multi", "reversed", "rgba"]
    )
    parser.add_argument("--native-padding-reference", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Exact native encoder fixture instead of synthetic cases",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    for source in prepared["sources"]:
        sys.path.insert(
            0,
            str(
                args.prepared
                / "sources"
                / f"{source['repository'].split('/')[-1]}-{source['revision']}"
                / "src"
            ),
        )
    sys.path.insert(0, str(args.prepared / "deps"))
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(20261020)
    torch.set_num_threads(2)
    model_path = args.prepared / "models/Qwen/Qwen-Image-2.1-Turbo"
    processor = AutoProcessor.from_pretrained(
        model_path / "processor", local_files_only=True
    )
    model = (
        Qwen3VLModel.from_pretrained(
            model_path / "text_encoder",
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .to("cuda")
    )
    recipe = json.loads(
        (
            Path(__file__).resolve().parents[1] / "config/qwen-image-encoder.json"
        ).read_text()
    )
    drop = len(
        processor.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": recipe["system_prompt"]}],
                }
            ],
            tokenize=True,
            return_dict=False,
        )[0]
    )
    current = {}
    positions_seen = []
    model.language_model.rotary_emb.register_forward_pre_hook(
        lambda m, a: positions_seen.append(a[1].detach().cpu().clone())
    )

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        current[name] = value.detach().cpu().contiguous()

    model.visual.patch_embed.register_forward_hook(
        lambda m, a, o: save("vision_patch-0", o)
    )
    model.visual.blocks[0].register_forward_pre_hook(
        lambda m, a: save("vision_input-0", a[0])
    )

    for i, layer in enumerate(model.visual.blocks):
        layer.register_forward_hook(lambda m, a, o, i=i: save(f"vision-{i}", o))
    for i, layer in enumerate(model.visual.deepstack_merger_list):
        layer.register_forward_hook(lambda m, a, o, i=i: save(f"deepstack-{i}", o))
    model.visual.merger.register_forward_hook(
        lambda m, a, o: save("vision_merged-0", o)
    )
    for i, layer in enumerate(model.language_model.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: save(f"text-{i}", o))
    model.language_model.register_forward_pre_hook(
        lambda m, a, kw: save("embeddings-0", kw["inputs_embeds"]), with_kwargs=True
    )
    model.language_model.norm.register_forward_pre_hook(
        lambda m, a: save("pre_norm", a[0])
    )
    model.language_model.norm.register_forward_hook(
        lambda m, a, o: save("post_norm", o)
    )
    y, x = np.mgrid[:256, :256]
    rgb = np.stack(((x * 3 + y) % 256, (y * 5) % 256, (x + y * 2) % 256), -1).astype(
        np.uint8
    )
    other = np.roll(rgb, 47, axis=1).copy()
    rgba = np.concatenate((rgb, ((x + y) % 256).astype(np.uint8)[..., None]), -1)
    cases = {
        "text": [],
        "reference": [rgb],
        "multi": [rgb, other],
        "reversed": [other, rgb],
        "rgba": [rgba],
        "resized": [rgb[:144, :96].copy()],
        "unicode": [],
        "empty": [],
    }
    for name in [args.fixture.stem] if args.fixture else args.cases:
        directory = args.output / name
        directory.mkdir(exist_ok=False)
        references = cases.get(name, [])
        prompt = "A red square beside a blue circle, clean white background."
        if name == "unicode":
            prompt = "Caffe\u0300, Ελληνικά, العربية, 中文, 日本語 🌈"
        if name == "empty":
            prompt = " "
        if args.fixture:
            fixture = args.fixture.read_bytes()
            length, count = struct.unpack_from("<2I", fixture)
            prompt = fixture[8 : 8 + length].decode()
            at = 8 + length
            references = []
            for _ in range(count):
                w, h, c = struct.unpack_from("<3I", fixture, at)
                at += 12
                references.append(
                    np.frombuffer(fixture, np.uint8, count=w * h * c, offset=at)
                    .reshape(h, w, c)
                    .copy()
                )
                at += w * h * c
            if at != len(fixture):
                raise ValueError("invalid encoder fixture")
        else:
            fixture = (
                struct.pack("<II", len(prompt.encode()), len(references))
                + prompt.encode()
            )
            for image in references:
                h, w, c = image.shape
                fixture += struct.pack("<III", w, h, c) + image.tobytes()
        (directory / "input.ds4i").write_bytes(fixture)
        images = []
        for array in references:
            image = Image.fromarray(array)
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, (255, 255, 255))
                white.paste(image, mask=image.getchannel("A"))
                image = white
            images.append(image)
        tags = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(images))
        )
        text = recipe["prefix"].format(**recipe) + tags + prompt + recipe["suffix"]
        inputs = processor(
            text=[text],
            images=images or None,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        np.savez(
            directory / "inputs.npz",
            **{k: v.numpy() for k, v in inputs.items() if isinstance(v, torch.Tensor)},
        )
        kwargs = {k: v.to("cuda") for k, v in inputs.items()}
        current.clear()
        positions_seen.clear()
        with torch.inference_mode():
            model(**kwargs, use_cache=False, output_hidden_states=True)
        mask = inputs.input_ids[0] == 151655
        with torch.inference_mode():
            for i in range(min(len(references) * 3, 3)):
                if f"deepstack-{i}" in current:
                    current[f"text-{i}"][0, mask] += current[f"deepstack-{i}"]
        current["conditioning-0"] = current["pre_norm"][0, drop:]
        for key, value in current.items():
            value.to(torch.bfloat16).view(torch.uint16).numpy().tofile(
                directory / f"{key}.bf16"
            )
        positions = positions_seen[-1]
        np.asarray(positions[:, 0].T, dtype=np.uint32).tofile(
            directory / "positions.u32"
        )
        report = {
            "case": name,
            "drop": drop,
            "tokens": inputs.input_ids.shape[-1],
            "references": len(images),
            "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
            "peak_device_bytes": torch.cuda.max_memory_allocated(),
            "pre_post_relative_l2": float(
                torch.linalg.vector_norm(
                    current["pre_norm"].float() - current["post_norm"].float()
                )
                / torch.linalg.vector_norm(current["pre_norm"].float())
            ),
        }
        (directory / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    if args.native_padding_reference:
        # The public adapter runs one item per lease. Explicit left padding
        # is compared at the same batch/sequence shape; batch-size-dependent
        # BF16 rounding is not treated as reference noise.
        prompts = [
            "A red square beside a blue circle, clean white background.",
            "Caffe\u0300, Ελληνικά, العربية, 中文, 日本語 🌈",
        ]
        directory = args.output / "padding"
        directory.mkdir()
        thresholds = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs/performance/native-suite-t00-20261020/tolerances.json"
            ).read_text()
        )["qwen_latent_boundary"]
        results = []
        for prompt, name in zip(prompts, ("text", "unicode")):
            text = recipe["prefix"].format(**recipe) + prompt + recipe["suffix"]
            inputs = processor(
                text=[text],
                padding="max_length",
                max_length=64,
                padding_side="left",
                return_tensors="pt",
            )
            current.clear()
            positions_seen.clear()
            with torch.inference_mode():
                model(**{k: v.to("cuda") for k, v in inputs.items()}, use_cache=False)
            positions = positions_seen[-1]
            np.savez(
                directory / f"{name}-inputs.npz",
                **{k: v.numpy() for k, v in inputs.items()},
            )
            valid = inputs.attention_mask[0].bool()
            expected = current["pre_norm"][0, valid][drop:].contiguous()
            expected.view(torch.uint16).numpy().tofile(directory / f"{name}.bf16")
            native = args.native_padding_reference / f"{name}-padded"
            bits = np.fromfile(native / "conditioning-0.bf16", dtype="<u2")
            actual = (
                (bits.astype(np.uint32) << 16).view(np.float32).reshape(expected.shape)
            )
            reference = expected.float().numpy()
            delta = actual.astype(np.float64) - reference
            relative = float(
                np.linalg.norm(delta) / np.linalg.norm(reference.astype(np.float64))
            )
            rmse = float(np.sqrt(np.mean(delta**2)))
            identity = np.array_equal(
                inputs.input_ids[0].numpy(),
                np.fromfile(native / "tokens.u32", dtype="<u4"),
            )
            identity &= np.array_equal(
                positions[:, 0].T.numpy(),
                np.fromfile(native / "positions.u32", dtype="<u4").reshape(-1, 3),
            )
            results.append(
                {
                    "case": name,
                    "tokens_positions": bool(identity),
                    "relative_l2": relative,
                    "rmse": rmse,
                    "verdict": "PASS"
                    if identity
                    and np.isfinite(actual).all()
                    and relative <= thresholds["relative_l2"]
                    and rmse <= thresholds["rmse_max"]
                    else "FAIL",
                }
            )
        (directory / "verification.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(results), flush=True)
        if any(r["verdict"] != "PASS" for r in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
