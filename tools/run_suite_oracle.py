#!/usr/bin/env python3
"""Development-only pinned Transformers/Diffusers oracle and repeatability dump.

Never imported by the DS4 server. Run GPU model processes serially. Use the
prepared references from prepare_suite_oracles.py; output must be a fresh dir.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["gemma", "qwen"], required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--recipe")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--latents", type=Path, help="Frozen packed latents NPZ with key latents"
    )
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeatability requires at least two samples after warmup")
    if args.model == "qwen" and args.dtype != "bfloat16":
        parser.error("full Qwen float32 is outside the frozen host/device budget")
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    for source in prepared["sources"]:
        name = source["repository"].split("/")[-1]
        sys.path.insert(
            0,
            str(
                (
                    args.prepared / "sources" / f"{name}-{source['revision']}" / "src"
                ).resolve()
            ),
        )
    sys.path.insert(0, str((args.prepared / "deps").resolve()))
    import numpy as np
    import torch

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20261020)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {
        "model": args.model,
        "prepared_sha256": hashlib.sha256(
            (args.prepared / "prepared.json").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "tf32": False,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "warmup": 1,
        "repeats": args.repeats,
        "cases": [],
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "tokenizers", "huggingface-hub", "accelerate")
        },
    }

    def dump(name, values):
        arrays = {
            k: v.detach().cpu().float().numpy()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.detach().cpu().numpy()
            if isinstance(v, torch.Tensor)
            else np.asarray(v)
            for k, v in values.items()
        }
        np.savez(args.output / f"{name}.npz", **arrays)

    def sample(name, run):
        baseline = None
        maximum = 0.0
        times = []
        for repeat in range(-1, args.repeats):
            torch.cuda.synchronize()
            start = time.monotonic()
            values = run(repeat)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            value = np.asarray(values["result"], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite oracle: {name}")
            if repeat == -1:
                continue
            times.append(elapsed)
            if baseline is None:
                baseline = value.copy()
                dump(name, values)
            maximum = max(maximum, float(np.max(np.abs(value - baseline))))
        report["cases"].append(
            {
                "id": name,
                "shape": list(baseline.shape),
                "max_repeat_abs": maximum,
                "seconds": times,
                "result_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
            }
        )
        (args.output / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, "max_repeat_abs", maximum, flush=True)

    if args.model == "gemma":
        from transformers import AutoModel, AutoProcessor

        path = args.prepared / "models/google/embeddinggemma-2"
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = (
            AutoModel.from_pretrained(
                path, dtype=dtype, attn_implementation="eager", local_files_only=True
            )
            .eval()
            .to("cuda")
        )
        # Synthetic stimuli isolate mechanics; they are not semantic retrieval
        # ground truth. Store the exact pixels/wave/frames, not just seeds.
        image = np.zeros((96, 144, 3), dtype=np.uint8)
        image[:, :72, 0] = 255
        image[:, 72:, 2] = 255
        audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(
            np.float32
        )
        video = np.stack([image, image[:, ::-1].copy()])
        dump("stimuli", {"image": image, "audio_16khz": audio, "video_1fps": video})
        requests = [
            (
                "text",
                dict(
                    text=[
                        "task: search result | query: Un cerchio blu",
                        "title: none | text: A blue circle",
                    ]
                ),
            ),
            ("image", dict(text="<|image|>", images=image)),
            ("audio", dict(text="<|audio|>", audio=audio, sampling_rate=16000)),
            ("video", dict(text="<|video|>", videos=video, do_sample_frames=False)),
            (
                "interleaved",
                dict(
                    text="Colori <|image|> movimento <|video|> suono <|audio|>",
                    images=image,
                    videos=video,
                    audio=audio,
                    sampling_rate=16000,
                    do_sample_frames=False,
                ),
            ),
        ]
        for name, request in requests:
            inputs = processor(**request, return_tensors="pt")
            dump(
                name + "-inputs",
                {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            )
            inputs = inputs.to(device="cuda")
            for key in inputs:
                if (
                    isinstance(inputs[key], torch.Tensor)
                    and inputs[key].is_floating_point()
                ):
                    inputs[key] = inputs[key].to(dtype)

            @torch.inference_mode()
            def forward(repeat):
                output = model(**inputs, output_hidden_states=True)
                hidden = output.last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                normalized = torch.nn.functional.normalize(pooled, dim=-1)
                values = {
                    "last_hidden_state": hidden,
                    "mean_f32": pooled,
                    "result": normalized.cpu().numpy(),
                }
                for dimension in (128, 256, 512, 768):
                    values[f"normalized_{dimension}"] = torch.nn.functional.normalize(
                        pooled[:, :dimension], dim=-1
                    )
                if output.hidden_states:
                    values["hidden_first"] = output.hidden_states[0]
                    values["hidden_last"] = output.hidden_states[-1]
                return values

            sample(name, forward)
        report["pooling"] = (
            "attention_mask mean including prompts, float32 reduction then L2; raw hidden also saved"
        )
        report["attention"] = "eager"
    else:
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(
            args.prepared / "models/Qwen/Qwen-Image-2.1-Turbo",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
        recipe = {
            "prompt": "A red square beside a blue circle, clean white background.",
            "height": 512,
            "width": 512,
            "steps": 8,
            "cfg": 1.0,
        }
        reference_images = None
        if args.recipe:
            if not args.corpus:
                parser.error("--recipe requires --corpus")
            corpus = json.loads(args.corpus.read_text())
            recipe = next(r for r in corpus["qwen_recipes"] if r["id"] == args.recipe)
            by_id = {i["id"]: i for i in corpus["items"]}
            if recipe["reference_ids"]:
                from PIL import Image

                reference_images = [
                    Image.open(
                        args.corpus.parent / by_id[i]["artifact"]["path"]
                    ).convert("RGB")
                    for i in recipe["reference_ids"]
                ]
        frozen_latents = None
        if args.latents:
            with np.load(args.latents) as values:
                frozen_latents = torch.from_numpy(values["latents"].copy()).to(
                    torch.bfloat16
                )
        # Capture the exact initial latent plus all eight scheduler boundaries.
        original_prepare = pipe.prepare_latents

        def prepare(*a, **kw):
            value = original_prepare(*a, **kw)
            dump("initial-latents", {"latents": value[0]})
            return value

        pipe.prepare_latents = prepare

        def forward(repeat):
            boundaries = {}

            def step(pipeline, index, timestep, values):
                boundaries[f"step_{index}"] = values["latents"].detach().cpu().float()
                return values

            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            output = pipe(
                prompt=recipe["prompt"],
                image=reference_images,
                height=recipe["height"],
                width=recipe["width"],
                num_inference_steps=recipe["steps"],
                true_cfg_scale=recipe["cfg"],
                generator=generator,
                latents=frozen_latents,
                output_type="np",
                callback_on_step_end=step,
            ).images
            if len(boundaries) != 8:
                raise ValueError(
                    f"expected Turbo eight-step recipe, observed {len(boundaries)}"
                )
            return {"result": output, **boundaries}

        sample(args.recipe or "image-generation-512", forward)
        report["recipe"] = dict(
            recipe, seed=args.seed, scheduler=dict(pipe.scheduler.config)
        )
    report["peak_device_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
