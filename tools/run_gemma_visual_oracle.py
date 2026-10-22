#!/usr/bin/env python3
"""Frozen image/video Gemma oracle with full vision and backbone boundaries."""

import argparse
import json
import re
import sys
from pathlib import Path

from suite_visual_cases import frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((args.prepared / "prepared.json").read_text())
    for source in lock["sources"]:
        if source["repository"] == "huggingface/transformers":
            sys.path.insert(
                0,
                str(
                    (
                        args.prepared
                        / "sources"
                        / f"transformers-{source['revision']}"
                        / "src"
                    ).resolve()
                ),
            )
    sys.path.insert(0, str((args.prepared / "deps").resolve()))
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    modelpath = args.prepared / "models/google/embeddinggemma-2"
    processor = AutoProcessor.from_pretrained(modelpath, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            modelpath,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    prompts = json.loads((modelpath / "config_sentence_transformers.json").read_text())[
        "prompts"
    ]
    prefixes = [
        "",
        *(
            prompts[k]
            for k in ("query", "document", "CodeRetrieval", "SentenceSimilarity")
        ),
    ]
    current = {"item": 0, "frame": 0, "soft": [], "record": True}

    def save(path, value):
        value.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(
            args.output / path
        )

    def dump(stage, layer, value):
        if not current["record"]:
            return
        save(f"item-{current['item']}-{stage}-{layer}.bf16", value)

    def vision(stage, layer, value):
        if not current["record"]:
            return
        if stage in ("vision-pooled", "vision-projected"):
            value = value.split(list(map(int, current["soft"])))
        for i, row in enumerate(value):
            save(f"frame-{current['frame'] + i}-{stage}-{layer}.bf16", row)

    model.vision_tower.patch_embedder.register_forward_hook(
        lambda m, a, o: vision("vision-patches", 0, o)
    )
    for i, layer in enumerate(model.vision_tower.encoder.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: vision("vision-layer", i, o))
    model.vision_tower.register_forward_hook(
        lambda m, a, o: vision("vision-pooled", 16, o.last_hidden_state)
    )
    model.embed_vision.register_forward_hook(
        lambda m, a, o: vision("vision-projected", 16, o)
    )
    layer0 = model.vision_tower.encoder.layers[0]
    for module, name in (
        (layer0.input_layernorm, "vision-norm"),
        (layer0.self_attn.q_norm, "vision-q"),
        (layer0.self_attn.k_norm, "vision-k"),
        (layer0.self_attn.v_norm, "vision-v"),
    ):
        module.register_forward_hook(lambda m, a, o, name=name: vision(name, 0, o))
    layer0.self_attn.o_proj.register_forward_pre_hook(
        lambda m, a: vision("vision-attention", 0, a[0])
    )
    layer0.pre_feedforward_layernorm.register_forward_pre_hook(
        lambda m, a: vision("vision-attn-out", 0, a[0])
    )
    layer0.mlp.down_proj.register_forward_pre_hook(
        lambda m, a: vision("vision-gate", 0, a[0])
    )
    from transformers.models.gemma4 import modeling_gemma4

    original_rope = modeling_gemma4.apply_multidimensional_rope

    def rope(x, *a, **kw):
        output = original_rope(x, *a, **kw)
        if current.get("rope", 0) < 2:
            vision(
                "vision-rope-q" if current.get("rope", 0) == 0 else "vision-rope-k",
                0,
                output,
            )
        current["rope"] = current.get("rope", 0) + 1
        return output

    modeling_gemma4.apply_multidimensional_rope = rope
    model.language_model.ple.register_forward_pre_hook(
        lambda m, a: dump("embeddings", 0, a[0])
    )
    model.language_model.ple.register_forward_hook(lambda m, a, o: dump("ple", 0, o))
    for i, layer in enumerate(model.language_model.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: dump("layer", i, o))
    model.language_model.embedding_projection.register_forward_hook(
        lambda m, a, o: dump("projected", 24, o)
    )
    text0 = model.language_model.layers[0]
    text0.input_layernorm.register_forward_hook(
        lambda m, a, o: dump("backbone-norm", 0, o)
    )
    for module, name in (
        (text0.self_attn.q_norm, "backbone-q"),
        (text0.self_attn.k_norm, "backbone-k"),
        (text0.self_attn.v_norm, "backbone-v"),
    ):
        module.register_forward_hook(
            lambda m, a, o, name=name: dump(name, 0, o.transpose(1, 2).contiguous())
        )
    text0.self_attn.o_proj.register_forward_pre_hook(
        lambda m, a: dump("backbone-attention", 0, a[0])
    )
    text0.pre_feedforward_layernorm.register_forward_pre_hook(
        lambda m, a: dump("backbone-attn-out", 0, a[0])
    )
    from transformers.models.embedding_gemma2 import modeling_embedding_gemma2

    original_text_rope = modeling_embedding_gemma2.apply_rotary_pos_emb

    def text_rope(q, k, *a, **kw):
        result = original_text_rope(q, k, *a, **kw)
        if not current.get("text_rope", False):
            dump("backbone-rope-q", 0, result[0].transpose(1, 2).contiguous())
            dump("backbone-rope-k", 0, result[1].transpose(1, 2).contiguous())
            current["text_rope"] = True
        return result

    modeling_embedding_gemma2.apply_rotary_pos_emb = text_rope
    # Process media separately, then use the official replacement and tokenizer.
    # This preserves per-media budgets and order without conflating video frames
    # and unrelated image placeholders in the high-level processor batching.
    vectors, official_vectors = [], []
    cases = json.loads(args.cases.read_text())
    with torch.inference_mode():
        for i, case in enumerate(cases):
            current["item"] = i
            current["text_rope"] = False
            prompt = case.get("prompt")
            if prompt is None:
                prompt = " ".join(f"<|{m['type']}|>" for m in case["media"])
            if not (
                prompt.startswith("task: ")
                or (prompt.startswith("title: ") and " | text: " in prompt)
            ):
                prompt = prefixes[case.get("task", 0)] + prompt
            parts = re.split(r"(<\|image\|>|<\|video\|>)", prompt)
            slots = iter(
                i for i, part in enumerate(parts) if part in ("<|image|>", "<|video|>")
            )
            image_features, video_features = [], []
            for j, media in enumerate(case["media"]):
                values, times = frames(media)
                marker = f"<|{media['type']}|>"
                if media["type"] == "image":
                    inputs = processor.image_processor(
                        values[0],
                        max_soft_tokens=media.get("soft") or 280,
                        return_tensors="pt",
                        input_data_format="channels_last",
                    )
                    pixels, positions = (
                        inputs["pixel_values"],
                        inputs["image_position_ids"],
                    )
                else:
                    inputs = processor.video_processor(
                        values,
                        do_sample_frames=False,
                        max_soft_tokens=media.get("soft") or 140,
                        return_tensors="pt",
                        input_data_format="channels_last",
                    )
                    pixels, positions = (
                        inputs["pixel_values_videos"],
                        inputs["video_position_ids"],
                    )
                soft = (positions[..., 0] >= 0).sum(-1).numpy() // 9
                current["soft"] = soft
                current["rope"] = 0
                pixels.numpy().astype("<f4").tofile(
                    args.output / f"item-{i}-media-{j}-pixels.f32"
                )
                positions.numpy().astype("<i4").tofile(
                    args.output / f"item-{i}-media-{j}-positions.i32"
                )
                blocks = []
                for n, time in zip(soft, times):
                    prefix = (
                        f"{int(time // 60):02d}:{int(time % 60):02d} "
                        if case.get("timestamps") and media["type"] == "video"
                        else ""
                    )
                    blocks.append(prefix + "<|image>" + marker * int(n) + "<image|>")
                slot = next(slots)
                assert parts[slot] == marker
                parts[slot] = (
                    " " if case.get("timestamps") and media["type"] == "video" else ""
                ).join(blocks)
                output = model.vision_tower(
                    pixel_values=pixels.to("cuda", torch.bfloat16),
                    pixel_position_ids=positions.to("cuda"),
                )
                feature = model.embed_vision(output.last_hidden_state)
                (image_features if media["type"] == "image" else video_features).append(
                    feature
                )
                current["frame"] += len(values)
            inputs = processor.tokenizer("".join(parts), return_tensors="pt").to("cuda")
            inputs["input_ids"].cpu().numpy().astype("<u4").tofile(
                args.output / f"item-{i}-tokens.u32"
            )
            ids = inputs["input_ids"]
            masked = ids.clone()
            masked[
                (ids == model.config.image_token_id)
                | (ids == model.config.video_token_id)
            ] = 0
            embedded = model.language_model.embed_tokens(masked)
            for token, features in (
                (model.config.image_token_id, image_features),
                (model.config.video_token_id, video_features),
            ):
                if features:
                    embedded[ids == token] = torch.cat(features)
            hidden = model.language_model(
                inputs_embeds=embedded, attention_mask=inputs["attention_mask"]
            ).last_hidden_state.float()
            pooled = hidden.mean(1)
            vectors.append(
                torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()[0]
            )
            # Independently exercise AutoProcessor + the complete official model.
            # Component traces use isolated media; this second gate also checks
            # batching and scatter performed by the original public entry point.
            current["record"] = False
            request = {"text": prompt, "return_tensors": "pt"}
            for kind, key in (("image", "images"), ("video", "videos")):
                selected = [m for m in case["media"] if m["type"] == kind]
                if selected:
                    values = [frames(m)[0] for m in selected]
                    request[key] = [v[0] for v in values] if kind == "image" else values
                    budgets = {
                        m.get("soft") or (280 if kind == "image" else 140)
                        for m in selected
                    }
                    assert len(budgets) == 1
                    request[key + "_kwargs"] = {
                        "max_soft_tokens": budgets.pop(),
                        "input_data_format": "channels_last",
                    }
                    if kind == "video":
                        request[key + "_kwargs"]["do_sample_frames"] = False
            official = processor(**request).to("cuda")
            for key in official:
                if (
                    isinstance(official[key], torch.Tensor)
                    and official[key].is_floating_point()
                ):
                    official[key] = official[key].bfloat16()
            assert torch.equal(official["input_ids"], ids)
            output = model(**official).last_hidden_state.float().mean(1)
            official_vectors.append(
                torch.nn.functional.normalize(output, dim=-1).cpu().numpy()[0]
            )
            current["record"] = True
    np.save(args.output / "vectors.npy", vectors)
    np.save(args.output / "official-vectors.npy", official_vectors)
    (args.output / "oracle.json").write_text(
        json.dumps(
            {
                "cases": len(cases),
                "frames": current["frame"],
                "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
                "dtype": "bfloat16",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
