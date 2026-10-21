#!/usr/bin/env python3
"""Frozen development oracle for the native Gemma text adapter (never serving).

Saves complete token/layer/projection arrays, plus padded-batch/single checks.
The JSON cases contain text and task (0 raw, 1 query, 2 document, 3 code, 4 STS).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((a.prepared / "prepared.json").read_text())
    for s in lock["sources"]:
        if s["repository"] == "huggingface/transformers":
            sys.path.insert(
                0,
                str(
                    (
                        a.prepared / "sources" / f"transformers-{s['revision']}" / "src"
                    ).resolve()
                ),
            )
    sys.path.insert(0, str((a.prepared / "deps").resolve()))
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    modelpath = a.prepared / "models/google/embeddinggemma-2"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
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
    tasks = json.loads((modelpath / "config_sentence_transformers.json").read_text())[
        "prompts"
    ]
    prefixes = [
        "",
        *(
            tasks[k]
            for k in ("query", "document", "CodeRetrieval", "SentenceSimilarity")
        ),
    ]
    items = json.loads(a.cases.read_text())
    current = {"item": 0}

    def dump(stage, layer, value):
        path = a.output / f"item-{current['item']}-{stage}-{layer}.bf16"
        value.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(path)

    model.language_model.embed_tokens.register_forward_hook(
        lambda m, args, o: dump("embeddings", 0, o)
    )
    model.language_model.ple.register_forward_hook(lambda m, args, o: dump("ple", 0, o))
    for i, layer in enumerate(model.language_model.layers):
        layer.register_forward_hook(lambda m, args, o, i=i: dump("layer", i, o))
    model.language_model.embedding_projection.register_forward_hook(
        lambda m, args, o: dump("projected", 24, o)
    )
    vectors = []
    formatted = []
    with torch.inference_mode():
        for i, item in enumerate(items):
            current["item"] = i
            text = item["text"]
            task = item.get("task", 0)
            if not (
                text.startswith("task: ")
                or (text.startswith("title: ") and " | text: " in text)
            ):
                text = prefixes[task] + text
            formatted.append(text)
            inputs = tokenizer(text, return_tensors="pt")
            inputs["input_ids"].numpy().astype("<u4").tofile(
                a.output / f"item-{i}-tokens.u32"
            )
            out = model(**inputs.to("cuda")).last_hidden_state.float()
            pooled = out.mean(1)
            vectors.append(
                torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()[0]
            )
    vectors = np.array(vectors)
    np.save(a.output / "vectors.npy", vectors)
    # Remove recording hooks before the padded batch reference.
    for module in model.modules():
        module._forward_hooks.clear()
    padded = []
    with torch.inference_mode():
        for start in range(0, len(formatted), 8):
            ins = tokenizer(
                formatted[start : start + 8], padding=True, return_tensors="pt"
            ).to("cuda")
            output = model(**ins).last_hidden_state.float()
            mask = ins["attention_mask"].unsqueeze(-1).float()
            pooled = (output * mask).sum(1) / mask.sum(1)
            padded.extend(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    np.save(a.output / "padded-vectors.npy", padded)
    summary = {
        "cases_sha256": hashlib.sha256(a.cases.read_bytes()).hexdigest(),
        "cases": len(items),
        "padded_single_max_abs": float(np.max(np.abs(vectors - np.array(padded)))),
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "dtype": "bfloat16",
    }
    (a.output / "oracle.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
