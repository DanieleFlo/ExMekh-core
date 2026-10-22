#!/usr/bin/env python3
"""Freeze Gemma vectors and bidirectional retrieval scores on the fixed corpus.

Development oracle only. This measures synthetic regression fixtures, not
real-world benchmark quality. It never modifies the corpus or golden scores.
"""

import argparse
import hashlib
import json
import math
import sys
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    for source in prepared["sources"]:
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
    import soxr
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.matmul.allow_tf32 = False
    # The native BF16 profile accumulates GEMMs in FP32. Record this explicitly;
    # the historical T00 r2 run inherited PyTorch's reduced-precision default.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.corpus.parent
    corpus = json.loads(args.corpus.read_text())
    for item in corpus["items"]:
        artifact = item.get("artifact")
        if (
            artifact
            and hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
            != artifact["sha256"]
        ):
            raise ValueError(f"corpus artifact changed: {item['id']}")
    path = args.prepared / "models/google/embeddinggemma-2"
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    cache = {}
    tokens = {}

    @torch.inference_mode()
    def encode(content):
        key = json.dumps(content, sort_keys=True)
        if key in cache:
            return cache[key]
        request = {}
        text = []
        images = []
        audio = []
        videos = []
        for part in content:
            kind = part["type"]
            if kind == "text":
                text.append(part["text"])
            elif kind == "image":
                images.append(Image.open(root / part["path"]).convert("RGB"))
                text.append("<|image|>")
            elif kind == "audio":
                # Canonical development decode/resample; implementation profile
                # must reproduce this or qualify a separately named processor.
                with wave.open(str(root / part["path"]), "rb") as wav:
                    if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                        raise ValueError("corpus requires mono PCM16")
                    rate = wav.getframerate()
                    data = (
                        np.frombuffer(
                            wav.readframes(wav.getnframes()), dtype="<i2"
                        ).astype(np.float32)
                        / 32768.0
                    )
                if rate != 16000:
                    data = soxr.resample(data, rate, 16000, quality="HQ")
                audio.append(data)
                text.append("<|audio|>")
            elif kind == "video":
                with np.load(root / part["path"]) as data:
                    fps = int(data["fps"])
                    videos.append(data["frames"][::fps].copy())
                text.append("<|video|>")
            else:
                raise ValueError(kind)
        request["text"] = " ".join(text)
        if images:
            request["images"] = images
        if audio:
            request.update(audio=audio, sampling_rate=16000)
        if videos:
            request.update(videos=videos, do_sample_frames=False)
        inputs = processor(**request, return_tensors="pt").to("cuda")
        for name in inputs:
            if (
                isinstance(inputs[name], torch.Tensor)
                and inputs[name].is_floating_point()
            ):
                inputs[name] = inputs[name].to(torch.bfloat16)
        count = int(inputs["attention_mask"].sum())
        if count > 8192:
            raise ValueError("oracle fixture exceeds operational context")
        hidden = model(**inputs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))[0].cpu().numpy()
        if not np.isfinite(pooled).all() or not np.linalg.norm(pooled):
            raise ValueError("invalid embedding")
        cache[key] = pooled
        tokens[key] = count
        return pooled

    vectors = {}
    for item in corpus["items"]:
        vectors[item["id"]] = encode(item["content"])
    metrics = {}
    rankings = {}
    for query in corpus["retrieval"]:
        link = query["link"]
        split = query["split"]
        if link.startswith("text_to_"):
            modality = link.removeprefix("text_to_")
            candidates = [
                i
                for i in corpus["items"]
                if i["split"] == split and i["id"].startswith(modality + "-")
            ]
            ids = [i["id"] for i in candidates]
            document_vectors = np.stack([vectors[i] for i in ids])
            query_vector = encode(
                [
                    {
                        "type": "text",
                        "text": "task: search result | query: " + query["text"],
                    }
                ]
            )
            relevant = set(query["relevant_ids"])
        else:
            modality = link.removesuffix("_to_text")
            candidates = [
                i
                for i in corpus["items"]
                if i["split"] == split and i["id"].startswith(modality + "-")
            ]
            ids = [i["id"] for i in candidates]
            document_vectors = np.stack(
                [
                    encode(
                        [
                            {
                                "type": "text",
                                "text": "title: none | text: " + i["caption"],
                            }
                        ]
                    )
                    for i in candidates
                ]
            )
            query_vector = vectors[query["input_id"]]
            relevant = {
                i["id"] for i in candidates if i["caption"] == query["relevant_text"]
            }
        for dimension in (128, 256, 512, 768):
            q = query_vector[:dimension]
            d = document_vectors[:, :dimension]
            scores = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))
            order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
            ranked = [ids[i] for i in order]
            dcg = sum(
                1 / math.log2(i + 2)
                for i, identity in enumerate(ranked[:10])
                if identity in relevant
            )
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
            key = f"{split}/{link}/{dimension}"
            metrics.setdefault(key, []).append(
                {
                    "recall_at_1": float(ranked[0] in relevant),
                    "recall_at_5": len(set(ranked[:5]) & relevant) / len(relevant),
                    "ndcg_at_10": dcg / ideal,
                }
            )
            rankings[f"{query['id']}/{dimension}"] = ranked
    report = {
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256(
            (args.prepared / "prepared.json").read_bytes()
        ).hexdigest(),
        "profile": "bf16-eager-fp32-reduction-mean-f32-l2",
        "numerics": {
            "bf16_reduced_precision_reduction": False,
            "matmul_tf32": False,
            "cudnn_tf32": False,
        },
        "decode": "wave mono PCM16, soxr HQ 16kHz; video frames selected at 1fps",
        "torch": torch.__version__,
        "soxr": soxr.__version__,
        "numpy": np.__version__,
        "items": len(vectors),
        "unique_processor_inputs": len(cache),
        "max_tokens": max(tokens.values()),
        "metrics": {
            k: {name: sum(x[name] for x in values) / len(values) for name in values[0]}
            for k, values in metrics.items()
        },
        "rankings": rankings,
        "quality_scope": corpus["quality_scope"],
    }
    for modality in ("image", "audio", "video"):
        subset = [v for k, v in vectors.items() if k.startswith(modality + "-")]
        report.setdefault("media_sensitivity", {})[modality] = len(
            {v.tobytes() for v in subset}
        )
        if report["media_sensitivity"][modality] < 2:
            raise ValueError(f"model insensitive to {modality}")
    np.savez(args.output / "vectors.npz", **vectors)
    (args.output / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            "items": len(vectors),
            "metric_groups": len(metrics),
            "media_sensitivity": report["media_sensitivity"],
        }
    )


if __name__ == "__main__":
    main()
