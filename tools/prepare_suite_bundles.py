#!/usr/bin/env python3
"""Create pinned bundle specifications from the frozen T00 source lock.

Requires the existing local snapshots for small asset SHA-256 values. Weight
hashes come from the frozen published LFS digests and are checked again by the
bundle compiler. No weights are downloaded or converted by this command.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from suite_bundle import compile_bundle


def prepare(lock: Path, root: Path, output: Path):
    frozen = json.loads(lock.read_text())
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for model in frozen["published"]:
        repository = model["repository"]
        if repository not in ("Qwen/Qwen-Image-2.1-Turbo", "google/embeddinggemma-2"):
            continue
        revision = model["revision"]
        image = repository.startswith("Qwen/")
        components = []
        if image:
            assignments = [
                ("text_encoder", "text_encoder/", ""),
                ("dit", "transformer/", ""),
                ("vae", "vae/", ""),
            ]
        else:
            assignments = [
                ("text", "", "language_model."),
                ("vision", "", "vision_tower."),
                ("vision_projection", "", "embed_vision."),
                ("audio", "", "audio_tower."),
                ("audio_projection", "", "embed_audio."),
            ]
        files = []
        assets = {}
        for artifact in model["artifacts"]:
            name = artifact["path"]
            is_weight = name.endswith(".safetensors")
            # Qwen's processor template is part of conditioning. Gemma's
            # standalone chat template is unrelated to its embedding recipe.
            is_asset = (
                name.endswith((".json", ".model", ".txt"))
                or name.startswith("LICENSE")
                or (image and name.endswith(".jinja"))
            )
            if not is_weight and not is_asset:
                continue
            relative = repository + "/" + name
            path = root / relative
            digest = artifact.get("published_sha256") or artifact.get("local_sha256")
            if not digest:
                data = path.read_bytes()
                git_blob = hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data
                ).hexdigest()
                if artifact.get("git_blob") != git_blob:
                    raise ValueError(
                        f"asset differs from pinned source lock: {relative}"
                    )
                digest = hashlib.sha256(data).hexdigest()
            files.append(
                {
                    "path": relative,
                    "format": "safetensors" if is_weight else "asset",
                    "revision": revision,
                    "bytes": artifact["bytes"],
                    "sha256": digest,
                    "url": f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
                }
            )
            if is_asset:
                assets[name] = relative
        for identity, directory, prefix in assignments:
            selectors = [
                {
                    "file": file["path"],
                    "prefix": prefix,
                    "group_regex": r"(?:layers|blocks|transformer_blocks)\.(\d+)\.",
                }
                for file in files
                if file["format"] == "safetensors"
                and file["path"].removeprefix(repository + "/").startswith(directory)
            ]
            components.append(
                {
                    "id": identity,
                    "revision": revision,
                    "for_model_revision": revision,
                    "compute_dtype": "bfloat16",
                    "selectors": selectors,
                }
            )
        spec = {
            "version": 1,
            "model": repository,
            "revision": revision,
            "profile": "bf16-native-v1",
            "assets": assets,
            "components": components,
            "files": files,
        }
        if image:
            if (
                "scheduler/scheduler_config.json" not in assets
                or "model_index.json" not in assets
            ):
                raise ValueError("incomplete Qwen image recipe")
        elif not any("processor" in name for name in assets):
            raise ValueError("missing multimodal processor")
        filename = (
            "qwen-image-2.1-turbo-bf16.json" if image else "embeddinggemma-2-bf16.json"
        )
        (output / filename).write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n"
        )
        outputs.append((filename, spec))
    if len(outputs) != 2:
        raise ValueError("both frozen model revisions are required")
    original = next(spec for _, spec in outputs if spec["model"].startswith("Qwen/"))
    quantized = next(
        model
        for model in frozen["published"]
        if model["repository"] == "unsloth/Qwen-Image-2.1-Turbo-GGUF"
    )
    artifact = next(
        file
        for file in quantized["artifacts"]
        if file["path"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    )
    candidate = copy.deepcopy(original)
    candidate["profile"] = "dit-q4-k-m-bf16-compute-v1"
    candidate["files"] = [
        file
        for file in candidate["files"]
        if not (file["format"] == "safetensors" and "/transformer/" in file["path"])
    ]
    path = quantized["repository"] + "/" + artifact["path"]
    candidate["files"].append(
        {
            "path": path,
            "format": "gguf",
            "revision": quantized["revision"],
            "sha256": artifact["published_sha256"],
            "bytes": artifact["bytes"],
            "url": f"https://huggingface.co/{quantized['repository']}/resolve/{quantized['revision']}/{artifact['path']}",
        }
    )
    dit = next(
        component for component in candidate["components"] if component["id"] == "dit"
    )
    dit["revision"] = quantized["revision"]
    dit["selectors"] = [
        {"file": path, "group_regex": r"(?:blocks|transformer_blocks)\.(\d+)\."}
    ]
    filename = "qwen-image-2.1-turbo-dit-q4-k-m.json"
    (output / filename).write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    )
    outputs.append((filename, candidate))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/performance/native-suite-t00-20261020/models.json"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-output", type=Path)
    args = parser.parse_args()
    for filename, spec in prepare(args.lock, args.root, args.output):
        print(filename, len(spec["files"]), "artifacts", flush=True)
        if args.compile_output:
            missing = [
                file["path"]
                for file in spec["files"]
                if not (args.root / file["path"]).exists()
            ]
            if missing:
                print(filename, "NOT_VERIFIED: missing artifacts", missing, flush=True)
                continue
            result = compile_bundle(
                spec, args.root, args.compile_output / filename.removesuffix(".json")
            )
            print(filename, len(result["tensors"]), "tensors compiled", flush=True)


if __name__ == "__main__":
    main()
