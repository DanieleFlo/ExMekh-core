#!/usr/bin/env python3
"""Export reviewable T00 manifests, excluding private worktree/process dumps."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(args.inputs.read_text())

    def read(key, name):
        return json.loads((Path(inputs[key]) / name).read_text())

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    frozen = read("freeze", "baseline.json")
    prepared = read("prepared", "prepared.json")
    write(
        "sources.json",
        {
            "sources": frozen["sources"],
            "head": frozen["local"]["head"]["stdout"].strip(),
            "baseline_c_sources": frozen["local"]["sources"],
            "binaries": frozen["local"]["binaries"],
            "local_weights": frozen["local"]["weights"],
        },
    )
    # Early preparation runs may predate Jinja inclusion. Inventory any pinned
    # supplemental template from the same model directory rather than omitting
    # a processor dependency or pretending it had a published LFS digest.
    for model in prepared["models"]:
        directory = Path(inputs["prepared"]) / "models" / model["repository"]
        existing = {f["path"] for f in model["files"]}
        for path in directory.rglob("*.jinja"):
            relative = path.relative_to(directory).as_posix()
            if relative not in existing:
                model["files"].append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "supplemental_pinned_template": True,
                    }
                )
    write(
        "models.json",
        {
            "published": frozen["models"],
            "locally_verified": prepared["models"],
            "reference_archives": prepared["sources"],
        },
    )
    write("tensors.json", read("headers", "tensor-inventory.json"))
    write("gemma-oracle.json", read("gemma", "oracle.json"))
    if "gemma_f32" in inputs:
        write("gemma-f32-oracle.json", read("gemma_f32", "oracle.json"))
    write("qwen-oracle.json", read("qwen", "oracle.json"))
    write("retrieval.json", read("retrieval", "retrieval.json"))
    write("noise.json", read("noise", "noise.json"))
    write("qwen27-baseline.json", read("qwen27", "model-runs.json"))
    write("qwen27-candidate.json", read("candidate", "model-runs.json"))
    flash = read("flash", "flash.json")
    write("flash.json", flash)
    if "payload_probe" in flash:
        write("flash-payload.json", flash["payload_probe"])
    hardware = frozen["local"]["hardware"]
    gpu = next(h["stdout"] for h in hardware if h["command"][0] == "nvidia-smi")

    def field(label):
        match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*(.+)$", gpu, re.M)
        return match[1].strip() if match else None

    write(
        "hardware.json",
        {
            "gpu": field("Product Name"),
            "driver": field("Driver Version"),
            "toolkit": next(
                h["stdout"] for h in hardware if h["command"][0].endswith("nvcc")
            ),
            "meminfo_at_freeze": frozen["local"]["meminfo"],
            "gpu_fb_memory_at_freeze": re.search(
                r"FB Memory Usage\s*\n([\s\S]*?)BAR1 Memory Usage", gpu
            )[1].strip(),
            "pinned_probe_bytes": 67108864,
            "initial_pinned_budget_bytes": 268435456,
            "legacy_pinned_staging": {
                "slots": 4,
                "bytes_per_slot": 67108864,
                "total_bytes": 268435456,
                "evidence": "runtime CUDA test: 4 x 64 MiB pinned staging",
            },
            "pinned_maximum": "NOT_VERIFIED; measured allocations are not a physical maximum",
            "pcie": read("freeze", "pcie.json"),
            "ssd": read("freeze", "ssd.json"),
        },
    )
    provenance = []
    labels = {
        "huggingface/transformers": ("LICENSE", "Apache-2.0"),
        "huggingface/diffusers": ("LICENSE", "Apache-2.0"),
        "ggml-org/llama.cpp": ("LICENSE", "MIT"),
        "unslothai/unsloth": ("LICENSE", "Apache-2.0"),
        "leejet/stable-diffusion.cpp": ("LICENSE", "MIT"),
        "Comfy-Org/ComfyUI": ("LICENSE", "GPL-3.0"),
        "sgl-project/sglang": ("LICENSE", "Apache-2.0"),
        "vllm-project/vllm-omni": ("LICENSE", "Apache-2.0"),
    }
    for source in frozen["sources"]:
        file, license_id = labels[source["repository"]]
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{file}"
        with urllib.request.urlopen(url, timeout=60) as response:
            license_text = response.read(128 * 1024)
        record = {
            "repository": source["repository"],
            "revision": source["revision"],
            "license_file": url,
            "license_sha256": hashlib.sha256(license_text).hexdigest(),
            "repository_license": license_id,
            "imported_code": False,
            "files": [],
        }
        for value in source["files"]:
            declared = (
                "AGPL-3.0-only"
                if source["repository"] == "unslothai/unsloth"
                and value["path"].startswith("studio/")
                else license_id
            )
            if (
                source["repository"] == "Comfy-Org/ComfyUI"
                and "qwen_image21" in value["path"]
            ):
                declared = "Apache-2.0 header; surrounding integration GPL-3.0"
            record["files"].append(dict(value, declared_license=declared, ported=False))
        provenance.append(record)
    write(
        "provenance.json",
        {
            "sources": provenance,
            "new_runtime": "Original C implementation; no external kernel port in T00-T02",
            "weights": [
                {
                    "model": "Qwen/Qwen-Image-2.1-Turbo",
                    "declared_license": "qwen-research",
                    "license_path": "LICENSE",
                },
                {
                    "model": "google/embeddinggemma-2",
                    "declared_license": "Apache-2.0",
                    "license_path": "README.md front matter",
                },
            ],
        },
    )
    artifacts = []
    for key in ("gemma", "qwen", "retrieval", "noise", "gemma_f32"):
        if key not in inputs:
            continue
        for path in sorted(Path(inputs[key]).glob("*.npz")):
            artifacts.append(
                {
                    "experiment": key,
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    write("artifacts.json", {"inputs": inputs, "arrays": artifacts})
    print("exported", args.output)


if __name__ == "__main__":
    main()
