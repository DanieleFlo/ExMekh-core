#!/usr/bin/env python3
"""Freeze source/model identities without downloading model weights.

Run from Linux/WSL. Remote LFS SHA-256 values describe the published artifacts;
they are deliberately distinguished from hashes computed on local bytes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from suite_artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "huggingface/transformers": [
        "src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py",
        "src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py",
        "src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
    ],
    "huggingface/diffusers": [
        "src/diffusers/models/transformers/transformer_qwenimage21.py",
        "src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py",
        "src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py",
        "src/diffusers/hooks/group_offloading.py",
    ],
    "ggml-org/llama.cpp": [
        "src/models/gemma-embedding2.cpp",
        "conversion/gemma.py",
        "ggml/src/ggml-cuda/mmq.cu",
        "ggml/src/ggml-cuda/fattn-tile.cuh",
    ],
    "unslothai/unsloth": [
        "unsloth/kernels/rms_layernorm.py",
        "unsloth/kernels/geglu.py",
        "studio/backend/core/inference/diffusion.py",
        "studio/backend/core/inference/diffusion_memory.py",
    ],
    "leejet/stable-diffusion.cpp": ["docs/qwen_image_2.1.md"],
    "Comfy-Org/ComfyUI": [
        "comfy/ldm/qwen_image21/model.py",
        "comfy/text_encoders/qwen_image21.py",
    ],
    "sgl-project/sglang": [
        "python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py",
    ],
    "vllm-project/vllm-omni": ["README.md"],
}
MODELS = [
    "Qwen/Qwen-Image-2.1-Turbo",
    "google/embeddinggemma-2",
    "unsloth/Qwen-Image-2.1-Turbo-GGUF",
    "unsloth/Qwen3-VL-8B-Instruct-GGUF",
]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DS4-baseline-audit"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def command(*args: str) -> dict:
    try:
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=90
        )
        return {
            "command": list(args),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(args), "error": str(exc)}


def freeze_source(repo: str, paths: list[str], directory: Path) -> dict:
    branch = json.loads(fetch(f"https://api.github.com/repos/{repo}"))["default_branch"]
    commit = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    revision = commit["sha"]
    files = []
    for path in paths:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        data = fetch(url)
        target = directory / "sources" / repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(
            {
                "path": path,
                "url": url,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "repository": repo,
        "revision": revision,
        "branch_at_capture": branch,
        "commit_date": commit["commit"]["committer"]["date"],
        "files": files,
    }


def freeze_model(repo: str, directory: Path) -> dict:
    info = json.loads(fetch(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    revision = info["sha"]
    artifacts = []
    for entry in info["siblings"]:
        path = entry["rfilename"]
        lfs = entry.get("lfs") or {}
        item = {
            "path": path,
            "bytes": entry.get("size"),
            "git_blob": entry.get("blobId"),
            "published_sha256": lfs.get("sha256"),
            "local_sha256": None,
        }
        # Read metadata/tokenizers, not weights, assets or executable remote code.
        if (
            path.endswith((".json", ".txt", ".model", ".jinja"))
            and (entry.get("size") or 0) < 32 * 1024**2
        ):
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
            data = fetch(url)
            target = directory / "models" / repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            item.update(
                local_sha256=hashlib.sha256(data).hexdigest(),
                local_bytes=len(data),
                url=url,
            )
        artifacts.append(item)
    return {
        "repository": repo,
        "revision": revision,
        "artifacts": artifacts,
        "weight_downloaded": False,
    }


def fetch_range(url: str, start: int, length: int) -> bytes:
    """Refuse an ignored Range header before reading a multi-GB weight body."""
    end = start + length - 1
    request = urllib.request.Request(
        url + f"?ds4_header={start}-{end}",
        headers={"User-Agent": "DS4-baseline-audit", "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise ValueError(
                f"server did not honor byte range: {response.status} {content_range}"
            )
        result = response.read(length + 1)
    if len(result) != length:
        raise ValueError("incomplete or excessive tensor header response")
    return result


def tensor_headers(manifest: dict, directory: Path) -> dict:
    inventories = []
    for model in manifest["models"]:
        if model["repository"] not in MODELS[:2]:
            continue
        files = []
        for artifact in model["artifacts"]:
            if not artifact["path"].endswith(".safetensors"):
                continue
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{artifact['path']}"
            size = struct.unpack("<Q", fetch_range(url, 0, 8))[0]
            if not 2 <= size <= 16 * 1024**2:
                raise ValueError(f"invalid safetensors header size: {size}")
            raw = fetch_range(url, 8, size)
            header = json.loads(raw)
            tensors = {
                name: value for name, value in header.items() if name != "__metadata__"
            }
            for name, tensor in tensors.items():
                if (
                    not isinstance(tensor.get("dtype"), str)
                    or not all(
                        isinstance(n, int) and n >= 0 for n in tensor.get("shape", [])
                    )
                    or len(tensor.get("data_offsets", [])) != 2
                ):
                    raise ValueError(f"invalid tensor directory: {name}")
                begin, end = tensor["data_offsets"]
                if not 0 <= begin <= end <= artifact["bytes"] - size - 8:
                    raise ValueError(f"tensor outside artifact: {name}")
            target = (
                directory / model["repository"] / (artifact["path"] + ".header.json")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files.append(
                {
                    "path": artifact["path"],
                    "header_sha256": hashlib.sha256(raw).hexdigest(),
                    "header_bytes": size,
                    "tensor_count": len(tensors),
                    "dtypes": sorted({v["dtype"] for v in tensors.values()}),
                    "tensors": tensors,
                }
            )
            print(model["repository"], artifact["path"], len(tensors), flush=True)
        inventories.append(
            {
                "repository": model["repository"],
                "revision": model["revision"],
                "files": files,
            }
        )
    return {
        "schema_version": 1,
        "weight_payload_downloaded": False,
        "models": inventories,
    }


def snapshot_local(directory: Path, hash_weights: bool) -> dict:
    tracked = command("git", "ls-files", "-z")
    if tracked.get("exit_code") != 0:
        raise RuntimeError(tracked)
    paths = [ROOT / name for name in tracked["stdout"].split("\0") if name]
    files = []
    for path in paths:
        if path.is_file() and path.suffix in {
            ".c",
            ".h",
            ".cu",
            ".cuh",
            ".m",
            ".metal",
            ".inc",
        }:
            files.append(
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            )
    binaries = []
    for name in (
        "ds4",
        "ds4-server",
        "ds4-bench",
        "ds4-eval",
        "tests/test_server_agentic_payload",
    ):
        source = ROOT / name
        if source.is_file():
            target = directory / "binaries" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            binaries.append(
                {
                    "path": name,
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                }
            )
    weights = []
    for path in sorted((ROOT / "gguf").glob("*.gguf")):
        weights.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path) if hash_weights else None,
                "hash_status": "COMPUTED" if hash_weights else "NOT_VERIFIED",
            }
        )
    hardware = [
        command("uname", "-a"),
        command("nvidia-smi", "-q"),
        command("/usr/local/cuda/bin/nvcc", "--version"),
        command("free", "-b"),
        command("ps", "-eo", "pid,comm,args"),
    ]
    return {
        "head": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--short"),
        "diff": command("git", "diff", "--binary"),
        "sources": files,
        "binaries": binaries,
        "weights": weights,
        "hardware": hardware,
        "meminfo": Path("/proc/meminfo").read_text()
        if sys.platform == "linux"
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument(
        "--headers-from",
        type=Path,
        help="Read frozen revisions and inspect safetensors headers only",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.headers_from:
        inventory = tensor_headers(
            json.loads(args.headers_from.read_text()), args.output
        )
        (args.output / "tensor-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        return 0
    manifest = {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "local": snapshot_local(args.output, args.hash_weights),
        "sources": [],
        "models": [],
    }
    (args.output / "baseline.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.remote:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                ("sources", pool.submit(freeze_source, repo, paths, args.output))
                for repo, paths in SOURCES.items()
            ]
            futures += [
                ("models", pool.submit(freeze_model, repo, args.output))
                for repo in MODELS
            ]
            for category, future in futures:
                try:
                    result = future.result()
                    manifest[category].append(result)
                    print(result["repository"], result["revision"], flush=True)
                except Exception as exc:
                    manifest.setdefault("errors", []).append(str(exc))
                (args.output / "baseline.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
    return 1 if manifest.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
