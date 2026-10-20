#!/usr/bin/env python3
"""Prepare pinned, isolated development references; never used by DS4 serving.

Run with a development Python containing huggingface_hub. The frozen manifest
selects revisions. No package in the caller's environment is installed/changed.
"""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.manifest.read_text())
    report = {"sources": [], "models": [], "production_dependency": False}
    for source in frozen["sources"]:
        if source["repository"] not in {
            "huggingface/transformers",
            "huggingface/diffusers",
        }:
            continue
        url = f"https://codeload.github.com/{source['repository']}/tar.gz/{source['revision']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(128 * 1024**2 + 1)
        if len(data) > 128 * 1024**2:
            raise ValueError("source archive exceeds limit")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(args.output / "sources", filter="data")
        report["sources"].append(
            {
                "repository": source["repository"],
                "revision": source["revision"],
                "url": url,
                "archive_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        print(source["repository"], "prepared", flush=True)
    from huggingface_hub import snapshot_download

    for model in frozen["models"]:
        if model["repository"] != "google/embeddinggemma-2" and not (
            args.qwen and model["repository"] == "Qwen/Qwen-Image-2.1-Turbo"
        ):
            continue
        path = args.output / "models" / model["repository"]
        snapshot_download(
            model["repository"],
            revision=model["revision"],
            local_dir=path,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.safetensors",
                "*.txt",
                "*.jinja",
                "README.md",
                "LICENSE*",
            ],
            max_workers=2,
        )
        artifacts = []
        for artifact in model["artifacts"]:
            file = path / artifact["path"]
            if not file.is_file():
                continue
            with file.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            expected = artifact.get("published_sha256") or artifact.get("local_sha256")
            if expected and digest != expected:
                raise ValueError(f"artifact SHA-256 mismatch: {file}")
            artifacts.append(
                {
                    "path": artifact["path"],
                    "sha256": digest,
                    "bytes": file.stat().st_size,
                }
            )
        report["models"].append(
            {
                "repository": model["repository"],
                "revision": model["revision"],
                "files": artifacts,
            }
        )
        print(model["repository"], "verified", flush=True)
    (args.output / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
