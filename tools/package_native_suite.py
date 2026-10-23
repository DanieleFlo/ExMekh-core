"""Package the native suite and a clean-build source tree, without model weights.

A release package requires every mandatory qualification gate to be PASS.
--development preserves a reviewable package with its incomplete matrix.
No Python interpreter is required to launch the resulting native service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANDATORY = (
    "image_quality",
    "embedding_quality",
    "residency",
    "fault_recovery",
    "stress_50",
    "legacy_regressions",
    "clean_build",
)
SOURCE_SUFFIXES = {
    ".c",
    ".h",
    ".cu",
    ".cuh",
    ".inc",
    ".m",
    ".metal",
    ".S",
    ".s",
    ".hpp",
}


def release_ready(qualification):
    return all(qualification.get("gates", {}).get(gate) == "PASS" for gate in MANDATORY)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_runtime(args, output):
    for directory in ("lib", "assets", "config", "var", "source"):
        (output / directory).mkdir()
    shutil.copy2(args.binary, output / "ds4-server-suite")
    config = json.loads(args.config.read_text())
    config["media_directory"] = "var"
    for i, model in enumerate(config["models"]):
        directory = output / "assets" / str(i)
        directory.mkdir()
        for key in ("index", "text_asset", "media_asset"):
            if key not in model:
                continue
            source = ROOT / model[key]
            shutil.copy2(source, directory / source.name)
            model[key] = (directory / source.name).relative_to(output).as_posix()
        model["root"] = "models"
    (output / "config/native-suite.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    for directory in (
        args.media_prefix / "lib",
        args.media_prefix / "usr/lib/x86_64-linux-gnu",
    ):
        if directory.is_dir():
            for source in sorted(directory.glob("*.so*")):
                if source.is_file():
                    shutil.copy2(source, output / "lib" / source.name)
    # CUDA/cuDNN .so files are explicit native dependencies. The qualification
    # obtains them from wheels, but the launcher never starts Python.
    launcher = """#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export LD_LIBRARY_PATH="$PWD/lib${DS4_NATIVE_LIBDIR:+:$DS4_NATIVE_LIBDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec ./ds4-server-suite --suite-config config/native-suite.json "$@"
"""
    (output / "run-suite").write_text(launcher)
    os.chmod(output / "run-suite", 0o755)
    os.chmod(output / "ds4-server-suite", 0o755)


def source_files():
    """Enumerate build inputs before creating the staging directory."""
    for path in ROOT.iterdir():
        if path.is_file() and (
            path.suffix in SOURCE_SUFFIXES
            or path.name in ("Makefile", "LICENSE", "AGENT.md", "PROJECT_INDEX.md")
        ):
            yield path
    for name in ("cuda", "metal", "gguf-tools", "third_party"):
        for path in (ROOT / name).rglob("*"):
            relative = path.relative_to(ROOT / name)
            if name == "gguf-tools" and relative.parts[:2] == (
                "quality-testing",
                "staging",
            ):
                continue
            if path.is_file() and (
                path.suffix in SOURCE_SUFFIXES
                or (
                    path.name.lower().startswith(("license", "copying"))
                    and path.suffix.lower()
                    in (
                        "",
                        ".txt",
                        ".md",
                        ".rst",
                        ".gpl",
                        ".lgpl",
                        ".lesser",
                        ".lib",
                        ".bsd",
                        ".apache",
                    )
                )
            ):
                yield path


def copy_documentation(args, output):
    for source in (args.qualification, args.candidates):
        shutil.copy2(source, output / source.name)
    for source in (ROOT / "config/suite-bundles").glob("*.json"):
        shutil.copy2(source, output / "config" / source.name)
    for source in (
        ROOT / "docs/architecture/native-suite-release.md",
        ROOT / "docs/architecture/native-suite-client.md",
    ):
        if source.is_file():
            shutil.copy2(source, output / source.name)


def write_manifest(output, qualification, dependencies):
    ready = release_ready(qualification)
    (output / "native-dependencies.txt").write_text(dependencies)
    manifest = {
        "version": 1,
        "release_qualified": ready,
        "kind": "release" if ready else "development",
        "model_weights_included": False,
        "python_runtime_required": False,
        "qualification": qualification,
        "external_native_libraries": sorted(
            set(re.findall(r"\b(lib(?:cuda|cublas|cudnn)[^\s]+)\s+=>", dependencies))
        ),
        "cudnn_split_libraries": "same installation/revision as libcudnn.so.9",
        "files": [
            {
                "path": p.relative_to(output).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(output.rglob("*"))
            if p.is_file()
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def package(args):
    qualification = json.loads(args.qualification.read_text())
    if not release_ready(qualification) and not args.development:
        raise ValueError(
            "release blocked: mandatory gates are incomplete; use --development for a labelled review package"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    dependencies = subprocess.run(
        ["ldd", str(args.binary.resolve())], check=True, capture_output=True, text=True
    ).stdout
    if "not found" in dependencies:
        raise ValueError("native library dependency is unresolved")
    sources = list(source_files())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publishing is one rename. Failed copies cannot masquerade as a package
    # or force the user to delete an incomplete output before retrying.
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}-", dir=args.output.parent
    ) as temporary:
        staging = Path(temporary)
        copy_runtime(args, staging)
        for source in sources:
            target = staging / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copy_documentation(args, staging)
        manifest = write_manifest(staging, qualification, dependencies)
        staging.rename(args.output)
    print(
        json.dumps(
            {
                "package": str(args.output),
                "kind": manifest["kind"],
                "files": len(manifest["files"]),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "ds4-server-suite")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/native-suite.example.json"
    )
    parser.add_argument("--media-prefix", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
