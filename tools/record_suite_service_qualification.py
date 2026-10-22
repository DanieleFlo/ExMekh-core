"""Record local T17/T18 evidence after the explicit acceptance commands pass."""

import argparse
import base64
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path
from zipfile import ZipFile

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/performance/native-suite-t17-t18-20261022"
ARTIFACTS = ROOT / "performance-results/native-suite-t17-t18-20261022"
LOGS = {
    "build_protocol": "/tmp/ds4-t17-final-build.log",
    "protocol": "/tmp/ds4-suite-protocol-final.log",
    "host": "/tmp/ds4-suite-host-final.log",
    "legacy_server": "/tmp/ds4-suite-server-final.log",
    "cpu_regression": "/tmp/ds4-suite-cpu-final.log",
    "openai_live": "/tmp/ds4-suite-final-acceptance.log",
    "exmekh_live": "/tmp/ds4-t18-exmekh-live.log",
    "library_offline": "/tmp/ds4-t18-lib-full.log",
    "frontend": "/tmp/ds4-t18-native-browser.log",
    "history": "/tmp/ds4-t18-history.log",
    "types": "/tmp/ds4-t18-types.log",
    "lint": "/tmp/ds4-t18-lint.log",
    "frontend_lint": "/tmp/ds4-t18-agent-lint.log",
    "docs": "/tmp/ds4-t17-docs-check.log",
    "product_text": "/tmp/ds4-t17-product-text.log",
    "wheel": "/tmp/ds4-t18-wheel.log",
    "wheel_install": "/tmp/ds4-t18-wheel-install.log",
    "native_server": "/tmp/ds4-suite-final-live.log",
    "extended_before_fix": "/tmp/ds4-suite-extended-before-fix.log",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(directory, *args, codes=(0,)):
    result = subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=directory,
        capture_output=True,
        check=False,
    )
    if result.returncode not in codes:
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def changed_files(directory):
    names = git(
        directory, "ls-files", "--modified", "--others", "--exclude-standard", "-z"
    )
    return sorted({name.decode() for name in names.split(b"\0") if name})


def inventory(directory):
    return {
        name: sha((directory / name).read_bytes())
        for name in changed_files(directory)
        if (directory / name).is_file()
        and not name.startswith("docs/performance/native-suite-t17-t18-20261022/")
    }


def receipt_summary(path):
    raw = path.read_bytes()
    response = json.loads(raw)
    items = []
    for row in response["data"]:
        if "b64_json" in row:
            image = base64.b64decode(row["b64_json"], validate=True)
            items.append(
                {
                    "png_bytes": len(image),
                    "png_sha256": sha(image),
                    "metadata": row["ds4"],
                }
            )
        elif "embedding" in row:
            values = row["embedding"]
            packed = struct.pack(f"<{len(values)}f", *values)
            items.append(
                {
                    "index": row["index"],
                    "dimensions": len(values),
                    "l2_norm": sum(v * v for v in values) ** 0.5,
                    "float32_sha256": sha(packed),
                }
            )
    return {
        "response_bytes": len(raw),
        "response_sha256": sha(raw),
        "model": response.get("model"),
        "metadata": response["ds4"],
        "items": items,
    }


def verify_wheel(external):
    wheel = ROOT / "agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl"
    lock = tomllib.loads((ROOT / "agent/uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "exmekh-lib")
    if package["wheels"][0]["hash"] != "sha256:" + sha(wheel.read_bytes()):
        raise RuntimeError("Wheel does not match the frontend lock")
    # Check the complete package even after its source repository is committed.
    names = git(
        external,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src/exmekh_lib",
    )
    sources = sorted(
        {
            p.decode()
            for p in names.split(b"\0")
            if p and (external / p.decode()).is_file()
        }
    )
    installed = ROOT / "agent/.venv/lib/python3.13/site-packages"
    with ZipFile(wheel) as archive:
        packaged = {
            name
            for name in archive.namelist()
            if name.startswith("exmekh_lib/") and not name.endswith("/")
        }
        if packaged != {name.removeprefix("src/") for name in sources}:
            raise RuntimeError("Wheel package inventory differs from source")
        for source in sources:
            name = source.removeprefix("src/")
            content = (external / source).read_bytes()
            if (
                archive.read(name) != content
                or (installed / name).read_bytes() != content
            ):
                raise RuntimeError(f"Wheel/installed source mismatch: {source}")
    return {
        "status": "PASS",
        "package_files_checked": len(sources),
        "lock_matches": True,
    }


def write_external_patch(directory, destination, base="HEAD"):
    """Snapshot tracked and new sources, checking that the patch matches them."""
    patch = git(directory, "diff", "--binary", base)
    untracked = git(directory, "ls-files", "--others", "--exclude-standard", "-z")
    for name in sorted(n.decode() for n in untracked.split(b"\0") if n):
        patch += git(
            directory,
            "diff",
            "--no-index",
            "--binary",
            "--",
            "/dev/null",
            name,
            codes=(0, 1),
        )
    destination.write_bytes(patch)
    git(directory, "apply", "--reverse", "--check", str(destination))
    return patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-directory", type=Path, required=True)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "logs").mkdir(parents=True, exist_ok=True)
    logs = {}
    for name, source in LOGS.items():
        raw = Path(source).read_bytes()
        text = raw.decode(errors="replace")
        destination = ARTIFACTS / "logs" / f"{name}.log"
        shutil.copyfile(source, destination)
        entry = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha(raw),
            "bytes": len(raw),
            "tail": text.splitlines()[-8:],
        }
        match = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
        if match:
            entry.update(
                tests=int(match[1]),
                seconds=float(match[2]),
                status="PASS" if text.rstrip().endswith("OK") else "FAIL",
            )
        match = re.search(
            r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) deselected)? in ([\d.]+)s",
            text,
        )
        if match:
            entry.update(
                tests=int(match[1]),
                skipped=int(match[2] or 0),
                deselected=int(match[3] or 0),
                seconds=float(match[4]),
                status="FAIL" if re.search(r"\d+ failed", text) else "PASS",
            )
        logs[name] = entry
    required = (
        "cpu_regression",
        "openai_live",
        "exmekh_live",
        "library_offline",
        "frontend",
        "history",
    )
    if any(logs[name].get("status") != "PASS" for name in required):
        raise RuntimeError("Final acceptance logs are incomplete or failing")
    external = ROOT / "exmekh_lib"
    patch_path = REPORT / "exmekh-lib.patch"
    patch = write_external_patch(external, patch_path)
    artifact_paths = [
        ROOT / "ds4-server-suite",
        ROOT / "agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl",
        ROOT / "config/native-suite.example.json",
    ]
    config = json.loads(artifact_paths[-1].read_text())
    for model in config["models"]:
        artifact_paths.extend(
            ROOT / model[key]
            for key in ("index", "text_asset", "media_asset")
            if key in model
        )
    result = {
        "schema_version": 1,
        "date": "2026-10-22",
        "tasks": {"T17": "PASS_LOCAL", "T18": "PASS_LOCAL"},
        "scope": "Ubuntu 24.04 WSL / RTX 3090 sm86 / serial CUDA / BF16 native suite / Qwen3.8-27B Q4_K_S no-MTP ctx2048",
        "root_base_commit": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "external_dependency": {
            "path": "exmekh_lib",
            "base_commit": git(external, "rev-parse", "HEAD").decode().strip(),
            "patch": str(patch_path.relative_to(ROOT)),
            "patch_sha256": sha(patch),
            "sources": inventory(external),
        },
        "wheel_source_check": verify_wheel(external),
        "source_sha256": inventory(ROOT),
        "artifact_sha256": {
            str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in artifact_paths
        },
        "logs": logs,
        "committed_response_samples": [
            receipt_summary(p) for p in sorted(args.media_directory.glob("rcp-*"))
        ],
        "limits": [
            "Service transport acceptance does not close T11 image quality or T19-T22 release gates.",
            "No Metal/ROCm/TP/distributed/SSD suite qualification; legacy compile-time path remains separate.",
            "No hardware device-loss or HTTP restore-failure injection in this task; executor host recovery faults are tested.",
            "PNG only. References preserve supplied aligned geometry; implicit resizing is rejected.",
            "Receipt durability is process-local for 600 seconds. No cross-restart replay guarantee.",
            "Transfer counters cover native host/device copy calls, excluding kernels and legacy model restoration.",
            "Gemma separate load/phase timings remain null. Wall times are descriptive, no speedup claim.",
        ],
    }
    (REPORT / "verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        f"Recorded {len(result['source_sha256'])} root sources and {len(result['external_dependency']['sources'])} external sources"
    )


if __name__ == "__main__":
    main()
