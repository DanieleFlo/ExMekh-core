"""Record the real image tool/browser follow-up without rewriting T17/T18 evidence."""

import json
import re
import shutil
from pathlib import Path

from record_suite_service_qualification import (
    ROOT,
    changed_files,
    git,
    sha,
    verify_wheel,
    write_external_patch,
)

REPORT = ROOT / "docs/performance/native-suite-image-e2e-20261022"
ARTIFACTS = ROOT / "performance-results/native-suite-image-e2e-20261022"
LOGS = {
    "browser": "/tmp/ds4-extra-browser.log",
    "tools_and_ranking": "/tmp/ds4-extra-tools.log",
    "library_offline": "/tmp/ds4-extra-library-offline.log",
    "frontend": "/tmp/ds4-extra-frontend-offline.log",
    "schema": "/tmp/ds4-extra-schema-final.log",
    "types": "/tmp/ds4-extra-types.log",
    "lint": "/tmp/ds4-extra-lint.log",
    "library_lint": "/tmp/ds4-extra-library-lint.log",
    "product_text": "/tmp/ds4-extra-product-text.log",
    "docs": "/tmp/ds4-extra-docs.log",
    "wheel": "/tmp/ds4-extra-wheel.log",
    "wheel_install": "/tmp/ds4-extra-wheel-install.log",
    "native_server": "/tmp/ds4-extra-e2e-server.log",
    "schema_before_fix": "/tmp/ds4-extra-schema-before.log",
    "history_before_fix": "/tmp/ds4-extra-history-before.log",
    "preview_before_fix": "/tmp/ds4-extra-preview-before.log",
    "browser_initial_fixture": "/tmp/ds4-extra-browser-before.log",
}


def manifest(paths):
    return {
        str(path.relative_to(ROOT)): {
            "sha256": sha(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
        if path.is_file()
    }


def source_manifest(directory):
    return {
        name: sha((directory / name).read_bytes())
        for name in changed_files(directory)
        if (directory / name).is_file()
        and not name.startswith("docs/performance/native-suite-")
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    logs = {}
    for name, source in LOGS.items():
        raw = Path(source).read_bytes()
        destination = ARTIFACTS / "logs" / f"{name}.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        logs[name] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha(raw),
            "tail": raw.decode(errors="replace").splitlines()[-8:],
        }
    for name in ("browser", "library_offline", "frontend"):
        text = Path(LOGS[name]).read_text()
        match = re.search(r"(\d+) passed(?:, (\d+) deselected)? in ([\d.]+)s", text)
        if not match or re.search(r"\d+ failed", text):
            raise RuntimeError(f"Incomplete or failing acceptance: {name}")
        logs[name].update(status="PASS", tests=int(match[1]), seconds=float(match[3]))
    for name in ("tools_and_ranking", "schema"):
        text = Path(LOGS[name]).read_text()
        match = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
        if not match or not text.rstrip().endswith("OK"):
            raise RuntimeError(f"Incomplete or failing acceptance: {name}")
        logs[name].update(status="PASS", tests=int(match[1]), seconds=float(match[2]))

    browser = json.loads((ARTIFACTS / "browser/browser.json").read_text())
    matrix = json.loads((ARTIFACTS / "tools/tools-and-ranking.json").read_text())
    if (
        len(browser["image_requests"]) != 1
        or browser["image_requests"][0]["body"]["n"] != 2
        or len(browser["provider_calls"]) < 3
        or browser["provider_calls"][-1]["input_image_sizes"]
        != [[128, 128], [128, 128]]
        or max(browser["ui_heartbeat_seconds"]) >= 2
        or {row["case"] for row in matrix}
        != {
            "text_only",
            "image_only",
            "text_and_image",
            "invalid_arguments",
            "native_ranking",
        }
    ):
        raise RuntimeError("Incomplete browser/tool observations")
    external = ROOT / "exmekh_lib"
    patch_path = REPORT / "exmekh-lib.patch"
    patch = write_external_patch(external, patch_path)
    result = {
        "schema_version": 1,
        "date": "2026-10-22",
        "status": "PASS_LOCAL_CUDA",
        "scope": "Funzionalità extra: registered image tool, real model/UI continuation and native web ranking",
        "environment": {
            "platform": "Ubuntu 24.04 WSL / RTX 3090 sm_86",
            "server": "ds4-server-suite",
            "chat_model": "Qwen3.8-27B-UD-Q4_K_S",
            "vision": "mmproj-Qwen3.8-27B-F16.gguf",
            "context": 8192,
            "mtp": False,
            "native_config": "config/native-suite.example.json",
            "execution": "serial CUDA; one large server process",
        },
        "main_base": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "external_base": git(external, "rev-parse", "HEAD").decode().strip(),
        "main_changed_sources": source_manifest(ROOT),
        "external_changed_sources": source_manifest(external),
        "external_patch_sha256": sha(patch),
        "external_patch_reverse_check": "PASS",
        "wheel": verify_wheel(external),
        "logs": logs,
        "browser": {
            "image_requests": browser["image_requests"],
            "image_responses": browser["image_responses"],
            "original_files": browser["files"],
            "provider_calls": [
                {
                    k: call.get(k)
                    for k in (
                        "input_image_sizes",
                        "tool_attachment_counts",
                        "output_text",
                        "tool_calls",
                    )
                }
                for call in browser["provider_calls"]
            ],
            "ui_heartbeat_seconds": browser["ui_heartbeat_seconds"],
        },
        "tools_and_ranking": matrix,
        "artifacts": manifest(
            [
                ROOT / "ds4-server-suite",
                ROOT / "agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl",
                ROOT / "config/native-suite.example.json",
                *[
                    p
                    for p in ARTIFACTS.rglob("*")
                    if "logs" not in p.parts and "wheel" not in p.parts
                ],
            ]
        ),
        "limits": [
            "No image quality or other-backend qualification; T11 image quality gates remain open.",
            "Multiple images share one request but the native recipe executes serially.",
            "The browser uses a test AgentDesign and a fixed auxiliary title; chat, tool and native APIs are real.",
            "The first browser fixture had an incorrect root exit instruction and insufficient context; its failed artifacts are retained separately.",
            "A slow 285-second generation exceeded the default 60-second tool wait and the model regenerated; the failed replay is retained. Final browser qualification explicitly uses wait_seconds=300, not asynchronous job retrieval.",
        ],
    }
    (REPORT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "wheel": result["wheel"]}))


if __name__ == "__main__":
    main()
