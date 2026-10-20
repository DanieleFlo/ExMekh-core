#!/usr/bin/env python3
"""Compare the available Flash LLM using its existing tiered launcher profile."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from qwen_flash_launcher import launcher_environment
from suite_artifacts import frontier_artifacts, probe_artifacts, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = args.output / "prompt.txt"
    prompt.write_text(
        "A red square beside a blue circle. Explain the colors and the shapes.\n" * 8
    )
    settings = launcher_environment()
    report = {"environment": settings, "runs": []}
    for name, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
        path = args.output / name
        path.mkdir()
        command = [
            str(binary.resolve()),
            "--cuda",
            "-m",
            "gguf/Qwen3.8-Flash-Next-Q2.gguf",
            "--prompt-file",
            str(prompt),
            "--ctx-start",
            "32",
            "--ctx-max",
            "32",
            "--ctx-alloc",
            "256",
            "--gen-tokens",
            "2",
            "--prefill-chunk",
            "32",
            "--repetitions",
            "1",
            "--dump-frontier-logits-dir",
            str(path),
            "--csv",
            str(path / "timing.csv"),
        ]
        with (path / "runtime.log").open("w") as log:
            result = subprocess.run(
                command,
                env=dict(os.environ, **settings),
                stdout=log,
                stderr=log,
                timeout=1200,
            )
        run = {
            "id": name,
            "command": command,
            "exit_code": result.returncode,
            "binary_sha256": sha256_file(binary),
            "logits": [],
        }
        if result.returncode == 0:
            try:
                run["logits"] = frontier_artifacts(path)
            except (OSError, ValueError) as error:
                run["validation_error"] = str(error)
        report["runs"].append(run)
        (args.output / "flash.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode or "validation_error" in run:
            return result.returncode or 1
    report["bit_exact"] = (
        bool(report["runs"][0]["logits"])
        and report["runs"][0]["logits"] == report["runs"][1]["logits"]
    )
    if args.payload_probe and report["bit_exact"]:
        path = args.output / "payload"
        path.mkdir()
        command = [
            "tools/probe_suite_baseline",
            "gguf/Qwen3.8-Flash-Next-Q2.gguf",
            "-",
            "-",
            "tests/vision-fixtures/glm53/spatial.png",
            str(path),
        ]
        with (
            (path / "result.json").open("w") as out,
            (path / "runtime.log").open("w") as log,
        ):
            result = subprocess.run(
                command,
                env=dict(os.environ, **settings),
                stdout=out,
                stderr=log,
                timeout=1200,
            )
        payload = {
            "command": command,
            "exit_code": result.returncode,
            "directory": path.as_posix(),
            "environment": settings,
            "baseline_timing": "post-refactor snapshot; pre-refactor logits compared independently above",
            "source_sha256": sha256_file(Path("ds4.c")),
            "probe_sha256": sha256_file(Path("tools/probe_suite_baseline")),
            "finite": False,
            "files": [],
        }
        if result.returncode == 0:
            try:
                payload["files"] = probe_artifacts(path)
                payload["finite"] = True
            except (OSError, ValueError) as error:
                payload["validation_error"] = str(error)
        report["payload_probe"] = payload
        if not payload["finite"]:
            report["bit_exact"] = False
    (args.output / "flash.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Flash bit_exact", report["bit_exact"])
    return 0 if report["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
