#!/usr/bin/env python3
"""Run the four Qwen27 lifecycle probes serially and hash their outputs."""

import argparse
import json
import subprocess
from pathlib import Path

from suite_artifacts import probe_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, mtp, vision in [
        ("text", False, False),
        ("text-mtp", True, False),
        ("vision", False, True),
        ("vision-mtp", True, True),
    ]:
        directory = args.output / name
        directory.mkdir(exist_ok=False)
        cmd = [
            "tools/probe_suite_baseline",
            "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf",
            "gguf/mtp-Qwen3.8-27B-Q4_0.gguf" if mtp else "-",
            "gguf/mmproj-Qwen3.8-27B-F16.gguf" if vision else "-",
            "tests/vision-fixtures/glm53/spatial.png",
            str(directory),
        ]
        with (
            (directory / "stderr.log").open("w") as log,
            (directory / "result.json").open("w") as out,
        ):
            result = subprocess.run(cmd, stdout=out, stderr=log)
        record = {
            "case": name,
            "command": cmd,
            "exit_code": result.returncode,
            "files": [],
        }
        if result.returncode == 0:
            try:
                record["files"] = probe_artifacts(
                    directory, args.compare / name if args.compare else None
                )
            except (OSError, ValueError) as error:
                record["validation_error"] = str(error)
        records.append(record)
        (args.output / "model-runs.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(name, result.returncode, flush=True)
        if result.returncode or "validation_error" in record:
            return result.returncode or 1
        if any(
            item.get("bit_exact") is False or item.get("finite") is False
            for item in record["files"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
