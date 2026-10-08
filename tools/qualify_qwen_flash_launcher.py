#!/usr/bin/env python3
"""Qualify launcher placement and prefill geometry with complete teacher logits."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from perf_qwen_flash_strata import ROOT, digest, environment, write_json
from qualify_qwen_flash_quality import CASES, tokenize
from qwen_flash_launcher import launcher_environment
from qwen_flash_metrics import compare_teacher_logits, teacher_quality_passes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-prompt", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[131072])
    parser.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--screen", action="store_true", help="One agentic case before the full domain suite")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Isolate the chunk using the ordinary placement")
    parser.add_argument("--env", action="append", default=[], help="Candidate override for causal diagnosis")
    parser.add_argument("--reference-results", type=Path, help="Reuse only identical probe/input/context reference dumps")
    parser.add_argument("--reference-probe", type=Path, help="Frozen reference binary, distinct from the candidate probe")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise RuntimeError("retain earlier gates; choose a new output")
    args.output.mkdir(parents=True)
    model = ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf"
    binary = ROOT / "tests/qwen_flash_quality_probe"
    fixtures = {}
    for name, (prompt, target) in CASES.items():
        if not prompt.startswith("<|im_start|>"):
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        fixtures[name] = {"prompt": prompt, "target": target}
    text = args.agent_prompt.read_text()
    if args.screen:
        fixtures = {"agent": {"prompt": text, "target": CASES["code"][1][:100]}}
    else:
        system_end = text.index("<|im_end|>") + len("<|im_end|>")
        for name, (prompt, target) in CASES.items():
            user = prompt if not prompt.startswith("<|im_start|>") else "Use the lookup tool for library opening hours."
            fixtures["agent-" + name] = {
                "prompt": text[:system_end] + f"\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
                "target": target,
            }
    for fixture in fixtures.values():
        fixture["prompt_ids"] = tokenize(ROOT / "ds4", model, fixture["prompt"])
        fixture["target_ids"] = tokenize(ROOT / "ds4", model, fixture["target"])
    write_json(args.output / "fixtures.json", fixtures)
    records = []
    candidate = ({} if args.plain else launcher_environment()) | dict(v.split("=", 1) for v in args.env)
    for context in args.contexts:
        for name, chunk, extra in [("reference", 256, {})] + [(f"candidate-{c}", c, candidate) for c in args.chunks]:
            dest = args.output / f"ctx-{context}-{name}"
            dest.mkdir()
            inputs = dest / "inputs.tsv"
            inputs.write_text("".join(
                ",".join(map(str, f["prompt_ids"])) + "|" + ",".join(map(str, f["target_ids"])) +
                f"\t{dest / (case + '.bin')}\n" for case, f in fixtures.items()))
            command = [str(binary), str(model), str(context), str(chunk), str(inputs), "1"]
            if name == "reference" and args.reference_results:
                previous = args.reference_results / f"ctx-{context}-reference"
                saved = json.loads((previous / "run.json").read_text())
                old_fixtures = json.loads((args.reference_results / "fixtures.json").read_text())
                expected_probe = args.reference_probe or binary
                if saved["binary_sha256"] != digest(expected_probe) or saved["exit_code"] or old_fixtures != fixtures or saved["command"][2:4] != command[2:4]:
                    raise RuntimeError("incompatible reference dumps")
                for case in fixtures:
                    shutil.copyfile(previous / (case + ".bin"), dest / (case + ".bin"))
                write_json(dest / "run.json", saved | {"reused_diagnostic_reference": str(previous)})
                continue
            env = environment({"DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "auto", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "auto"} | extra)
            with (dest / "runtime.log").open("wb") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=3600, check=False)
            write_json(dest / "run.json", {"command": command, "environment": extra,
                       "exit_code": result.returncode, "binary_sha256": digest(binary)})
            if result.returncode:
                raise RuntimeError(f"{name} failed; retained runtime.log")
            print(f"ctx={context} {name}: dump complete", flush=True)
            if name == "reference":
                continue
            for case, fixture in fixtures.items():
                rows = len(fixture["target_ids"])
                x, y = [np.fromfile(args.output / f"ctx-{context}-{variant}" / (case + ".bin"), dtype=np.float32).reshape(rows, -1).astype(np.float64)
                        for variant in ("reference", name)]
                record = {"context": context, "chunk": chunk, "case": case, "prompt_tokens": len(fixture["prompt_ids"]),
                          **compare_teacher_logits(x, y, fixture["target_ids"])}
                record["pass"] = teacher_quality_passes(record)
                records.append(record)
                print(json.dumps(record), flush=True)
            write_json(args.output / "report.json", {"cases": records, "all_pass": all(r["pass"] for r in records),
                       "status": "IN_PROGRESS", "candidate_environment": candidate, "source_sha256": digest(Path(__file__))})
    write_json(args.output / "report.json", {"cases": records, "all_pass": all(r["pass"] for r in records),
               "status": "COMPLETE", "candidate_environment": candidate, "source_sha256": digest(Path(__file__))})
    if args.require_pass and any(not r["pass"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
