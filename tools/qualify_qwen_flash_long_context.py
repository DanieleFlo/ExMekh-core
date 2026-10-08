#!/usr/bin/env python3
"""Use actual long frontiers and compare complete logits, serially on one GPU."""
import argparse
import json
import shutil
from pathlib import Path
from perf_qwen_flash_strata import ROOT, run_bench, compare_logits, write_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "ds4-bench")
    p.add_argument("--baseline", type=Path, default=ROOT / "performance-results/flash-strata-20261006/baseline/ds4-bench")
    p.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf")
    p.add_argument("--prompt", type=Path, default=ROOT / "speed-bench/promessi_sposi.txt")
    p.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536, 131072])
    p.add_argument("--resident", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--require-pass", action="store_true", help="Fail after retaining any incomplete or non-exact frontier")
    p.add_argument("--baseline-results", type=Path, help="Reuse compatible completed baseline frontiers for diagnostics; absent frontiers are measured afresh")
    p.add_argument("--env", action="append", default=[], help="Identical memory/policy overrides for both variants")
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    a.warmup = 0; a.samples = 1
    configuration = dict(value.split("=", 1) for value in a.env)
    records = []
    for context in a.contexts:
        a.start = a.end = context; a.context = context + a.tokens + 8
        entry = {"actual_frontier": context, "resident": a.resident, "configuration_environment": configuration, "status": "IN_PROGRESS"}
        records.append(entry)
        for variant, binary, extra in (("baseline", a.baseline, configuration), ("candidate", a.binary,
                configuration | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_KV_RESIDENT": str(a.resident),
                 "DS4_CUDA_QWEN4_CACHE_STATS": "1"})):
            name = f"context-{context}-{variant}"
            try:
                recorded = a.baseline_results / name if a.baseline_results and variant == "baseline" else None
                if recorded and (recorded / "run.json").is_file():
                    saved = json.loads((recorded / "run.json").read_text())
                    command = saved["command"]
                    expected = {"--ctx-start": context, "--ctx-max": context, "--ctx-alloc": a.context,
                                "--gen-tokens": a.tokens, "--prefill-chunk": a.chunk, "--repetitions": 1}
                    compatible = saved.get("exit_code") == 0 and saved.get("samples") == 1 and saved.get("warmup") == 0 and \
                        saved.get("binary_sha256") == digest(binary) and saved.get("model", {}).get("bytes") == a.model.stat().st_size and \
                        all(str(value) == command[command.index(option) + 1] for option, value in expected.items()) and \
                        command[command.index("--prompt-file") + 1] == str(a.prompt.resolve()) and \
                        command[command.index("-m") + 1] == str(a.model.resolve()) and "--teacher-forced-decode" in command
                    if not compatible: raise ValueError("recorded baseline is incompatible with the requested workload")
                    shutil.copytree(recorded, a.output / name, dirs_exist_ok=True)
                    saved["reused_diagnostic_baseline"] = str(recorded.resolve())
                    entry[variant] = saved
                else:
                    entry[variant] = run_bench(a, name, binary, extra)
            except Exception as error:
                entry[variant] = {"error": str(error)}
            write_json(a.output / "report.json", {"records": records, "status": "IN_PROGRESS"})
        if all("error" not in entry[v] for v in ("baseline", "candidate")):
            entry["logits"] = compare_logits(a.output / f"context-{context}-baseline/logits",
                                             a.output / f"context-{context}-candidate/logits")
            entry["status"] = "PASS" if all(q["bit_exact"] for q in entry["logits"]) else "NUMERICAL_DIFFERENCE"
        else:
            entry["status"] = "INCOMPLETE"
        write_json(a.output / "report.json", {"records": records,
                   "status": "DIAGNOSTICS; one sample qualifies capacity/numerics, not performance promotion"})
        print(json.dumps({k: entry[k] for k in ("actual_frontier", "resident", "status")}), flush=True)
    if a.require_pass and any(entry["status"] != "PASS" for entry in records):
        p.exit(1, "Long-context qualification failed; retained measurements describe the failure\n")


if __name__ == "__main__":
    main()
