#!/usr/bin/env python3
"""Exercise actual dense Qwen frontiers and full logits in every capacity mode."""
import argparse
from argparse import Namespace
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from perf_qwen_flash_strata import ROOT, run_bench, compare_logits, write_json


def measure(args, name, binary, extra):
    stop = threading.Event()
    memory = []

    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output([
                    "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                memory.append(float(value.strip().splitlines()[0]))
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            stop.wait(2)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        result = run_bench(args, name, binary, extra)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        result = {"error": str(error)}
    finally:
        stop.set()
        worker.join()
    result["gpu_memory_peak_mib_sampled_2s"] = max(memory) if memory else None
    result["gpu_memory_samples_mib"] = memory
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "ds4-bench")
    parser.add_argument("--baseline", type=Path, default=ROOT / "performance-results/flash-strata-20261006/baseline/ds4-bench")
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")
    parser.add_argument("--prompt", type=Path, default=ROOT / "speed-bench/promessi_sposi.txt")
    parser.add_argument("--contexts", type=int, nargs="+", default=[65536])
    parser.add_argument("--resident", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--progress", action="store_true", help="Log completed chunks without extra GPU synchronization; disable for speed promotion")
    parser.add_argument("--modes", nargs="+", choices=["off", "grow", "hybrid-attention", "reference-attention"],
                        default=["off", "grow", "hybrid-attention"])
    a = parser.parse_args()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    a.samples = 1; a.warmup = 0
    records = []
    for context in a.contexts:
        a.start = a.end = context; a.context = context + a.tokens + 8
        entry = {"actual_frontier": context, "resident": a.resident, "modes": {}, "status": "IN_PROGRESS"}
        records.append(entry)
        for mode in a.modes:
            binary = a.baseline if mode == "off" else a.binary
            extra = {"DS4_CUDA_QWEN_KV_CAPACITY_MODE": "hybrid-attention" if mode == "reference-attention" else mode,
                     "DS4_CUDA_QWEN_KV_RESIDENT": str(a.resident)}
            if mode == "reference-attention":
                # The original resident attention kernels see a complete,
                # contiguous layer freshly staged from the logical KV. This
                # is an independent decode arithmetic oracle for tiling when
                # all 16 resident layers exceed the available VRAM at 64K.
                extra["DS4_CUDA_QWEN_KV_REFERENCE_ATTENTION"] = "1"
            if a.progress: extra["DS4_BENCH_PROGRESS"] = "1"
            name = f"context-{context}-{mode}"
            entry["modes"][mode] = {"status": "IN_PROGRESS", "environment": extra}
            write_json(a.output / "report.json", {"records": records, "status": "IN_PROGRESS"})
            entry["modes"][mode] = measure(a, name, binary, extra)
            if "error" not in entry["modes"][mode]:
                files = list((a.output / name / "logits").glob("*.json"))
                payloads = [json.loads(path.read_text()) for path in files]
                valid = len(payloads) == 2 and all(
                    data.get("frontier_tokens") == context and
                    len(data.get("logits", [])) == data.get("vocab") and
                    all(math.isfinite(value) for value in data["logits"]) for data in payloads)
                decode = [data for data in payloads if "generated_count" in data]
                valid &= len(decode) == 1 and decode[0]["generated_count"] == a.tokens and decode[0]["final_position"] == context + a.tokens
                entry["modes"][mode]["capacity_verified"] = bool(valid)
                if not valid: entry["modes"][mode]["error"] = "missing or invalid actual frontier/decode logits"
            write_json(a.output / "report.json", {"records": records, "status": "IN_PROGRESS"})
        # Also support running capacity modes first and the resident reference
        # last. A capacity-only pass must not silently become a numeric pass.
        reference_mode = next((mode for mode in ("off", "reference-attention")
                               if mode in entry["modes"] and "error" not in entry["modes"][mode]), None)
        if reference_mode:
            reference = a.output / f"context-{context}-{reference_mode}/logits"
            entry["numeric_reference"] = reference_mode
            for mode, run in entry["modes"].items():
                if mode != reference_mode and "error" not in run:
                    run["logits"] = compare_logits(reference, a.output / f"context-{context}-{mode}/logits")
        success = [r for r in entry["modes"].values() if "error" not in r]
        numerics = all(all(q["bit_exact"] for q in r.get("logits", [])) for r in success)
        complete = len(success) == len(a.modes) and numerics
        compared = reference_mode is not None and all(mode == reference_mode or bool(run.get("logits")) for mode, run in entry["modes"].items())
        entry["status"] = ("PASS" if compared else "CAPACITY_ONLY") if complete else "INCOMPLETE"
        print(json.dumps({"actual_frontier": context, "status": entry["status"]}), flush=True)
    write_json(a.output / "report.json", {"records": records,
        "status": "CAPACITY_AND_NUMERICS_DIAGNOSTIC; one sample per mode does not qualify speed promotion"})
    if any(entry["status"] not in {"PASS", "CAPACITY_ONLY"} for entry in records):
        parser.exit(1, "Capacity qualification incomplete; inspect retained measurements\n")


if __name__ == "__main__":
    main()
