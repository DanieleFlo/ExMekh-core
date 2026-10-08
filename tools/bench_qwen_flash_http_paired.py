#!/usr/bin/env python3
"""Alternate local HTTP A/B runs and verify that embedded MTP executes."""
import argparse
import json
from pathlib import Path
import subprocess
from perf_qwen_flash_strata import ROOT, stats, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=ROOT / "performance-results/flash-strata-20261006/baseline/ds4-server")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--env", action="append", default=[])
    a = p.parse_args()
    if a.samples < 5: p.error("at least five pairs required")
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    runs = {"baseline": [], "candidate": []}
    for i in range(a.samples):
        for variant in (("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")):
            out = a.output / f"pair-{i}-{variant}"
            binary = a.baseline if variant == "baseline" else a.binary
            command = ["python3", "tools/bench_qwen_flash_http.py", "--binary", str(binary.resolve()),
                       "--fixture", str(a.fixture.resolve()), "--output", str(out), "--context", str(a.context),
                       "--new-prompts"]
            if variant == "candidate":
                for value in a.env: command += ["--env", value]
            print(f"HTTP pair {i}: {variant}", flush=True)
            with (a.output / f"pair-{i}-{variant}.log").open("wb") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, timeout=1200)
            if result.returncode: raise RuntimeError(f"HTTP run failed: {out}")
            report = json.loads((out / "report.json").read_text())
            if not report.get("mtp_session_totals"):
                raise RuntimeError(f"no executed MTP cycles: {out}")
            runs[variant].append(report)
            write_json(a.output / "report.json", {"runs": runs, "status": "IN_PROGRESS"})
    quality = []
    phases = {}
    for request in range(3):
        name = runs["baseline"][0]["records"][request]["requested_phase"]
        timing, throughput = {}, {}
        for variant in runs:
            rows = [r["records"][request] for r in runs[variant]]
            timing[variant] = stats([r["ttft_seconds"] for r in rows])
            throughput[variant] = stats([(r["usage"]["output_tokens"] - 1) / (r["seconds"] - r["ttft_seconds"]) for r in rows])
        phases[name] = {"ttft": timing, "ttft_reduction": 1 - timing["candidate"]["median"] / timing["baseline"]["median"],
                        "http_tokens_after_first_per_second": throughput,
                        "throughput_gain": throughput["candidate"]["median"] / throughput["baseline"]["median"] - 1,
                        "observed_cached_tokens": {v: [r["records"][request]["usage"]["input_tokens_details"]["cached_tokens"] for r in runs[v]] for v in runs}}
        for i in range(a.samples):
            x, y = [runs[v][i]["records"][request] for v in ("baseline", "candidate")]
            quality.append({"pair": i, "phase": name, "same_generated_text": x["generated_text"] == y["generated_text"],
                            "same_token_usage": x["usage"] == y["usage"]})
    all_pass = all(q["same_generated_text"] and q["same_token_usage"] for q in quality)
    write_json(a.output / "report.json", {"runs": runs, "phases": phases, "quality": quality, "all_pass": all_pass,
               "status": "HTTP_MTP_MEASUREMENTS; generated text checks supplement full-logit state gates",
               "throughput_definition": "output tokens after first divided by remaining HTTP time, includes protocol/finalization"})
    if not all_pass:
        raise RuntimeError("HTTP generated text/token usage differs; measurements retained")


if __name__ == "__main__": main()
