#!/usr/bin/env python3
"""Run model gates serially; retain every failure without weakening checks."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from perf_qwen_flash_strata import ROOT, environment, digest, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "performance-results/flash-strata-20261006")
    p.add_argument("--gate", action="append", help="Run only these gate names (repeatable)")
    p.add_argument("--require-pass", action="store_true", help="Exit unsuccessfully after recording any failed selected gate")
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    flash = "gguf/Qwen3.8-Flash-Next-Q2.gguf"
    dense = "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf"
    state = {"DS4_TEST_MODEL": flash, "DS4_TEST_GLM_MTP": "1", "DS4_TEST_MTP_EXACT": "1"}
    gates = [
        ("decode-ranks-oracle", ["./tests/test_qwen4_ranks"], {}),
        ("tuning-workloads", ["./tests/test_qwen4_tuning"], {}),
        ("decode-graph-state-banks", ["./tests/test_qwen4_cuda"],
            {"DS4_TEST_QWEN4_GRAPH_BANKS": "1", "DS4_CUDA_DECODE_GRAPHS": "1"}),
        ("pcie-current", [str(a.output / "pcie-probe")], {}),
        ("primitives-final", ["./tests/test_qwen4_cuda"], {}),
        ("prefill-ring-primitives", ["./tests/test_qwen4_cuda"],
            {"DS4_TEST_QWEN4_PREFILL_RING": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6",
             "DS4_CUDA_QWEN4_RAM_CACHE_MB": "4"}),
        ("prefill-ring-ram-zero", ["./tests/test_qwen4_cuda"],
            {"DS4_TEST_QWEN4_PREFILL_RING": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6",
             "DS4_CUDA_QWEN4_RAM_CACHE_MB": "0"}),
        ("kv-primitives-final", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_KV_STREAMING": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6"}),
        ("kv-primitives-decode-batch", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_KV_STREAMING": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6",
            "DS4_CUDA_QWEN4_KV_DECODE_BATCH": "1"}),
        ("qsa-register-ordered-ids", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_QSA_REGISTER_KEYS": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6"}),
        ("hybrid-primitives-final", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_HYBRID": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1",
            "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "64"}),
        ("hybrid-ram-zero-final", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_HYBRID": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1",
            "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "0"}),
        ("hybrid-ram-pressure-final", ["./tests/test_qwen4_cuda"], {"DS4_TEST_QWEN4_HYBRID": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1",
            "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1", "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "6", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "4",
            "DS4_CUDA_QWEN4_STAGING_WORKERS": "2", "DS4_CUDA_QWEN4_CACHE_STATS": "1"}),
        ("server-unit-final", ["./ds4_server_test"], {}),
        ("static-regression-final", ["python3", "-m", "unittest", "tests.test_qwen38_compatibility",
                                     "tests.test_server_sampling_matrix.SamplingMatrixFixtureTests", "-v"], {}),
        ("flash-state-ordinary", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"], state),
        ("flash-state-default-nextn", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind", "--session-rewind-resample"],
            state | {"DS4_TEST_QWEN4_MTP_AUTO": "1", "DS4_CUDA_QWEN4_KV_STREAMING": "1",
                     "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable", "DS4_CUDA_QWEN4_PREFILL_RING": "1",
                     "DS4_CUDA_QWEN4_PREFILL_AHEAD": "1", "DS4_CUDA_QWEN4_PREFILL_AHEAD_MIN": "9"}),
        ("flash-state-resident-nextn", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind", "--session-rewind-resample"],
            state | {"DS4_TEST_QWEN4_MTP_AUTO": "1", "DS4_CUDA_QWEN4_MTP_RESIDENT": "1",
                     "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable",
                     "DS4_CUDA_QWEN4_DECODE_RANKS": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1",
                     "DS4_CUDA_QWEN4_PCIE_FRAC": "1", "DS4_CUDA_QWEN4_CHECK_VICTIMS": "1"}),
        ("flash-state-resident-nextn-pressure", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_MTP_RESIDENT": "1", "DS4_CUDA_QWEN4_EXPERT_CACHE_MB": "256",
                     "DS4_CUDA_QWEN4_RAM_CACHE_MB": "256", "DS4_CUDA_QWEN4_CHECK_VICTIMS": "1",
                     "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable"}),
        ("flash-state-kv", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1"}),
        ("flash-state-stable", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable"}),
        ("flash-state-ring", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable",
                     "DS4_CUDA_QWEN4_PREFILL_RING": "1", "DS4_CUDA_QWEN4_PREFILL_RING_MB": "128"}),
        ("flash-state-ring-ahead", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable",
                     "DS4_CUDA_QWEN4_PREFILL_RING": "1", "DS4_CUDA_QWEN4_PREFILL_RING_MB": "128",
                     "DS4_CUDA_QWEN4_PREFILL_AHEAD": "1", "DS4_CUDA_QWEN4_PREFILL_AHEAD_MIN": "9"}),
        ("flash-state-stable-pressure", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable",
                     "DS4_CUDA_QWEN4_EXPERT_CACHE_MB": "256", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "256"}),
        ("flash-state-plan-gpu", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "1", "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1"}),
        ("flash-state-plan-gpu-compact", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "1",
                     "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1", "DS4_CUDA_QWEN4_COMPACT_LISTS": "1"}),
        ("flash-state-plan-cpu", ["./ds4_test", "--qwen4-prefill-checkpoints", "--session-snapshot", "--session-rewind"],
            {"DS4_TEST_MODEL": flash, "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-state-plan-cpu-mtp", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-state-plan-cpu-prefill", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5",
                     "DS4_CUDA_QWEN4_HYBRID_PREFILL": "1"}),
        ("flash-state-decode-graphs", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-state-decode-graphs-kv", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "1",
                     "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_KV_RESIDENT": "4096", "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable"}),
        ("flash-state-decode-batch", ["./ds4_test", "--qwen4-prefill-checkpoints", "--qwen4-restore-reuse", "--session-snapshot", "--session-rewind"],
            state | {"DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5",
                     "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_KV_RESIDENT": "4096", "DS4_CUDA_QWEN4_KV_DECODE_BATCH": "1"}),
        ("flash-mtp-limits", ["python3", "tests/test_qwen4_mtp_limits.py", "--model", flash, "--output", str(a.output / "mtp-limits")],
            {"DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-mtp-limits-decode-graphs", ["python3", "tests/test_qwen4_mtp_limits.py", "--model", flash, "--output", str(a.output / "mtp-limits-graphs")],
            {"DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-mtp-limits-decode-batch", ["python3", "tests/test_qwen4_mtp_limits.py", "--model", flash, "--output", str(a.output / "mtp-limits-batch")],
            {"DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_KV_DECODE_BATCH": "1",
             "DS4_CUDA_QWEN4_QSA_REGISTER_KEYS": "1", "DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5"}),
        ("flash-teacher-quality", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-hybrid"),
            "--env", "DS4_CUDA_QWEN4_HYBRID_MOE=1", "--env", "DS4_CUDA_QWEN4_PCIE_FRAC=0.5"], {}),
        ("flash-teacher-cpu-prefill", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-cpu-prefill"),
            "--env", "DS4_CUDA_QWEN4_HYBRID_MOE=1", "--env", "DS4_CUDA_QWEN4_PCIE_FRAC=0.5",
            "--env", "DS4_CUDA_QWEN4_HYBRID_PREFILL=1"], {}),
        ("flash-teacher-kv", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-kv"),
            "--env", "DS4_CUDA_QWEN4_KV_STREAMING=1"], {}),
        ("flash-teacher-ring", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-ring"),
            "--env", "DS4_CUDA_QWEN4_KV_STREAMING=1", "--env", "DS4_CUDA_QWEN4_EXPERT_POLICY=stable",
            "--env", "DS4_CUDA_QWEN4_PREFILL_RING=1"], {}),
        ("flash-teacher-decode-graphs", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-decode-graphs"),
            "--env", "DS4_CUDA_QWEN4_DECODE_GRAPHS=1", "--env", "DS4_CUDA_QWEN4_HYBRID_MOE=1", "--env", "DS4_CUDA_QWEN4_PCIE_FRAC=0.5"], {}),
        ("flash-teacher-decode-batch", ["python3", "tools/qualify_qwen_flash_quality.py", "--output", str(a.output / "quality-decode-batch"),
            "--env", "DS4_CUDA_QWEN4_DECODE_GRAPHS=1", "--env", "DS4_CUDA_QWEN4_HYBRID_MOE=1", "--env", "DS4_CUDA_QWEN4_PCIE_FRAC=0.5",
            "--env", "DS4_CUDA_QWEN4_KV_STREAMING=1", "--env", "DS4_CUDA_QWEN4_KV_DECODE_BATCH=1", "--env", "DS4_CUDA_QWEN4_QSA_REGISTER_KEYS=1"], {}),
        ("qwen27-mtp-inherited-final", ["./tests/test_qwen_mtp_callback", dense, "gguf/mtp-Qwen3.8-27B-Q4_0.gguf"],
            {"DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_DECODE_GRAPHS": "1", "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "0.5",
             "DS4_CUDA_QWEN4_KV_DECODE_BATCH": "1", "DS4_CUDA_QWEN4_QSA_REGISTER_KEYS": "1",
             "DS4_CUDA_QWEN4_EXPERT_POLICY": "stable", "DS4_CUDA_QWEN4_PLE_OVERLAP": "1",
             "DS4_CUDA_QWEN4_PREFILL_RING": "1", "DS4_CUDA_QWEN4_PREFILL_AHEAD": "1",
             "DS4_CUDA_QWEN4_PREFILL_AHEAD_ALL": "1", "DS4_CUDA_QWEN4_COMPACT_LISTS": "1",
             "DS4_CUDA_QWEN4_MTP_RESIDENT": "1"}),
        ("qwen27-nextn-catchup", ["./tests/qwen_mtp_catchup_probe", dense, "gguf/mtp-Qwen3.8-27B-Q4_0.gguf"],
            {"DS4_CUDA_QWEN_MTP_BATCH_CATCHUP": "1"}),
    ]
    if a.gate:
        unknown = set(a.gate) - {name for name, _, _ in gates}
        if unknown:
            p.error("unknown gates: " + ", ".join(sorted(unknown)))
        gates = [gate for gate in gates if gate[0] in a.gate]
    records = []
    teacher_reports = {"flash-teacher-quality": "quality-hybrid", "flash-teacher-cpu-prefill": "quality-cpu-prefill",
        "flash-teacher-kv": "quality-kv", "flash-teacher-ring": "quality-ring", "flash-teacher-decode-graphs": "quality-decode-graphs",
        "flash-teacher-decode-batch": "quality-decode-batch"}
    for name, command, extra in gates:
        print(name, flush=True); started = time.monotonic(); error = None
        with (a.output / (name + ".log")).open("wb") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=environment(extra), stdout=log, stderr=log, timeout=1200)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = -1; error = "timeout"
        if code == 0 and name in teacher_reports:
            # The teacher tool emits diagnostics even when distributions differ.
            # Its successful process exit is not itself a quality gate pass.
            quality = json.loads((a.output / teacher_reports[name] / "report.json").read_text())
            cases = quality.get("cases", [])
            if not cases or not all(case["bit_exact"] for case in cases):
                code = 1; error = "full-vocabulary teacher logits differ; diagnostics retained"
        records.append({"gate": name, "command": command, "environment": extra, "exit_code": code,
                        "seconds": time.monotonic() - started, "error": error})
        write_json(a.output / "qualification.json", {"gates": records, "status": "IN_PROGRESS"})
        print(f"{name}: exit={code}", flush=True)
    write_json(a.output / "qualification.json", {"gates": records, "status": "COMPLETE_MEASUREMENTS",
               "all_pass": all(r["exit_code"] == 0 for r in records),
               "binary_sha256": {n: digest(ROOT / n) for n in ("ds4", "ds4-server", "ds4-bench", "ds4-eval")}})
    if a.require_pass and any(r["exit_code"] != 0 for r in records):
        p.exit(1, "Selected gates failed; details retained in qualification.json\n")

if __name__ == "__main__": main()
