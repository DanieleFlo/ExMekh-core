#!/usr/bin/env python3
"""Freeze, benchmark and calibrate opt-in Flash scheduling on one CUDA GPU.

Uses complete forwards and matched teacher-forced continuations. Never changes
launchers, system settings, weights, golden data or the normal runtime defaults.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a"
BASE_ENV = {
    "DS4_CUDA_QWEN4_TIERED_CACHE": "1",
    "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "20",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stats(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("incomplete or invalid benchmark samples")
    mean = statistics.mean(values)
    return {"samples": values, "median": statistics.median(values), "mean": mean,
            "cv": statistics.stdev(values) / mean if len(values) > 1 else None}


def gain_stats(values):
    """Signed relative changes; zero/negative gains are valid measurements."""
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("incomplete or invalid paired gains")
    return {"samples": values, "median": statistics.median(values), "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else None}


def summarize_csv(path, warmup):
    groups = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(int(row["ctx_tokens"]), []).append(row)
    result = {}
    for ctx, rows in groups.items():
        measured = rows[warmup:]
        result[str(ctx)] = {key: stats([float(row[column]) for row in measured])
                           for key, column in (("prefill", "prefill_tps"), ("decode", "gen_steady_tps"))}
        result[str(ctx)]["first_prefill_tps"] = float(rows[0]["prefill_tps"])
        if "gen_tps" in measured[0]:
            result[str(ctx)]["decode_total"] = stats([float(row["gen_tps"]) for row in measured])
    if not result:
        raise ValueError("benchmark produced no completed frontiers")
    return result


def compare_logits(reference, candidate):
    reports = []
    for path in sorted(reference.glob("*.json")):
        peer = candidate / path.name
        left, right = json.loads(path.read_text()), json.loads(peer.read_text())
        for key in ("ctx", "frontier_tokens", "vocab", "generated_tokens"):
            if key in left and left[key] != right.get(key):
                raise ValueError(f"unmatched numerical inputs: {path.name}: {key}")
        a, b = np.asarray(left["logits"], dtype=np.float64), np.asarray(right["logits"], dtype=np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid logits: {peer}")
        la = a - np.max(a); lb = b - np.max(b)
        la -= np.log(np.exp(la).sum()); lb -= np.log(np.exp(lb).sum())
        p = np.exp(la)
        reports.append({"file": path.name, "bit_exact": np.array_equal(a, b),
                        "max_abs": float(np.max(np.abs(a - b))), "kl": max(0.0, float(np.sum(p * (la - lb)))),
                        "argmax_match": int(np.argmax(a)) == int(np.argmax(b))})
    if not reports:
        raise ValueError("missing reference logits")
    return reports


_known_flash_environment = None


def environment(extra):
    global _known_flash_environment
    if _known_flash_environment is None:
        _known_flash_environment = set()
        for source in ROOT.glob("ds4*.*"):
            if source.suffix in {".c", ".h", ".cu", ".cuh"}:
                _known_flash_environment.update(re.findall(
                    r'"(DS4_CUDA_QWEN4_[A-Z0-9_]+)"', source.read_text()))
    unknown = sorted(key for key in extra if key.startswith("DS4_CUDA_QWEN4_")
                     and key not in _known_flash_environment)
    if unknown:
        raise ValueError("Unknown Flash environment controls: " + ", ".join(unknown))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("DS4_CUDA_QWEN4_", "DS4_MTP_QWEN_", "DS4_CUDA_QWEN_MTP_")) or key in {"DS4_QWEN4_TIMING", "DS4_QWEN4_MOE_PROFILE",
                "DS4_QWEN_PREFILL_PHASES", "DS4_MTP_CYCLE_TRACE", "DS4_BENCH_DECODE_WARMUP", "DS4_BENCH_DECODE_TIMES", "DS4_BENCH_SPEC_TRACE",
                "DS4_CUDA_QWEN_KV_CAPACITY_MODE", "DS4_CUDA_QWEN_KV_RESIDENT", "DS4_CUDA_QWEN_KV_REFERENCE_ATTENTION", "DS4_CUDA_QWEN_MTP_KV_MODE", "DS4_QWEN_EXTENDED_DRAFT",
                "DS4_CUDA_QWEN_WEIGHT_STREAMING", "DS4_CUDA_QWEN_WEIGHT_RESIDENT_MB", "DS4_CUDA_QWEN_WEIGHT_STAGE_MB"}:
            del env[key]
    env.update(BASE_ENV)
    env.update(extra)
    return env


def run_bench(args, name, binary, extra, samples=None):
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    logits = output / "logits"
    logits.mkdir(exist_ok=True)
    samples = args.samples if samples is None else samples
    extra = dict(extra)
    profile_key = "DS4_CUDA_QWEN4_EXPERT_PROFILE"
    profile_identity = None
    if extra.get(profile_key) and Path(extra[profile_key]).is_file():
        original_profile = Path(extra[profile_key]).resolve()
        profile_identity = digest(original_profile)
        # The engine learns on close. Each A/B run must start from the same
        # frozen ranking rather than inherit previous benchmark continuations.
        local_profile = output / "expert-profile.bin"
        shutil.copyfile(original_profile, local_profile)
        extra[profile_key] = str(local_profile.resolve())
    command = [str(binary.resolve()), "--cuda", "-m", str(args.model.resolve()),
               "--prompt-file", str(args.prompt.resolve()), "--ctx-start", str(args.start),
               "--ctx-max", str(args.end), "--ctx-alloc", str(args.context),
               "--step-incr", str(getattr(args, "step", None) or max(1, args.end - args.start)), "--gen-tokens", str(args.tokens),
               "--teacher-forced-decode", "--prefill-chunk", str(args.chunk),
               "--repetitions", str(args.warmup + samples), "--csv", str(output / "speed.csv"),
               "--dump-frontier-logits-dir", str(logits)]
    if getattr(args, "mtp", False):
        command.remove("--teacher-forced-decode")
        command.append("--mtp")
        if getattr(args, "mtp_draft", None) is not None:
            command.extend(["--mtp-draft", str(args.mtp_draft)])
        # CPU logging uses the existing timing boundaries, outside forwards.
        extra["DS4_BENCH_DECODE_TIMES"] = "1"
    print(f"{name}: complete forward benchmark ({args.warmup} warm-up + {samples} samples)", flush=True)
    identity = digest(binary)
    start = time.monotonic()
    write_json(output / "run.json", {"status": "IN_PROGRESS", "command": command,
        "environment": BASE_ENV | extra, "binary_sha256": identity,
        "started_unix": time.time(), "timeout_seconds": args.timeout})
    with (output / "runtime.log").open("wb") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment(extra), stdout=log, stderr=log, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            write_json(output / "run.json", {"status": "TIMEOUT", "command": command,
                "environment": BASE_ENV | extra, "binary_sha256": identity,
                "wall_seconds": time.monotonic() - start})
            raise
    report = {"status": "COMPLETE" if completed.returncode == 0 else "FAILED",
              "command": command, "environment": BASE_ENV | extra, "exit_code": completed.returncode,
              "wall_seconds": time.monotonic() - start, "binary_sha256": identity,
              "model": {"path": str(args.model.resolve()), "bytes": args.model.stat().st_size},
              "samples": samples, "warmup": args.warmup}
    if profile_identity:
        report["initial_expert_profile_sha256"] = profile_identity
    if completed.returncode == 0:
        report["frontiers"] = summarize_csv(output / "speed.csv", args.warmup)
        if getattr(args, "mtp", False):
            runtime = (output / "runtime.log").read_text(errors="replace")
            if "speculative decoding enabled" not in runtime or "future cycles will use target-only" in runtime:
                raise RuntimeError("MTP calibration did not enable its speculative runtime")
            calls = re.findall(r"decode cycle[^\n]*rows=(\d+)", runtime)
            report["mtp_calls"] = len(calls)
            report["mtp_multi_token_calls"] = sum(int(n) > 1 for n in calls)
    write_json(output / "run.json", report)
    if digest(binary) != identity:
        raise RuntimeError("benchmark binary changed during the run")
    if completed.returncode:
        raise RuntimeError(f"{name} failed; see {output / 'runtime.log'}")
    print(f"{name}: done in {report['wall_seconds']:.1f}s", flush=True)
    return report


def freeze(args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline = args.output / "baseline"
    binaries = {str(p.relative_to(ROOT)): digest(p) for p in baseline.iterdir() if p.is_file() and os.access(p, os.X_OK)}
    sources = {}
    for name in ("ds4.c", "ds4_cuda.cu", "ds4_qwen4_cuda.cuh", "ds4_qwen4_expert_cache.h", "Makefile"):
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        sources[name] = hashlib.sha256(data).hexdigest()
    stat = args.model.stat()
    model = {"path": str(args.model.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "sha256": digest(args.model) if args.hash_model else "NOT_REHASHED"}
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current", "--format=csv"], text=True)
    report = {"baseline_head": commit, "baseline_sources": sources, "baseline_binaries": binaries,
              "model": model, "gpu": gpu, "strata_commit": SOURCE_COMMIT,
              "strata_license": "MIT; architectural reference, no copied implementation",
              "strata_files": [f"https://github.com/Niko1221/Strata/blob/{SOURCE_COMMIT}/{p}" for p in
                               ("src/core/session.cpp", "src/core/expert_cache.cpp", "src/kernels/cuda/kv_stream.cu", "src/kernels/cpu/pool.cpp")]}
    write_json(args.output / "baseline-manifest.json", report)
    print("Baseline identity saved", flush=True)


def freeze_candidate(args):
    # Read the actual checkout bytes independently of Git/Windows CRLF defaults.
    git_read = ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false"]
    sources = [".gitignore", "Makefile", "ds4.c", "ds4.h", "tests/ds4_test.c", "ds4_cuda.cu", "ds4_gpu.h", "ds4_server.c", "ds4_cli.c", "ds4_bench.c", "ds4_env.h", "ds4_help.c", "ds4_qwen4_cuda.cuh",
               "ds4_qwen4_cpu.c", "ds4_qwen4_cpu.h", "ds4_qwen4_cpu_internal.h", "ds4_qwen4_cpu_sign_masks.inc", "ds4_qwen4_hybrid.cuh",
               "ds4_qwen4_kv.cuh", "ds4_qwen4_profile.cuh", "ds4_qwen4_residency.cuh", "ds4_qwen4_transfer.cuh", "ds4_qwen4_tuning.cuh", "ds4_qwen4_ranks.cuh", "ds4_qwen4_ring.cuh", "tests/test_qwen4_ranks.cpp",
               "ds4_qwen4_runtime.h", "ds4_qwen4_expert_cache.h", "tests/test_qwen4_kernels.c", "tests/test_qwen4_cpu.c", "tests/test_qwen4_tuning.cpp", "tests/test_qwen_mtp_callback.c", "tests/qwen_mtp_catchup_probe.c", "tests/test_server_agentic_payload.c",
               "tests/test_qwen4_mtp_limits.py", "tests/test_qwen4_cli_vision.py", "tests/test_qwen4_steering.py",
               "tools/perf_qwen_flash_strata.py", "tools/qualify_qwen_flash_quality.py", "tools/bench_qwen_flash_http.py", "tools/bench_qwen_flash_http_paired.py",
               "tools/qwen_pcie_probe.cu", "tools/qwen_qsa_probe.py", "tools/qualify_qwen_flash_strata.py", "tools/qualify_qwen_flash_long_context.py",
               "tools/qualify_qwen_flash_profile.py", "tools/qualify_qwen_flash_vision.py",
               "tools/build_docs_indexes.py", "tests/test_docs_wiki.py",
               "tools/bench_qwen_flash_agent.py", "tools/qualify_qwen_flash_launcher.py",
               "tools/qwen_flash_launcher.py", "tools/qwen_flash_metrics.py", "tests/test_qwen_flash_tools.py",
               "agent/src/exmekh_agent/exmekh_bridge/agents/collector.py", "agent/src/exmekh_agent/state/active_turns.py",
               "agent/src/exmekh_agent/exmekh_bridge/agents/agent_runner.py", "agent/src/exmekh_agent/exmekh_bridge/capabilities/agent_design.py",
               "agent/tests/test_agent_runner.py",
               "agent/docs/architecture.md", "docs/architecture/clean-code-review-20261008.md",
               "tests/qwen_flash_quality_probe.c", "tests/test_flash_launcher.py",
               "start-ds4-server.sh", "agent/start-ds4-server-flash.sh", "agent/start-ds4-server.sh",
               "docs/performance/qwen-flash-agent-launcher-20261008.md",
               "docs/performance/qwen-flash-prefill-strata-20261008.md",
               "docs/experiments/qwen27-4bis-20261008/README.md", "docs/experiments/qwen27-4bis-20261008/manifest.json",
               "docs/experiments/qwen27-4bis-20261008/shared-integration.patch", "docs/architecture/qwen-flash-hybrid-kv-20261006.md",
               "docs/performance/qwen-flash-strata-20261006.md", "docs/performance/qwen-flash-strata-audit-20261006.md", "docs/performance/qwen-flash-strata-audit-20261007.md"]
    diff = subprocess.check_output([*git_read, "diff", "HEAD", "--", *sources], cwd=ROOT)
    tracked = set(subprocess.check_output([*git_read, "ls-files", "--", *sources], cwd=ROOT, text=True).splitlines())
    for name in sources:
        if name not in tracked:
            part = subprocess.run([*git_read, "diff", "--no-index", "--", "/dev/null", name], cwd=ROOT, capture_output=True)
            if part.returncode not in (0, 1): raise RuntimeError("cannot capture new source diff")
            diff += part.stdout
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidate.diff").write_bytes(diff)
    baseline = json.loads((args.output / "baseline-manifest.json").read_text())
    stat = args.model.stat()
    if stat.st_size != baseline["model"]["bytes"] or stat.st_mtime_ns != baseline["model"]["mtime_ns"]:
        raise ValueError("model changed since the frozen baseline")
    write_json(args.output / "candidate-manifest.json", {
        "baseline_head": subprocess.check_output([*git_read, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output([*git_read, "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: digest(ROOT / name) for name in sources},
        "binary_sha256": {name: digest(ROOT / name) for name in ("ds4", "ds4-server", "ds4-bench", "ds4-eval",
            "ds4_test", "ds4_server_test", "tests/test_qwen4_cuda", "tests/test_qwen4_cpu", "tests/test_qwen4_ranks",
            "tests/test_qwen4_tuning", "tests/test_qwen_mtp_callback", "tests/qwen_mtp_catchup_probe",
            "tests/qwen_flash_quality_probe", "tests/test_server_agentic_payload") if (ROOT / name).is_file()},
        "diff_sha256": digest(args.output / "candidate.diff"), "model": baseline["model"],
        "build": "make, GCC QUALITY_CFLAGS/ffp-contract=off for CPU; CUDA12.4 nvcc --use_fast_math sm_86; see build.log",
        "scope": "Flash qualified launcher/runtime and scheduling diagnostics; 27B 4 BIS retired, ordinary 27B retained as a regression gate"})


def read_tuning_profile(path, key):
    result = {phase: (1, 4, 0) for phase in ("decode", "mtp", "prefill")}
    if not path.is_file(): return result
    parts = path.read_text().splitlines()
    if not parts or parts[0].split() != ["DS4QWEN4TUNE2", key] or len(parts) != 4:
        return result
    try:
        parsed = {}
        for phase, line in zip(result, parts[1:]):
            label, frac, workers, enabled = line.split()
            frac, workers, enabled = float(frac), int(workers), int(enabled)
            if label != phase or not math.isfinite(frac) or not 0 <= frac <= 1 or not 1 <= workers <= 8 or enabled not in (0, 1):
                return result
            parsed[phase] = (frac, workers, enabled)
        return parsed
    except ValueError:
        return result


def tuning_lines(key, selections):
    return [f"DS4QWEN4TUNE2 {key}"] + [
        f"{phase} {frac:g} {workers} {enabled}" for phase, (frac, workers, enabled) in selections.items()]


def calibrate(args):
    common = dict(value.split("=", 1) for value in args.env)
    grid_controls = {"DS4_CUDA_QWEN4_" + name for name in
                     ("HYBRID_MOE", "HYBRID_PREFILL", "CPU_WORKERS", "PCIE_FRAC", "TUNING_PROFILE", "CALIBRATE_INFO")}
    if common.keys() & grid_controls:
        raise ValueError("calibration owns grid controls; use --env for memory, KV and expert-profile configuration")
    reference_environment = common | {"DS4_CUDA_QWEN4_HYBRID_MOE": "0"}
    identity = digest(args.binary)
    cpu_qualified = False
    cpu_prefill_qualified = False
    if args.cpu_quality_report:
        qualification = json.loads(args.cpu_quality_report.read_text())
        gates = {g["gate"]: g for g in qualification.get("gates", [])}
        cpu_qualified = qualification.get("binary_sha256", {}).get("ds4-bench") == identity and all(
            name in gates and gates[name]["exit_code"] == 0
            for name in ("flash-state-plan-cpu", "flash-state-plan-cpu-mtp"))
        cpu_prefill_qualified = cpu_qualified and gates.get("flash-state-plan-cpu-prefill", {}).get("exit_code") == 0
    # Passing replay checks alone cannot promote mixed-prefill arithmetic.
    # Screen every teacher-forced vocabulary row using the same KL/argmax
    # screen as the grid, and bind the candidate CLI to this qualification.
    cpu_prefill_screen = False
    if cpu_prefill_qualified and args.cpu_prefill_quality_report:
        path = args.cpu_prefill_quality_report
        quality = json.loads(path.read_text())
        run = json.loads((path.parent / "candidate/run.json").read_text())
        cases = quality.get("cases", [])
        cpu_prefill_screen = run.get("binary_sha256") == qualification.get("binary_sha256", {}).get("ds4") and \
            bool(cases) and all(c["argmax_rows_equal"] == c["rows"] and c["kl_max"] <= 0.001 for c in cases)
    cpu_prefill_qualified &= cpu_prefill_screen
    baseline = run_bench(args, "calibration-baseline", args.baseline, reference_environment)
    diagnostic_args = None; diagnostic_baseline = None
    if args.diagnostic_frontier and not cpu_qualified:
        diagnostic_args = copy.copy(args)
        diagnostic_args.start = args.diagnostic_frontier
        diagnostic_args.end = args.diagnostic_frontier * 2
        diagnostic_baseline = run_bench(diagnostic_args, "diagnostic-baseline", args.baseline, reference_environment)
    decode_phase = "mtp" if args.mtp else "decode"
    phases = (decode_phase, "prefill")
    rows = []
    for workers in (1, 2, 4, 8):
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            name = f"calibration-w{workers}-p{fraction:g}"
            # Decode quota qualification must start from the qualified GPU
            # prefill. Otherwise an already rejected mixed-prefill distribution
            # contaminates every decode comparison and excludes a correct CPU
            # decode scheduler for the wrong reason. GPU-only prefill retains
            # its separate grid; mixed CPU prefill needs its teacher gate first.
            measure_hybrid_prefill = fraction == 1 or cpu_prefill_qualified
            extra = common | {"DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_HYBRID_PREFILL": str(int(measure_hybrid_prefill)),
                     "DS4_CUDA_QWEN4_CPU_WORKERS": str(workers), "DS4_CUDA_QWEN4_PCIE_FRAC": str(fraction),
                     "DS4_CUDA_QWEN4_CALIBRATE_INFO": "1"}
            compact = fraction < 1 and diagnostic_args is not None
            measured_args = diagnostic_args if compact else args
            reference = diagnostic_baseline if compact else baseline
            reference_name = "diagnostic-baseline" if compact else "calibration-baseline"
            candidate = run_bench(measured_args, name, args.binary, extra)
            if candidate["binary_sha256"] != identity:
                raise RuntimeError("calibration binary changed between grid settings")
            quality = compare_logits(args.output / reference_name / "logits", args.output / name / "logits")
            # This diagnostic threshold never replaces primitive/task-quality
            # gates. A setting failing it cannot be an automatic preference.
            numerical = all(q["argmax_match"] and q["kl"] <= 0.001 for q in quality)
            entry = {"name": name, "fraction": fraction, "workers": workers,
                     "numerical_screen": numerical, "quality": quality, "phases": {},
                     "workload_frontiers": [measured_args.start, measured_args.end], "diagnostic_only": compact,
                     "hybrid_prefill_measured": measure_hybrid_prefill,
                     "cpu_prefill_excluded_before_grid": fraction < 1 and not cpu_prefill_qualified}
            for phase in phases:
                # A small KL is insufficient when placement changes fail the
                # existing checkpoint/replay tolerances. Measure the entire
                # grid, but only qualified arithmetic may become automatic.
                state_qualified = cpu_qualified if phase == decode_phase else cpu_prefill_qualified
                gains, eligible = [], numerical and (fraction == 1 or state_qualified)
                if args.mtp: eligible &= candidate.get("mtp_multi_token_calls", 0) > 0
                for ctx, ref in reference["frontiers"].items():
                    metric = "decode_total" if phase == decode_phase and "decode_total" in ref else "prefill" if phase == "prefill" else "decode"
                    a, b = ref[metric], candidate["frontiers"][ctx][metric]
                    gains.append(b["median"] / a["median"] - 1)
                    eligible &= a["cv"] is not None and b["cv"] is not None and max(a["cv"], b["cv"]) <= 0.05
                entry["phases"][phase] = {"gain": min(gains), "eligible": bool(eligible and min(gains) >= args.min_gain),
                                          "minimum_gain": args.min_gain}
            text = (args.output / name / "runtime.log").read_text(errors="replace")
            match = re.search(r"Qwen tuning key=([0-9a-f]{16})", text)
            entry["key"] = match[1] if match else None
            entry["cpu_state_qualified"] = cpu_qualified
            entry["cpu_prefill_state_qualified"] = cpu_prefill_qualified
            rows.append(entry)
            write_json(args.output / "calibration.json", {"baseline": baseline, "settings": rows, "status": "IN_PROGRESS"})
    keys = {r["key"] for r in rows}
    if None in keys or len(keys) != 1:
        raise ValueError("inconsistent calibration identity")
    key = keys.pop()
    best = {}
    for phase in phases:
        eligible = [r for r in rows if r["phases"][phase]["eligible"]]
        best[phase] = max(eligible, key=lambda r: r["phases"][phase]["gain"]) if eligible else None
    selections = read_tuning_profile(args.tuning_profile, key)
    changed = (decode_phase,) if args.mtp else phases
    for phase in changed:
        row = best[phase]
        selections[phase] = (row['fraction'], row['workers'], 1) if row else (1, 4, 0)
    lines = tuning_lines(key, selections)
    args.tuning_profile.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_profile.write_text("\n".join(lines) + "\n")
    combined = run_bench(args, "calibration-combined", args.binary,
        common | {"DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "auto",
         "DS4_CUDA_QWEN4_TUNING_PROFILE": str(args.tuning_profile.resolve())})
    quality = compare_logits(args.output / "calibration-baseline/logits", args.output / "calibration-combined/logits")
    joint_ok = all(q["argmax_match"] and q["kl"] <= 0.001 for q in quality)
    for ctx, ref in baseline["frontiers"].items():
        for metric in ("decode_total", "prefill"):
            joint_ok &= combined["frontiers"][ctx][metric]["median"] >= ref[metric]["median"] * 0.95
    if not joint_ok:
        for phase in changed: selections[phase] = (1, 4, 0)
        lines = tuning_lines(key, selections)
        args.tuning_profile.write_text("\n".join(lines) + "\n")
        for phase in changed: best[phase] = None
    write_json(args.output / "calibration.json", {"baseline": baseline, "settings": rows, "best": best,
               "combined": combined, "combined_quality": quality, "combined_screen_pass": bool(joint_ok),
               "status": "EXPERIMENTAL_PREFERENCES_ONLY", "tuning_profile": str(args.tuning_profile),
               "configuration_environment": common,
               "cpu_state_qualified": cpu_qualified, "cpu_prefill_state_qualified": cpu_prefill_qualified,
               "cpu_prefill_teacher_screen_pass": bool(cpu_prefill_screen),
               "decode_workload": decode_phase, "profile_selections": selections,
               "mtp_prefill_selection_preserved": bool(args.mtp), "minimum_gain": args.min_gain,
               "cpu_quality_report": str(args.cpu_quality_report) if args.cpu_quality_report else None})
    print(f"Experimental preferences saved to {args.tuning_profile}; task-quality gates remain required", flush=True)


def paired(args):
    """Alternate fresh processes, warming each one before one measured sweep."""
    if args.samples < 5:
        raise ValueError("paired acceptance requires at least five A/B samples")
    extra = dict(value.split("=", 1) for value in args.env)
    baseline_extra = dict(value.split("=", 1) for value in args.baseline_env)
    runs = {"baseline": [], "candidate": []}
    quality = []
    for i in range(args.samples):
        order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        names = {}
        for variant in order:
            name = f"{args.name}-{i:02d}-{variant}"
            names[variant] = name
            runs[variant].append(run_bench(args, name, args.baseline if variant == "baseline" else args.binary,
                                           baseline_extra if variant == "baseline" else extra, samples=1))
        quality += compare_logits(args.output / names["baseline"] / "logits", args.output / names["candidate"] / "logits")
    frontiers = {}
    for ctx in runs["baseline"][0]["frontiers"]:
        frontiers[ctx] = {}
        for phase in ("prefill", "decode", "decode_total"):
            if any(phase not in r["frontiers"][ctx] for rows in runs.values() for r in rows):
                continue
            a, b = (stats([r["frontiers"][ctx][phase]["median"] for r in runs[v]]) for v in ("baseline", "candidate"))
            pair_gains = [c["frontiers"][ctx][phase]["median"] / r["frontiers"][ctx][phase]["median"] - 1
                          for r, c in zip(runs["baseline"], runs["candidate"])]
            frontiers[ctx][phase] = {"baseline": a, "candidate": b, "gain": b["median"] / a["median"] - 1,
                "paired_gains": gain_stats(pair_gains), "positive_pairs": sum(value > 0 for value in pair_gains),
                "minimum_gain": args.min_gain,
                "meets_gain_threshold": b["median"] / a["median"] - 1 >= args.min_gain}
    write_json(args.output / f"{args.name}-paired.json", {"runs": runs, "quality": quality, "frontiers": frontiers,
               "status": "MEASUREMENTS_ONLY; task quality and all regression gates still required"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "freeze-candidate", "bench", "calibrate", "compare", "paired"))
    parser.add_argument("--output", type=Path, default=ROOT / "performance-results/flash-strata-20261006")
    parser.add_argument("--binary", type=Path, default=ROOT / "ds4-bench")
    parser.add_argument("--baseline", type=Path, default=ROOT / "performance-results/flash-strata-20261006/baseline/ds4-bench")
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf")
    parser.add_argument("--prompt", type=Path, default=ROOT / "speed-bench/promessi_sposi.txt")
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--baseline-env", action="append", default=[], help="Explicit paired reference overrides for matched policy/budget comparisons")
    parser.add_argument("--start", type=int, default=256)
    parser.add_argument("--end", type=int, default=512)
    parser.add_argument("--step", type=int, help="Optional fixed frontier step for traces with several domain changes")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--mtp", action="store_true", help="Measure real embedded Flash MTP separately from ordinary decode")
    parser.add_argument("--mtp-draft", type=int, help="Explicit MTP depth; omission keeps the existing runtime default")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--min-gain", type=float, default=0.01,
                        help="Minimum fractional performance gain, default 1%%; quality gates remain unchanged")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--tuning-profile", type=Path, default=ROOT / "performance-results/flash-strata-20261006/tuning.txt")
    parser.add_argument("--cpu-quality-report", type=Path, help="Matching binary identity and passing ordinary/MTP CPU state gates are required for automatic CPU quotas")
    parser.add_argument("--cpu-prefill-quality-report", type=Path, help="Matching full teacher-forced logits are also required before preferring CPU prefill")
    parser.add_argument("--diagnostic-frontier", type=int, help="Compact full forwards for already unqualified CPU quotas; automatic choices still use the full requested workload")
    args = parser.parse_args()
    if not 0 <= args.min_gain <= 1:
        parser.error("--min-gain must be between 0 and 1")
    args.output = args.output.resolve()
    if args.action == "freeze":
        freeze(args)
    elif args.action == "freeze-candidate":
        freeze_candidate(args)
    elif args.action == "bench":
        run_bench(args, args.name, args.binary, dict(value.split("=", 1) for value in args.env))
    elif args.action == "calibrate":
        if args.samples < 5 or args.warmup < 1:
            parser.error("calibration requires warm-up and at least five measured samples")
        if args.diagnostic_frontier and (args.diagnostic_frontier < 9 or 2 * args.diagnostic_frontier + args.tokens >= args.context):
            parser.error("diagnostic frontier must exercise prefill and fit the configured context")
        calibrate(args)
    elif args.action == "paired":
        paired(args)
    else:
        if not args.reference or not args.candidate:
            parser.error("compare requires --reference and --candidate logit directories")
        report = compare_logits(args.reference, args.candidate)
        write_json(args.output / "numerical-comparison.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
