#!/usr/bin/env python3
"""Validate learned routing, incompatible fallback and matched policy budgets."""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
from perf_qwen_flash_strata import ROOT, digest, run_bench, compare_logits, write_json
from qualify_qwen_flash_quality import CASES, tokenize


def inspect_profile(path):
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError("truncated profile")
    magic, fingerprint, count, checksum = struct.unpack_from("<4Q", data)
    if magic != 0x31504851345344 or len(data) != 32 + count * 8:
        raise ValueError("invalid profile geometry")
    computed = 14695981039346656037
    for byte in data[32:]:
        computed = ((computed ^ byte) * 1099511628211) & ((1 << 64) - 1)
    if checksum != computed:
        raise ValueError("invalid checksum")
    counts = struct.unpack_from(f"<{count}Q", data, 32)
    return {"sha256": digest(path), "fingerprint": f"{fingerprint:016x}", "entries": count,
            "checksum_valid": True, "layer_totals": [sum(counts[i:i + 512]) for i in range(0, count, 512)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "ds4-bench")
    p.add_argument("--cli", type=Path, default=ROOT / "ds4")
    p.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf")
    p.add_argument("--prompt", type=Path, default=ROOT / "speed-bench/promessi_sposi.txt")
    p.add_argument("--include-stable", action="store_true", help="Also qualify static per-layer placement with and without a learned profile")
    p.add_argument("--skip-domain-shift", action="store_true", help="Skip the additional code/prose/tool trace; retain ordinary policy qualification")
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    original = a.profile.read_bytes(); identity = digest(a.profile)
    report = {"input_profile": inspect_profile(a.profile), "policies": {}, "fallbacks": {}}
    learned_fixtures = a.profile.parent / "quality-profile-learn/fixtures.json"
    if learned_fixtures.is_file():
        fixtures = json.loads(learned_fixtures.read_text())
        expected = sum(len(f["prompt_ids"]) + len(f["target_ids"]) - 1 for f in fixtures.values()) * 10
        totals = report["input_profile"]["layer_totals"]
        report["count_once"] = {"expected_per_target_layer": expected,
                                "pass": len(totals) == 49 and totals[:48] == [expected] * 48 and totals[48] == 0}
        if not report["count_once"]["pass"]:
            raise ValueError("router counts were duplicated or omitted")
    a.baseline = a.binary; a.start = 256; a.end = 512; a.context = 2048
    a.chunk = 256; a.tokens = 8; a.warmup = 2; a.samples = 5; a.timeout = 1200
    policies = [("recent", {}), ("lru", {"DS4_CUDA_QWEN4_EXPERT_POLICY": "lru"}),
                ("learned", {"DS4_CUDA_QWEN4_EXPERT_PROFILE": str(a.profile.resolve())})]
    if a.include_stable:
        policies.extend([("stable", {"DS4_CUDA_QWEN4_EXPERT_POLICY": "stable"}),
                         ("stable-learned", {"DS4_CUDA_QWEN4_EXPERT_POLICY": "stable",
                          "DS4_CUDA_QWEN4_EXPERT_PROFILE": str(a.profile.resolve())})])
    def isolated_profile(extra, name):
        # The runtime atomically saves learned counts to its configured path.
        # Every policy starts from the same immutable training artifact; a
        # previous benchmark must not train the next policy's initial ranking.
        extra = dict(extra)
        if "DS4_CUDA_QWEN4_EXPERT_PROFILE" in extra:
            path = a.output / (name + "-profile.bin")
            path.write_bytes(original)
            extra["DS4_CUDA_QWEN4_EXPERT_PROFILE"] = str(path)
        return extra
    for policy, extra in policies:
        report["policies"][policy] = run_bench(a, "policy-" + policy, a.binary, isolated_profile(extra, "policy-" + policy))
        if policy != "recent":
            report["policies"][policy]["logits"] = compare_logits(a.output / "policy-recent/logits",
                                                                  a.output / ("policy-" + policy) / "logits")
        write_json(a.output / "report.json", report)
    if not a.skip_domain_shift:
        trace = copy.copy(a); trace.start = 256; trace.end = 768; trace.step = 256
        trace.prompt = a.output / "domain-shift.txt"
        text = ""; boundaries = []
        for domain, (_, target) in CASES.items():
            unit = target + "\n"
            count = len(tokenize(a.cli, a.model, unit))
            text += unit * ((256 + count - 1) // count + 1)
            boundaries.append({"domain": domain, "ends_at_token": len(tokenize(a.cli, a.model, text))})
        text += CASES["tool"][1] * 2
        total = len(tokenize(a.cli, a.model, text))
        if not (boundaries[0]["ends_at_token"] < 512 and boundaries[1]["ends_at_token"] < 768 and total >= 768 + a.tokens):
            raise ValueError("domain trace does not cross code/prose/tool at the measured frontiers")
        trace.prompt.write_text(text)
        report["domain_shift"] = {"boundaries": boundaries, "total_tokens": total,
                                  "frontiers": [256, 512, 768], "policies": {}}
        for policy, extra in policies:
            name = "domain-" + policy
            run = run_bench(trace, name, a.binary, isolated_profile(extra, name))
            if policy != "recent":
                run["logits"] = compare_logits(a.output / "domain-recent/logits", a.output / name / "logits")
            report["domain_shift"]["policies"][policy] = run
            write_json(a.output / "report.json", report)
    source = a.output / "quality-profile.bin"; source.write_bytes(original)
    command = ["python3", "tools/qualify_qwen_flash_quality.py", "--binary", str(a.cli.resolve()),
               "--output", str(a.output / "quality"), "--env", "DS4_CUDA_QWEN4_EXPERT_PROFILE=" + str(source)]
    with (a.output / "quality-driver.log").open("wb") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, timeout=1200)
    if result.returncode:
        raise RuntimeError("loaded profile quality failed")
    report["quality"] = json.loads((a.output / "quality/report.json").read_text())
    report["saved_profile"] = inspect_profile(source)
    fixtures = json.loads((a.output / "quality/fixtures.json").read_text())
    expected = sum(len(f["prompt_ids"]) + len(f["target_ids"]) - 1 for f in fixtures.values()) * 10
    before = report["input_profile"]["layer_totals"]
    after = report["saved_profile"]["layer_totals"]
    delta = [right - left for left, right in zip(before, after)]
    report["learned_count_once"] = {"domains": list(fixtures), "expected_per_target_layer": expected,
        "delta_per_layer": delta, "pass": len(delta) == 49 and delta[:48] == [expected] * 48 and delta[48] == 0}
    small = copy.copy(a); small.start = 16; small.end = 32; small.warmup = 0; small.samples = 1
    run_bench(small, "fallback-reference", a.binary, {})
    wrong_model = bytearray(original); wrong_model[8] ^= 1
    corrupt = bytearray(original); corrupt[-1] ^= 1
    for name, data in (("truncated", original[:40]), ("fingerprint", wrong_model), ("checksum", corrupt)):
        path = a.output / (name + ".bin"); path.write_bytes(data)
        extra = {"DS4_CUDA_QWEN4_EXPERT_PROFILE": str(path), "DS4_CUDA_QWEN4_PROFILE_PRELOAD": "0"}
        run = run_bench(small, "fallback-" + name, a.binary, extra)
        numeric = compare_logits(a.output / "fallback-reference/logits", a.output / ("fallback-" + name) / "logits")
        ignored = "incompatible/corrupt Qwen expert profile ignored" in (a.output / ("fallback-" + name) / "runtime.log").read_text()
        report["fallbacks"][name] = {"ignored": ignored, "logits": numeric, "run": run}
    report["source_unchanged"] = identity == digest(a.profile)
    report["all_pass"] = report["source_unchanged"] and all(c["bit_exact"] for c in report["quality"]["cases"])
    report["all_pass"] &= all(v["ignored"] and all(q["bit_exact"] for q in v["logits"]) for v in report["fallbacks"].values())
    report["all_pass"] &= all(all(q["bit_exact"] for q in v.get("logits", [])) for v in report["policies"].values())
    report["all_pass"] &= report["learned_count_once"]["pass"]
    report["all_pass"] &= all(all(q["bit_exact"] for q in v.get("logits", []))
        for v in report.get("domain_shift", {}).get("policies", {}).values())
    report["status"] = "QUALIFICATION; policy timings are diagnostic, no automatic promotion"
    write_json(a.output / "report.json", report)
    if not report["all_pass"]:
        raise RuntimeError("expert profile qualification failed")


if __name__ == "__main__":
    main()
