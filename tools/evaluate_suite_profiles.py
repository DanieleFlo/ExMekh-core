"""Evaluate explicit numerical candidates without changing the public catalog.

Missing measurements are NEED_MORE_DATA. KEEP requires all numerical, held-out
task and end-to-end performance gates; a storage ratio cannot promote compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.qualify_suite_release import timing
except ModuleNotFoundError:
    from qualify_suite_release import timing

GATES = (
    "input_identity",
    "kernel",
    "model",
    "heldout",
    "retrieval_or_image_quality",
    "memory",
    "llm_regression",
)


def evaluate(candidate, measurement):
    result = {
        "id": candidate["id"],
        "public": False,
        "verdict": "NEED_MORE_DATA",
        "missing": [],
    }
    if measurement.get("native_binding") == "UNSUPPORTED":
        return {
            **result,
            "verdict": "REJECT",
            "reason": "No native binding for this storage/compute profile",
        }
    gates = measurement.get("gates", {})
    unknown = set(gates) - set(GATES)
    if unknown or any(
        value not in ("PASS", "FAIL", "NOT_VERIFIED") for value in gates.values()
    ):
        raise ValueError("unknown or invalid candidate gate")
    failed = [gate for gate in GATES if gates.get(gate) == "FAIL"]
    if failed:
        return {**result, "verdict": "REJECT", "failed": failed}
    result["missing"] = [gate for gate in GATES if gates.get(gate) != "PASS"]
    if measurement.get("native_binding") != "PASS":
        result["missing"].append("native_binding")
    if not measurement.get("artifact_sha256") or not measurement.get("sources"):
        result["missing"].append("artifact_and_sources")
    if measurement.get("heldout_split") != "held-out":
        result["missing"].append("heldout_split")
    if not measurement.get("warmup") or not measurement.get("platform"):
        result["missing"].append("warmup_and_platform")
    for name in ("baseline", "candidate"):
        samples = measurement.get(f"{name}_seconds", [])
        if samples:
            result[name] = timing(samples)
            if result[name]["verdict"] != "STABLE":
                result["missing"].append(f"{name}_stable_five_samples")
        else:
            result["missing"].append(f"{name}_stable_five_samples")
    if "baseline" in result and "candidate" in result:
        speedup = result["baseline"]["median"] / result["candidate"]["median"] - 1
        result["speedup"] = speedup
        noise = max(result["baseline"]["cv"], result["candidate"]["cv"])
        if speedup < -0.05:
            return {
                **result,
                "verdict": "REJECT",
                "reason": "End-to-end regression exceeds 5%",
            }
        # A memory-saving profile may explicitly trade latency for feasibility.
        if candidate.get("objective") == "latency" and speedup <= noise:
            result["missing"].append("improvement_above_noise")
    if not result["missing"]:
        result["verdict"] = "KEEP"
        result["public"] = bool(candidate.get("offer_when_qualified", False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=Path("config/native-suite-candidates.json")
    )
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    measurements = json.loads(args.measurements.read_text())
    identifiers = [c["id"] for c in registry["candidates"]]
    if len(identifiers) != len(set(identifiers)) or set(measurements) - set(
        identifiers
    ):
        raise ValueError("duplicate or unknown candidate")
    report = {
        "version": 1,
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "measurements_sha256": hashlib.sha256(
            args.measurements.read_bytes()
        ).hexdigest(),
        "candidates": [
            evaluate(c, measurements.get(c["id"], {})) for c in registry["candidates"]
        ],
        "excluded": registry["excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
