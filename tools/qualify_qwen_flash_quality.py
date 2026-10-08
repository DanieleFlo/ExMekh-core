#!/usr/bin/env python3
"""Matched full-logit and teacher-forced loss diagnostics for Flash variants."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from perf_qwen_flash_strata import ROOT, digest, environment, write_json
from qwen_flash_metrics import compare_teacher_logits

CASES = {
    "code": ("Write a short C function that clamps an integer to a range.",
             "int clamp(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }"),
    "prose": ("Riassumi in italiano perché una cache può accelerare un programma.",
              "Una cache conserva dati già letti o calcolati, riducendo accessi ripetuti alla memoria più lenta."),
    "tool": (("<|im_start|>system\nUse the tool only when requested. Available tool: lookup(query: string).<|im_end|>\n"
             "<|im_start|>user\nLook up the opening hours of the town library.<|im_end|>\n"
             "<|im_start|>assistant\n"),
             '<tool_call>\n{"name":"lookup","arguments":{"query":"town library opening hours"}}\n</tool_call>'),
}

def tokenize(binary, model, text):
    data = subprocess.check_output([str(binary.resolve()), "-m", str(model.resolve()),
                                  "--dump-tokens", "--raw-prompt", "-p", text], cwd=ROOT)
    return json.loads(data.splitlines()[0])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf")
    p.add_argument("--baseline", type=Path, default=ROOT / "performance-results/flash-strata-20261006/baseline/ds4")
    p.add_argument("--binary", type=Path, default=ROOT / "ds4")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", action="append", default=[])
    a = p.parse_args()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for name, (prompt, target) in CASES.items():
        if not prompt.startswith("<|im_start|>"):
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        fixtures[name] = {"prompt": prompt, "target": target, "prompt_ids": tokenize(a.baseline, a.model, prompt),
                          "target_ids": tokenize(a.baseline, a.model, target)}
    write_json(a.output / "fixtures.json", fixtures)
    for name, binary, extra in (("baseline", a.baseline, {}),
                                ("candidate", a.binary, dict(v.split("=", 1) for v in a.env))):
        out = a.output / name; out.mkdir(exist_ok=True)
        manifest = out / "inputs.tsv"
        manifest.write_text("".join(
            ",".join(map(str, f["prompt_ids"])) + "|" + ",".join(map(str, f["target_ids"])) +
            f"\t{out / (case + '.bin')}\n" for case, f in fixtures.items()))
        env = environment(extra) | {"DS4_QWEN4_GPU": "1", "DS4_QWEN4_FT_LIST": str(manifest)}
        command = [str(binary.resolve()), "--cuda", "-m", str(a.model.resolve()),
                   "--ctx", "8192", "--first-token-test", "-p", "hello"]
        with (out / "runtime.log").open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=900, check=False)
        write_json(out / "run.json", {"command": command, "environment": extra,
                   "binary_sha256": digest(binary), "exit_code": result.returncode})
        if result.returncode: raise RuntimeError(f"{name} failed: {out / 'runtime.log'}")
    reports = []
    for case, fixture in fixtures.items():
        target = fixture["target_ids"]
        x, y = [np.fromfile(a.output / name / (case + ".bin"), dtype=np.float32).reshape(len(target), -1).astype(np.float64)
                for name in ("baseline", "candidate")]
        metrics = compare_teacher_logits(x, y, target)
        metrics["loss_baseline"] = metrics.pop("loss_reference")
        reports.append({"case": case, **metrics})
    write_json(a.output / "report.json", {"cases": reports, "status": "DIAGNOSTICS; no tolerance or golden change"})
    print(json.dumps(reports, indent=2))

if __name__ == "__main__":
    main()
