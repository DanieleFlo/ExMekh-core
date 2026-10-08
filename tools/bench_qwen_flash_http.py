#!/usr/bin/env python3
"""Local HTTP TTFT measurements with protocol metadata excluded from TTFT."""
import argparse
import copy
import json
import re
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from perf_qwen_flash_strata import ROOT, digest, environment, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=ROOT / "ds4-server")
    p.add_argument("--model", type=Path, default=ROOT / "gguf/Qwen3.8-Flash-Next-Q2.gguf")
    p.add_argument("--fixture", type=Path, default=ROOT / "performance-results/flash-prefill-async-20261004/real-agent-ciao-request.json")
    p.add_argument("--vision", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=131072)
    p.add_argument("--requests", type=int, default=3)
    p.add_argument("--new-prompts", action="store_true", help="Alternate new prompts at warm expert cache and prefix hits")
    p.add_argument("--env", action="append", default=[])
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    cmd = [str(a.binary.resolve()), "--cuda", "-m", str(a.model.resolve()), "--ctx", str(a.context),
           "--prefill-chunk", "256", "--mtp", "--mtp-timing", "--host", "127.0.0.1", "--port", str(port), "--kv-cache-min-tokens", "128"]
    # Flash has no automatic disk prefix cache. A repeated request ends before
    # the live generated suffix, so it needs the saved prompt frontier.
    cmd += ["--kv-disk-dir", str(a.output / "kv-cache"), "--kv-disk-space-mb", "512"]
    if a.vision: cmd += ["--vision", str(a.vision.resolve())]
    env = environment(dict(v.split("=", 1) for v in a.env))
    identity = digest(a.binary); records = []
    with (a.output / "server.log").open("wb") as log:
        started = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            while True:
                if proc.poll() is not None: raise RuntimeError("server exited during startup")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=1) as r: json.load(r)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() - started > 300: raise RuntimeError("startup timeout")
                    time.sleep(0.25)
            startup = time.monotonic() - started
            fixture = json.loads(a.fixture.read_text())
            for i in range(a.requests):
                body = copy.deepcopy(fixture)
                phase = "cold" if i == 0 else "prefix_reuse"
                if a.new_prompts and i:
                    phase = "new_prompt_warm_experts" if i % 2 else "prefix_reuse"
                    tag = f"Benchmark case {(i + 1) // 2}.\n"
                    content = body["input"][0]["content"]
                    if isinstance(content, str):
                        body["input"][0]["content"] = tag + content
                    elif isinstance(content, list):
                        part = next(c for c in content if "text" in c)
                        part["text"] = tag + part["text"]
                    else:
                        raise ValueError("fixture needs a text first message for --new-prompts")
                body.update(stream=True, temperature=0, max_output_tokens=32)
                # Same body exercises prefix reuse. Classification uses reported
                # cached tokens; do not infer a hit just from request order.
                write_json(a.output / f"request-{i}.json", body)
                req = urllib.request.Request(base + "/v1/responses", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
                t0 = time.monotonic(); first = None; first_type = None; events = []
                with urllib.request.urlopen(req, timeout=900) as response:
                    for raw in response:
                        if not raw.startswith(b"data:"): continue
                        data = raw[5:].strip()
                        if data == b"[DONE]": continue
                        event = json.loads(data); events.append(event)
                        if first is None and event.get("type") in ("response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta",
                                                                   "response.function_call_arguments.delta") and event.get("delta"):
                            first = time.monotonic() - t0
                            first_type = event["type"]
                final = next((e["response"] for e in reversed(events) if "response" in e and "usage" in e["response"]), {})
                if first is None or final.get("status") not in ("completed", "incomplete"):
                    raise RuntimeError("missing generated delta or failed response")
                row = {"request": i, "requested_phase": phase, "startup_seconds": startup, "ttft_seconds": first,
                       "first_delta_type": first_type,
                       "seconds": time.monotonic() - t0, "usage": final.get("usage"), "status": final.get("status"),
                       "generated_text": "".join(e.get("delta", "") for e in events
                            if e.get("type") in ("response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta"))}
                records.append(row); write_json(a.output / f"events-{i}.json", events)
                print(json.dumps(row), flush=True)
                write_json(a.output / "report.json", {"binary_sha256": identity, "command": cmd,
                           "environment": dict(v.split("=", 1) for v in a.env), "records": records})
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=60)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    if digest(a.binary) != identity: raise RuntimeError("server binary changed during measurement")
    text = (a.output / "server.log").read_text(errors="replace")
    mtp = [{"cycles": int(c), "accepted": int(n), "acceptance_percent": float(rate)}
           for c, n, rate in re.findall(r"Qwen3\.8 mtp: (\d+) verify cycles, (\d+) drafts accepted \(([0-9.]+)%\)", text)]
    write_json(a.output / "report.json", {"binary_sha256": identity, "command": cmd,
               "environment": dict(v.split("=", 1) for v in a.env), "records": records,
               "mtp_session_totals": mtp, "status": "HTTP_DIAGNOSTICS; not five paired samples"})

if __name__ == "__main__": main()
