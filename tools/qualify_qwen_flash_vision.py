#!/usr/bin/env python3
"""Run local vision/API and multimodal checkpoint gates serially on one GPU."""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.request
from perf_qwen_flash_strata import ROOT, environment, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+", choices=("flash", "27b"), default=["flash", "27b"])
    p.add_argument("--server", type=Path, default=ROOT / "ds4-server")
    p.add_argument("--payload", type=Path, default=ROOT / "tests/test_server_agentic_payload")
    p.add_argument("--qwen-support", type=Path, help="Explicit Qwen 27B support model for MTP/vision qualification")
    p.add_argument("--default-flash-mtp", action="store_true", help="Exercise the experimental automatic Flash activation and --no-mtp override rather than explicit --mtp")
    p.add_argument("--env", action="append", default=[], help="Additional candidate-only experimental controls")
    p.add_argument("--chunk", type=int, default=256, help="Candidate production prefill chunk; ordinary payload reference retains 256")
    p.add_argument("--require-pass", action="store_true")
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    records = []
    env = environment({"DS4_CUDA_QWEN4_HYBRID_MOE": "1", "DS4_CUDA_QWEN4_PCIE_FRAC": "1",
                       "DS4_CUDA_QWEN4_KV_STREAMING": "1", "DS4_TEST_CTX": "8192", "DS4_TEST_PREFIX_TOKENS": "512"} |
                      dict(value.split("=", 1) for value in a.env))
    if a.default_flash_mtp: env["DS4_QWEN4_MTP_AUTO"] = "1"
    for family in a.families:
        model = ROOT / ("gguf/Qwen3.8-Flash-Next-Q2.gguf" if family == "flash" else "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")
        vision = ROOT / ("gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf" if family == "flash" else "gguf/mmproj-Qwen3.8-27B-F16.gguf")
        for mtp in (False, True):
            name = family + ("-mtp" if mtp else "-target")
            cmd = [str(a.payload.resolve()), str(model), "--vision", str(vision), "--image", str(ROOT / "tests/vision-fixtures/glm53/earth.jpg")]
            if mtp:
                if family=="27b" and a.qwen_support: cmd += ["--mtp-model",str(a.qwen_support.resolve())]
                else: cmd += ["--embedded-mtp" if family == "flash" else "--mtp"]
            payload_pass = True
            for variant in ("ordinary", "candidate"):
                dump = a.output / (name + "-" + variant + "-frontier.bin")
                extra = {"DS4_TEST_CTX": "8192", "DS4_TEST_PREFIX_TOKENS": "512",
                         "DS4_TEST_FRONTIER_LOGITS_PATH": str(dump)}
                payload_env = (environment(extra) if variant == "ordinary" else env | extra | {"DS4_TEST_PREFILL_CHUNK": str(a.chunk)})
                started = time.monotonic()
                with (a.output / (name + "-" + variant + "-payload.log")).open("wb") as log:
                    result = subprocess.run(cmd, cwd=ROOT, env=payload_env, stdout=log, stderr=log, timeout=1200)
                payload_pass &= result.returncode == 0
                records.append({"gate": name + "-" + variant + "-payload", "exit_code": result.returncode,
                                "seconds": time.monotonic() - started, "command": cmd})
                write_json(a.output / "report.json", {"records": records, "status": "IN_PROGRESS"})
                print(records[-1], flush=True)
            if payload_pass:
                x, y = [(a.output / (name + "-" + variant + "-frontier.bin")).read_bytes()
                        for variant in ("ordinary", "candidate")]
                magic, vocab, tokens = struct.unpack_from("<III", x)
                valid = magic == 0x31465651 and len(x) == 12 + 4 * (vocab + tokens)
                valid = valid and all(math.isfinite(v) for (v,) in struct.iter_unpack("<f", x[12 + 4 * tokens:]))
                records.append({"gate": name + "-full-frontier", "exit_code": 0 if valid and x == y else 1,
                                "vocab": vocab, "tokens": tokens, "bit_exact": valid and x == y,
                                "comparison": "same frozen binary, ordinary versus experimental paths"})
                print(records[-1], flush=True)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            server_cmd = [str(a.server.resolve()), "--cuda", "-m", str(model), "--vision", str(vision),
                          "--ctx", "8192", "--prefill-chunk", str(a.chunk), "--host", "127.0.0.1", "--port", str(port), "--kv-cache-min-tokens", "128"]
            if mtp:
                if family=="27b" and a.qwen_support: server_cmd += ["--mtp-model",str(a.qwen_support.resolve())]
                elif family != "flash" or not a.default_flash_mtp: server_cmd += ["--mtp"]
            elif family == "flash" and a.default_flash_mtp:
                server_cmd += ["--no-mtp"]
            with (a.output / (name + "-server.log")).open("wb") as log:
                started = time.monotonic(); proc = subprocess.Popen(server_cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
                try:
                    while True:
                        if proc.poll() is not None: raise RuntimeError("vision server exited during startup")
                        try:
                            with urllib.request.urlopen(base + "/v1/models", timeout=1) as response: json.load(response)
                            break
                        except (OSError, TimeoutError):
                            if time.monotonic() - started > 300: raise RuntimeError("vision server startup timeout")
                            time.sleep(0.25)
                    cmd = ["python3", "tests/test_qwen35_vision_api.py", "--url", base, "--model", model.stem,
                           "--report", str(a.output / (name + "-api.json"))]
                    with (a.output / (name + "-api.log")).open("wb") as api_log:
                        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=api_log, stderr=api_log, timeout=1200)
                    records.append({"gate": name + "-api", "exit_code": result.returncode,
                                    "seconds": time.monotonic() - started, "command": server_cmd})
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            write_json(a.output / "report.json", {"records": records, "status": "IN_PROGRESS"})
            print(records[-1], flush=True)
            if family == "flash" and a.default_flash_mtp:
                auto_enabled = "Flash NextN enabled by default" in (a.output / (name + "-server.log")).read_text()
                records.append({"gate": name + "-default-nextn-selection", "exit_code": 0 if auto_enabled == mtp else 1,
                                "automatic_nextn_enabled": auto_enabled, "expected": mtp})
    write_json(a.output / "report.json", {"records": records, "all_pass": all(r["exit_code"] == 0 for r in records),
               "binary_sha256": {str(p): digest(p) for p in (a.server, a.payload)}, "status": "COMPLETE"})
    if a.require_pass and any(r["exit_code"] != 0 for r in records):
        p.exit(1, "Vision qualification failed; every result is retained in report.json\n")


if __name__ == "__main__": main()
