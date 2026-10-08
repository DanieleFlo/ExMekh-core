#!/usr/bin/env python3
"""Measure real ExMekh turns through the unchanged Flash shell launcher.

Run with agent/.venv/bin/python. User state is neither loaded nor saved.
One owned server runs at a time; log boundaries distinguish agent requests,
prefix reuse, fresh prefill chunks and generation.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Self
from unittest.mock import patch

from exmekh_agent.config import config
from exmekh_agent.exmekh_bridge.agents.agent_runner import execute_agent_turn_sync
from exmekh_agent.exmekh_bridge.models import ChatMessage
from exmekh_agent.exmekh_bridge.providers.ds4_prompt import (
    apply_ds4_scope,
    render_ds4_qwen_prompt,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.providers.local import LocalProvider
from perf_qwen_flash_strata import ROOT, digest, environment, stats, write_json
from qwen_flash_launcher import launcher_environment

PROFILES = {
    "launcher": {"DS4_FLASH_PROFILE": "legacy"},
    "launcher-4096": {"DS4_FLASH_PROFILE": "legacy", "DS4_PREFILL_CHUNK": "4096"},
    "standard": {"DS4_FLASH_PROFILE": "standard", "DS4_PREFILL_CHUNK": "2048"},
    "stream-ring-256": {"DS4_FLASH_PROFILE": "legacy", "DS4_PREFILL_CHUNK": "256"},
}
PROFILES["stream-ring-4096"] = PROFILES["stream-ring-256"] | {"DS4_PREFILL_CHUNK": "4096"}
PROFILES["stream-ring-2048"] = PROFILES["stream-ring-256"] | {"DS4_PREFILL_CHUNK": "2048"}
PROFILES["stream-ring-1024"] = PROFILES["stream-ring-256"] | {"DS4_PREFILL_CHUNK": "1024"}
CHUNK = re.compile(r"\b(prefill|system prefill|history prefill) chunk (\d+)/(\d+).*?chunk=([\d.]+) t/s avg=([\d.]+) t/s ([\d.]+)s")
DECODE = re.compile(r"\bgen=(\d+).*?decoding chunk=([\d.]+) t/s avg=([\d.]+) t/s ([\d.]+)s")
DONE = re.compile(r"\bprompt done ([\d.]+)s")


def comparison(records):
    measured = [row for row in records if not row["warmup"]]
    groups = {}
    for row in measured:
        turns = row["turns"]
        group = groups.setdefault(row["profile"], {"prefill_full_tps": [], "ttft_seconds": [],
                                                  "followup_seconds": [], "code_decode_tps": [], "prose_decode_tps": []})
        group["prefill_full_tps"].append(turns[0]["prefill_chunks"][-1]["total"] / turns[0]["prompt_done_seconds"][-1])
        group["ttft_seconds"].append(turns[0]["first_visible_token_seconds"])
        group["followup_seconds"].append(turns[1]["wall_seconds"])
        if len(turns) >= 4:
            for key, index in (("code_decode_tps",2),("prose_decode_tps",3)):
                group[key].append(turns[index]["decode_progress"][-1]["avg_tps"])
    exact = []
    for sample in sorted({row["sample"] for row in measured}):
        pair = [row for row in measured if row["sample"] == sample]
        if len(pair)==2:
            left, right = pair[0]["turns"], pair[1]["turns"]
            if len(left) != len(right):
                raise ValueError("paired agent runs have different turn counts")
            exact.append({"sample": sample, "answers_equal": [a["answer"] == b["answer"]
                          for a,b in zip(left, right)],
                          "initial_prompt_tokens": [row["turns"][0]["prefill_chunks"][-1]["total"] for row in pair]})
    return {"profiles": {name:{key: stats(values) | {"min": min(values), "max": max(values)}
                             for key,values in group.items() if values} for name,group in groups.items()},
            "paired_outputs": exact, "five_measured_pairs": len(exact)>=5}


class GreedyProvider(LocalProvider):
    """Keep the real provider, tools and streaming; fix only sampling for A/B."""
    def __init__(self, *args, capture_directory: Path | None = None, thinking: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_directory = capture_directory
        self._thinking = thinking

    def call(self, request):
        if self._capture_directory is not None:
            messages = [message.message_dump() for message in request.get_messages()]
            extra = request.get_extra() or {}
            if extra.get("agentic"):
                apply_ds4_scope(messages, extra["agentic"])
            tools = OpenAIResponseToolMapper.map_tools(request.get_tools()) if request.get_tools() else []
            number = len(list(self._capture_directory.glob("request-*.json")))
            write_json(self._capture_directory / f"request-{number}.json", {"messages": messages, "tools": tools, "extra": extra})
            (self._capture_directory / f"request-{number}-prompt.txt").write_text(
                render_ds4_qwen_prompt(messages, tools, thinking=self._thinking))
        fixed = ProviderRequest(
            request.get_messages(), extra=(request.get_extra() or {}) | {"temperature": 0, "seed": 424242},
            tool_choice=request.get_tool_choice(), tools=request.get_tools(), stream=request.get_streaming(),
            on_event=request.on_event, reasoning_effort_override=request.get_reasoning_effort_override(),
        )
        return super().call(fixed)


class OwnedServer:
    def __init__(self, directory: Path, env: dict[str, str], port: int) -> None:
        self.directory, self.port = directory, port
        self.work_root = directory.parent / "work"
        self._env = dict(env)
        self._stop_monitor = threading.Event()
        self._samples = []
        self._process = None
        self._monitor = None
        self._log = None
        self._closed = False

    def __enter__(self) -> Self:
        if self._closed or self._process is not None:
            raise RuntimeError("owned server already used; create a new instance")
        # A model catalog on someone else's port must never count as our startup.
        with socket.socket() as probe:
            # Like the server listener, permit our previous connection's TIME_WAIT.
            # This still rejects an active listener on the same address.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.port))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.command = [str(ROOT / "agent/start-ds4-server-flash.sh"), "--host", "127.0.0.1"]
        if self._env.get("DS4_AGENT_BENCH_TRACE") == "1":
            self.command += ["--trace", str(self.directory / "server-trace.txt")]
        started = time.monotonic()
        try:
            self._log = (self.directory / "server.log").open("wb")
            self._process = subprocess.Popen(self.command, cwd=ROOT, env=self._env,
                                             stdout=self._log, stderr=self._log, start_new_session=True)
            while time.monotonic() - started < 300:
                if self._process.poll() is not None:
                    raise RuntimeError(f"launcher exited {self._process.returncode}; see {self.directory / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=1) as response:
                        models = json.load(response)
                    self.model = models["data"][0]["id"]
                    self.startup = time.monotonic() - started
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("launcher model startup timed out")
            self._monitor = threading.Thread(target=self._sample_gpu, daemon=True)
            self._monitor.start()
            return self
        except BaseException:
            self.close()
            raise

    def _sample_gpu(self) -> None:
        while not self._stop_monitor.is_set():
            try:
                result = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                self._samples.append({"time": time.time(), "error": str(error)})
                break
            if result.returncode == 0:
                self._samples.append({"time": time.time(), "used_free_util": result.stdout.strip()})
            self._stop_monitor.wait(2)

    def __exit__(self, *_):
        self.close()

    def _signal_process(self, signal_number: int) -> None:
        try:
            os.killpg(self._process.pid, signal_number)
        except ProcessLookupError:
            # The child can exit between poll/wait and either signal.
            pass

    def close(self) -> None:
        """Drain only this instance's process and monitor, even after failed startup."""
        if self._closed:
            return
        self._stop_monitor.set()
        if self._monitor:
            self._monitor.join(timeout=6)
        if self._process and self._process.poll() is None:
            self._signal_process(signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._signal_process(signal.SIGKILL)
                self._process.wait(timeout=10)
        if self._log:
            self._log.close()
        write_json(self.directory / "gpu-memory.json", self._samples)
        self._closed = True

    def log_offset(self) -> int:
        return (self.directory / "server.log").stat().st_size

    def log_since(self, offset: int) -> str:
        with (self.directory / "server.log").open("rb") as log:
            log.seek(offset)
            return log.read().decode(errors="replace")


def turn(server, directory, prompt, history, context_id, reasoning):
    directory.mkdir(parents=True, exist_ok=True)
    work_root = server.work_root
    work_root.mkdir(exist_ok=True)
    started = time.monotonic()
    first = []
    raw, tree, contexts, tools = [], [], [], []
    cancellation = threading.Event()
    offset = server.log_offset()

    def token(_):
        if not first:
            first.append(time.monotonic() - started)

    def capture(new_history, new_tree):
        raw[:] = new_history
        tree[:] = new_tree

    def capture_context(usage):
        contexts.append({"elapsed": time.monotonic() - started, **usage.model_dump()})

    def capture_tool(event):
        tools.append(event.model_dump(mode="json"))

    deadline = threading.Timer(1200, cancellation.set)
    deadline.start()
    error = None
    try:
        answer = execute_agent_turn_sync(
            user_prompt=prompt, model=server.model,
            base_url=f"http://127.0.0.1:{server.port}/v1", api_key="local", api_type="responses",
            reasoning_effort_str=reasoning, agent_mode="deep_agent",
            capabilities_path_str=str(config.capabilities_path), name_hds="main",
            on_token=token, on_reasoning=lambda *_: None, on_tool_event=capture_tool,
            on_raw_history=capture, on_context=capture_context,
            history=history, agent_context_id=context_id, work_root=str(work_root),
            tool_approval_mode="manual", on_request_manual_approval=lambda _: False,
            cancellation_event=cancellation,
        )
    except Exception as exc:  # noqa: BLE001 -- retain diagnostics for any agent failure
        error = f"{type(exc).__name__}: {exc}"
        answer = ""
    finally:
        deadline.cancel()
    elapsed = time.monotonic() - started
    text = server.log_since(offset)
    (directory / "server-slice.log").write_text(text)
    chunks = [{"phase": m[1], "processed": int(m[2]), "total": int(m[3]),
               "chunk_tps": float(m[4]), "avg_tps": float(m[5]), "elapsed": float(m[6])}
              for m in CHUNK.finditer(text)]
    decode = [{"generated": int(m[1]), "chunk_tps": float(m[2]), "avg_tps": float(m[3]), "elapsed": float(m[4])}
              for m in DECODE.finditer(text)]
    write_json(directory / "history.json", {"history": raw, "skills": tree, "tools": tools})
    report = {"prompt": prompt, "answer": answer, "error": error, "wall_seconds": elapsed,
              "first_visible_token_seconds": first[0] if first else None,
              "contexts": contexts, "prefill_chunks": chunks, "decode_progress": decode,
              "prompt_done_seconds": [float(m[1]) for m in DONE.finditer(text)]}
    write_json(directory / "report.json", report)
    print(json.dumps({"turn": directory.name, "seconds": round(elapsed, 3), "error": error,
                      "prefill": chunks[-1] if chunks else None}), flush=True)
    if error:
        raise RuntimeError(error)
    return [*history, ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=answer, raw_history=raw, skill_history_tree=tree)], report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--greedy", action="store_true", help="Fix temperature=0/seed through the real provider for repeatable A/B")
    parser.add_argument("--decode-prompts", action="store_true", help="Add code/prose generations; one-token answers are not decode benchmarks")
    parser.add_argument("--brief-decode", action="store_true", help="Use shorter code/prose outputs for repeated agent A/B")
    parser.add_argument("--warmup-pairs", type=int, default=0, help="Exclude initial pairs from the final measured comparison")
    parser.add_argument("--trace", action="store_true", help="Capture the server's exact rendered prompt for numerical qualification")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config.data_dir = args.output / "agent-isolated-data"
    identity = {"server_sha256": digest(ROOT / "ds4-server"),
                "launcher_sha256": digest(ROOT / "agent/start-ds4-server-flash.sh"),
                "shared_launcher_sha256": digest(ROOT / "start-ds4-server.sh"),
                "context_capacity": 131072, "api": "responses", "agent_mode": "deep_agent",
                "reasoning": args.reasoning, "greedy": args.greedy, "mtp": True, "vision_loaded": True,
                "decode_prompts": args.decode_prompts, "brief_decode": args.brief_decode,
                "trace": args.trace, "warmup_pairs": args.warmup_pairs}
    identity["harness_source_sha256"] = digest(Path(__file__))
    (args.output / "harness-source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    standard_environment = launcher_environment()
    for sample in range(args.repeats):
        order = args.profiles if sample % 2 == 0 else list(reversed(args.profiles))
        for profile in order:
            dest = args.output / f"sample-{sample}-{profile}"
            if dest.exists():
                raise RuntimeError("retain earlier samples; choose a new output directory")
            placement = standard_environment if profile.startswith("stream-ring-") else {}
            extra = placement | PROFILES[profile] | {"DS4_CTX": "131072", "DS4_PREFILL_CHUNK": PROFILES[profile].get("DS4_PREFILL_CHUNK", "256"),
                "DS4_SERVER_PORT": str(args.port), "DS4_KV_DISK_DIR": str(dest / "disk-kv"),
                "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "auto", "DS4_CUDA_QWEN4_RAM_CACHE_MB": "auto"}
            if args.trace:
                extra["DS4_AGENT_BENCH_TRACE"] = "1"
            env = environment(extra)
            print(f"Starting sample={sample} profile={profile} via agent/start-ds4-server-flash.sh", flush=True)
            with OwnedServer(dest, env, args.port) as server:
                records.append({"sample": sample, "warmup": sample < args.warmup_pairs, "profile": profile, "startup_seconds": server.startup,
                                "command": server.command, "environment": extra, "turns": []})
                context_id = "flash-agent-prefill-test"
                history = []
                prompts = [
                    ("This is a latency test. Answer only with READY. Do not inspect files, run tools, "
                     "delegate work or open skills. End the turn normally."),
                    ("This is a follow-up latency test. Answer only with DONE. Do not inspect files, run tools, "
                     "delegate work or open skills. End the turn normally."),
                ]
                if args.decode_prompts:
                    prompts.extend([
                        "Without using tools or delegating, write an iterative binary search in Python and explain it in about 80 words.",
                        "Without using tools or delegating, explain how refraction and reflection in water droplets produce a rainbow in about 100 words.",
                    ])
                    if args.brief_decode:
                        prompts[-2:] = [
                            "Without tools or delegation, write only a Python iterative binary search function that returns the index or -1. No explanation.",
                            "Without tools or delegation, explain how a rainbow forms in about 40 words. Mention refraction and reflection.",
                        ]
                for index, prompt in enumerate(prompts):
                    turn_directory = dest / f"turn-{index}"
                    turn_directory.mkdir(exist_ok=True)
                    provider_factory = partial(GreedyProvider, capture_directory=turn_directory,
                                               thinking=args.reasoning.strip().lower() != "none")
                    provider_scope = patch("exmekh_agent.exmekh_bridge.agents.agent_runner.LocalProvider", provider_factory) if args.greedy else nullcontext()
                    with provider_scope:
                        history, report = turn(server, dest / f"turn-{index}", prompt, history, context_id, args.reasoning)
                    records[-1]["turns"].append(report)
                    write_json(args.output / "report.json", {"identity": identity, "records": records, "status": "IN_PROGRESS"})
            if digest(ROOT / "ds4-server") != identity["server_sha256"]:
                raise RuntimeError("server binary changed during paired trials")
    write_json(args.output / "report.json", {"identity": identity, "records": records, "comparison": comparison(records),
               "status": "COMPLETE_MEASUREMENTS; numerical/state gates are separate"})


if __name__ == "__main__":
    main()
