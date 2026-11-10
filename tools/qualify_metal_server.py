#!/usr/bin/env python3
"""Bound a native 8192/64 server plus an actual frontend/HTTP client check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

if __package__:
    from .qualification_checkpoint import (
        atomic_json,
        digest_file,
        exclusive_lock,
        qualification_output,
    )
    from .qualification_client import QualificationClient
    from .qualification_memory import MemoryPolicy
    from .qualification_timeouts import add_timeout_multiplier
    from .qualify_metal_command import (
        interrupt_on_termination,
        sample,
        stop,
    )
else:
    from qualification_checkpoint import (
        atomic_json,
        digest_file,
        exclusive_lock,
        qualification_output,
    )
    from qualification_client import QualificationClient
    from qualification_memory import MemoryPolicy
    from qualification_timeouts import add_timeout_multiplier
    from qualify_metal_command import (
        interrupt_on_termination,
        sample,
        stop,
    )

ROOT = Path(__file__).resolve().parents[1]
MAX_QUALIFICATION_SECONDS = 21600


def cleanup_processes(
    report: dict, client: subprocess.Popen | None, server: subprocess.Popen | None
) -> None:
    """Record every cleanup outcome without promoting forced or failed shutdown."""
    errors, forced = [], []
    for process in (client, server):
        if process is None:
            continue
        try:
            if not stop(process, grace_seconds=30 if process is server else 5):
                forced.append(process.pid)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append({"pid": process.pid, "error": repr(error)})
    if errors:
        report["cleanup_errors"] = errors
    report["cleanup_forced"] = forced
    report["server_exit_code"] = server.returncode if server is not None else None
    if report["status"] in ("PASS", "PASS_SEGMENTED"):
        if errors or forced:
            report["status"] = "NOT_VERIFIED"
        elif report["server_exit_code"] != 0:
            report.update(status="NOT_VERIFIED", error="server did not exit cleanly")


def prepare_suite_configuration(
    source: Path, output: Path, multiplier: int
) -> tuple[Path, int]:
    """Scale only this qualification's copy of the service operation deadline."""
    configuration = json.loads(source.read_text())
    base_timeout = configuration.get("timeout_seconds", 600)
    if (
        type(base_timeout) is not int
        or base_timeout <= 0
        or multiplier not in (1, 2, 4)
        or base_timeout * multiplier > 86400
    ):
        raise ValueError("scaled suite operation timeout must be in (0,86400]")
    (output / "media").mkdir()
    configuration["media_directory"] = str(output / "media")
    configuration["timeout_seconds"] = base_timeout * multiplier
    prepared = output / "suite-config.json"
    prepared.write_text(json.dumps(configuration, indent=2) + "\n")
    return prepared, base_timeout


def resume_profile(
    args,
    report: dict,
    invocation: QualificationClient,
    output: Path,
    environment: dict | None = None,
) -> Path:
    """Pin actual executable, model bytes, assets, policy and test code."""
    assets = {}
    for source in (ROOT / "metal").glob("*.metal"):
        assets[str(source.resolve())] = digest_file(source)
    for library in (ROOT / "build/suite-media/lib").glob("*.dylib"):
        path = library.resolve()
        if str(path) not in assets:
            assets[str(path)] = digest_file(path)
    if args.suite_config:
        configuration = json.loads(args.suite_config.read_text())
        for model in configuration.get("models", []):
            for key in ("index", "text_asset", "media_asset"):
                if model.get(key):
                    path = (ROOT / model[key]).resolve()
                    assets[str(path)] = digest_file(path)
    command = report["command"]
    for option in ("-m", "--mtp-model", "--vision"):
        if option in command:
            path = Path(command[command.index(option) + 1])
            assets[str(path.resolve())] = digest_file(path)
    sources = dict(report["client_sources_sha256"])
    for name in (
        "tools/qualify_metal_server.py",
        "tools/qualify_suite_release.py",
        "tools/qualification_checkpoint.py",
        "tools/qualification_suite.py",
        "tools/qualification_memory.py",
        "tools/qualification_client.py",
        "tools/qualification_timeouts.py",
        "tools/qualify_metal_command.py",
    ):
        sources[name] = digest_file(ROOT / name)
    hardware = {
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for name in ("hw.model", "hw.memsize", "machdep.cpu.brand_string"):
            hardware[name] = subprocess.check_output(
                ["sysctl", "-n", name], text=True
            ).strip()
    environment = os.environ if environment is None else environment
    # These controller-owned paths intentionally change with each attempt.
    # Hash other native overrides so reports never expose their raw values.
    attempt_paths = {
        "DS4_QUALIFICATION_PROFILE",
        "DS4_SUITE_E2E_ARTIFACTS",
        "DS4_AGENTIC_TEST_CHECKPOINT_ROOT",
        "DS4_SUITE_TEST_PHASE_BARRIER",
        "DS4_SUITE_PHASE_CONTROL",
    }
    identity = {
        "version": 1,
        "executables": report["executables_sha256"],
        "assets": assets,
        "sources": sources,
        "hardware": hardware,
        "runtime_environment": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in environment.items()
            if key.startswith(("DS4_", "MTL_", "METAL_", "GGML_", "DYLD_"))
            and key not in attempt_paths
        },
        "configuration": report["qualification_profile"],
        "memory_policy": {k: report[k] for k in MemoryPolicy().as_dict()},
        "suite_config_sha256": report.get("suite_config_sha256"),
        "client": invocation.identity_command(),
        "mtp": args.mtp,
        "vision": args.vision,
        "resource_metrics": report["resource_metrics"],
        "phase_profile": report["phase_profile"],
        "phase_server": bool(getattr(args, "phase_server", False)),
        "kv_seed": report["kv_seed"],
        "footprint_limit_gib": report["footprint_limit_gib"],
    }
    root_profile = args.output.resolve() / "qualification-profile.json"
    if (
        getattr(args, "resume", False)
        and json.loads(root_profile.read_text())["identity"] != identity
    ):
        raise ValueError("qualification identity changed; use a new output directory")
    profile = {"version": 1, "identity": identity, "server_instance": uuid.uuid4().hex}
    path = output / "qualification-profile.json"
    atomic_json(path, profile)
    return path


def run(args: argparse.Namespace) -> dict:
    resume = bool(getattr(args, "resume", False))
    with qualification_output(args.output.resolve(), resume) as output:
        return run_attempt(args, output)


def run_attempt(args: argparse.Namespace, output: Path) -> dict:
    client_command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if client_command:
        client_cwd = ROOT
    else:
        client_command = [
            str(ROOT / "agent/.venv/bin/python"),
            "-m",
            "pytest",
            "-n",
            "0",
            "--no-cov",
            "-s",
            "tests/test_live_e2e.py::test_live_e2e_quick_chat_waiting_for_response",
        ]
        client_cwd = ROOT / "agent"
    invocation = QualificationClient.prepare(
        client_command, client_cwd, output / "client-junit.xml"
    )
    resume = bool(getattr(args, "resume", False))
    if resume and not invocation.is_suite_release:
        raise ValueError("--resume requires the native suite release client")
    policy = MemoryPolicy.from_environment()
    # Detailed ledgers belong to resource checks. Ordinary client/tool flows
    # use the service's production response, keeping diagnostics out of LLM
    # context. Both automatic selections are explicit in the saved profile.
    resource_check = args.phase_server or invocation.is_suite_release
    resource_metrics = (
        resource_check if args.resource_metrics is None else args.resource_metrics
    )
    multiplier = args.timeout_multiplier
    timeout = args.timeout * multiplier
    tool_start_timeout = (
        args.tool_start_timeout or (480 if args.vision else 120)
    ) * multiplier
    kv_seed = []
    if args.kv_seed:
        # Copy completed public disk-cache entries only. Agentic checkpoints
        # belong to the originating server and must not seed a fresh instance.
        (output / "kv").mkdir()
        for source in sorted(args.kv_seed.glob("*.kv")):
            if source.is_file() and not source.is_symlink():
                target = output / "kv" / source.name
                shutil.copyfile(source, target)
                with target.open("rb") as cache:
                    digest = hashlib.file_digest(cache, "sha256").hexdigest()
                kv_seed.append(
                    {
                        "name": source.name,
                        "bytes": target.stat().st_size,
                        "sha256": digest,
                    }
                )
        if not kv_seed:
            raise ValueError("KV seed directory has no completed cache entries")
    model = "Qwen3.8-" + (
        "Flash-Next-Q2" if args.family == "flash" else "27B-UD-Q4_K_S"
    )
    executable = (
        "tests/probe_suite_phase_server"
        if args.phase_server
        else ("ds4-server-suite" if args.suite_config else "ds4-server")
    )
    command = [
        str(ROOT / executable),
        "--metal",
        "-m",
        str(ROOT / "gguf" / (model + ".gguf")),
        "--ssd-streaming",
        "--ssd-streaming-cold",
        "--ctx",
        "8192",
        "--prefill-chunk",
        "64",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--kv-disk-dir",
        str(output / "kv"),
        "--trace",
        str(output / "trace.log"),
    ]
    if not args.automatic_cache:
        command += ["--ssd-streaming-cache-experts", "2GB"]
    if args.mtp:
        if args.family == "qwen27":
            command += [
                "--mtp-model",
                str(ROOT / "gguf/mtp-Qwen3.8-27B-Q4_0.gguf"),
                "--mtp-draft",
                "2",
            ]
        else:
            command += ["--glm-mtp", "--mtp-draft", "2"]
    else:
        command += ["--no-mtp"]
    suite_timeout = None
    if args.suite_config:
        prepared_config, suite_timeout = prepare_suite_configuration(
            args.suite_config, output, multiplier
        )
        command += ["--suite-config", str(prepared_config)]
    if args.vision:
        projector = (
            "mmproj-Qwen3.8-"
            + ("Flash-Next-Q8_0" if args.family == "flash" else "27B-F16")
            + ".gguf"
        )
        command += ["--vision", str(ROOT / "gguf" / projector)]
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DS4_", "MTL_", "CUDA_"))
    }
    environment.update(
        BASE_URL=f"http://127.0.0.1:{args.port}/v1",
        DEFAULT_MODEL=model,
        CONTEXT_LENGTH="8192",
        REASONING_EFFORT="none",
        API_TYPE="responses",
        API_KEY="local",
        EXMEKH_AGENT_RUN_LIVE_TESTS="1",
        DS4_SUITE_RESOURCE_METRICS="1" if resource_metrics else "0",
        DS4_SERVER_PHASE_PROFILE="1" if args.phase_profile else "0",
        DS4_SUITE_HTTP_URL=f"http://127.0.0.1:{args.port}/v1",
        DS4_SUITE_HTTP_TIMEOUT_SECONDS=str(timeout),
        DS4_METAL_TIMEOUT_MULTIPLIER=str(multiplier),
        EXMEKH_INTERACTIVE_TIMEOUT_SECONDS=str(300 * multiplier),
        DS4_SUITE_PHASE_TIMEOUT_SECONDS=str(180 * multiplier),
        DS4_SUITE_LLM_MODEL=model,
        DS4_SUITE_E2E_ARTIFACTS=str(output / "client-artifacts"),
        DS4_AGENTIC_BASE_URL=f"http://127.0.0.1:{args.port}",
        DS4_AGENTIC_MODEL=model,
        DS4_AGENTIC_REASONING_EFFORT="none",
        DS4_AGENTIC_TEST_CHECKPOINT_ROOT=str(output / "kv"),
        DS4_INTERACTIVE_BASE_URL=f"http://127.0.0.1:{args.port}",
        DS4_INTERACTIVE_MODEL=model,
        DS4_INTERACTIVE_TIMEOUT_SECONDS=str(300 * multiplier),
        DS4_SUITE_LLM_START_TIMEOUT_SECONDS=str(tool_start_timeout),
    )
    environment.update(policy.environment())
    if args.vision:
        environment["DS4_QWEN4_IMAGE_MAX_TOKENS"] = "128"
    if args.phase_server:
        environment["DS4_SUITE_TEST_PHASE_BARRIER"] = str(output / "phase-control")
        environment["DS4_SUITE_PHASE_CONTROL"] = str(output / "phase-control")
    invocation = invocation.for_suite_server(environment["BASE_URL"], resume)
    server = client = None
    started = time.monotonic()
    report = {
        "status": "NOT_VERIFIED",
        "command": command,
        "client_command": list(invocation.command),
        "client_cwd": str(invocation.cwd),
        "client_junit": str(invocation.junit) if invocation.junit is not None else None,
        "client_sources_sha256": invocation.source_hashes(),
        "qualification_tool_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "reason": args.reason,
        "timeout_seconds": timeout,
        "base_timeout_seconds": args.timeout,
        "timeout_multiplier": multiplier,
        "footprint_limit_gib": args.max_footprint_gib or None,
        **policy.as_dict(),
        "mtp": args.mtp,
        "resource_metrics": resource_metrics,
        "phase_profile": args.phase_profile,
        "kv_seed": {"directory": str(args.kv_seed.resolve()), "entries": kv_seed}
        if args.kv_seed
        else None,
        "qualification_profile": {
            "context_tokens": 8192,
            "prefill_chunk": 64,
            "ssd_cache_bytes": None if args.automatic_cache else 2**31,
            "ssd_cache_mode": "automatic" if args.automatic_cache else "explicit",
            "reasoning_effort": "none",
            "tool_start_timeout_seconds": tool_start_timeout,
            "http_timeout_seconds": timeout,
            "interactive_timeout_seconds": 300 * multiplier,
            "phase_timeout_seconds": 180 * multiplier,
            "suite_operation_base_timeout_seconds": suite_timeout,
            "suite_operation_timeout_seconds": suite_timeout * multiplier
            if suite_timeout is not None
            else None,
        },
        "samples": [],
        "source_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "executables_sha256": {
            command[0]: hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
        },
    }
    if args.suite_config:
        report["suite_config_sha256"] = hashlib.sha256(
            args.suite_config.read_bytes()
        ).hexdigest()
        report["prepared_suite_config_sha256"] = hashlib.sha256(
            prepared_config.read_bytes()
        ).hexdigest()
    try:
        with (
            (output / "server.log").open("w") as log,
            (output / "client.log").open("w") as client_log,
            (output / "memory.jsonl").open("w") as memory,
        ):
            # Reject a pre-existing listener before loading weights or accepting
            # its catalog as readiness evidence for our own server.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", args.port))
            if invocation.is_suite_release:
                profile = resume_profile(args, report, invocation, output, environment)
                environment["DS4_QUALIFICATION_PROFILE"] = str(profile)
                report["checkpoint_profile"] = str(profile)
                checkpoint_directory = invocation.suite_output
                if checkpoint_directory.exists():
                    with exclusive_lock(checkpoint_directory / ".cycles.lock"):
                        pass
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            bad = 0

            def observe():
                nonlocal bad
                current = sample(server.pid)
                report["samples"].append(current)
                memory.write(json.dumps(current) + "\n")
                memory.flush()
                unsafe = policy.is_unsafe(current, report["samples"][0])
                bad = bad + 1 if unsafe else 0
                footprint = current["footprint_bytes"] or 0
                if (
                    args.max_footprint_gib
                    and footprint > args.max_footprint_gib * 2**30
                ) or bad >= policy.unsafe_samples:
                    report["status"] = "SKIP_RESOURCE_LIMIT"
                    raise MemoryError(
                        "footprint, pressure, swap growth or host reserve limit"
                    )

            while time.monotonic() - started < min(120 * multiplier, timeout):
                observe()
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                try:
                    with urllib.request.urlopen(
                        environment["BASE_URL"] + "/models", timeout=1
                    ) as response:
                        catalog = json.load(response)
                    (output / "catalog.json").write_text(
                        json.dumps(catalog, indent=2) + "\n"
                    )
                    selected = next(
                        row["id"]
                        for row in catalog["data"]
                        if row["id"]
                        not in ("Qwen/Qwen-Image-2.1-Turbo", "google/embeddinggemma-2")
                    )
                    environment.update(
                        DEFAULT_MODEL=selected,
                        DS4_SUITE_LLM_MODEL=selected,
                        DS4_AGENTIC_MODEL=selected,
                        DS4_INTERACTIVE_MODEL=selected,
                    )
                    report["api_model"] = selected
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("server readiness deadline")
            client = subprocess.Popen(
                invocation.command,
                cwd=invocation.cwd,
                env=environment,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while client.poll() is None:
                observe()
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("client deadline")
                if server.poll() is not None:
                    raise RuntimeError("server exited during client operation")
                time.sleep(1)
            report["client_exit_code"] = client.returncode
            report["status"] = "PASS" if client.returncode == 0 else "FAIL"
            if invocation.is_suite_release and client.returncode == 0:
                summary = json.loads(
                    (invocation.suite_output / "summary.json").read_text()
                )
                if summary["status"] not in ("PASS", "PASS_SEGMENTED"):
                    raise ValueError("release client has no completed qualification")
                report["status"] = summary["status"]
                report["cycle_progress"] = summary
    except KeyboardInterrupt as error:
        report["error"] = str(error) or "interrupted"
        raise
    except TimeoutError as error:
        report["error"] = str(error) or "interrupted"
    except MemoryError as error:
        report["error"] = str(error)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="FAIL", error=repr(error))
    finally:
        cleanup_processes(report, client, server)
        report["elapsed_seconds"] = time.monotonic() - started
        invocation.record_results(report)
        report["log_sha256"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("server.log", "client.log", "trace.log")
            if (output / name).exists()
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("flash", "qwen27"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep prior attempts and continue compatible completed suite cycles",
    )
    parser.add_argument("--suite-config", type=Path)
    parser.add_argument(
        "--resource-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Detailed ledgers; default on for phase/release-memory checks, off for ordinary clients",
    )
    parser.add_argument(
        "--phase-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record existing decode/constraint phase telemetry in the server log",
    )
    parser.add_argument(
        "--kv-seed",
        type=Path,
        help="Copy completed SSD prefix entries from a previous instance; record hashes",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Matching real encoder, explicit 128 image-token cap",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="Explicit matching predictor; baseline remains target-only",
    )
    parser.add_argument(
        "--phase-server",
        action="store_true",
        help="Qualification binary with a fenced cancellation boundary",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--automatic-cache",
        action="store_true",
        help="Use the production dynamic SSD cache; default is the frozen 2 GiB cache ceiling",
    )
    add_timeout_multiplier(parser)
    parser.add_argument(
        "--tool-start-timeout",
        type=float,
        help="Explicit browser tool-start deadline; default 480 s with vision, otherwise 120 s",
    )
    parser.add_argument(
        "--max-footprint-gib",
        type=float,
        default=0,
        help="0 uses available RAM after DS4_METAL_HOST_RESERVE_BYTES (default 2 GiB)",
    )
    parser.add_argument("--reason", default="initial bounded qualification")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.timeout <= MAX_QUALIFICATION_SECONDS or not 0 < args.port <= 65535:
        parser.error(
            f"timeout must be in (0,{MAX_QUALIFICATION_SECONDS}] and port in [1,65535]"
        )
    if (
        args.tool_start_timeout is not None
        and not 0 < args.tool_start_timeout <= args.timeout
    ):
        parser.error(
            "tool-start timeout must be positive and fit within the overall timeout"
        )
    if not math.isfinite(args.max_footprint_gib) or args.max_footprint_gib < 0:
        parser.error("footprint cap must be finite and nonnegative")
    if args.phase_server and not args.suite_config:
        parser.error("--phase-server requires --suite-config")
    with interrupt_on_termination():
        result = run(args)
    print(result["status"], f"{result['elapsed_seconds']:.1f}s")
    raise SystemExit(0 if result["status"] in ("PASS", "PASS_SEGMENTED") else 1)


if __name__ == "__main__":
    main()
