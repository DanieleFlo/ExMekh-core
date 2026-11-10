#!/usr/bin/env python3
"""Run one bounded native qualification with process and host memory evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .qualification_memory import MemoryPolicy
    from .qualification_timeouts import add_timeout_multiplier, local_timeout_multiplier
else:
    from qualification_memory import MemoryPolicy
    from qualification_timeouts import add_timeout_multiplier, local_timeout_multiplier


@contextmanager
def interrupt_on_termination():
    """Allow CLI SIGTERM to run the same evidence/child cleanup as Ctrl-C."""

    def interrupt(signum, frame):
        raise KeyboardInterrupt("termination requested")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class ProcessMemory(ctypes.Structure):
    # sys/resource.h, rusage_info_v0. Querying our child needs no task port.
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "package_wakeups",
            "interrupt_wakeups",
            "pageins",
            "wired",
            "resident",
            "footprint",
            "start",
            "exit",
        )
    ]


def sample(pid: int) -> dict:
    usage = ProcessMemory()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    result = lib.proc_pid_rusage(pid, 0, ctypes.byref(usage))
    values = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage"],
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    ).stdout.splitlines()
    match = re.search(r"used\s*=\s*([\d.]+)([MG])", values[1])
    if not match:
        raise ValueError("cannot measure host swap usage")
    vm = subprocess.run(
        ["vm_stat"], text=True, capture_output=True, check=True, timeout=5
    ).stdout
    page_size = re.search(r"page size of (\d+)", vm)
    pages = {
        key: int(value)
        for key, value in re.findall(r"^([^:]+):\s+(\d+)\.", vm, re.MULTILINE)
    }
    if (
        not page_size
        or not {"Pages free", "Pages inactive", "File-backed pages"} <= pages.keys()
    ):
        raise ValueError("cannot measure reclaimable host memory")
    host_available = (
        pages["Pages free"] + min(pages["Pages inactive"], pages["File-backed pages"])
    ) * int(page_size[1])
    return {
        "monotonic": time.monotonic(),
        "pressure": int(values[0]),
        "host_available_bytes": host_available,
        "swap_bytes": int(float(match[1]) * (2**20 if match[2] == "M" else 2**30)),
        "footprint_bytes": usage.footprint if result == 0 else None,
        "resident_bytes": usage.resident if result == 0 else None,
    }


def process_group_finished(group: int) -> bool:
    """Zombies retain a group ID, but no longer own model or worker storage."""
    listing = subprocess.run(
        ["ps", "-axo", "pgid=,stat="],
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    return not any(
        len(fields := line.split()) == 2
        and int(fields[0]) == group
        and not fields[1].startswith("Z")
        for line in listing.stdout.splitlines()
    )


def signal_group(process: subprocess.Popen, sig: int) -> bool:
    """Return false if the group has already finished, retaining signal errors."""
    try:
        os.killpg(process.pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can reject a group signal after its last live member exits.
        if process.poll() is not None and process_group_finished(process.pid):
            return False
        raise


def stop(process: subprocess.Popen, grace_seconds: float = 5) -> bool:
    """Drain the entire group; return false when SIGKILL was required."""
    deadline = time.monotonic() + grace_seconds
    if not signal_group(process, signal.SIGTERM):
        process.wait()
        return True
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # A reaped parent can leave workers alive. Give them the remaining grace
    # period too, rather than immediately killing them after the parent exits.
    while process.poll() is not None:
        if process_group_finished(process.pid):
            return True
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    forced = signal_group(process, signal.SIGKILL)
    process.wait()
    return not forced


def run(
    command: list[str],
    output: Path,
    timeout: float,
    footprint_gib: float,
    reason: str = "initial bounded qualification",
    *,
    timeout_multiplier: int | None = None,
) -> dict:
    multiplier = (
        local_timeout_multiplier() if timeout_multiplier is None else timeout_multiplier
    )
    if multiplier not in (1, 2, 4):
        raise ValueError("timeout multiplier must be 1, 2 or 4")
    base_timeout = timeout
    policy = MemoryPolicy.from_environment()
    environment = {**os.environ, **policy.environment()}
    timeout *= multiplier
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    samples = []
    stop_status = None
    process = None
    binaries = {}
    operation_error = None
    try:
        for argument in command:
            candidate = Path(argument)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                with candidate.open("rb") as executable:
                    binaries[argument] = hashlib.file_digest(
                        executable, "sha256"
                    ).hexdigest()
        with (output / "command.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            bad_samples = 0
            while process.poll() is None:
                current = sample(process.pid)
                samples.append(current)
                footprint = current["footprint_bytes"] or 0
                unsafe = policy.is_unsafe(current, samples[0])
                bad_samples = bad_samples + 1 if unsafe else 0
                if (
                    footprint_gib and footprint > footprint_gib * 2**30
                ) or bad_samples >= policy.unsafe_samples:
                    stop_status = "SKIP_RESOURCE_LIMIT"
                    break
                if time.monotonic() - started >= timeout:
                    stop_status = "NOT_VERIFIED"
                    break
                time.sleep(1)
    except BaseException as error:
        stop_status = "NOT_VERIFIED"
        operation_error = repr(error)
        raise
    finally:
        cleanup_error = None
        cleanup_forced = False
        if process is not None:
            # A successful parent may leave descendants in its process group.
            # Drain the whole group on every path, preserving its exit status.
            try:
                cleanup_forced = not stop(process)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_error = repr(error)
                stop_status = stop_status or "NOT_VERIFIED"
            code = process.returncode
            if cleanup_forced and code == 0:
                stop_status = stop_status or "NOT_VERIFIED"
        else:
            code = None
        source = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        ).stdout.strip()
        report = {
            "command": command,
            "cwd": str(Path.cwd()),
            "source_head": source,
            "environment": {
                k: v
                for k, v in environment.items()
                if k.startswith(("DS4_", "MTL_", "CUDA_"))
            },
            "timeout_seconds": timeout,
            "base_timeout_seconds": base_timeout,
            "timeout_multiplier": multiplier,
            "reason": reason,
            "footprint_limit_gib": footprint_gib or None,
            "memory_policy": policy.as_dict(),
            "exit_code": code,
            "elapsed_seconds": time.monotonic() - started,
            "status": stop_status or ("PASS" if code == 0 else "FAIL"),
            "cleanup_error": cleanup_error,
            "cleanup_forced": cleanup_forced,
            "samples": samples,
        }
        if operation_error is not None:
            report["error"] = operation_error
        log_path = output / "command.log"
        report["executables_sha256"] = binaries
        if log_path.exists():
            report["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    add_timeout_multiplier(parser)
    parser.add_argument(
        "--max-footprint-gib",
        type=float,
        default=0,
        help="Optional process cap; 0 uses available RAM after DS4_METAL_HOST_RESERVE_BYTES (default 2 GiB)",
    )
    parser.add_argument("--reason", default="initial bounded qualification")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or not 0 < args.timeout <= 600
        or not math.isfinite(args.max_footprint_gib)
        or args.max_footprint_gib < 0
    ):
        parser.error(
            "command, timeout in (0,600] and nonnegative footprint cap required"
        )
    with interrupt_on_termination():
        result = run(
            command,
            args.output,
            args.timeout,
            args.max_footprint_gib,
            args.reason,
            timeout_multiplier=args.timeout_multiplier,
        )
    print(result["status"], f"{result['elapsed_seconds']:.1f}s")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
