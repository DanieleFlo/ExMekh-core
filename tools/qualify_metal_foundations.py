#!/usr/bin/env python3
"""Record Metal foundations and optional T04/T05 checks without large-model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .qualification_memory import MemoryPolicy
else:
    from qualification_memory import MemoryPolicy

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("gguf-tools/quality-testing/data/qwen38-metadata/target.json")
SOURCES = (
    "Makefile",
    "start-ds4-server.sh",
    "agent/start-ds4-server.sh",
    "agent/start-ds4-server-flash.sh",
    "ds4.c",
    "ds4.h",
    "ds4_sha256.h",
    "tests/test_sha256.c",
    "ds4_lifecycle.inc",
    "ds4_server.c",
    "ds4_gpu.h",
    "ds4_resources.c",
    "ds4_metal.m",
    "ds4_metal_memory.h",
    "ds4_metal_device.h",
    "ds4_numeric_metal.m",
    "ds4_numeric_metal_args.h",
    "metal/numeric.metal",
    "metal/numeric_ordered.metal",
    "ds4_qwen_image_encoder.c",
    "ds4_qwen35_metal.inc",
    "ds4_qwen4_metal_stream.inc",
    "ds4_file_reader.h",
    "ds4_read_queue.h",
    "tests/test_file_reader.c",
    "tests/test_read_queue.c",
    "ds4_server_suite.inc",
    "tests/probe_qwen_metal_stream.c",
    "tests/test_flash_metal_stream_graph.c",
    "tests/test_metal_stream_qualification.py",
    "config/flash-metal-streamed.env.example",
    "config/qwen27-metal-streamed.env.example",
    "config/native-suite-metal.example.json",
    "tools/qualify_metal_command.py",
    "tools/qualification_memory.py",
    "tests/test_qualification_memory.py",
    "tools/qualification_timeouts.py",
    "tools/embed_metal_numeric.py",
    "tools/bench_suite_numeric.py",
    "tools/qualify_metal_server.py",
    "tools/qualify_metal_suite.py",
    "tools/suite_offloaded_oracle.py",
    "tools/run_qwen_image_pipeline_oracle.py",
    "tools/qualify_qwen_image_pipeline.py",
    "tools/diagnose_qwen_image_encoder.py",
    "tools/diagnose_qwen_image_vision.py",
    "tests/test_suite_offloaded_oracle.py",
    "tests/test_suite_pipeline_qualification.py",
    "tests/probe_qwen_image_encoder.c",
    "tests/probe_qwen_image_graph.c",
    "tests/test_suite_http_live.py",
    "tests/test_qwen_metal_api_live.py",
    "tests/test_suite_embedding_tools_e2e.py",
    "tests/test_suite_exmekh_live.py",
    "tests/test_agentic_api.py",
    "tests/test_suite_phase_live.py",
    "tests/probe_suite_phase_server.c",
    "ds4_server_suite_types.h",
    "metal/qwen35.metal",
    "metal/qwen35_quants.metal",
    "tests/test_metal_memory_policy.c",
    "tests/test_qwen35_metal.c",
    "tests/qwen_numerics_probe.c",
    "tests/qwen_half_reference.h",
    "tests/test_qwen35_metal_graph.c",
    "tests/test_qwen35_mrope.c",
    "tests/test_qwen_metal_lifecycle.c",
    "tests/test_flash_launcher.py",
    "tests/test_metal_foundations.py",
    "tools/qualify_metal_foundations.py",
)
FEATURE_SOURCES = (
    "agent/tests/test_live_e2e.py",
    "agent/tests/file_workflow_profile.py",
    "agent/src/exmekh_agent/capabilities/hds/coding-agent.md",
    "agent/tests/test_agent_design.py",
    "agent/tests/test_auto_compaction.py",
    "agent/vendor-patches/exmekh-lib-budget.patch",
    "agent/uv.lock",
    "agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl",
    "agent/tests/browser_support.py",
    "agent/tests/test_native_image_generation_e2e.py",
    "agent/tests/test_tabs_screenshots_e2e.py",
    "ds4_qwen4_vision.h",
    "ds4_image.c",
    "ds4_image_resize.inc",
    "tests/test_image_decode.c",
    "tests/test_qwen4_vision.c",
    "tests/qwen4_vision_ref.py",
    "tests/qwen35_metal_fixture.h",
    "tests/qwen_test_profile.h",
    "tests/test_qwen35_metal_features.c",
    "tests/test_qwen_test_profile.c",
    "tests/test_qwen4_kernels.c",
    "tests/test_agentic_checkpoint.c",
    "tests/test_qwen_mtp_callback.c",
    "tools/qualify_qwen_metal_vision.py",
    "tools/qwen_vision_reference.py",
    "tests/test_qwen4_vision_ref.py",
    "tests/test_metal_vision_qualification.py",
)
FEATURE_TARGETS = (
    "tests/test_image_decode",
    "tests/test_qwen4_vision",
    "ds4-server-suite",
    "ds4_server_test",
    "tests/test_server_interactive",
    "tests/test_sampling",
    "tests/test_qwen35_metal_features",
    "tests/test_qwen4_kernels",
    "tests/test_qwen_test_profile",
    "tests/test_agentic_checkpoint",
    "tests/test_qwen_mtp_callback",
)
# (elements per block, bytes per block), matching the audited GGML layouts.
QUANT_LAYOUTS = {
    "F32": (1, 4),
    "Q8_0": (32, 34),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "IQ2_XS": (256, 74),
    "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98),
    "IQ3_S": (256, 110),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
}
ENVIRONMENT_COMMANDS = {
    "source": ["git", "rev-parse", "HEAD"],
    "macos": ["sw_vers"],
    "toolchain": ["clang", "--version"],
    "sdk": ["xcrun", "--show-sdk-version"],
    "hardware": ["sysctl", "hw.model", "hw.memsize"],
    "ssd": ["df", "-k", str(ROOT)],
}
MEMORY_COMMAND = ["sysctl", "kern.memorystatus_vm_pressure_level", "vm.swapusage"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_operations(tensor: dict) -> list[str]:
    name = tensor["name"]
    if name == "token_embd.weight":
        return ["embedding", "gather"]
    if name == "output.weight":
        return ["output-head", "decode-matvec", "prefill-matmul"]
    if name.endswith("ssm_conv1d.weight"):
        return ["causal-convolution"]
    if len(tensor["shape"]) > 1:
        return ["decode-matvec", "prefill-matmul"]
    return ["normalization-or-recurrence"]


def build_tensor_inventory(snapshot: dict) -> list[dict]:
    """Validate pinned metadata before assigning execution roles to tensors."""
    tensors = snapshot["tensors"]
    if len(tensors) != snapshot["gguf"]["tensor_count"]:
        raise ValueError("snapshot tensor count mismatch")
    if Counter(t["type"] for t in tensors) != snapshot["tensor_type_counts"]:
        raise ValueError("snapshot tensor type counts mismatch")
    metadata = snapshot["metadata"]
    target_layers = (
        metadata["qwen35.block_count"] - metadata["qwen35.nextn_predict_layers"]
    )
    names = set()
    inventory = []
    for tensor in tensors:
        name, shape = tensor["name"], tensor["shape"]
        if name in names:
            raise ValueError("duplicate tensor: " + name)
        names.add(name)
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("invalid tensor shape: " + name)
        try:
            block_elements, block_bytes = QUANT_LAYOUTS[tensor["type"]]
        except KeyError as error:
            raise ValueError("unsupported tensor format: " + name) from error
        if shape[0] % block_elements:
            raise ValueError("invalid quantized row: " + name)
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements // block_elements * block_bytes
        offset = tensor["offset"]
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid tensor offset: " + name)
        if (
            snapshot["gguf"]["data_offset_bytes"] + offset + size
            > snapshot["artifact"]["size_bytes"]
        ):
            raise ValueError("tensor outside pinned artifact: " + name)
        layer = re.match(r"blk\.(\d+)\.", name)
        stage = "mtp" if layer and int(layer[1]) >= target_layers else "target"
        inventory.append(
            tensor
            | {"bytes": size, "operations": tensor_operations(tensor), "stage": stage}
        )
    return inventory


class QualificationReport:
    """Own evidence persistence and the final verdict, including validation failures."""

    def __init__(self, output: Path, *, features: bool = False):
        output.mkdir(parents=True, exist_ok=False)
        self._path = output / "verification.json"
        self._validations: dict[str, str] = {}
        self._document = {
            "schema": "exmekh-metal-features-v1"
            if features
            else "exmekh-metal-foundations-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "T00-T03 foundations; T04/T05 synthetic Metal sessions and host protocol fixtures; no real LLM inference"
                if features
                else "T00-T02 foundations and T03 synthetic numerics; no real LLM inference"
            ),
            "host_reserve_bytes": MemoryPolicy.from_environment().host_reserve_bytes,
            "hybrid_device_cap_bytes": 128 << 20,
            "lifecycle_device_cap_bytes": 1 << 30,
            "checks": [],
            "not_verified": [
                "Qwen27 real teacher-forced logits and streamed 8192/64 (T06-B)",
                "Flash streamed inference (T06-A)",
                "T04-T08 real workflows",
                "CUDA, ROCm, distributed inference and other Apple devices",
            ],
        }
        self._save()

    def _save(self) -> None:
        write_json(self._path, self._document)

    def record_section(self, name: str, value: object) -> None:
        self._document[name] = value
        self._save()

    def record_validation(self, name: str, value: dict) -> None:
        self._validations[name] = value["status"]
        self.record_section(name, value)

    def record_check(self, result: dict) -> bool:
        self._document["checks"].append(result)
        self._save()
        return result["status"] == "PASS"

    def finish(self) -> bool:
        checks = self._document["checks"]
        passed = (
            bool(checks)
            and all(row["status"] == "PASS" for row in checks)
            and all(status == "PASS" for status in self._validations.values())
        )
        self.record_section("all_executed_checks_pass", passed)
        return passed


class CommandRunner:
    """Own subprocess groups, bounded execution and command evidence."""

    def __init__(self, root: Path, output: Path, report: QualificationReport):
        self._root = root
        self._output = output
        self._report = report
        self._environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("DS4_", "MTL_", "CUDA_"))
        }

    def capture(self, command: list[str]) -> dict:
        try:
            result = subprocess.run(
                command,
                cwd=self._root,
                env=self._environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            return {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": str(error),
            }

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The parent can exit before a descendant that ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def run(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path | None = None,
        extra: dict | None = None,
        timeout: float = 240,
    ) -> bool:
        log = self._output / (name + ".log")
        before = self.capture(MEMORY_COMMAND)
        started = time.monotonic()
        timed_out = False
        code = None
        with log.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd or self._root,
                    env=self._environment | (extra or {}),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as error:
                stream.write(str(error) + "\n")
            else:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._stop_process(process)
                    code = process.returncode
                except KeyboardInterrupt:
                    self._stop_process(process)
                    raise
        status = "NOT_VERIFIED" if timed_out else "PASS" if code == 0 else "FAIL"
        result = {
            "name": name,
            "command": command,
            "cwd": str(cwd or self._root),
            "environment": extra or {},
            "status": status,
            "exit_code": code,
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": log.name,
            "sha256": digest(log),
            "memory_before": before,
            "memory_after": self.capture(MEMORY_COMMAND),
        }
        print(name, status, flush=True)
        return self._report.record_check(result)


def freeze_inventory(output: Path, snapshot: dict) -> dict:
    inventory = build_tensor_inventory(snapshot)
    path = output / "qwen27-tensors.json"
    write_json(
        path,
        {
            "snapshot": str(SNAPSHOT),
            "snapshot_sha256": digest(ROOT / SNAPSHOT),
            "artifact": snapshot["artifact"],
            "tensor_type_counts": snapshot["tensor_type_counts"],
            "tensors": inventory,
        },
    )
    return {
        "file": path.name,
        "sha256": digest(path),
        "count": len(inventory),
        "source": "pinned metadata snapshot",
    }


def artifact_identity(model: Path, expected: dict) -> dict:
    artifact = {
        "path": str(model),
        "bytes": model.stat().st_size,
        "sha256": digest(model),
    }
    matches = (
        artifact["bytes"] == expected["size_bytes"]
        and artifact["sha256"] == expected["sha256"]
    )
    return artifact | {
        "status": "PASS" if matches else "FAIL",
        "matches_pinned_qwen27": matches,
        "snapshot": str(SNAPSHOT),
        "snapshot_sha256": digest(ROOT / SNAPSHOT),
    }


def symbol_verdict(header: str, symbols: dict) -> dict:
    required = set(re.findall(r"\b(ds4_gpu_qwen35_\w+)\s*\(", header))
    defined = set(re.findall(r"\bT\s+_(ds4_gpu_qwen35_\w+)", symbols["stdout"]))
    passed = symbols["exit_code"] == 0 and bool(required) and not required - defined
    return {
        "required": sorted(required),
        "defined": sorted(defined),
        "missing": sorted(required - defined),
        "status": "PASS" if passed else "FAIL",
    }


def copy_cpu_sources(destination: Path) -> None:
    destination.mkdir()
    for source in ROOT.iterdir():
        if source.is_file() and (
            source.suffix in (".c", ".h", ".inc") or source.name == "Makefile"
        ):
            shutil.copy2(source, destination / source.name)
    shutil.copytree(ROOT / "third_party", destination / "third_party")


def run_gates(runner: CommandRunner, cpu_build: bool) -> None:
    if cpu_build:
        with tempfile.TemporaryDirectory(prefix="exmekh-metal-cpu-") as directory:
            isolated = Path(directory) / "cpu"
            copy_cpu_sources(isolated)
            if runner.run("cpu-build", ["make", "-j2", "cpu"], cwd=isolated):
                runner.run("cpu-link", ["otool", "-L", str(isolated / "ds4")])
    runner.run(
        "host",
        [
            "make",
            "test-runtime",
            "test-sha256",
            "test-resources",
            "test-executor",
            "test-metal-memory-policy",
        ],
    )
    runner.run(
        "launchers",
        [
            "python3",
            "-m",
            "unittest",
            "tests.test_flash_launcher",
            "tests.test_metal_foundations",
            "tests.test_metal_stream_qualification",
            "-v",
        ],
    )
    for name, wrapper in [
        ("root", "start-ds4-server.sh"),
        ("agent", "agent/start-ds4-server.sh"),
        ("flash-agent", "agent/start-ds4-server-flash.sh"),
    ]:
        runner.run(
            "launcher-" + name,
            ["bash", wrapper, "--print-command"],
            extra={"DS4_BACKEND": "metal"},
        )
    for name, command, extra in [
        ("quant-gdn", ["./tests/test_qwen35_metal"], {}),
        ("attention", ["./tests/test_qwen35_mrope"], {}),
        ("hybrid", ["./tests/test_qwen35_metal_graph"], {}),
        (
            "hybrid-general",
            ["./tests/test_qwen35_metal_graph"],
            {"DS4_METAL_DISABLE_METAL4": "1"},
        ),
        ("flash-hybrid-ssd", ["./tests/test_flash_metal_stream_graph"], {}),
        (
            "lifecycle-qwen27-ssd",
            ["./tests/test_qwen_metal_lifecycle", "qwen27-stream"],
            {},
        ),
        ("lifecycle-qwen27", ["./tests/test_qwen_metal_lifecycle", "qwen27"], {}),
        ("lifecycle-flash", ["./tests/test_qwen_metal_lifecycle", "flash"], {}),
    ]:
        runner.run(name, command, extra=extra)


def run_feature_gates(runner: CommandRunner, frontend: bool) -> None:
    # HTTP/dialect/constraint fixtures have no model inference. Keep their
    # evidence distinct from the actual Metal scheduler and lifecycle checks.
    for name, command, extra in [
        ("native-protocol", ["./ds4_server_test"], {}),
        ("interactive-protocol", ["./tests/test_server_interactive"], {}),
        ("sampling", ["./tests/test_sampling"], {}),
        ("image-preprocessing", ["./tests/test_image_decode"], {}),
        ("live-test-profile", ["./tests/test_qwen_test_profile"], {}),
        ("agentic-mtp-vision-synthetic", ["./tests/test_qwen35_metal_features"], {}),
        (
            "agentic-mtp-vision-general",
            ["./tests/test_qwen35_metal_features"],
            {"DS4_METAL_DISABLE_METAL4": "1"},
        ),
        (
            "flash-mtp-staging",
            ["./tests/test_qwen4_kernels"],
            {"DS4_TEST_QWEN4_MTP": "1"},
        ),
        (
            "flash-ssd-primitives",
            ["./tests/test_qwen4_kernels"],
            {"DS4_TEST_QWEN4_METAL_STREAM": "1"},
        ),
    ]:
        runner.run(name, command, extra=extra)
    runner.run(
        "qwen-contracts",
        [
            "python3",
            "-m",
            "unittest",
            "tests.test_qwen38_compatibility",
            "tests.test_server_sampling_matrix.SamplingMatrixFixtureTests",
            "-v",
        ],
    )
    runner.run(
        "vision-qualification-contracts",
        [
            str(ROOT / "build/metal-venv/bin/python"),
            "-m",
            "unittest",
            "tests.test_qwen4_vision_ref",
            "tests.test_metal_vision_qualification",
            "-v",
        ],
    )
    if frontend:
        runner.run(
            "frontend-mock",
            ["uv", "run", "--frozen", "pytest"],
            cwd=ROOT / "agent",
            extra={
                "EXMEKH_AGENT_RUN_LIVE_TESTS": "0",
                "PYTEST_XDIST_AUTO_NUM_WORKERS": "4",
            },
            timeout=600,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; never overwrite evidence",
    )
    parser.add_argument(
        "--cpu-build",
        action="store_true",
        help="Build CPU binaries in a fresh source copy",
    )
    parser.add_argument(
        "--model", type=Path, help="Optional Qwen27 artifact: inspect and hash only"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Also run bounded T04/T05 session features and host protocol fixtures",
    )
    parser.add_argument(
        "--frontend",
        action="store_true",
        help="With --features, run the frontend including mock Playwright cases",
    )
    args = parser.parse_args()
    if args.frontend and not args.features:
        parser.error("--frontend requires --features")
    if platform.system() != "Darwin":
        parser.error(
            "Metal qualification requires Darwin; make test-metal-memory-policy also runs on CPU"
        )
    if args.model and not args.model.is_file():
        parser.error("model is not a readable file: " + str(args.model))
    output = args.output.resolve()
    report = QualificationReport(output, features=args.features)
    runner = CommandRunner(ROOT, output, report)
    try:
        report.record_section(
            "environment",
            {
                name: runner.capture(command)
                for name, command in ENVIRONMENT_COMMANDS.items()
            },
        )
        report.record_section(
            "sources",
            {
                name: digest(ROOT / name)
                for name in SOURCES + (FEATURE_SOURCES if args.features else ())
            },
        )
        snapshot = json.loads((ROOT / SNAPSHOT).read_text())
        report.record_section("tensor_inventory", freeze_inventory(output, snapshot))
        built = runner.run(
            "metal-build",
            [
                "make",
                "-j2",
                "all",
                "tests/test_qwen35_metal",
                "tests/test_qwen35_mrope",
                "tests/test_qwen35_metal_graph",
                "tests/test_flash_metal_stream_graph",
                "tests/test_qwen_metal_lifecycle",
                *(FEATURE_TARGETS if args.features else ()),
            ],
        )
        if built:
            run_gates(runner, args.cpu_build)
            if args.features:
                run_feature_gates(runner, args.frontend)
            if args.model:
                model = args.model.resolve()
                report.record_validation(
                    "model", artifact_identity(model, snapshot["artifact"])
                )
                runner.run("qwen27-inspect", ["./ds4", "--inspect", "-m", str(model)])
            report.record_validation(
                "qwen35_symbols",
                symbol_verdict(
                    (ROOT / "ds4_gpu.h").read_text(),
                    runner.capture(["nm", "-g", "ds4_metal.o"]),
                ),
            )
            report.record_section(
                "binaries",
                {
                    name: digest(ROOT / name)
                    for name in ("ds4", "ds4-server", "ds4-bench", "ds4-eval")
                },
            )
    except KeyboardInterrupt:
        report.record_validation("interruption", {"status": "NOT_VERIFIED"})
        report.finish()
        return 130
    except (OSError, ValueError, KeyError) as error:
        report.record_validation("error", {"status": "FAIL", "message": str(error)})
    return 0 if report.finish() else 1


if __name__ == "__main__":
    raise SystemExit(main())
