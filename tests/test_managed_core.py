"""Regression coverage for the headless management HTTP and artifact receipt."""

import hashlib
import http.client
import json
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from managed_assets import Artifact, PinnedStore, Problem  # noqa: E402
from managed_catalog import (  # noqa: E402
    MODELS,
    SUITE_MANIFESTS,
    ModelCatalog,
    modes,
)
from managed_hardware import HardwareProbe, cuda_device  # noqa: E402
from managed_http import ManagedHTTP  # noqa: E402
from managed_memory import (  # noqa: E402
    CUDA_RESERVE,
    GIB,
    MODEL_MEMORY,
    PREFILL_ROWS,
    ReplacementMemory,
    context_bytes,
    max_context,
)
from managed_server import Core  # noqa: E402
from managed_worker import NativeLaunch, NativeModel, NativeWorker  # noqa: E402
from qwen_flash_launcher import FLASH_CUDA_STANDARD  # noqa: E402


class ServerFixture:
    def __init__(self, server):
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


class ManagedMemoryTests(unittest.TestCase):
    def cuda_host(self, free=22 * GIB):
        return {
            "backend": "cuda",
            "system": "Linux",
            "execution_device_uuid": "GPU-example",
            "devices": [
                {
                    "compute": "8.6",
                    "uuid": "GPU-example",
                    "total_bytes": 24 * GIB,
                    "free_bytes": free,
                }
            ],
        }

    def test_cuda_limit_is_maximal_and_vision_uses_the_same_budget(self):
        host = self.cuda_host()
        model = MODELS["qwen27"]
        weights = model["artifact"].size
        vision = model["vision"].size
        mode = next(item for item in modes("qwen27", host) if item["mode_id"] == "vram")
        limit = mode["estimated_max_context_tokens"]
        self.assertEqual(mode["recommended_context_tokens"], limit)
        self.assertGreater(limit, 22593)
        budget = host["devices"][0]["free_bytes"] - weights - CUDA_RESERVE
        for tokens, fits in ((limit, True), (limit + 1, False)):
            scores = tokens * 2048 * 4
            self.assertEqual(
                context_bytes("qwen27", "vram", tokens) + scores <= budget, fits
            )
        vision_limit = max_context("qwen27", "vram", host, weights, vision)
        self.assertLess(vision_limit, limit)
        self.assertEqual(mode["estimated_max_context_tokens_with_vision"], vision_limit)
        self.assertEqual(max_context("qwen27", "vram", self.cuda_host(GIB), weights), 0)
        large = self.cuda_host(128 * GIB)
        large["devices"][0]["total_bytes"] = 128 * GIB
        self.assertEqual(max_context("qwen27", "vram", large, weights, vision), 262144)
        self.assertEqual(max_context("flash", "vram", large, weights), 0)

        profile = {
            "model_id": "qwen27",
            "mode_id": "vram",
            "context_tokens": limit,
            "features": [],
        }
        catalog = ModelCatalog(None, None, suite_store=None, native_stores={})
        catalog.validate(profile, host)
        with self.assertRaisesRegex(Problem, f"{vision_limit:,} tokens"):
            catalog.validate({**profile, "features": ["vision"]}, host)
        with self.assertRaisesRegex(Problem, "262,144 tokens"):
            catalog.parse_profile({**profile, "context_tokens": 262145})

    def test_flash_cuda_streaming_uses_available_vram_and_ram(self):
        host = {**self.cuda_host(), "ram_available_bytes": 24 * GIB}
        flash = MODELS["flash"]
        weights = flash["artifact"].size
        vision = flash["vision"].size
        option = next(
            item for item in modes("flash", host) if item["mode_id"] == "tiered"
        )
        self.assertEqual(option["estimated_max_context_tokens"], 262144)
        self.assertEqual(option["recommended_context_tokens"], 65536)
        self.assertEqual(option["estimated_max_context_tokens_with_vision"], 262144)

        host["ram_available_bytes"] = 4 * GIB
        limit = max_context("flash", "tiered", host, weights, vision)
        self.assertEqual(limit, ((2 * GIB) // (13 * 4 * 2048)) * 4)
        host["devices"][0]["free_bytes"] = 6 * GIB
        self.assertEqual(max_context("flash", "tiered", host, weights), 0)

    def test_flash_streaming_accepts_native_suite_features(self):
        cuda = {**self.cuda_host(), "ram_available_bytes": 24 * GIB}
        metal = {
            "backend": "metal",
            "system": "Darwin",
            "unified_memory": True,
            "ram_bytes": 24 * GIB,
            "ram_available_bytes": 16 * GIB,
            "ram_stream_available_bytes": 16 * GIB,
            "metal": {"working_set_bytes": 18 * GIB, "max_buffer_bytes": 2 * GIB},
        }
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "gemma-media.ds4p"
            catalog = ModelCatalog(None, media, suite_store=None, native_stores={})
            for host, mode in ((cuda, "tiered"), (metal, "ssd")):
                with self.subTest(mode=mode):
                    profile = {
                        "model_id": "flash",
                        "mode_id": mode,
                        "context_tokens": 8192,
                        "features": ["images"],
                    }
                    catalog.validate(profile, host)
                    both = {**profile, "features": ["images", "embeddings"]}
                    with self.assertRaisesRegex(Problem, "media asset"):
                        catalog.validate(both, host)
                    media.touch()
                    catalog.validate(both, host)
                    media.unlink()

    def test_managed_launch_preserves_qualified_prefill_and_memory_geometry(self):
        profiles = (
            ("qwen27", "cuda", "vram", 2048, None),
            ("qwen27", "metal", "unified", 64, "64"),
            ("qwen27", "metal", "ssd", 64, "64"),
            ("flash", "cuda", "tiered", 2048, "2048"),
            ("flash", "metal", "ssd", 64, "64"),
        )
        for model, backend, mode, rows, override in profiles:
            with self.subTest(model=model, backend=backend, mode=mode):
                profile = {"model_id": model, "mode_id": mode, "context_tokens": 8192}
                launch = NativeLaunch.for_profile(
                    Path("ds4-server"),
                    backend,
                    profile,
                    Path("weights.gguf"),
                    None,
                    None,
                    Path("kv"),
                )
                self.assertEqual(PREFILL_ROWS[mode], rows)
                self.assertIn(f"--{backend}", launch.argv)
                if override is None:
                    self.assertNotIn("--prefill-chunk", launch.argv)
                else:
                    index = launch.argv.index("--prefill-chunk")
                    self.assertEqual(launch.argv[index + 1], override)
                self.assertEqual("--ssd-streaming" in launch.argv, mode == "ssd")
                self.assertEqual("--ssd-streaming-cold" in launch.argv, mode == "ssd")
                self.assertEqual("--warm-weights" in launch.argv, mode == "unified")
                self.assertEqual("--mtp" in launch.argv, mode == "tiered")
                self.assertEqual(
                    dict(launch.environment),
                    FLASH_CUDA_STANDARD if mode == "tiered" else {},
                )

    @patch.dict(os.environ, {"DS4_METAL_HOST_RESERVE_BYTES": str(2 * GIB)})
    def test_metal_resident_and_streaming_use_separate_ram_samples(self):
        host = {
            "backend": "metal",
            "system": "Darwin",
            "unified_memory": True,
            "ram_bytes": 24 * GIB,
            "ram_available_bytes": 20 * GIB,
            "ram_stream_available_bytes": 14 * GIB,
            "metal": {"working_set_bytes": 18 * GIB, "max_buffer_bytes": 2 * GIB},
            "devices": [],
        }
        weights = MODELS["qwen27"]["artifact"].size
        resident = max_context("qwen27", "unified", host, weights)
        stream = max_context("qwen27", "ssd", host, weights)
        flash_weights = MODELS["flash"]["artifact"].size
        self.assertGreater(resident, 4096)
        self.assertGreater(stream, resident)
        self.assertGreater(max_context("flash", "ssd", host, flash_weights), 8192)
        self.assertEqual(max_context("flash", "unified", host, flash_weights), 0)
        small = {
            **host,
            "ram_bytes": 16 * GIB,
            "ram_available_bytes": 12 * GIB,
            "ram_stream_available_bytes": 9 * GIB,
            "metal": {**host["metal"], "working_set_bytes": 11 * GIB},
        }
        self.assertEqual(max_context("qwen27", "unified", small, weights), 0)
        self.assertGreater(max_context("qwen27", "ssd", small, weights), 0)
        self.assertLess(max_context("qwen27", "ssd", small, weights), stream)
        pressured = {
            **host,
            "ram_available_bytes": 0,
            "ram_stream_available_bytes": GIB,
        }
        self.assertEqual(max_context("qwen27", "unified", pressured, weights), 0)
        self.assertEqual(max_context("qwen27", "ssd", pressured, weights), 0)
        small_buffer = {
            **host,
            "metal": {**host["metal"], "max_buffer_bytes": GIB // 2},
        }
        self.assertEqual(max_context("flash", "ssd", small_buffer, flash_weights), 0)
        large = {
            **host,
            "ram_bytes": 128 * GIB,
            "ram_available_bytes": 110 * GIB,
            "ram_stream_available_bytes": 100 * GIB,
            "metal": {"working_set_bytes": 96 * GIB, "max_buffer_bytes": 2 * GIB},
        }
        self.assertEqual(
            max_context("flash", "ssd", large, flash_weights),
            MODEL_MEMORY["flash"].max_tokens,
        )

    def test_replacement_recovers_only_memory_of_the_worker_on_the_same_gpu(self):
        weights = MODELS["qwen27"]["artifact"].size
        profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 16384}
        resident = weights + context_bytes("qwen27", "vram", profile["context_tokens"])
        cold = self.cuda_host()
        active = self.cuda_host(22 * GIB - resident)
        replacement = ReplacementMemory.observed(
            cold, active, profile, weights, 0, device_bytes=resident
        )
        replacement.credit(active)
        self.assertEqual(active, cold)
        busy = self.cuda_host(21 * GIB - resident)
        replacement.credit(busy)
        self.assertEqual(busy, self.cuda_host(21 * GIB))
        changed = self.cuda_host(GIB)
        changed["devices"][0]["uuid"] = "GPU-other"
        changed["execution_device_uuid"] = "GPU-other"
        replacement.credit(changed, worker_vram=resident)
        self.assertEqual(changed["devices"][0]["free_bytes"], GIB)
        live = self.cuda_host(GIB)
        replacement.credit(live, worker_vram=20 * GIB)
        self.assertEqual(live["devices"][0]["free_bytes"], 21 * GIB)
        warm = self.cuda_host(21 * GIB - resident)
        replacement.credit(warm, device_bytes=resident + GIB)
        self.assertEqual(warm, cold)

    def test_flash_replacement_credits_anonymous_ram_to_the_plan(self):
        profile = {"model_id": "flash", "mode_id": "tiered", "context_tokens": 65536}
        cold = self.cuda_host()
        active = {
            **self.cuda_host(3 * GIB),
            "ram_bytes": 24 * GIB,
            "ram_available_bytes": 6 * GIB,
        }
        replacement = ReplacementMemory.observed(
            cold, active, profile, MODELS["flash"]["artifact"].size, 0
        )
        with (
            patch.object(HardwareProbe, "worker_vram", return_value=18 * GIB),
            patch.object(
                HardwareProbe, "worker_anonymous_ram", return_value=8 * GIB
            ) as ram,
        ):
            replacement.credit(active, *replacement.sample(HardwareProbe, None, 123))
        ram.assert_called_once_with(123)
        self.assertEqual(active["ram_available_bytes"], 14 * GIB)
        self.assertEqual(active["devices"][0]["free_bytes"], 21 * GIB)

    def test_streaming_cache_growth_is_credited_once_in_unified_memory(self):
        profile = {"model_id": "flash", "mode_id": "ssd", "context_tokens": 4096}
        cold = {
            "devices": [],
            "unified_memory": True,
            "ram_bytes": 24 * GIB,
            "ram_available_bytes": 20 * GIB,
            "ram_stream_available_bytes": 14 * GIB,
        }
        active = {
            **cold,
            "ram_available_bytes": 19 * GIB,
            "ram_stream_available_bytes": 13 * GIB,
        }
        replacement = ReplacementMemory.observed(
            cold, active, profile, MODELS["flash"]["artifact"].size, 0, device_bytes=GIB
        )
        warm = {
            **active,
            "ram_available_bytes": 15 * GIB,
            "ram_stream_available_bytes": 9 * GIB,
        }
        replacement.credit(warm, device_bytes=5 * GIB)
        self.assertEqual(warm, cold)
        critical = {**active, "ram_available_bytes": 0, "ram_stream_available_bytes": 0}
        replacement.credit(critical, device_bytes=5 * GIB)
        self.assertEqual(critical["ram_available_bytes"], 0)
        self.assertEqual(critical["ram_stream_available_bytes"], 0)

    def test_worker_memory_probe_accepts_only_positive_device_bytes(self):
        worker = NativeWorker()
        with patch.object(worker, "_get_json", return_value={"device_bytes": GIB}):
            self.assertEqual(worker.device_memory(), GIB)
        with patch.object(worker, "_get_json", return_value={"device_bytes": True}):
            self.assertIsNone(worker.device_memory())
        with patch.object(worker, "_get_json", side_effect=ValueError("older worker")):
            self.assertIsNone(worker.device_memory())

    @patch(
        "managed_hardware.command_output",
        return_value="77, GPU-other, 500\n88, GPU-example, 600\n77, GPU-example, 12000",
    )
    def test_nvml_sample_uses_only_the_managed_pid_and_device(self, _):
        self.assertEqual(HardwareProbe.worker_vram(77, "GPU-example"), 12000 * 2**20)
        self.assertIsNone(HardwareProbe.worker_vram(33, "GPU-example"))

    @patch("managed_hardware.platform.system", return_value="Linux")
    @patch(
        "managed_hardware.Path.read_text",
        return_value="MemTotal: 67108864 kB\nMemAvailable: 50331648 kB\n",
    )
    @patch("managed_hardware.cuda_device_uuid", return_value="GPU-second")
    @patch(
        "managed_hardware.command_output",
        return_value="Large GPU, 49152, 48000, 9.0, GPU-first\n"
        "Selected GPU, 24576, 22528, 8.6, GPU-second",
    )
    def test_cuda_capacity_uses_driver_device_with_multiple_gpus(
        self, _, selected, _meminfo, _system
    ):
        probe = HardwareProbe("auto", ROOT)
        host = probe.snapshot()
        self.assertEqual(len(host["devices"]), 2)
        device = cuda_device(host)
        self.assertEqual(device["uuid"], "GPU-second")
        weights = MODELS["qwen27"]["artifact"].size
        limit = max_context("qwen27", "vram", host, weights)
        self.assertEqual(
            limit, max_context("qwen27", "vram", self.cuda_host(), weights)
        )

        active = probe.snapshot()
        cuda_device(active)["free_bytes"] -= 16 * GIB
        profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 8192}
        replacement = ReplacementMemory.observed(host, active, profile, weights, 0)
        replacement.credit(active, worker_vram=16 * GIB)
        self.assertEqual(active, host)

        selected.return_value = "GPU-first"
        self.assertGreater(
            max_context("qwen27", "vram", probe.snapshot(refresh=True), weights), limit
        )
        selected.return_value = None
        self.assertEqual(
            max_context("qwen27", "vram", probe.snapshot(refresh=True), weights), 0
        )


class ManagedCoreTests(unittest.TestCase):
    def test_native_suite_config_makes_local_images_and_embeddings_plannable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "config" / "suite-bundles"
            manifests.mkdir(parents=True)
            native = root / "native-models"
            config = {"version": 1, "models": []}
            media = root / "gemma-media.ds4p"
            for filename in SUITE_MANIFESTS.values():
                spec = json.loads(
                    (ROOT / "config" / "suite-bundles" / filename).read_text()
                )
                for item in spec["files"]:
                    source = native / item["path"]
                    source.parent.mkdir(parents=True, exist_ok=True)
                    payload = item["path"].encode()
                    source.write_bytes(payload)
                    item["bytes"] = len(payload)
                    item["sha256"] = hashlib.sha256(payload).hexdigest()
                (manifests / filename).write_text(json.dumps(spec))
                config["models"].append(
                    {"model": spec["model"], "root": "native-models"}
                )
            config["models"][-1]["media_asset"] = media.name
            config_path = root / "native.json"
            config_path.write_text(json.dumps(config))
            host = {
                "backend": "cuda",
                "system": "Linux",
                "ram_bytes": 64 * 2**30,
                "disk_free_bytes": 100 * 2**30,
                "execution_device_uuid": "GPU-test",
                "devices": [
                    {
                        "uuid": "GPU-test",
                        "compute": "8.6",
                        "free_bytes": 24 * GIB,
                        "total_bytes": 24 * GIB,
                    }
                ],
            }
            profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 512}
            with (
                patch("managed_catalog.ROOT", root),
                patch("managed_suite.ROOT", root),
                patch.object(Core, "host", return_value=host),
            ):
                core = Core(
                    root / "state", root / "models", "cuda", suite_config=config_path
                )
                try:

                    def features():
                        model = next(
                            item
                            for item in core.catalog()["models"]
                            if item["id"] == "qwen27"
                        )
                        return {item["id"]: item for item in model["features"]}

                    self.assertFalse(features()["embeddings"]["available"])
                    self.assertIn("media asset", features()["embeddings"]["reason"])
                    optional = {
                        item["id"]: item for item in core.catalog()["optional_models"]
                    }
                    self.assertTrue(optional["embeddings"]["present"])
                    self.assertFalse(optional["embeddings"]["removable"])
                    self.assertFalse(optional["embeddings"]["active"])
                    media.write_bytes(b"fixture media processor")
                    for feature in ("images", "embeddings"):
                        entry = features()[feature]
                        self.assertTrue(entry["available"])
                        self.assertEqual(entry["missing_bytes"], 0)
                        self.assertFalse(entry["installed"])
                    self.assertEqual(
                        core.plan({**profile, "features": ["images", "embeddings"]})[
                            "missing_bytes"
                        ],
                        core.plan(profile)["missing_bytes"],
                    )
                    (native / "google/embeddinggemma-2/model.safetensors").unlink()
                    self.assertGreater(features()["embeddings"]["missing_bytes"], 0)
                    self.assertEqual(features()["images"]["missing_bytes"], 0)
                    self.assertEqual(
                        features()["embeddings"]["available_modes"], ["vram"]
                    )
                    spec = json.loads(
                        (manifests / SUITE_MANIFESTS["embeddings"]).read_text()
                    )
                    managed_files = [
                        root / "models" / "suite" / item["path"]
                        for item in spec["files"]
                    ]
                    for path, item in zip(managed_files, spec["files"]):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(item["path"].encode())
                    removal = {"kind": "remove", "model_id": "embeddings"}
                    with core._lock:
                        core._saved = {**profile, "features": ["embeddings"]}
                    with self.assertRaises(Problem) as in_use:
                        core.submit(removal)
                    self.assertEqual(in_use.exception.code, "MODEL_IN_USE")
                    with core._lock:
                        core._saved = {**profile, "features": []}
                    optional = {
                        item["id"]: item for item in core.catalog()["optional_models"]
                    }
                    self.assertTrue(optional["embeddings"]["removable"])
                    job = core.submit(removal)
                    deadline = time.monotonic() + 5
                    while core.get_jobs(job["id"])["state"] not in {
                        "completed",
                        "failed",
                    }:
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.02)
                    self.assertEqual(core.get_jobs(job["id"])["state"], "completed")
                    self.assertFalse(any(path.exists() for path in managed_files))
                    self.assertTrue(
                        (native / "google/embeddinggemma-2/tokenizer.json").is_file()
                    )
                    (manifests / SUITE_MANIFESTS["images"]).write_text("[]")
                    self.assertFalse(features()["images"]["available"])
                    self.assertTrue(features()["embeddings"]["available"])
                    with self.assertRaises(Problem) as invalid:
                        core.plan({**profile, "features": ["images"]})
                    self.assertEqual(invalid.exception.code, "SUITE_MANIFEST_INVALID")
                finally:
                    core.close()

    def test_local_suite_store_verifies_in_place_and_never_modifies_the_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            source = native / "model.safetensors"
            source.write_bytes(b"verified weights")
            artifact = Artifact(
                source.name,
                "",
                source.stat().st_size,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            store = PinnedStore(native, local_receipts=root / "verification")
            reports = []
            self.assertEqual(store.remaining(artifact), 0)
            self.assertEqual(
                store.ensure(
                    artifact, lambda phase, *_: reports.append(phase), lambda: None
                ),
                source,
            )
            self.assertTrue(store.installed(artifact))
            self.assertEqual(reports, ["verifying"])
            self.assertEqual(list(native.iterdir()), [source])
            with self.assertRaises(Problem) as read_only:
                store.remove(artifact)
            self.assertEqual(read_only.exception.code, "ARTIFACT_READ_ONLY")
            source.write_bytes(b"corrupted bytes!")
            with self.assertRaises(Problem) as invalid:
                store.ensure(artifact, lambda *_: None, lambda: None)
            self.assertEqual(invalid.exception.code, "CHECKSUM_MISMATCH")
            self.assertEqual(source.read_bytes(), b"corrupted bytes!")

    def test_root_launcher_opens_managed_http_with_or_without_saved_profile(self):
        for configured, native_env in product((False, True), repeat=2):
            with (
                self.subTest(configured=configured, native_env=native_env),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                state = root / "state"
                profile = {
                    "model_id": "qwen27",
                    "mode_id": "vram",
                    "context_tokens": 512,
                }
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("DS4_")
                }
                env.update(
                    DS4_STATE_DIR=str(state),
                    DS4_GGUF_DIR=os.path.relpath(root / "gguf", ROOT),
                    DS4_BACKEND="cpu",
                    DS4_SERVER_PORT="0",
                )
                if native_env:
                    env["DS4_MODEL_PROFILE"] = "flash"
                if configured:
                    state.mkdir()
                    (state / "profile.json").write_text(
                        json.dumps({"version": 1, "revision": 1, "profile": profile})
                    )
                with subprocess.Popen(
                    ["bash", str(ROOT / "start-ds4-server.sh")],
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ) as process:
                    try:
                        self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                        line = process.stdout.readline()
                        endpoint = re.search(r"http://127\.0\.0\.1:\d+/v1", line)
                        self.assertIsNotNone(endpoint, line)
                        with urllib.request.urlopen(
                            endpoint[0] + "/manage/status", timeout=2
                        ) as response:
                            status = json.load(response)
                        self.assertFalse(status["chat_ready"])
                        self.assertTrue((root / "gguf").is_dir())
                        if configured:
                            self.assertIn(status["state"], {"starting", "error"})
                            self.assertEqual(
                                status["saved_profile"]["context_tokens"], 512
                            )
                        else:
                            self.assertEqual(status["state"], "empty")
                            self.assertIsNone(status["saved_profile"])
                            self.assertEqual(list((root / "gguf").glob("*.gguf*")), [])
                        with self.assertRaises(urllib.error.HTTPError) as unavailable:
                            urllib.request.urlopen(endpoint[0] + "/models", timeout=2)
                        self.assertEqual(unavailable.exception.code, 503)
                    finally:
                        process.terminate()
                        process.communicate(timeout=5)

    def test_http_reports_saved_profile_restore_while_model_work_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 512}
            (state / "profile.json").write_text(
                json.dumps({"version": 1, "revision": 1, "profile": profile})
            )
            entered = threading.Event()
            release = threading.Event()

            def prepare(_core, _job):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release the restore")

            with patch.object(Core, "_run_job", prepare):
                core = Core(state, root / "gguf", "cpu")
                try:
                    server = ManagedHTTP(("127.0.0.1", 0), core, None)
                    core.restore_saved_profile()
                    self.assertTrue(entered.wait(2))
                    with ServerFixture(server) as base:
                        with urllib.request.urlopen(
                            base + "/v1/manage/status", timeout=2
                        ) as response:
                            status = json.load(response)
                        self.assertEqual(status["state"], "starting")
                        self.assertFalse(status["chat_ready"])
                        job = status["activation_job"]
                        self.assertEqual(job["id"], status["job_id"])
                        self.assertEqual(job["kind"], "restore")
                        self.assertEqual(job["model_id"], profile["model_id"])
                        with urllib.request.urlopen(
                            base + "/v1/manage/jobs/" + job["id"], timeout=2
                        ) as response:
                            self.assertEqual(json.load(response)["id"], job["id"])
                        with self.assertRaises(urllib.error.HTTPError) as unavailable:
                            urllib.request.urlopen(base + "/v1/models", timeout=2)
                        self.assertEqual(unavailable.exception.code, 503)
                finally:
                    release.set()
                    core.close()

    def test_saved_profile_can_repair_an_inactive_corrupt_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, models = root / "state", root / "gguf"
            state.mkdir()
            models.mkdir()
            model = models / "Qwen3.8-27B-UD-Q4_K_S.gguf"
            model.write_bytes(b"broken")
            (state / "profile.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "revision": 1,
                        "profile": {
                            "model_id": "qwen27",
                            "mode_id": "vram",
                            "context_tokens": 512,
                        },
                    }
                )
            )
            core = Core(state, models, "cpu")
            try:
                entry = next(
                    item for item in core.catalog()["models"] if item["id"] == "qwen27"
                )
                self.assertTrue(entry["removable"])
                job = core.submit(
                    {"kind": "remove", "model_id": "qwen27"}, "repair-key"
                )
                self.assertNotIn("idempotency", job)
                for _ in range(100):
                    result = core.get_jobs(job["id"])
                    if result["state"] == "completed":
                        break
                    time.sleep(0.01)
                self.assertEqual(result["state"], "completed")
                self.assertEqual(result["allowed_actions"], [])
                self.assertFalse(model.exists())
                self.assertEqual(core.status()["saved_profile"]["model_id"], "qwen27")
            finally:
                core.close()

    def test_activation_waits_for_a_conflicting_removal(self):
        profile = {"model_id": "qwen27", "mode_id": "vram", "context_tokens": 512}
        payload = b"model fixture"
        artifact = Artifact(
            "model.gguf", "", len(payload), hashlib.sha256(payload).hexdigest()
        )
        host = {
            "backend": "cuda",
            "system": "Linux",
            "disk_free_bytes": 100 * GIB,
            "execution_device_uuid": "GPU-test",
            "devices": [
                {
                    "uuid": "GPU-test",
                    "compute": "8.6",
                    "free_bytes": 24 * GIB,
                    "total_bytes": 24 * GIB,
                }
            ],
        }
        entered = threading.Event()
        release = threading.Event()
        remove_file = PinnedStore.remove

        def remove(store, item):
            entered.set()
            release.wait(5)
            remove_file(store, item)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(Core, "host", return_value=host),
            patch.object(PinnedStore, "remove", remove),
            patch.dict(MODELS["qwen27"], artifact=artifact),
        ):
            root = Path(directory)
            core = Core(root / "state", root / "gguf", "cuda")
            try:
                (root / "gguf" / artifact.path).write_bytes(payload)
                plan = core.plan(profile)
                self.assertEqual(plan["missing_bytes"], 0)
                job = core.submit({"kind": "remove", "model_id": profile["model_id"]})
                self.assertTrue(entered.wait(2))
                with self.assertRaises(Problem) as conflict:
                    core.submit({"kind": "apply", "plan_id": plan["id"]})
                self.assertEqual(conflict.exception.code, "MODEL_BUSY")
                release.set()
                deadline = time.monotonic() + 5
                while core.get_jobs(job["id"])["state"] != "completed":
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.02)
                with self.assertRaises(Problem) as stale:
                    core.submit({"kind": "apply", "plan_id": plan["id"]})
                self.assertEqual(stale.exception.code, "PLAN_STALE")
                self.assertEqual(core.plan(profile)["missing_bytes"], len(payload))
            finally:
                release.set()
                core.close()

    def test_memory_change_at_launch_preserves_chat_when_only_vision_no_longer_fits(
        self,
    ):
        payload = b"fixture"
        digest = hashlib.sha256(payload).hexdigest()
        model = Artifact("model.gguf", "", len(payload), digest)
        vision = Artifact("vision.gguf", "", len(payload), digest)
        tokens = 8192
        profile = {
            "model_id": "qwen27",
            "mode_id": "vram",
            "context_tokens": tokens,
            "features": ["vision"],
        }
        host = {
            "backend": "cuda",
            "system": "Linux",
            "disk_free_bytes": GIB,
            "execution_device_uuid": "GPU-test",
        }
        device = {
            "uuid": "GPU-test",
            "compute": "8.6",
            "free_bytes": 24 * GIB,
            "total_bytes": 24 * GIB,
        }
        chat_budget = (
            model.size
            + CUDA_RESERVE
            + context_bytes("qwen27", "vram", tokens)
            + tokens * 2048 * 4
        )
        enough = {**host, "devices": [device]}
        reduced = {**host, "devices": [{**device, "free_bytes": chat_budget}]}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                MODELS,
                {"qwen27": {**MODELS["qwen27"], "artifact": model, "vision": vision}},
            ),
            patch.object(
                Core, "host", side_effect=[enough, enough, reduced, reduced, reduced]
            ),
            patch.dict(os.environ, {"DS4_MANAGED_SERVER_BIN": sys.executable}),
            patch.object(
                NativeWorker,
                "start",
                return_value=NativeModel("test", "ds4.c", frozenset()),
            ) as start,
            patch.object(NativeWorker, "is_running", return_value=True),
        ):
            root = Path(directory)
            models = root / "gguf"
            models.mkdir()
            for artifact in (model, vision):
                (models / artifact.path).write_bytes(payload)
            core = Core(root / "state", models, "cuda")
            try:
                plan = core.plan(profile)
                job = core.submit({"kind": "apply", "plan_id": plan["id"]})
                for _ in range(100):
                    if core.get_jobs(job["id"])["state"] == "completed":
                        break
                    time.sleep(0.01)
                status = core.status()
                self.assertTrue(status["chat_ready"], status)
                self.assertEqual(status["saved_profile"]["features"], ["vision"])
                self.assertEqual(status["active_profile"]["available_features"], [])
                self.assertIn("vision", status["feature_errors"])
                self.assertNotIn("--vision", start.call_args.args[0].argv)
            finally:
                core.close()

    def test_other_store_cannot_mutate_an_artifact_during_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"fixture"
            artifact = Artifact(
                "model.gguf", "", len(payload), hashlib.sha256(payload).hexdigest()
            )
            (root / artifact.path).write_bytes(payload)
            verifier, other = PinnedStore(root), PinnedStore(root)

            def report(*_):
                with self.assertRaises(Problem) as busy:
                    other.remove(artifact)
                self.assertEqual(busy.exception.code, "ARTIFACT_BUSY")

            verifier.ensure(artifact, report, lambda: None, import_only=True)
            self.assertTrue(verifier.installed(artifact))

    def test_worker_releases_its_listener_on_invalid_discovery(self):
        script = """
import os
import socket

with socket.socket(fileno=int(os.environ["DS4_MANAGED_LISTENER_FD"])) as listener:
    os.write(int(os.environ["DS4_MANAGED_CONTROL_FD"]), b"opening_model\\nready\\n")
    connection, _ = listener.accept()
    with connection:
        connection.recv(1024)
        connection.sendall(b"HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\n{}")
    listener.accept()
"""
        worker = NativeWorker()
        with self.assertRaises(Problem) as error:
            worker.start(
                NativeLaunch((sys.executable, "-c", script)),
                lambda *_: None,
                lambda: None,
            )
        self.assertEqual(error.exception.code, "WORKER_DISCOVERY_FAILED")
        self.assertFalse(worker.is_running())
        with self.assertRaises(OSError):
            worker.connect()

    def test_worker_discovers_configured_operations_with_request_admission(self):
        worker = NativeWorker()
        models = [
            {"id": "flash"},
            {
                "id": "image",
                "ds4": {
                    "ready": False,
                    "configured": True,
                    "operations": ["image_generate", "image_edit"],
                    "limits": {"admission_required": True},
                },
            },
            {
                "id": "gemma",
                "ds4": {
                    "configured": True,
                    "ready": False,
                    "accepting_requests": False,
                    "operations": ["embed"],
                    "limits": {"admission_required": True},
                },
            },
        ]
        with patch.object(worker, "_get_json", return_value={"data": models}):
            native = worker._discover_model()
        self.assertEqual(native.name, "flash")
        self.assertEqual(
            native.available_features(["vision", "images", "embeddings"]),
            ["vision", "images"],
        )
        self.assertEqual(native.available_features(["embeddings"]), [])
        models[-1]["ds4"]["accepting_requests"] = True
        self.assertEqual(
            NativeModel.from_catalog(models).available_features(["embeddings"]),
            ["embeddings"],
        )

    def test_empty_core_and_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            core = Core(path / "state", path / "gguf", "cpu")
            try:
                with ServerFixture(
                    ManagedHTTP(("127.0.0.1", 0), core, "secret")
                ) as base:

                    def read(path, **headers):
                        return urllib.request.urlopen(
                            urllib.request.Request(base + path, headers=headers),
                            timeout=5,
                        )

                    with self.assertRaises(urllib.error.HTTPError) as unauthenticated:
                        read("/v1/manage/status")
                    self.assertEqual(unauthenticated.exception.code, 401)
                    with read(
                        "/v1/manage/status", Authorization="Bearer secret"
                    ) as result:
                        self.assertEqual(json.load(result)["state"], "empty")
                    with self.assertRaises(urllib.error.HTTPError) as unavailable:
                        read("/v1/models")
                    self.assertEqual(unavailable.exception.code, 503)
                    self.assertEqual(
                        json.load(unavailable.exception)["error"]["code"],
                        "MODEL_NOT_READY",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as wrong_origin:
                        read(
                            "/v1/manage/status",
                            Authorization="Bearer secret",
                            Origin="https://other.example",
                        )
                    self.assertEqual(wrong_origin.exception.code, 403)
            finally:
                core.close()

    def test_resume_download_and_require_verified_receipt(self):
        payload = b"fixture GGUF bytes"
        requests = []

        class Downloads(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.headers.get("Range"))
                offset = 5
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
                self.send_header("Content-Length", str(len(payload) - offset))
                self.end_headers()
                self.wfile.write(payload[offset:])

        with tempfile.TemporaryDirectory() as directory:
            store = PinnedStore(Path(directory))
            with ServerFixture(ThreadingHTTPServer(("127.0.0.1", 0), Downloads)) as url:
                artifact = Artifact(
                    "model.gguf", url, len(payload), hashlib.sha256(payload).hexdigest()
                )
                partial = store.root / "model.gguf.part"
                partial.write_bytes(payload[:5])
                states = []
                self.assertFalse(store.installed(artifact))
                self.assertEqual(store.remaining(artifact), len(payload) - 5)
                final = store.ensure(
                    artifact, lambda phase, *_: states.append(phase), lambda: None
                )
                self.assertEqual(requests, ["bytes=5-"])
                self.assertIn("downloading", states)
                self.assertIn("verifying", states)
                self.assertTrue(store.installed(artifact))
                self.assertFalse(partial.exists())
                final.write_bytes(b"!" * len(payload))
                self.assertFalse(store.installed(artifact))
                with self.assertRaises(Problem) as mismatch:
                    store.ensure(
                        artifact, lambda *_: None, lambda: None, import_only=True
                    )
                self.assertEqual(mismatch.exception.code, "CHECKSUM_MISMATCH")

    def test_changed_partial_is_not_certified_after_checksum(self):
        payload = b"fixture weights"
        artifact = Artifact(
            "model.gguf", "", len(payload), hashlib.sha256(payload).hexdigest()
        )
        with tempfile.TemporaryDirectory() as directory:
            store = PinnedStore(Path(directory))
            partial = store.root / "model.gguf.part"
            partial.write_bytes(payload)
            verified = False

            def report(phase, completed, total):
                nonlocal verified
                verified = phase == "verifying" and completed == total

            def check():
                if verified:
                    partial.write_bytes(b"!" * len(payload))

            with self.assertRaises(Problem) as changed:
                store.ensure(artifact, report, check)
            self.assertEqual(changed.exception.code, "ARTIFACT_CHANGED")
            self.assertFalse(store.installed(artifact))
            self.assertFalse((store.root / artifact.path).exists())

    def test_proxy_preserves_multipart_and_propagates_disconnect(self):
        multipart = (
            b'--fixture\r\nContent-Disposition: form-data; name="image";'
            b' filename="sample.png"\r\n\r\n\x00\xff\r\n--fixture--\r\n'
        )
        received = []
        disconnect = threading.Event()

        with (
            tempfile.TemporaryDirectory() as directory,
            socket.socket(socket.AF_UNIX) as native,
        ):
            path = str(Path(directory) / "native.sock")
            native.bind(path)
            native.listen()
            native.settimeout(5)

            class WorkerCore:
                def is_closing(self):
                    return False

                @contextmanager
                def open_inference(self, *, processing=False):
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(5)
                        connection.connect(path)
                        yield connection

            def accept():
                connection, _ = native.accept()
                with connection, connection.makefile("rb") as reader:
                    connection.settimeout(5)
                    request_line = reader.readline()
                    headers = http.client.parse_headers(reader)
                    body = reader.read(int(headers["Content-Length"]))
                    received.append(
                        (
                            request_line,
                            headers["Content-Type"],
                            headers["Idempotency-Key"],
                            body,
                        )
                    )
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                        b"event: operation.started\ndata: {}\n\n"
                    )
                    if connection.recv(1) == b"":
                        disconnect.set()

            worker = threading.Thread(target=accept)
            worker.start()
            with ServerFixture(
                ManagedHTTP(("127.0.0.1", 0), WorkerCore(), None)
            ) as base:
                request = urllib.request.Request(
                    base + "/v1/images/edits",
                    data=multipart,
                    headers={
                        "Content-Type": "multipart/form-data; boundary=fixture",
                        "Idempotency-Key": "test-upload",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.readline(), b"event: operation.started\n")
                self.assertTrue(disconnect.wait(5))
            worker.join(5)
            self.assertEqual(
                received,
                [
                    (
                        b"POST /v1/images/edits HTTP/1.1\r\n",
                        "multipart/form-data; boundary=fixture",
                        "test-upload",
                        multipart,
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
