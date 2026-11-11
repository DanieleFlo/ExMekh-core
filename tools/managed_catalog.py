"""Catalog identity, hardware snapshots and managed profile compatibility."""

import copy
import json
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from managed_assets import ROOT, Artifact, Problem
from suite_bundle import validate_spec

GIB = 2**30
HF = "https://huggingface.co"
QWEN_REV = "4ca720788d1e01f1bff70c033e0d0028fd02e502"


def qwen_artifact(path, remote, size, digest):
    return Artifact(
        path, f"{HF}/unsloth/Qwen3.8-27B-GGUF/resolve/{QWEN_REV}/{remote}", size, digest
    )


MODELS = {
    "qwen27": {
        "name": "Qwen3.8-27B UD-Q4_K_S",
        "artifact": qwen_artifact(
            "Qwen3.8-27B-UD-Q4_K_S.gguf",
            "Qwen3.8-27B-UD-Q4_K_S.gguf",
            15358213024,
            "75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017",
        ),
        "vision": qwen_artifact(
            "mmproj-Qwen3.8-27B-F16.gguf",
            "mmproj-F16.gguf",
            927607488,
            "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e",
        ),
    },
    "flash": {
        "name": "Qwen3.8 Flash Next Q2",
        "artifact": Artifact(
            "Qwen3.8-Flash-Next-Q2.gguf",
            f"{HF}/antirez/qwen3.8-flash-next-gguf/resolve/main/Qwen3.8-Flash-Next-Q2.gguf",
            147207127040,
            "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9",
        ),
    },
}
SUITE_IDS = {
    "images": "Qwen/Qwen-Image-2.1-Turbo",
    "embeddings": "google/embeddinggemma-2",
}
SUITE_MANIFESTS = {
    "images": "qwen-image-2.1-turbo-bf16.json",
    "embeddings": "embeddinggemma-2-bf16.json",
}


def command_output(*argv):
    try:
        return subprocess.check_output(
            argv, text=True, stderr=subprocess.DEVNULL, timeout=4
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware(backend, directory):
    total = available = None
    devices = []
    if platform.system() == "Darwin":
        value = command_output("sysctl", "-n", "hw.memsize").strip()
        total = int(value) if value.isdecimal() else None
        vm = command_output("vm_stat")
        page = re.search(r"page size of (\d+) bytes", vm)
        if page:
            counts = re.findall(r"Pages (?:free|inactive|speculative):\s*(\d+)", vm)
            available = int(page[1]) * sum(map(int, counts)) if counts else None
    elif platform.system() == "Linux":
        mem = dict(
            re.findall(
                r"^(MemTotal|MemAvailable):\s+(\d+) kB",
                Path("/proc/meminfo").read_text(),
                re.M,
            )
        )
        total = int(mem["MemTotal"]) * 1024 if "MemTotal" in mem else None
        available = int(mem["MemAvailable"]) * 1024 if "MemAvailable" in mem else None
    if backend in {"auto", "cuda"} and platform.system() != "Darwin":
        gpu = command_output(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        )
        for line in gpu.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4 and all(field.isdecimal() for field in fields[1:3]):
                name, size, free, capability = fields
                devices.append(
                    {
                        "name": name,
                        "total_bytes": int(size) * 2**20,
                        "free_bytes": int(free) * 2**20,
                        "compute": capability,
                    }
                )
    if backend == "auto":
        backend = (
            "metal" if platform.system() == "Darwin" else "cuda" if devices else "cpu"
        )
    return {
        "backend": backend,
        "system": platform.system(),
        "ram_bytes": total,
        "ram_available_bytes": available,
        "unified_memory": backend == "metal",
        "devices": devices,
        "disk_free_bytes": shutil.disk_usage(directory).free,
    }


def modes(model, host, vision=False):
    """Expose qualified envelopes. The worker performs native admission at load."""
    backend = host["backend"]
    metal = backend == "metal" and host["system"] == "Darwin"
    cuda = backend == "cuda" and len(host["devices"]) == 1
    ram = host["ram_bytes"] or 0
    free = host["devices"][0]["free_bytes"] if cuda else 0
    vision_bytes = (
        MODELS[model]["vision"].size if vision and "vision" in MODELS[model] else 0
    )
    # Qwen27 has 16 full-attention layers and two FP32 KV tensors per layer.
    # The headroom covers scratch, driver and allocations outside the raw KV.
    # Keep the resulting estimate inside the existing launcher envelope.
    kv_bytes_per_token = 16 * 2 * 4 * 256 * 4
    vision_overhead = vision_bytes + GIB // 2 if vision_bytes else 0
    room = free - MODELS[model]["artifact"].size - 3 * GIB - vision_overhead
    cuda_limit = min(22593, room // kv_bytes_per_token) if model == "qwen27" else 0
    # Keep admission decisions in the native runtime.
    limits = {
        "ssd": 8192 if metal and ram >= 24 * GIB else 0,
        "ram": 0,
        "vram": cuda_limit if cuda and cuda_limit >= 512 else 0,
        "unified": 4096 if model == "qwen27" and metal and ram >= 24 * GIB else 0,
    }
    labels = {
        "ssd": "SSD streaming",
        "ram": "Fully in RAM",
        "vram": "Fully in VRAM",
        "unified": "Resident in unified memory",
    }
    reasons = {
        "ssd": "The managed SSD profile requires a qualified Metal host with at least 24 GiB.",
        "ram": "A guaranteed full-RAM profile is unavailable for this model and backend.",
        "vram": "The managed VRAM profile requires Qwen3.8-27B and a single CUDA GPU with sufficient free VRAM.",
        "unified": "The managed resident profile requires Qwen3.8-27B on Metal with at least 24 GiB.",
    }
    return [
        {
            "mode_id": key,
            "label": labels[key],
            "available": bool(limit),
            "reason": None if limit else reasons[key],
            "estimated_max_context_tokens": limit or None,
            "recommended_context_tokens": min(limit, 8192) if limit else None,
        }
        for key, limit in limits.items()
        if key != "unified" or metal
    ]


class HardwareProbe:
    def __init__(self, backend, directory):
        self._backend = backend
        self._directory = directory
        self._lock = threading.Lock()
        self._sample = None
        self._sampled_at = 0

    def snapshot(self, *, refresh=False):
        with self._lock:
            if (
                refresh
                or self._sample is None
                or time.monotonic() - self._sampled_at > 5
            ):
                self._sample = hardware(self._backend, self._directory)
                self._sampled_at = time.monotonic()
            return copy.deepcopy(self._sample)

    def invalidate(self):
        with self._lock:
            self._sample = None


class ModelCatalog:
    def __init__(self, store, gemma_media):
        self._store = store
        self._gemma_media = gemma_media

    @staticmethod
    def has_model(model_id):
        return isinstance(model_id, str) and model_id in MODELS

    @staticmethod
    def model_file(model_id):
        return MODELS[model_id]["artifact"]

    @staticmethod
    def vision_file(model_id):
        return MODELS[model_id]["vision"]

    def files_for_suite(self, feature):
        spec = self._suite_manifest(feature)
        validate_spec(spec)
        return [self._suite_artifact(item) for item in spec["files"]]

    @staticmethod
    def parse_profile(payload):
        if not isinstance(payload, dict):
            raise Problem("PROFILE_INVALID", "Expected a profile object.")
        model, mode, context = (
            payload.get(key) for key in ("model_id", "mode_id", "context_tokens")
        )
        features = payload.get("features", [])
        if (
            not isinstance(model, str)
            or model not in MODELS
            or mode not in ("ssd", "ram", "vram", "unified")
        ):
            raise Problem("PROFILE_INVALID", "Choose a known model and memory mode.")
        if type(context) is not int or not 1 <= context <= 262144:
            raise Problem("CONTEXT_INVALID", "Enter a positive integer context length.")
        if not isinstance(features, list) or any(
            feature not in ("vision", "images", "embeddings") for feature in features
        ):
            raise Problem("FEATURE_INVALID", "Choose supported features.")
        return {
            "model_id": model,
            "mode_id": mode,
            "context_tokens": context,
            "features": sorted(set(features)),
        }

    def _features(self, model_id, host, mode=None):
        cuda = host["backend"] == "cuda" and len(host["devices"]) == 1
        metal = host["backend"] == "metal"
        suite = (
            cuda and host["devices"][0]["compute"] == "8.6" and mode in (None, "vram")
        )
        vision = "vision" in MODELS[model_id] and (cuda or metal)
        return {
            "vision": (
                vision,
                None
                if vision
                else "A qualified matching encoder is unavailable for this profile.",
            ),
            "images": (
                suite,
                None
                if suite
                else "This suite profile requires the qualified CUDA SM 8.6 configuration.",
            ),
            "embeddings": (
                suite and bool(self._gemma_media and self._gemma_media.is_file()),
                None
                if suite and self._gemma_media and self._gemma_media.is_file()
                else "Requires the qualified CUDA suite and a prepared Gemma media asset.",
            ),
        }

    def describe(self, host):
        data = []
        for key, model in MODELS.items():
            artifact = model["artifact"]
            installed = self._store.installed(artifact)
            features = self._features(key, host)
            data.append(
                {
                    "id": key,
                    "name": model["name"],
                    "filename": artifact.path,
                    "size_bytes": artifact.size,
                    "installed": installed,
                    "importable": not installed
                    and self._store.has_local_file(artifact),
                    "modes": modes(key, host),
                    "features": [
                        self._feature_entry(key, name, available, reason)
                        for name, (available, reason) in features.items()
                    ],
                }
            )
        return data

    def _feature_entry(self, model, name, available, reason):
        entry = {"id": name, "available": available, "reason": reason}
        if available:
            assets = list(self.artifacts({"model_id": model, "features": [name]}))[1:]
            entry["size_bytes"] = sum(item.size for item in assets)
            entry["installed"] = all(self._store.installed(item) for item in assets)
        return entry

    def artifacts(self, profile):
        yield self.model_file(profile["model_id"])
        if "vision" in profile["features"]:
            yield self.vision_file(profile["model_id"])
        for feature in profile["features"]:
            if feature in SUITE_IDS:
                yield from self.files_for_suite(feature)

    @staticmethod
    def _suite_manifest(feature):
        return json.loads(
            (ROOT / "config" / "suite-bundles" / SUITE_MANIFESTS[feature]).read_text()
        )

    @staticmethod
    def _suite_artifact(item):
        return Artifact(
            "suite/" + item["path"], item["url"], item["bytes"], item["sha256"]
        )

    def validate(self, profile, host):
        model = profile["model_id"]
        options = modes(model, host, vision="vision" in profile["features"])
        option = next(
            (item for item in options if item["mode_id"] == profile["mode_id"]), None
        )
        if option is None or not option["available"]:
            raise Problem(
                "MODE_UNAVAILABLE",
                option["reason"] if option else "This memory mode is unavailable.",
            )
        if profile["context_tokens"] > option["estimated_max_context_tokens"]:
            raise Problem(
                "CONTEXT_LIMIT", "The requested context exceeds the estimated limit."
            )
        features = self._features(model, host, profile["mode_id"])
        for name in profile["features"]:
            if not features[name][0]:
                raise Problem("FEATURE_UNAVAILABLE", features[name][1])
