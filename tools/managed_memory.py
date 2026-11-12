"""Memory bounds for the pinned managed models and their native launch profiles."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from managed_hardware import cuda_device
from qwen_flash_launcher import FLASH_CUDA_STANDARD

GIB = 2**30
CUDA_RESERVE = 2 * GIB  # Driver, dynamic workspaces and checkpoints outside the graph.
METAL_RUNTIME = GIB // 4
VISION_RUNTIME = GIB // 2
PREFILL_ROWS = {
    "vram": 2048,
    "unified": 64,
    "ssd": 64,
    "tiered": int(FLASH_CUDA_STANDARD["DS4_PREFILL_CHUNK"]),
}
CUDA_MODES = frozenset({"vram", "tiered"})
FLASH_DENSE_RESERVE = 21 * GIB // 4  # Pinned Flash Q2 has 5.14 GiB fixed CUDA weights.


def qwen27_graph_bytes(tokens, rows):
    # ds4.c:qwen_graph_context_memory_estimate. Graph tensors and GDN are FP32.
    kv = 16 * 2 * tokens * 4 * 256 * 4
    projection = max(10240 + 6144 + 48 + 48, 12288 + 1024 + 1024)
    scratch = rows * (7 * 5120 + projection + 6144 + 2 * 17408 + 1) * 4
    recurrent = 48 * ((6144 + 2 * 16 * 128) * 4 + 48 * 128 * 128) * 4
    return kv + scratch + recurrent + (5120 + 248320) * 4


def flash_graph_bytes(tokens, rows):
    # ds4.c:ds4_context_memory_estimate_with_prefill_mode includes snapshot headroom.
    blocks = tokens // 4 + 1
    raw = 13 * tokens * 2 * 2 * 256 * 2 + tokens * 16
    index = 13 * (tokens * 128 * 4 + blocks * 128 * 2)
    scratch = rows * (12 * 10240 + 24 * 2560 + 11 * (2560 + 2 * 640) + 2 * blocks) * 4
    recurrent = 3 * 36 * (48 * 128 * 128 + 3 * 10240) * 4
    return raw + index + scratch + recurrent


@dataclass(frozen=True)
class ModelMemory:
    max_tokens: int
    stream_binding_bytes: int
    metal_kv_row_bytes: int
    modes: frozenset[str]
    graph_bytes: Callable[[int, int], int]


# GGUF metadata and tensor sizes for the exact hashes pinned in managed_catalog.
MODEL_MEMORY = {
    "qwen27": ModelMemory(
        max_tokens=262144,
        stream_binding_bytes=1042944000,
        metal_kv_row_bytes=4096,
        modes=frozenset({"vram", "unified", "ssd"}),
        graph_bytes=qwen27_graph_bytes,
    ),
    "flash": ModelMemory(
        max_tokens=262144,
        stream_binding_bytes=675430400,
        metal_kv_row_bytes=1024,
        modes=frozenset({"ssd", "tiered"}),
        graph_bytes=flash_graph_bytes,
    ),
}


@dataclass(frozen=True)
class ReplacementMemory:
    gpu_uuid: str | None = None
    vram_bytes: int = 0
    ram_bytes: int = 0
    stream_ram_bytes: int = 0
    startup_device_bytes: int | None = None
    sample_anonymous_ram: bool = False

    @classmethod
    def observed(cls, before, after, profile, weights, vision, device_bytes=None):
        model, mode, tokens = (
            profile[key] for key in ("model_id", "mode_id", "context_tokens")
        )
        if mode == "ssd":
            weights = MODEL_MEMORY[model].stream_binding_bytes
        resident = (
            weights
            + (vision + VISION_RUNTIME if vision else 0)
            + context_bytes(model, mode, tokens)
            + (CUDA_RESERVE if mode in CUDA_MODES else METAL_RUNTIME)
        )

        def released(old, current):
            return (
                min(resident, max(0, old - current))
                if old is not None and current is not None
                else 0
            )

        if mode in CUDA_MODES:
            previous, current = cuda_device(before), cuda_device(after)
            if previous and current and previous["uuid"] == current["uuid"]:
                return cls(
                    gpu_uuid=previous["uuid"],
                    vram_bytes=released(previous["free_bytes"], current["free_bytes"]),
                    startup_device_bytes=device_bytes,
                    sample_anonymous_ram=mode == "tiered",
                )
        if (
            mode in {"unified", "ssd"}
            and before.get("unified_memory")
            and after.get("unified_memory")
        ):
            return cls(
                ram_bytes=released(
                    before.get("ram_available_bytes"), after.get("ram_available_bytes")
                ),
                stream_ram_bytes=released(
                    before.get("ram_stream_available_bytes"),
                    after.get("ram_stream_available_bytes"),
                ),
                startup_device_bytes=device_bytes,
            )
        return cls()

    def sample(self, hardware, worker, pid):
        vram = hardware.worker_vram(pid, self.gpu_uuid) if self.gpu_uuid else None
        device_bytes = (
            worker.device_memory()
            if vram is None and self.startup_device_bytes is not None
            else None
        )
        ram = hardware.worker_anonymous_ram(pid) if self.sample_anonymous_ram else None
        return vram, device_bytes, ram

    def credit(self, host, worker_vram=None, device_bytes=None, worker_ram=None):
        delta = (
            device_bytes - self.startup_device_bytes
            if device_bytes is not None and self.startup_device_bytes is not None
            else 0
        )
        if self.gpu_uuid:
            device = cuda_device(host)
            if device and device["uuid"] == self.gpu_uuid:
                reclaim = (
                    max(0, self.vram_bytes + delta)
                    if worker_vram is None
                    else worker_vram
                )
                device["free_bytes"] = min(
                    device["total_bytes"], device["free_bytes"] + reclaim
                )
        if (
            worker_ram is not None
            and host.get("ram_available_bytes") is not None
            and host.get("ram_bytes")
        ):
            host["ram_available_bytes"] = min(
                host["ram_bytes"], host["ram_available_bytes"] + worker_ram
            )
        if host.get("unified_memory") and host.get("ram_bytes"):
            for key, reclaim in (
                ("ram_available_bytes", self.ram_bytes),
                ("ram_stream_available_bytes", self.stream_ram_bytes),
            ):
                if host.get(key):
                    host[key] = min(
                        host["ram_bytes"], host[key] + max(0, reclaim + delta)
                    )


def context_bytes(model, mode, tokens):
    """Use the native graph geometry, including the launcher's prefill capacity."""
    return MODEL_MEMORY[model].graph_bytes(tokens, min(tokens, PREFILL_ROWS[mode]))


def metal_reserve():
    value = os.environ.get("DS4_METAL_HOST_RESERVE_BYTES", "")
    if value.isascii() and value.isdecimal() and len(value) <= 20:
        reserve = int(value)
        if reserve <= 2**64 - 1:
            return reserve
    return 2 * GIB


def max_context(model, mode, host, weight_bytes, vision_bytes=0):
    """Find the greatest fitting integer at or below the model's own maximum."""
    descriptor = MODEL_MEMORY[model]
    if mode not in descriptor.modes:
        return 0
    vision = vision_bytes + VISION_RUNTIME if vision_bytes else 0
    if mode == "vram":
        device = cuda_device(host)
        if not device:
            return 0
        budget = (
            min(device["free_bytes"], device["total_bytes"])
            - weight_bytes
            - CUDA_RESERVE
            - vision
        )

        def fits(tokens):
            rows = min(tokens, PREFILL_ROWS[mode])
            # ds4_cuda.cu:qwen35_attention_gemm_rows_launch reuses one FP32 score matrix.
            scores = tokens * rows * 4 if rows >= 512 else 0
            return context_bytes(model, mode, tokens) + scores <= budget

    elif mode == "tiered":
        device = cuda_device(host)
        available_ram = host.get("ram_available_bytes")
        if not device or device["compute"] != "8.6" or not available_ram:
            return 0
        budget = (
            min(device["free_bytes"], device["total_bytes"])
            - FLASH_DENSE_RESERVE
            - CUDA_RESERVE
            - vision
        )

        def fits(tokens):
            # Budget host K/V for every attention layer. Count full raw KV
            # in the GPU estimate too, allowing for transfer workspaces.
            host_kv = 13 * ((tokens + 3) // 4 * 4) * 2048
            return (
                host_kv + CUDA_RESERVE <= available_ram
                and context_bytes(model, mode, tokens) <= budget
            )

    elif mode in {"ssd", "unified"}:
        metal = host.get("metal") or {}
        available = host.get(
            "ram_stream_available_bytes" if mode == "ssd" else "ram_available_bytes"
        )
        if (
            host["backend"] != "metal"
            or host["system"] != "Darwin"
            or not host["unified_memory"]
            or not host["ram_bytes"]
            or not available
            or not metal.get("working_set_bytes")
            or not metal.get("max_buffer_bytes")
        ):
            return 0
        minimum_weights = (
            descriptor.stream_binding_bytes if mode == "ssd" else weight_bytes
        )
        if descriptor.stream_binding_bytes > metal["max_buffer_bytes"]:
            return 0
        budget = (
            min(available, host["ram_bytes"], metal["working_set_bytes"])
            - metal_reserve()
            - METAL_RUNTIME
            - minimum_weights
            - vision
        )
        graphs = (
            2 if mode == "ssd" else 1
        )  # Live graph and the transactional restore graph.

        def fits(tokens):
            # Per-layer K/V and Flash QSA keys are separate buffers.
            return (
                tokens * descriptor.metal_kv_row_bytes <= metal["max_buffer_bytes"]
                and graphs * context_bytes(model, mode, tokens) <= budget
            )

    else:
        return 0

    low, high = 0, descriptor.max_tokens
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low
