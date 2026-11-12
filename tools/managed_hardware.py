"""Live hardware samples used for admission and worker replacement."""

import copy
import ctypes
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path


def command_output(*argv):
    try:
        return subprocess.check_output(
            argv, text=True, stderr=subprocess.DEVNULL, timeout=4
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def cuda_device_uuid():
    """Identify the device that the single-GPU native worker opens."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None)
        if get_uuid is None:
            get_uuid = driver.cuDeviceGetUuid
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        device = ctypes.c_int()
        identifier = (ctypes.c_ubyte * 16)()
        if (
            driver.cuInit(0) == 0
            and driver.cuDeviceGet(ctypes.byref(device), 0) == 0
            and get_uuid(identifier, device) == 0
        ):
            return f"GPU-{uuid.UUID(bytes=bytes(identifier))}"
    except (OSError, AttributeError):
        pass
    return None


def cuda_device(host):
    selected = host.get("execution_device_uuid")
    if host["backend"] == "cuda" and selected:
        return next(
            (item for item in host["devices"] if item["uuid"] == selected), None
        )
    return None


def metal_available_memory():
    """Mirror the resident and streaming host samples in ds4_metal_device.h."""
    vm = command_output("vm_stat")
    page = re.search(r"page size of (\d+) bytes", vm)
    inactive = re.search(r"Pages inactive:\s*(\d+)", vm)
    if not page or not inactive:
        return None, None
    pages = command_output(
        "sysctl",
        "-n",
        "vm.page_free_count",
        "vm.page_speculative_count",
        "vm.page_pageable_external_count",
    ).split()
    if len(pages) != 3 or not all(value.isdecimal() for value in pages):
        return None, None
    pressure = command_output(
        "sysctl", "-n", "kern.memorystatus_vm_pressure_level"
    ).strip()
    if pressure.isdecimal() and int(pressure) >= 4:
        return 0, 0
    free, speculative, external = map(int, pages)
    page_bytes, inactive_pages = int(page[1]), int(inactive[1])
    file_pages = max(0, external - speculative)
    resident = (free + speculative + min(inactive_pages, file_pages)) * page_bytes
    floor_data = command_output(
        "sysctl",
        "-n",
        "vm.vm_page_filecache_min",
        "vm.page_pageable_internal_count",
        "vm.page_throttled_count",
    ).split()
    if len(floor_data) != 3 or not all(value.isdecimal() for value in floor_data):
        return resident, None
    floor, internal, throttled = map(int, floor_data)
    floor = max(floor, max(0, free + external + internal - throttled) * 10 // 27)
    streaming = (
        free + speculative + min(inactive_pages, max(0, file_pages - floor))
    ) * page_bytes
    return resident, streaming


def hardware(backend, directory):
    total = available = stream_available = metal = None
    execution_device_uuid = None
    devices = []
    system = platform.system()
    if system == "Darwin":
        value = command_output("sysctl", "-n", "hw.memsize").strip()
        total = int(value) if value.isdecimal() else None
        apple_silicon = (
            platform.machine() == "arm64"
            or command_output("sysctl", "-n", "hw.optional.arm64").strip() == "1"
        )
        if backend in {"metal", "auto"} and apple_silicon:
            metal = metal_device_limits()
            if metal and metal["unified_memory"]:
                available, stream_available = metal_available_memory()
    elif system == "Linux":
        mem = dict(
            re.findall(
                r"^(MemTotal|MemAvailable):\s+(\d+) kB",
                Path("/proc/meminfo").read_text(),
                re.MULTILINE,
            )
        )
        total = int(mem["MemTotal"]) * 1024 if "MemTotal" in mem else None
        available = int(mem["MemAvailable"]) * 1024 if "MemAvailable" in mem else None
    if backend in {"auto", "cuda"} and system == "Linux":
        execution_device_uuid = cuda_device_uuid()
        gpu = command_output(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        )
        for line in gpu.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5 and all(field.isdecimal() for field in fields[1:3]):
                name, size, free, capability, uuid = fields
                devices.append(
                    {
                        "name": name,
                        "total_bytes": int(size) * 2**20,
                        "free_bytes": int(free) * 2**20,
                        "compute": capability,
                        "uuid": uuid,
                    }
                )
    if backend == "auto":
        backend = (
            "metal"
            if system == "Darwin"
            else "cuda"
            if execution_device_uuid
            else "cpu"
        )
    return {
        "backend": backend,
        "system": system,
        "ram_bytes": total,
        "ram_available_bytes": available,
        "ram_stream_available_bytes": stream_available,
        "metal": metal,
        "unified_memory": backend == "metal"
        and bool(metal and metal["unified_memory"]),
        "devices": devices,
        "execution_device_uuid": execution_device_uuid,
        "disk_free_bytes": shutil.disk_usage(directory).free,
    }


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

    @staticmethod
    def worker_vram(pid, uuid):
        """Return only the given worker's framebuffer usage when NVML exposes it."""
        output = command_output(
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if (
                len(fields) == 3
                and fields[:2] == [str(pid), uuid]
                and fields[2].isdecimal()
                and int(fields[2]) > 0
            ):
                return int(fields[2]) * 2**20
        return None

    @staticmethod
    def worker_anonymous_ram(pid):
        """Read the worker's anonymous resident pages, excluding mapped GGUF files."""
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return None
        resident = re.search(r"^RssAnon:\s+(\d+) kB$", status, re.MULTILINE)
        return int(resident[1]) * 1024 if resident else None


def metal_device_limits():
    """Read Metal's limits without creating an inference device or loading weights."""
    try:
        ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        send_bool = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        objc.objc_autoreleasePoolPush.argtypes = []
        objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
        objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
        objc.objc_autoreleasePoolPop.restype = None
        objc.objc_release.argtypes = [ctypes.c_void_p]
        objc.objc_release.restype = None
        metal.MTLCreateSystemDefaultDevice.argtypes = []
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        pool = objc.objc_autoreleasePoolPush()
        device = None
        try:
            device = metal.MTLCreateSystemDefaultDevice()
            if device:
                return {
                    "unified_memory": send_bool(
                        device, objc.sel_registerName(b"hasUnifiedMemory")
                    ),
                    "working_set_bytes": send(
                        device, objc.sel_registerName(b"recommendedMaxWorkingSetSize")
                    ),
                    "max_buffer_bytes": send(
                        device, objc.sel_registerName(b"maxBufferLength")
                    ),
                }
        finally:
            if device:
                objc.objc_release(device)
            objc.objc_autoreleasePoolPop(pool)
    except (OSError, AttributeError):
        pass
    return None
