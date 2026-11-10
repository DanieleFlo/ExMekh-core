"""Native suite bindings for development tests, diagnostics and benchmarks.

The owner handles ABI declarations, tensor storage and context teardown.
Importing this module does not configure PyTorch or load a native library.
"""

import ctypes as C
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class Status(IntEnum):
    OK = 0
    INVALID = 1
    UNSUPPORTED = 2
    NOT_RESIDENT = 3
    NOT_READY = 4
    BUSY = 5
    NOMEM = 6
    CANCELLED = 7
    INFERENCE = 8
    RESTORE = 9
    DEVICE_LOST = 10
    CONTEXT_LENGTH = 11


class NativeError(AssertionError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"native status {status}")


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("count", C.c_uint64), ("type", C.c_uint32)]


class Attention(C.Structure):
    _fields_ = [
        (k, C.c_uint32)
        for k in ("batch", "queries", "keys", "q_heads", "kv_heads", "dim")
    ] + [
        ("scale", C.c_float),
        ("softcap", C.c_float),
        ("round_scores", C.c_bool),
        ("round_probabilities", C.c_bool),
        ("relative_keys", View),
        ("relative_rows", C.c_uint32),
        ("relative_offset", C.c_int32),
    ]


class Reindex(C.Structure):
    _fields_ = [
        ("rank", C.c_uint32),
        ("shape", C.c_uint64 * 8),
        ("stride", C.c_uint64 * 8),
    ]


class Resample(C.Structure):
    _fields_ = [
        ("batch", C.c_uint32),
        ("channels", C.c_uint32),
        ("input", C.c_uint32 * 3),
        ("output", C.c_uint32 * 3),
    ]


class Conv(C.Structure):
    _fields_ = [
        (k, C.c_uint32) for k in ("batch", "in_channels", "out_channels", "groups")
    ] + [
        (k, C.c_uint32 * 3)
        for k in ("input", "kernel", "stride", "dilation", "pad_before", "pad_after")
    ]


class Budget(C.Structure):
    _fields_ = [
        ("limit", C.c_uint64),
        ("margin", C.c_uint64),
        ("global_free", C.c_uint64),
        ("global_known", C.c_bool),
    ]


class MemoryUsage(NamedTuple):
    live_bytes: int
    peak_bytes: int


class Numeric:
    def __init__(self, backend="cpu", *, library_path=None):
        if backend not in ("cpu", "cuda", "metal"):
            raise ValueError("backend must be cpu, cuda or metal")
        self.backend = backend
        self.lib = self._load_library(backend, library_path)
        self.resources, self.ctx = C.c_void_p(), C.c_void_p()
        self.buffers = []
        try:
            self.ok(self.lib.ds4_resources_create(C.byref(self.resources)))
            if backend == "metal":
                self.ok(self.lib.ds4_resources_set_unified_memory(self.resources, True))
            self.ok(
                getattr(self.lib, f"ds4_numeric_{backend}")(
                    C.byref(self.ctx), self.resources, 100
                )
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _load_library(backend, library_path):
        path = library_path or ROOT / f"tests/libds4_numeric_{backend}.so"
        lib = C.CDLL(str(path))
        ptr = C.c_void_p
        signatures = {
            "ds4_resources_create": [C.POINTER(ptr)],
            "ds4_resources_close": [ptr],
            "ds4_resources_set_unified_memory": [ptr, C.c_bool],
            "ds4_numeric_cpu": [C.POINTER(ptr), ptr, C.c_uint64],
            "ds4_numeric_close": [ptr],
            "ds4_numeric_fence": [ptr],
            "ds4_num_alloc": [ptr, C.c_uint64, C.c_uint32, C.POINTER(ptr)],
            "ds4_num_free": [ptr],
            "ds4_num_data": [ptr],
            "ds4_num_slice": [View, C.c_uint64, C.c_uint64],
            "ds4_num_write": [ptr, View, ptr, C.c_uint64],
            "ds4_num_read": [ptr, View, ptr, C.c_uint64],
            "ds4_num_linear": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
            ],
            "ds4_num_matmul_batch": [
                ptr,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
            ],
            "ds4_num_matmul_batch_layout": [
                ptr,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
                C.c_bool,
            ],
            "ds4_num_channel_l2": [ptr, View, View, C.c_uint32, C.c_uint32, C.c_float],
            "ds4_num_channel_rms": [
                ptr,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_float,
            ],
            "ds4_num_norm": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_float,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
            ],
            "ds4_num_point": [ptr, View, View, View, C.c_uint32, C.c_float],
            "ds4_num_rope": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
                C.c_bool,
                C.c_bool,
            ],
            "ds4_num_rope_complex": [
                ptr,
                View,
                View,
                View,
                View,
                C.c_uint32,
                C.c_uint32,
            ],
            "ds4_num_attend": [
                ptr,
                View,
                View,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.POINTER(Attention),
            ],
            "ds4_num_conv": [ptr, View, View, View, View, C.POINTER(Conv)],
            "ds4_num_dequantize": [ptr, C.c_uint64, ptr, C.c_uint64, C.c_uint32],
            "ds4_num_reorder": [ptr, View, View, C.POINTER(Reindex)],
            "ds4_num_pool": [
                ptr,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_float,
            ],
            "ds4_num_nearest": [ptr, View, View, C.POINTER(Resample)],
            "ds4_num_softmax": [
                ptr,
                View,
                View,
                C.POINTER(C.c_uint32),
                C.c_uint32,
                C.c_uint32,
                C.c_float,
            ],
        }
        if backend != "cpu":
            signatures[f"ds4_numeric_{backend}"] = signatures["ds4_numeric_cpu"]
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = View if name in ("ds4_num_data", "ds4_num_slice") else C.c_int
        lib.ds4_numeric_memory.argtypes = [
            ptr,
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint64),
        ]
        lib.ds4_numeric_memory.restype = None
        lib.ds4_numeric_linear_workspace.argtypes = [
            ptr,
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
        ]
        lib.ds4_numeric_linear_workspace.restype = C.c_uint64
        lib.ds4_num_buffer_stage_file.argtypes = [
            ptr,
            C.c_uint64,
            C.c_int,
            C.c_uint64,
            C.c_uint64,
        ]
        lib.ds4_num_buffer_stage_file.restype = C.c_int
        for name in ("get", "set"):
            function = getattr(lib, f"ds4_resources_{name}_budget")
            function.argtypes = [ptr, C.c_int, C.POINTER(Budget)]
            function.restype = C.c_int
        if backend == "metal":
            lib.ds4_numeric_metal_budget.argtypes = [ptr, C.c_uint64]
            lib.ds4_numeric_metal_budget.restype = C.c_int
        return lib

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def ok(status):
        if status:
            raise NativeError(status)

    @staticmethod
    def _format(dtype):
        if dtype == torch.float32:
            return 0, 4
        if dtype == torch.bfloat16:
            return 30, 2
        raise ValueError("native tensors require float32 or bfloat16")

    def tensor(self, tensor):
        dtype, _ = self._format(tensor.dtype)
        tensor = tensor.detach().cpu().contiguous()
        p = C.c_void_p()
        self.ok(self.lib.ds4_num_alloc(self.ctx, tensor.numel(), dtype, C.byref(p)))
        self.buffers.append(p)
        v = self.lib.ds4_num_data(p)
        array = (
            tensor.view(torch.uint16).numpy() if dtype == 30 else tensor.float().numpy()
        )
        self.ok(self.lib.ds4_num_write(self.ctx, v, array.ctypes.data, array.nbytes))
        return v

    def slice(self, view, offset, count):
        """Borrow a native subview while the original allocation remains owned."""
        if (
            type(offset) is not int
            or type(count) is not int
            or not 0 <= offset < 2**64
            or not 0 < count < 2**64
        ):
            raise ValueError("invalid numeric slice")
        result = self.lib.ds4_num_slice(view, offset, count)
        if not result.data:
            raise ValueError("invalid numeric slice")
        return result

    def file_tensor(self, path, count, dtype, *, offset=0):
        """Load file bytes with the same public PyTorch dtypes as tensor()."""
        native_type, unit = self._format(dtype)
        if (
            type(count) is not int
            or type(offset) is not int
            or not 0 < count <= (2**63 - 1) // unit
            or not 0 <= offset <= 2**63 - 1 - count * unit
        ):
            raise ValueError("invalid numeric file range")
        with open(path, "rb") as source:
            buffer = C.c_void_p()
            self.ok(
                self.lib.ds4_num_alloc(self.ctx, count, native_type, C.byref(buffer))
            )
            self.buffers.append(buffer)
            self.ok(
                self.lib.ds4_num_buffer_stage_file(
                    buffer,
                    0,
                    source.fileno(),
                    offset,
                    count * unit,
                )
            )
        return self.lib.ds4_num_data(buffer)

    def read(self, v, shape):
        arr = np.empty(v.count, dtype=np.uint16 if v.type == 30 else np.float32)
        self.ok(self.lib.ds4_num_read(self.ctx, v, arr.ctypes.data, arr.nbytes))
        result = torch.from_numpy(arr)
        return (
            (result.view(torch.bfloat16) if v.type == 30 else result)
            .float()
            .reshape(shape)
        )

    def call(self, name, *args):
        self.ok(getattr(self.lib, "ds4_num_" + name)(self.ctx, *args))

    def fence(self):
        self.ok(self.lib.ds4_numeric_fence(self.ctx))

    def memory_usage(self):
        live, peak = C.c_uint64(), C.c_uint64()
        self.lib.ds4_numeric_memory(self.ctx, C.byref(live), C.byref(peak))
        return MemoryUsage(live.value, peak.value)

    def linear_workspace(self, rows, inner, columns):
        return self.lib.ds4_numeric_linear_workspace(self.ctx, rows, inner, columns)

    def configure_metal_budget(self, *, reserve_bytes, storage_limit_bytes):
        if self.backend != "metal":
            raise ValueError("Metal memory policy requires a Metal context")
        if (
            not 0 <= reserve_bytes < 2**64
            or not 0 < storage_limit_bytes < 2**64 - reserve_bytes
        ):
            raise ValueError("invalid memory budget")
        self.ok(self.lib.ds4_numeric_metal_budget(self.ctx, reserve_bytes))
        # Unified DEVICE storage also consumes HOST capacity. Both ledgers
        # retain their global-free guard and reserve while sharing this cap.
        for location in (0, 3):
            budget = Budget()
            self.ok(
                self.lib.ds4_resources_get_budget(
                    self.resources, location, C.byref(budget)
                )
            )
            budget.limit = min(budget.limit, storage_limit_bytes + budget.margin)
            self.ok(
                self.lib.ds4_resources_set_budget(
                    self.resources, location, C.byref(budget)
                )
            )

    def linear(self, output, inputs, weights, rows, inner, columns):
        self.call("linear", output, inputs, weights, View(), rows, inner, columns)

    def rms_norm(self, output, inputs, scale, width, epsilon):
        self.call("norm", output, inputs, scale, View(), width, epsilon, 0, 1, False)

    def scale(self, output, inputs, factor):
        self.call("point", output, inputs, View(), 6, factor)

    def clear(self):
        while self.buffers:
            # A failed free retains ownership in the C API. Keep the handle
            # too so recovery can retry rather than losing a live allocation.
            self.ok(self.lib.ds4_num_free(self.buffers[-1]))
            self.buffers.pop()

    def close(self):
        self.clear()
        if self.ctx:
            self.ok(self.lib.ds4_numeric_close(self.ctx))
            self.ctx = C.c_void_p()
        if self.resources:
            self.ok(self.lib.ds4_resources_close(self.resources))
            self.resources = C.c_void_p()
