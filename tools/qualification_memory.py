"""Memory limits shared by native command and server qualification runners."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class MemoryPolicy:
    """Snapshot configuration once; evaluate and report the same limits."""

    host_reserve_bytes: int = 2**31
    critical_pressure: int = 4
    swap_growth_bytes: int = 256 * 2**20
    unsafe_samples: int = 2
    reserve_variable: ClassVar[str] = "DS4_METAL_HOST_RESERVE_BYTES"

    def __post_init__(self) -> None:
        for value in (self.host_reserve_bytes, self.swap_growth_bytes):
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError("memory limits must be unsigned 64-bit byte counts")
        for value in (self.critical_pressure, self.unsafe_samples):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "pressure and consecutive sample limits must be positive integers"
                )

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "MemoryPolicy":
        """Match C policy: empty, malformed and overflowing values use 2 GiB."""
        environment = os.environ if environment is None else environment
        value = environment.get(cls.reserve_variable, "")
        digits = value.lstrip("0")
        if value and value.isascii() and value.isdecimal() and len(digits) <= 20:
            reserve = int(digits or "0")
            if reserve < 2**64:
                return cls(reserve)
        return cls()

    def is_unsafe(self, current: dict, baseline: dict) -> bool:
        return (
            current["pressure"] >= self.critical_pressure
            or current["swap_bytes"] > baseline["swap_bytes"] + self.swap_growth_bytes
            or current.get("host_available_bytes", 2**64 - 1) < self.host_reserve_bytes
        )

    def environment(self) -> dict[str, str]:
        """Propagate the resolved value, including zero, to the native owner."""
        return {self.reserve_variable: str(self.host_reserve_bytes)}

    def as_dict(self) -> dict[str, int]:
        return {
            "host_reserve_bytes": self.host_reserve_bytes,
            "critical_pressure": self.critical_pressure,
            "swap_growth_bytes": self.swap_growth_bytes,
            "unsafe_samples": self.unsafe_samples,
        }
