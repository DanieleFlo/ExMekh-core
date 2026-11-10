"""The child process, report and guard share one immutable memory profile."""

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from tools.qualification_memory import MemoryPolicy


class MemoryPolicyTests(unittest.TestCase):
    def test_unsigned_environment_values_match_the_native_policy(self):
        cases = (
            ("0", 0),
            ("0" * 5000, 0),
            ("1073741824", 2**30),
            ("18446744073709551615", 2**64 - 1),
            ("", 2**31),
            ("-1", 2**31),
            ("1GB", 2**31),
            (" 0", 2**31),
            ("1.5", 2**31),
            ("٠", 2**31),
            ("18446744073709551616", 2**31),
        )
        self.assertEqual(MemoryPolicy.from_environment({}).host_reserve_bytes, 2**31)
        for value, expected in cases:
            with self.subTest(value=value[:30]):
                policy = MemoryPolicy.from_environment(
                    {MemoryPolicy.reserve_variable: value}
                )
                self.assertEqual(policy.host_reserve_bytes, expected)
                self.assertEqual(policy.as_dict()["host_reserve_bytes"], expected)
                self.assertEqual(
                    MemoryPolicy.from_environment(policy.environment()), policy
                )

    def test_zero_reserve_keeps_pressure_and_swap_guards(self):
        policy = MemoryPolicy(0)
        baseline = {"pressure": 1, "swap_bytes": 100, "host_available_bytes": 0}
        self.assertFalse(policy.is_unsafe(baseline, baseline))
        self.assertFalse(policy.is_unsafe({**baseline, "pressure": 2}, baseline))
        self.assertTrue(policy.is_unsafe({**baseline, "pressure": 4}, baseline))
        limit = baseline["swap_bytes"] + policy.swap_growth_bytes
        self.assertFalse(policy.is_unsafe({**baseline, "swap_bytes": limit}, baseline))
        self.assertTrue(
            policy.is_unsafe({**baseline, "swap_bytes": limit + 1}, baseline)
        )
        self.assertTrue(MemoryPolicy(1).is_unsafe(baseline, baseline))

    def test_configuration_is_snapshotted_and_cannot_change_during_a_run(self):
        with patch.dict(os.environ, DS4_METAL_HOST_RESERVE_BYTES="0"):
            policy = MemoryPolicy.from_environment()
        self.assertEqual(policy.host_reserve_bytes, 0)
        with self.assertRaises(FrozenInstanceError):
            policy.host_reserve_bytes = 1
        for invalid in (-1, True, 2**64):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MemoryPolicy(invalid)
