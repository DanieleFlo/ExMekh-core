"""Frozen numerical and residency rules of the native release suite.

Persistence and server lifetime accounting belong to qualification_checkpoint.
This owner restores and validates the same suite history on fresh/resumed runs.
"""

import hashlib
import io
import json
import math
import statistics


def timing(samples):
    values = sorted(samples)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite timing samples required")

    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = deviation / mean
    return {
        "samples": samples,
        "min": values[0],
        "p10": percentile(0.1),
        "median": statistics.median(values),
        "mean": mean,
        "p90": percentile(0.9),
        "max": values[-1],
        "stddev": deviation,
        "cv": cv,
        "verdict": "NEED_MORE_DATA" if len(values) < 5 or cv > 0.05 else "STABLE",
    }


class SuiteReleaseHistory:
    def __init__(self, records, image_size):
        self._image_size = tuple(map(int, image_size.split("x")))
        self._images, self._vectors, self._snapshots = {}, {}, {}
        self._samples = {"image": [], "embedding": [], "cycle": []}
        self.restore(records)

    def restore(self, records):
        for row in records:
            self.validate_resources(row["embedding"]["resources"])
            self.record(row)

    @staticmethod
    def seed(cycle):
        return 17 if cycle % 2 else 1000 + cycle

    @staticmethod
    def dimensions(cycle):
        return (128, 256, 512, 768)[cycle % 4]

    @staticmethod
    def _remember(known, key, digest, message):
        if key in known and known[key] != digest:
            raise AssertionError(message)
        known[key] = digest
        return digest

    def image_digest(self, cycle, png):
        from PIL import Image

        with Image.open(io.BytesIO(png)) as decoded:
            if decoded.size != self._image_size or decoded.mode != "RGBA":
                raise AssertionError("unexpected image geometry or alpha channel")
        return self._remember(
            self._images,
            self.seed(cycle),
            hashlib.sha256(png).hexdigest(),
            "residency changed deterministic image output",
        )

    def embedding_digest(self, cycle, matrix):
        dimensions = self.dimensions(cycle)
        if len(matrix) != 5 or any(len(row) != dimensions for row in matrix):
            raise AssertionError("missing modality or vector dimension")
        for row in matrix:
            if (
                any(not math.isfinite(v) for v in row)
                or abs(math.sqrt(sum(v * v for v in row)) - 1) > 1e-5
            ):
                raise AssertionError("embedding norm or finite gate failed")
        return self._remember(
            self._vectors,
            dimensions,
            hashlib.sha256(
                json.dumps(matrix, separators=(",", ":")).encode()
            ).hexdigest(),
            "residency changed deterministic multimodal vectors",
        )

    @staticmethod
    def validate_resources(snapshot):
        if any(any(owner["reserved"]) for owner in snapshot["owners"]):
            raise AssertionError("operation left a live reservation")

    def record(self, row):
        self._remember(
            self._images,
            self.seed(row["cycle"]),
            row["image_sha256"],
            "residency changed deterministic image output",
        )
        self._remember(
            self._vectors,
            row["dimensions"],
            row["embedding_sha256"],
            "residency changed deterministic multimodal vectors",
        )
        self._snapshots.setdefault(row["segment"], []).append(
            row["embedding"]["resources"]
        )
        if not row["warmup"]:
            for name, field in (
                ("image", "image_seconds"),
                ("embedding", "embedding_seconds"),
                ("cycle", "seconds"),
            ):
                self._samples[name].append(row[field])

    @property
    def owned_memory_observed(self):
        return bool(self._snapshots)

    def timings(self):
        return {
            name: timing(values) for name, values in self._samples.items() if values
        }

    def validate_resting_state(self) -> None:
        baselines = []
        for snapshots in self._snapshots.values():
            native = [
                next(o for o in s["owners"] if o["owner"] == 500) for s in snapshots[4:]
            ]
            if native:
                baselines.extend(native)
        if baselines and any(owner != baselines[0] for owner in baselines):
            raise AssertionError(
                "native owner did not return to its bounded warm state"
            )
