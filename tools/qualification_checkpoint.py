"""Durable, identity-checked progress for native suite qualification."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


def digest_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".checkpoint-", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


@contextmanager
def exclusive_lock(path: Path):
    with path.open("a+") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"qualification already running: {path.parent}"
            ) from error
        owner.seek(0)
        owner.truncate()
        owner.write(str(os.getpid()) + "\n")
        owner.flush()
        try:
            yield
        finally:
            fcntl.flock(owner, fcntl.LOCK_UN)


@contextmanager
def qualification_output(root: Path, resume: bool):
    """Retain every server attempt; a live owner prevents a duplicate model."""
    root.mkdir(parents=True, exist_ok=resume)
    with exclusive_lock(root / ".qualification.lock"):
        output = root
        if resume:
            if not (root / "qualification-profile.json").is_file():
                raise ValueError("cannot resume an unversioned qualification directory")
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            index = 1
            while (output := attempts / f"{index:04d}").exists():
                index += 1
            output.mkdir()
        yield output


class ReleaseCheckpoint:
    """Own completed-cycle durability, artifact integrity and process continuity.

    The journal is authoritative. A torn last write is archived and retried;
    a damaged completed record or artifact is an error, never dropped progress.
    """

    def __init__(self, output: Path, identity: dict, segment: str, resume: bool):
        if not isinstance(segment, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", segment
        ):
            raise ValueError("invalid qualification server instance")
        self.output = output
        self._identity = deepcopy(identity)
        self.segment = segment
        self.resume = resume
        self._records = []
        self._last_hash = digest_json(self._identity)
        self._lease = None

    def __enter__(self):
        self._lease = exclusive_lock(self.output / ".cycles.lock")
        self._lease.__enter__()
        try:
            manifest = self.output / "checkpoint.json"
            journal = self.output / "cycles.jsonl"
            if (self.output / "gate-failure.json").exists():
                raise ValueError(
                    "a validation gate failed; use a new qualification directory"
                )
            if self.resume and manifest.exists():
                saved = json.loads(manifest.read_text())
                if saved != {"version": 1, "identity": self._identity}:
                    raise ValueError(
                        "checkpoint identity changed: binary, inputs or qualification policy"
                    )
                self._load()
            else:
                # Startup can be interrupted before creating the manifest.
                # Only an absent/empty journal is safe to initialize again;
                # unversioned completed cycles cannot establish provenance.
                if manifest.exists() or (
                    journal.exists() and (not self.resume or journal.stat().st_size)
                ):
                    raise FileExistsError("checkpoint already exists or is unversioned")
                atomic_bytes(journal, b"")
                atomic_json(manifest, {"version": 1, "identity": self._identity})
            return self
        except BaseException:
            self._lease.__exit__(None, None, None)
            raise

    def __exit__(self, *exception):
        self._lease.__exit__(*exception)

    @property
    def records(self) -> tuple[dict, ...]:
        return tuple(deepcopy(self._records))

    @property
    def completed(self) -> int:
        return sum(not row["warmup"] for row in self._records)

    def needs_warmup(self) -> bool:
        return not any(
            row["segment"] == self.segment and row["warmup"] for row in self._records
        )

    def pending_cycles(self):
        warmup = int(not self._identity["configuration"]["bounded"])
        target = self._identity["configuration"]["cycles"]
        pending = list(range(self.completed + warmup, target + warmup))
        return ([(0, True)] if pending and warmup and self.needs_warmup() else []) + [
            (cycle, False) for cycle in pending
        ]

    def completion(self):
        required = self._identity["configuration"]["cycles"]
        if self.completed < required:
            raise ValueError("qualification still has incomplete cycles")
        bounded = self._identity["configuration"]["bounded"]
        continuous = self.continuity()["segments"].get(self.segment, 0) >= required
        return {
            "status": "PASS" if bounded or continuous else "PASS_SEGMENTED",
            "continuous_lifetime": "NOT_APPLICABLE"
            if bounded
            else "PASS"
            if continuous
            else "NOT_VERIFIED",
        }

    def _artifact(self, name: str) -> Path:
        path = (self.output / name).resolve()
        if not path.is_relative_to(self.output.resolve()):
            raise ValueError("checkpoint artifact escapes its directory")
        return path

    def _validate_artifacts(self, row: dict) -> None:
        if digest_file(self._artifact(row["image_file"])) != row["image_sha256"]:
            raise ValueError("completed image artifact changed")
        matrix = json.loads(
            self._artifact(f"vectors-{row['dimensions']}.json").read_text()
        )
        digest = hashlib.sha256(
            json.dumps(matrix, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != row["embedding_sha256"]:
            raise ValueError("completed embedding artifact changed")

    def _accept(self, row: dict) -> None:
        payload = {
            k: v for k, v in row.items() if k not in ("chain_sha256", "previous_sha256")
        }
        if row.get("previous_sha256") != self._last_hash or row.get(
            "chain_sha256"
        ) != digest_json(payload):
            raise ValueError("completed checkpoint record changed")
        if row.get("status") != "PASS" or type(row.get("warmup")) is not bool:
            raise ValueError("checkpoint contains an unvalidated cycle")
        cycle = row["cycle"]
        warmup = not self._identity["configuration"]["bounded"]
        if row["warmup"] and (
            not warmup
            or any(
                r["warmup"] and r["segment"] == row["segment"] for r in self._records
            )
        ):
            raise ValueError("duplicate or unexpected checkpoint warmup")
        if type(cycle) is not int or cycle != (
            0 if row["warmup"] else self.completed + int(warmup)
        ):
            raise ValueError("checkpoint cycle sequence is incomplete")
        if row["dimensions"] != (128, 256, 512, 768)[cycle % 4]:
            raise ValueError("checkpoint dimensions changed")
        if (
            self._records
            and row["segment"] != self._records[-1]["segment"]
            and any(previous["segment"] == row["segment"] for previous in self._records)
        ):
            raise ValueError("checkpoint server instance is not continuous")
        self._validate_artifacts(row)
        self._records.append(row)
        self._last_hash = row["chain_sha256"]

    def _load(self) -> None:
        path = self.output / "cycles.jsonl"
        raw = path.read_bytes()
        complete = raw.rfind(b"\n") + 1
        # Completed lines are verified before repairing a torn tail.
        for line in raw[:complete].splitlines():
            self._accept(json.loads(line))
        if complete != len(raw):
            atomic_bytes(
                self.output / f"interrupted-tail-{uuid.uuid4().hex}.jsonl",
                raw[complete:],
            )
            with path.open("r+b") as journal:
                journal.truncate(complete)
                journal.flush()
                os.fsync(journal.fileno())

    def save_image(self, cycle: int, warmup: bool, png: bytes) -> str:
        name = f"warmup-{self.segment}.png" if warmup else f"image-{cycle:03d}.png"
        if any(row["image_file"] == name for row in self._records):
            if self._artifact(name).read_bytes() != png:
                raise ValueError("completed image cannot be overwritten")
        else:
            atomic_bytes(self._artifact(name), png)
        return name

    def save_vectors(self, dimensions: int, matrix) -> None:
        path = self._artifact(f"vectors-{dimensions}.json")
        if not any(row["dimensions"] == dimensions for row in self._records):
            atomic_json(path, matrix)

    def append(self, record: dict) -> dict:
        row = {**deepcopy(record), "segment": self.segment}
        row["chain_sha256"] = digest_json(row)
        row["previous_sha256"] = self._last_hash
        previous_hash = self._last_hash
        self._accept(row)
        try:
            with (self.output / "cycles.jsonl").open("ab") as journal:
                journal.write(
                    (
                        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    ).encode()
                )
                journal.flush()
                os.fsync(journal.fileno())
        except BaseException:
            self._records.pop()
            self._last_hash = previous_hash
            raise
        return deepcopy(row)

    def continuity(self) -> dict:
        measured = {}
        for row in self._records:
            measured.setdefault(row["segment"], 0)
            measured[row["segment"]] += not row["warmup"]
        return {
            "segments": measured,
            "longest_continuous_cycles": max(measured.values(), default=0),
        }

    def save_summary(self, report: dict) -> None:
        report.update(
            completed_cycles=self.completed,
            warmup_cycles=sum(row["warmup"] for row in self._records),
            **self.continuity(),
        )
        atomic_json(self.output / "summary.json", report)
        atomic_json(self.output / f"segment-{self.segment}.json", report)

    def reject_gate(self, error: BaseException) -> None:
        atomic_json(
            self.output / "gate-failure.json",
            {"error": repr(error), "segment": self.segment},
        )
