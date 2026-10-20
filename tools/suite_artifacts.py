"""Validate development probe outputs before promoting numerical evidence."""

import array
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _invalid_constant(value: str):
    raise ValueError(f"non-standard JSON number: {value}")


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=_finite_float,
        parse_constant=_invalid_constant,
    )


def _positive_integer(value) -> bool:
    return type(value) is int and value > 0


def _validate_tokens(tokens, vocab: int) -> None:
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or not 0 <= token < vocab for token in tokens)
    ):
        raise ValueError("missing or invalid generated tokens")


def probe_artifacts(directory: Path, compare: Path | None = None) -> list[dict]:
    metadata = read_json(directory / "result.json")
    if not isinstance(metadata, dict):
        raise ValueError("probe metadata must be an object")
    vocab = metadata.get("vocab")
    if not _positive_integer(vocab) or not _positive_integer(
        metadata.get("prefix_tokens")
    ):
        raise ValueError("missing or invalid probe shape")
    tokens = metadata.get("tokens")
    _validate_tokens(tokens, vocab)
    logits = directory / "logits.f32"
    if logits.stat().st_size != (len(tokens) + 1) * vocab * 4:
        raise ValueError("logits byte length does not match probe shape")
    values = array.array("f")
    values.frombytes(logits.read_bytes())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite probe logits")
    if not (directory / "session.payload").stat().st_size:
        raise ValueError("empty session payload")
    records = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        record = file_record(path)
        if path.name == "logits.f32":
            record["finite"] = True
        if compare and path.name in {"logits.f32", "session.payload"}:
            record["bit_exact"] = record["sha256"] == sha256_file(compare / path.name)
        records.append(record)
    return records


def frontier_artifacts(directory: Path) -> list[dict]:
    logits = sorted(directory.glob("frontier_*.logits.json"))
    decode = sorted(directory.glob("frontier_*.decode.json"))
    if not logits or {p.name.removesuffix(".logits.json") for p in logits} != {
        p.name.removesuffix(".decode.json") for p in decode
    }:
        raise ValueError("missing or incomplete frontier outputs")
    records = []
    for path in sorted(logits + decode):
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("frontier metadata must be an object")
        vocab, values = data.get("vocab"), data.get("logits")
        if (
            not _positive_integer(vocab)
            or not isinstance(values, list)
            or len(values) != vocab
            or any(type(value) not in (int, float) for value in values)
        ):
            raise ValueError(f"invalid full-vocabulary frontier: {path.name}")
        if path.name.endswith(".decode.json"):
            _validate_tokens(data.get("generated_tokens"), vocab)
            if data.get("generated_count") != len(data["generated_tokens"]):
                raise ValueError("generated token count mismatch")
        record = file_record(path)
        # Keep the original manifest format for comparisons with frozen runs.
        records.append({"path": record["path"], "sha256": record["sha256"]})
    return records
