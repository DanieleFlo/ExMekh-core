#!/usr/bin/env python3
"""Offline bundle compilation, lossless conversion and pinned downloads.

The native runtime consumes bundle.ds4b and original file mappings. This tool
is never invoked by inference. Shapes use GGUF order, with dimension zero the
contiguous dimension; BF16/FP32 bit patterns and GGUF mixed blocks are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

LAYOUT = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, 56),
    30: (1, 2),
    39: (32, 17),
}
SAFE_TYPES = {
    "F32": 0,
    "F16": 1,
    "BF16": 30,
    "I8": 24,
    "I16": 25,
    "I32": 26,
    "I64": 27,
    "F64": 28,
}
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
MAX_INDEX_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_matches(path: Path, artifact) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == artifact["bytes"]
        and sha256(path) == artifact["sha256"]
    )


def relative_path(value: str) -> str:
    parts = value.split("/")
    if (
        not value
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"invalid bundle-relative path: {value}")
    return str(PurePosixPath(value))


def exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("truncated tensor directory")
    return value


def number(stream, code):
    return struct.unpack("<" + code, exact(stream, struct.calcsize("<" + code)))[0]


def gguf_string(stream):
    count = number(stream, "Q")
    if count > 64 * 1024 * 1024:
        raise ValueError("GGUF string too large")
    return exact(stream, count).decode("utf-8")


def skip_value(stream, kind):
    if kind == 8:
        gguf_string(stream)
    elif kind == 9:
        child, count = number(stream, "I"), number(stream, "Q")
        if count > 1 << 24 or child == 9:
            raise ValueError("unsupported GGUF metadata array")
        if child in SCALAR:
            stream.seek(SCALAR[child] * count, 1)
        elif child == 8:
            for _ in range(count):
                gguf_string(stream)
        else:
            raise ValueError("unsupported GGUF metadata type")
    elif kind in SCALAR:
        stream.seek(SCALAR[kind], 1)
    else:
        raise ValueError("unsupported GGUF metadata type")


def layout(kind, shape):
    if (
        kind not in LAYOUT
        or len(shape) > 8
        or any(not isinstance(n, int) or n <= 0 for n in shape)
    ):
        raise ValueError("unsupported tensor storage or dimensions")
    block, width = LAYOUT[kind]
    if not shape:
        if block != 1:
            raise ValueError("quantized scalar storage is unsupported")
        return block, width, width, []
    if shape[0] % block:
        raise ValueError("quantized rows must contain whole blocks")
    stride = [width]
    size = shape[0] // block * width
    for dimension in shape[1:]:
        stride.append(size)
        size *= dimension
    if size >= 1 << 64:
        raise ValueError("tensor extent overflow")
    return block, width, size, stride


def inventory(path: Path, storage: str):
    size = path.stat().st_size
    tensors = []
    with path.open("rb") as stream:
        if storage == "gguf":
            if exact(stream, 4) != b"GGUF" or number(stream, "I") != 3:
                raise ValueError("only little-endian GGUF v3 is accepted")
            count, metadata = number(stream, "Q"), number(stream, "Q")
            if count > size // 24 or metadata > size // 12:
                raise ValueError("invalid GGUF directory count")
            alignment = 32
            scalar_tensors = set()
            keys = set()
            for _ in range(metadata):
                key, kind = gguf_string(stream), number(stream, "I")
                if key in keys:
                    raise ValueError("duplicate GGUF metadata")
                keys.add(key)
                if key == "general.alignment":
                    if kind != 4:
                        raise ValueError("invalid GGUF alignment type")
                    alignment = number(stream, "I")
                elif key == "ds4.scalar_tensors":
                    if kind != 9 or number(stream, "I") != 8:
                        raise ValueError("invalid scalar tensor metadata")
                    scalar_tensors = {
                        gguf_string(stream) for _ in range(number(stream, "Q"))
                    }
                else:
                    skip_value(stream, kind)
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError("invalid GGUF alignment")
            for _ in range(count):
                name, rank = gguf_string(stream), number(stream, "I")
                if rank < 1 or rank > 8:
                    raise ValueError("invalid tensor rank")
                shape = [number(stream, "Q") for _ in range(rank)]
                kind, offset = number(stream, "I"), number(stream, "Q")
                if name in scalar_tensors:
                    if shape != [1]:
                        raise ValueError("invalid converted scalar dimensions")
                    shape = []
                tensors.append(
                    dict(
                        name=name,
                        shape=shape,
                        type=kind,
                        offset=offset,
                        alignment=alignment,
                    )
                )
            start = (stream.tell() + alignment - 1) // alignment * alignment
            for tensor in tensors:
                tensor["offset"] += start
        elif storage == "safetensors":
            count = number(stream, "Q")
            if count > 64 * 1024 * 1024 or count > size - 8:
                raise ValueError("invalid safetensors header size")

            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("duplicate safetensors key")
                    result[key] = value
                return result

            header = json.loads(exact(stream, count), object_pairs_hook=pairs)
            start = count + 8
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                kind = SAFE_TYPES.get(spec["dtype"])
                if kind is None:
                    raise ValueError(f"unsupported safetensors dtype: {spec['dtype']}")
                begin, end = spec["data_offsets"]
                shape = list(reversed(spec["shape"]))
                _, _, extent, _ = layout(kind, shape)
                if end - begin != extent or begin < 0:
                    raise ValueError("invalid safetensors extent")
                tensors.append(
                    dict(
                        name=name,
                        shape=shape,
                        type=kind,
                        offset=start + begin,
                        alignment=1,
                    )
                )
        else:
            raise ValueError("unknown native storage format")
    names = set()
    for tensor in tensors:
        if tensor["name"] in names:
            raise ValueError("duplicate tensor name")
        names.add(tensor["name"])
        block, width, extent, stride = layout(tensor["type"], tensor["shape"])
        tensor.update(
            block_elements=block, block_bytes=width, bytes=extent, stride=stride
        )
        if (
            tensor["offset"] % tensor["alignment"]
            or tensor["offset"] < start
            or tensor["offset"] + extent > size
        ):
            raise ValueError("tensor is outside its original artifact")
    ordered = sorted(tensors, key=lambda t: (t["offset"], t["bytes"]))
    previous = None
    for tensor in ordered:
        if previous and tensor["offset"] < previous["offset"] + previous["bytes"]:
            if any(
                tensor[key] != previous[key]
                for key in ("offset", "bytes", "type", "shape")
            ):
                raise ValueError("partially overlapping tensor payloads")
        previous = tensor
    return sorted(tensors, key=lambda t: t["name"])


def packed_string(value):
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > 4096 or b"\0" in encoded:
        raise ValueError("invalid manifest string")
    return struct.pack("<I", len(encoded)) + encoded


def atomic_write(path, data):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_spec(spec):
    if spec.get("version") != 1 or not spec.get("model") or not spec.get("profile"):
        raise ValueError("invalid bundle identity")
    if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
        raise ValueError("bundle requires a pinned source revision")
    components = spec.get("components", [])
    if not components or len({c["id"] for c in components}) != len(components):
        raise ValueError("invalid component set")
    for component in components:
        if component["for_model_revision"] != spec["revision"] or component[
            "compute_dtype"
        ] not in ("float32", "bfloat16"):
            raise ValueError("incompatible component revision or compute dtype")
        if not re.fullmatch(r"[0-9a-f]{40}", component["revision"]):
            raise ValueError("component requires a pinned source revision")
    paths = set()
    for file in spec.get("files", []):
        relative_path(file["path"])
        if file["path"] in paths or file["format"] not in (
            "gguf",
            "safetensors",
            "asset",
        ):
            raise ValueError("invalid or duplicate bundle artifact")
        paths.add(file["path"])
        if (
            not re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
            or file["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{40}", file["revision"])
        ):
            raise ValueError("artifact requires exact revision, size and SHA-256")
    if not paths:
        raise ValueError("empty bundle")
    if not spec.get("assets") or any(
        path not in paths for path in spec["assets"].values()
    ):
        raise ValueError("bundle must pin its configuration/processor assets")
    files = {file["path"]: file for file in spec["files"]}
    if any(
        files[path]["format"] != "asset" or files[path]["revision"] != spec["revision"]
        for path in spec["assets"].values()
    ):
        raise ValueError("configuration/processor assets must match the model revision")
    for component in components:
        if not component.get("selectors") or any(
            s["file"] not in paths for s in component["selectors"]
        ):
            raise ValueError("component selector references an absent artifact")
        for selector in component["selectors"]:
            file = files[selector["file"]]
            if file["format"] == "asset" or file["revision"] != component["revision"]:
                raise ValueError(
                    "component selector requires weights from its pinned revision"
                )
            if selector.get("group_regex"):
                try:
                    pattern = re.compile(selector["group_regex"])
                except re.error as error:
                    raise ValueError("invalid layer group expression") from error
                if pattern.groups != 1:
                    raise ValueError("layer group expression requires one capture")


def compile_bundle(spec, root: Path, output: Path):
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=True)
    tensors = []
    for file_id, file in enumerate(spec["files"]):
        path = root / relative_path(file["path"])
        if not artifact_matches(path, file):
            raise ValueError(f"artifact identity mismatch: {path}")
        if file["format"] == "asset":
            continue
        for tensor in inventory(path, file["format"]):
            matches = []
            for index, component in enumerate(spec["components"]):
                for selector in component["selectors"]:
                    if selector["file"] == file["path"] and tensor["name"].startswith(
                        selector.get("prefix", "")
                    ):
                        matches.append((index, selector))
            if len(matches) != 1:
                raise ValueError(
                    f"tensor must have exactly one component owner: {tensor['name']}"
                )
            component_id, selector = matches[0]
            match = (
                re.search(selector["group_regex"], tensor["name"])
                if selector.get("group_regex")
                else None
            )
            group = int(match.group(1)) + 1 if match else 0
            if not 0 <= group < 1 << 32:
                raise ValueError("layer group exceeds the native index range")
            tensors.append(
                dict(tensor, file=file_id, component=component_id, group=group)
            )
    for i, component in enumerate(spec["components"]):
        if not any(t["component"] == i for t in tensors):
            raise ValueError(f"component has no tensors: {component['id']}")
    if len({(t["component"], t["name"]) for t in tensors}) != len(tensors):
        raise ValueError("duplicate tensor name across component shards")
    tensors.sort(key=lambda t: (t["component"], t["group"], t["name"]))
    data = bytearray(
        packed_string(spec["model"])
        + packed_string(spec["revision"])
        + packed_string(spec["profile"])
    )
    data += struct.pack(
        "<III", len(spec["files"]), len(spec["components"]), len(tensors)
    )
    for file in spec["files"]:
        data += b"".join(packed_string(file[k]) for k in ("path", "revision", "format"))
        data += struct.pack("<Q", file["bytes"]) + bytes.fromhex(file["sha256"])
    for component in spec["components"]:
        data += b"".join(
            packed_string(component[k]) for k in ("id", "revision", "compute_dtype")
        )
    for tensor in tensors:
        data += packed_string(tensor["name"])
        data += struct.pack(
            "<8I2Q",
            tensor["file"],
            tensor["component"],
            tensor["group"],
            tensor["type"],
            len(tensor["shape"]),
            tensor["block_elements"],
            tensor["block_bytes"],
            tensor["alignment"],
            tensor["offset"],
            tensor["bytes"],
        )
        data += struct.pack("<" + "Q" * len(tensor["shape"]), *tensor["shape"])
        data += struct.pack("<" + "Q" * len(tensor["stride"]), *tensor["stride"])
    if len(data) > MAX_INDEX_BYTES:
        raise ValueError("bundle directory exceeds the native index limit")
    header = (
        b"DS4BND\1\0"
        + struct.pack("<QQ", 1, len(data))
        + hashlib.sha256(data).digest()
        + bytes(8)
    )
    atomic_write(output / "bundle.ds4b", header + data)
    manifest = dict(
        spec,
        tensors=tensors,
        native_index_sha256=hashlib.sha256(header + data).hexdigest(),
    )
    atomic_write(
        output / "bundle.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def convert_safetensors(source: Path, output: Path, architecture: str, revision: str):
    tensors = inventory(source, "safetensors")

    def gs(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("conversion requires a pinned source revision")
    metadata = [
        ("general.architecture", architecture),
        ("ds4.source.revision", revision),
    ]
    scalars = [t["name"] for t in tensors if not t["shape"]]
    header = b"GGUF" + struct.pack(
        "<IQQ", 3, len(tensors), len(metadata) + bool(scalars)
    )
    for key, value in metadata:
        header += gs(key) + struct.pack("<I", 8) + gs(value)
    if scalars:
        header += gs("ds4.scalar_tensors") + struct.pack("<IIQ", 9, 8, len(scalars))
        header += b"".join(gs(name) for name in scalars)
    offset = 0
    for tensor in tensors:
        offset = (offset + 31) // 32 * 32
        tensor["converted_offset"] = offset
        shape = tensor["shape"] or [1]
        header += gs(tensor["name"]) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", tensor["type"], offset)
        offset += tensor["bytes"]
    header += bytes((-len(header)) % 32)
    temporary = output.with_name(output.name + ".partial")
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            dst.write(header)
            for tensor in tensors:
                target = len(header) + tensor["converted_offset"]
                dst.write(bytes(target - dst.tell()))
                src.seek(tensor["offset"])
                remaining = tensor["bytes"]
                while remaining:
                    block = exact(src, min(65536, remaining))
                    dst.write(block)
                    remaining -= len(block)
            dst.flush()
            os.fsync(dst.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def download_bundle(spec, destination: Path, components=()):
    validate_spec(spec)
    selected = set(components)
    if selected - {c["id"] for c in spec["components"]}:
        raise ValueError("unknown explicit component target")
    needed = {
        s["file"]
        for c in spec["components"]
        if not selected or c["id"] in selected
        for s in c["selectors"]
    }
    for file in spec["files"]:
        if file["format"] != "asset" and file["path"] not in needed:
            continue
        path = destination / relative_path(file["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_matches(path, file):
            continue
        partial = path.with_name(path.name + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > file["bytes"]:
            raise ValueError("partial artifact exceeds pinned size")
        if offset == file["bytes"]:
            if sha256(partial) != file["sha256"]:
                raise ValueError("completed partial artifact has a wrong SHA-256")
            partial.replace(path)
            continue
        url = file.get("url") or (
            "https://huggingface.co/"
            + file["repository"]
            + "/resolve/"
            + file["revision"]
            + "/"
            + file["path"]
        )
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "huggingface.co"
            and os.environ.get("HF_TOKEN")
        ):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            append = offset and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise ValueError("invalid resume response")
            with partial.open("ab" if append else "wb") as stream:
                while block := response.read(65536):
                    stream.write(block)
                    if stream.tell() > file["bytes"]:
                        raise ValueError("download exceeds pinned size")
                stream.flush()
                os.fsync(stream.fileno())
        if partial.stat().st_size != file["bytes"] or sha256(partial) != file["sha256"]:
            raise ValueError("download SHA-256 or size mismatch; partial preserved")
        partial.replace(path)


def discover(spec, root):
    validate_spec(spec)
    # Shared weight files and recipe assets are hashed once per discovery,
    # regardless of how many components refer to them.
    verified_files = {
        f["path"]: artifact_matches(root / f["path"], f) for f in spec["files"]
    }
    assets = set(spec["assets"].values())
    components = []
    for component in spec["components"]:
        paths = {s["file"] for s in component["selectors"]} | assets
        verified = all(verified_files[p] for p in paths)
        components.append(
            {
                "id": component["id"],
                "artifacts_verified": verified,
                "host_loaded": False,
                "device_resident": False,
            }
        )
    return {
        "model": spec["model"],
        "revision": spec["revision"],
        "profile": spec["profile"],
        "configured": True,
        "ready": False,
        "qualified_backends": [],
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "download", "discover"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--root", type=Path, required=True)
        if name == "build":
            sub.add_argument("--output", type=Path, required=True)
        if name == "download":
            sub.add_argument("--component", action="append", default=[])
    sub = commands.add_parser("convert")
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--architecture", required=True)
    sub.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.command == "convert":
        convert_safetensors(args.source, args.output, args.architecture, args.revision)
    else:
        spec = json.loads(args.manifest.read_text())
        if args.command == "build":
            compile_bundle(spec, args.root, args.output)
        elif args.command == "download":
            download_bundle(spec, args.root, args.component)
        else:
            print(json.dumps(discover(spec, args.root), indent=2))


if __name__ == "__main__":
    main()
