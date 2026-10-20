import copy
import hashlib
import http.server
import json
import os
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema

from tools.suite_bundle import (
    LAYOUT,
    compile_bundle,
    convert_safetensors,
    discover,
    download_bundle,
    inventory,
    sha256,
)


class SuiteBundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        header = {
            "layer.0.weight": {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [0, 16],
            },
            "norm.weight": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        (self.root / "weights.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + bytes(range(32))
        )
        (self.root / "config.json").write_text('{"layers":1,"norm_eps":1e-6}')
        self.revision = "a" * 40
        self.spec = {
            "version": 1,
            "model": "fixture",
            "revision": self.revision,
            "profile": "bf16-f32",
            "assets": {"config": "config.json"},
            "components": [
                {
                    "id": "encoder",
                    "revision": self.revision,
                    "for_model_revision": self.revision,
                    "compute_dtype": "bfloat16",
                    "selectors": [
                        {
                            "file": "weights.safetensors",
                            "prefix": "",
                            "group_regex": r"layer\.(\d+)\.",
                        }
                    ],
                }
            ],
            "files": [
                {
                    "path": name,
                    "revision": self.revision,
                    "format": kind,
                    "sha256": sha256(self.root / name),
                    "bytes": (self.root / name).stat().st_size,
                }
                for name, kind in [
                    ("weights.safetensors", "safetensors"),
                    ("config.json", "asset"),
                ]
            ],
        }

    def native(self, path, mode="test"):
        binary = os.environ.get("DS4_BUNDLE_TEST_BINARY", "./tests/test_bundle")
        return subprocess.run(
            [binary, str(path), str(self.root), mode],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_deterministic_inventory_and_native_group_lifetime(self):
        one = compile_bundle(self.spec, self.root, self.root / "one")
        compile_bundle(self.spec, self.root, self.root / "two")
        self.assertEqual(
            (self.root / "one/bundle.ds4b").read_bytes(),
            (self.root / "two/bundle.ds4b").read_bytes(),
        )
        self.assertEqual(
            (self.root / "one/bundle.json").read_bytes(),
            (self.root / "two/bundle.json").read_bytes(),
        )
        rows = self.native(self.root / "one/bundle.ds4b").splitlines()
        self.native(self.root / "one/bundle.ds4b", "host-deny")
        self.assertEqual(len(rows), len(one["tensors"]))
        for row, tensor in zip(rows, one["tensors"]):
            fields = row.split("\t")
            self.assertEqual(fields[:2], ["encoder", tensor["name"]])
            self.assertEqual(
                list(map(int, fields[2:])),
                [tensor[k] for k in ("group", "type", "offset", "bytes")]
                + tensor["shape"],
            )

    def test_lossless_conversion_repeated_and_mixed_types(self):
        for target in ("one.gguf", "two.gguf"):
            convert_safetensors(
                self.root / "weights.safetensors",
                self.root / target,
                "fixture",
                self.revision,
            )
        self.assertEqual(sha256(self.root / "one.gguf"), sha256(self.root / "two.gguf"))
        original = inventory(self.root / "weights.safetensors", "safetensors")
        converted = inventory(self.root / "one.gguf", "gguf")
        self.assertEqual([t["type"] for t in original], [30, 0])
        for a, b in zip(original, converted):
            self.assertEqual(
                (a["name"], a["shape"], a["type"]), (b["name"], b["shape"], b["type"])
            )
            self.assertEqual(
                (self.root / "weights.safetensors").read_bytes()[
                    a["offset"] : a["offset"] + a["bytes"]
                ],
                (self.root / "one.gguf").read_bytes()[
                    b["offset"] : b["offset"] + b["bytes"]
                ],
            )
        spec = copy.deepcopy(self.spec)
        spec["files"][0].update(
            path="one.gguf",
            format="gguf",
            sha256=sha256(self.root / "one.gguf"),
            bytes=(self.root / "one.gguf").stat().st_size,
        )
        spec["components"][0]["selectors"][0]["file"] = "one.gguf"
        compile_bundle(spec, self.root, self.root / "native")
        self.native(self.root / "native/bundle.ds4b")

    def test_reject_artifacts_changed_during_upload(self):
        compile_bundle(self.spec, self.root, self.root / "native")
        self.native(self.root / "native/bundle.ds4b", "mutate")
        self.assertTrue((self.root / "config.json").read_bytes().endswith(b"changed"))

    def test_discovery_verifies_shared_artifacts_once(self):
        spec = copy.deepcopy(self.spec)
        spec["components"][0]["selectors"][0]["prefix"] = "layer."
        other = copy.deepcopy(spec["components"][0])
        other["id"] = "projection"
        other["selectors"][0]["prefix"] = "norm."
        spec["components"].append(other)
        with patch("tools.suite_bundle.sha256", wraps=sha256) as hashed:
            result = discover(spec, self.root)
        self.assertTrue(all(c["artifacts_verified"] for c in result["components"]))
        self.assertEqual(hashed.call_count, len(spec["files"]))

    def test_invalid_group_and_oversized_directory_fail_before_publication(self):
        for expression in ("[", "layer", r"layer\.(\d+)\.(\d+)"):
            spec = copy.deepcopy(self.spec)
            spec["components"][0]["selectors"][0]["group_regex"] = expression
            with self.assertRaises(ValueError):
                compile_bundle(spec, self.root, self.root / "invalid")
        with (
            patch("tools.suite_bundle.MAX_INDEX_BYTES", 16),
            self.assertRaises(ValueError),
        ):
            compile_bundle(self.spec, self.root, self.root / "oversized")
        self.assertFalse((self.root / "oversized/bundle.ds4b").exists())

    def test_mixed_gguf_uses_each_actual_block_geometry(self):
        def gs(value):
            value = value.encode()
            return struct.pack("<Q", len(value)) + value

        header = b"GGUF" + struct.pack("<IQQ", 3, len(LAYOUT), 0)
        payload = bytearray()
        for kind, (block, width) in sorted(LAYOUT.items()):
            payload += bytes((-len(payload)) % 32)
            header += gs(f"type{kind}.weight") + struct.pack(
                "<IQQIQ", 2, block * 2, 3, kind, len(payload)
            )
            payload += bytes([kind]) * (width * 6)
        header += bytes((-len(header)) % 32)
        path = self.root / "mixed.gguf"
        path.write_bytes(header + payload)
        spec = copy.deepcopy(self.spec)
        spec["files"][0].update(
            path=path.name,
            format="gguf",
            sha256=sha256(path),
            bytes=path.stat().st_size,
        )
        spec["components"][0]["selectors"] = [{"file": path.name}]
        manifest = compile_bundle(spec, self.root, self.root / "mixed")
        self.assertEqual({t["type"] for t in manifest["tensors"]}, set(LAYOUT))
        self.native(self.root / "mixed/bundle.ds4b")

    def test_scalar_calibration_is_preserved(self):
        path = self.root / "scalar.safetensors"
        header = json.dumps(
            {"scale": {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x80\x3f")
        convert_safetensors(path, self.root / "scalar.gguf", "fixture", self.revision)
        original, converted = (
            inventory(path, "safetensors"),
            inventory(self.root / "scalar.gguf", "gguf"),
        )
        self.assertEqual(original[0]["shape"], converted[0]["shape"])
        self.assertEqual(converted[0]["bytes"], 2)

    def test_storage_geometry_matches_vendored_ggml(self):
        program = '#include <assert.h>\n#define GGML_COMMON_DECL_C\n#include "cuda/mmq/ggml-common.h"\n#include <stdio.h>\nint main(void) { printf("%zu %zu", sizeof(block_q8_1), sizeof(block_iq1_s)); }\n'
        binary = self.root / "layout"
        subprocess.run(
            ["cc", "-std=c11", "-I.", "-x", "c", "-o", str(binary), "-"],
            input=program,
            text=True,
            check=True,
        )
        actual = subprocess.check_output([str(binary)], text=True).split()
        self.assertEqual(list(map(int, actual)), [LAYOUT[9][1], LAYOUT[19][1]])

    def test_pinned_bundle_profiles_conform_to_schema(self):
        schema = json.loads(Path("schemas/ds4-suite-bundle-v1.schema.json").read_text())
        profiles = list(Path("config/suite-bundles").glob("*.json"))
        self.assertEqual(len(profiles), 3)
        for path in profiles:
            spec = json.loads(path.read_text())
            jsonschema.validate(spec, schema)
            self.assertTrue(
                all(
                    c["for_model_revision"] == spec["revision"]
                    for c in spec["components"]
                )
            )

    def test_reject_unknown_incompatible_and_corrupt(self):
        compile_bundle(self.spec, self.root, self.root / "native")
        path = self.root / "native/bundle.ds4b"
        data = bytearray(path.read_bytes())
        for position in (0, 8, 24, 56, 64, len(data) - 1):
            damaged = bytearray(data)
            damaged[position] ^= 127
            path.write_bytes(damaged)
            self.native(path, "reject")
        path.write_bytes(data)
        # Checksum-valid but structurally invalid payload must fail without leaks.
        damaged = bytearray(data)
        damaged[64:68] = struct.pack("<I", 0xFFFFFFFF)
        damaged[24:56] = hashlib.sha256(damaged[64:]).digest()
        path.write_bytes(damaged)
        self.native(path, "reject")
        path.write_bytes(data)
        # A valid checksum does not authorize mixing tensor/component revisions.
        damaged = bytearray(data)
        revision_at = damaged.index(b"encoder") + len(b"encoder") + 4
        damaged[revision_at : revision_at + 40] = b"b" * 40
        damaged[24:56] = hashlib.sha256(damaged[64:]).digest()
        path.write_bytes(damaged)
        self.native(path, "reject")
        path.write_bytes(data)
        source = self.root / "weights.safetensors"
        original = source.read_bytes()
        source.write_bytes(original[:-1] + b"X")
        self.native(path, "reject")
        source.write_bytes(original)
        for mutation in ("revision", "asset_revision", "owner", "dtype", "path"):
            invalid = copy.deepcopy(self.spec)
            if mutation == "revision":
                invalid["components"][0]["for_model_revision"] = "b" * 40
            elif mutation == "asset_revision":
                invalid["files"][1]["revision"] = "b" * 40
            elif mutation == "owner":
                invalid["components"][0]["selectors"][0]["prefix"] = "absent"
            elif mutation == "dtype":
                invalid["components"][0]["compute_dtype"] = "float16"
            else:
                invalid["files"][0]["path"] = "../outside"
            with self.assertRaises(ValueError):
                compile_bundle(invalid, self.root, self.root / "invalid")

    def test_explicit_download_checksum_and_resume_without_selection(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            offsets = []

            def do_GET(handler):
                file = self.root / handler.path.removeprefix("/")
                data = file.read_bytes()
                offset = int(
                    handler.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0]
                )
                Handler.offsets.append(offset)
                handler.send_response(206 if offset else 200)
                if offset:
                    handler.send_header(
                        "Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}"
                    )
                handler.end_headers()
                handler.wfile.write(data[offset:])

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        spec = copy.deepcopy(self.spec)
        for file in spec["files"]:
            file["url"] = f"http://127.0.0.1:{server.server_port}/{file['path']}"
        target = self.root / "download"
        target.mkdir()
        (target / "weights.safetensors.partial").write_bytes(
            (self.root / "weights.safetensors").read_bytes()[:13]
        )
        selected = self.root / "ds4flash.gguf"
        selected.write_bytes(b"selected conversational model")
        download_bundle(spec, target, ["encoder"])
        self.assertIn(13, Handler.offsets)
        self.assertEqual(selected.read_bytes(), b"selected conversational model")
        self.assertTrue(discover(spec, target)["components"][0]["artifacts_verified"])
        self.assertFalse(discover(spec, target)["ready"])
        download_bundle(spec, target, ["encoder"])
        with self.assertRaises(ValueError):
            download_bundle(spec, target, ["unknown"])


if __name__ == "__main__":
    unittest.main()
