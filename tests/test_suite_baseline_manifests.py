"""Protect frozen identities and corpus membership; never regenerate goldens."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/performance/native-suite-t00-20261020"
CORPUS = ROOT / "tests/fixtures/native-suite-v1/corpus.json"


class BaselineManifests(unittest.TestCase):
    def test_pinned_sources_and_local_weight_hashes(self):
        data = json.loads((BASE / "sources.json").read_text())
        self.assertEqual(len(data["sources"]), 8)
        for source in data["sources"]:
            self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
            for file in source["files"]:
                self.assertIn(source["revision"], file["url"])
                self.assertRegex(file["sha256"], r"^[0-9a-f]{64}$")
        for weight in data["local_weights"]:
            self.assertEqual(weight["hash_status"], "COMPUTED")
            self.assertRegex(weight["sha256"], r"^[0-9a-f]{64}$")

    def test_published_and_local_official_weights_match(self):
        data = json.loads((BASE / "models.json").read_text())
        published = {m["repository"]: m for m in data["published"]}
        for model in data["locally_verified"]:
            expected = {
                f["path"]: f for f in published[model["repository"]]["artifacts"]
            }
            self.assertEqual(
                model["revision"], published[model["repository"]]["revision"]
            )
            for file in model["files"]:
                if file["path"].endswith(".safetensors"):
                    self.assertEqual(
                        file["sha256"], expected[file["path"]]["published_sha256"]
                    )
        qwen = next(
            m for m in data["locally_verified"] if m["repository"].startswith("Qwen/")
        )
        self.assertTrue(any(f["path"].endswith(".jinja") for f in qwen["files"]))

    def test_oracles_and_legacy_gates(self):
        for name in ("gemma-oracle.json", "qwen-oracle.json"):
            data = json.loads((BASE / name).read_text())
            self.assertEqual(data["warmup"], 1)
            self.assertEqual(data["repeats"], 5)
            for case in data["cases"]:
                self.assertEqual(case["max_repeat_abs"], 0.0)
        candidate = json.loads((BASE / "qwen27-candidate.json").read_text())
        self.assertEqual(len(candidate), 4)
        checked = 0
        for case in candidate:
            self.assertEqual(case["exit_code"], 0)
            for file in case["files"]:
                if file["path"] in ("logits.f32", "session.payload"):
                    self.assertTrue(file["bit_exact"])
                    checked += 1
                if file["path"] == "logits.f32":
                    self.assertTrue(file["finite"])
        self.assertEqual(checked, 8)
        self.assertTrue(json.loads((BASE / "flash.json").read_text())["bit_exact"])

    def test_corpus_hashes_counts_and_split_ownership(self):
        data = json.loads(CORPUS.read_text())
        by_id = {i["id"]: i for i in data["items"]}
        self.assertEqual(len(by_id), 104)
        digests = {"dev": set(), "heldout": set()}
        for item in data["items"]:
            artifact = item.get("artifact")
            if artifact:
                digest = hashlib.sha256(
                    (CORPUS.parent / artifact["path"]).read_bytes()
                ).hexdigest()
                self.assertEqual(digest, artifact["sha256"])
                digests[item["split"]].add(digest)
            for part in item["content"]:
                if "path" in part:
                    owner = next(
                        i
                        for i in data["items"]
                        if i.get("artifact", {}).get("path") == part["path"]
                    )
                    self.assertEqual(owner["split"], item["split"])
        self.assertFalse(digests["dev"] & digests["heldout"])
        self.assertEqual(len(data["qwen_recipes"]), 24)
        for recipe in data["qwen_recipes"]:
            self.assertEqual(len(recipe["seeds"]), 2)
            for identity in recipe["reference_ids"]:
                self.assertEqual(by_id[identity]["split"], recipe["split"])
        retrieval = json.loads((BASE / "retrieval.json").read_text())
        self.assertEqual(
            retrieval["corpus_sha256"], hashlib.sha256(CORPUS.read_bytes()).hexdigest()
        )
        self.assertEqual(retrieval["items"], len(by_id))


if __name__ == "__main__":
    unittest.main()
