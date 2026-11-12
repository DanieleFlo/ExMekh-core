"""Prepare verified native suite bundles and their runtime configuration."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from managed_assets import ROOT, PinnedStore
from managed_catalog import SUITE_IDS, SUITE_MANIFESTS
from managed_worker import run_command


@dataclass(frozen=True)
class NativeSuiteSource:
    root: Path | None
    text_asset: Path | None
    media_asset: Path | None


def _suite_path(value):
    return (ROOT / value).resolve() if isinstance(value, str) and value else None


def native_suite_sources(config_path):
    """Read local source paths from the same configuration as the native launcher."""
    config_path = ROOT / config_path
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(config, dict) or config.get("version") != 1:
        return {}
    entries = config.get("models")
    if not isinstance(entries, list):
        return {}
    models = {
        entry.get("model"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }
    return {
        feature: NativeSuiteSource(
            root=_suite_path(models[model_id].get("root")),
            text_asset=_suite_path(models[model_id].get("text_asset")),
            media_asset=_suite_path(models[model_id].get("media_asset")),
        )
        for feature, model_id in SUITE_IDS.items()
        if model_id in models
    }


class SuiteAssets:
    def __init__(self, directory, model_root, token, config_path, gemma_media, closing):
        self._directory = directory
        self._closing = closing
        self._sources = native_suite_sources(config_path) if config_path else {}
        embeddings = self._sources.get("embeddings")
        configured_media = embeddings.media_asset if embeddings else None
        self.gemma_media = gemma_media or configured_media
        self.store = PinnedStore(model_root / "suite", token)
        self.native_stores = {
            feature: PinnedStore(
                source.root, local_receipts=directory / "native-suite" / feature
            )
            for feature, source in self._sources.items()
            if source.root and source.root != self.store.root and source.root.is_dir()
        }

    def prepare(self, feature, root):
        directory = self._directory / feature
        directory.mkdir(mode=0o700, exist_ok=True)
        index = directory / "bundle.ds4b"
        if not index.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "tools" / "suite_bundle.py"),
                    "build",
                    "--manifest",
                    str(ROOT / "config" / "suite-bundles" / SUITE_MANIFESTS[feature]),
                    "--root",
                    str(root),
                    "--output",
                    str(directory),
                ],
                self._closing,
            )
        asset = self._suite_text_asset(feature, root, directory)
        result = {
            "model": SUITE_IDS[feature],
            "index": str(index),
            "root": str(root),
            "text_asset": str(asset),
        }
        if feature == "embeddings":
            result["media_asset"] = str(self.gemma_media)
        return result

    def _suite_text_asset(self, feature, root, directory):
        prepared = self._sources.get(feature)
        if prepared and prepared.text_asset and prepared.text_asset.is_file():
            return prepared.text_asset
        if feature == "images":
            asset = directory / "tokenizer.ds4q"
            source = (
                root / "Qwen" / "Qwen-Image-2.1-Turbo" / "processor" / "tokenizer.json"
            )
            compiler = "compile_qwen_image_tokenizer.py"
        else:
            asset = directory / "gemma.ds4g"
            source = root / "google" / "embeddinggemma-2"
            compiler = "compile_embedding_gemma2.py"
        if not asset.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "tools" / compiler),
                    str(source),
                    str(asset),
                ],
                self._closing,
            )
        return asset

    def config(self, models):
        media = self._directory / "media"
        media.mkdir(mode=0o700, exist_ok=True)
        return {
            "version": 1,
            "media_directory": str(media),
            "storage_bytes": 4 * 2**30,
            "timeout_seconds": 600,
            "image_timeout_seconds": 1200,
            "models": models,
        }
