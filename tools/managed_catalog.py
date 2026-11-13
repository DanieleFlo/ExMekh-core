"""Pinned models, artifact selection and managed profile compatibility."""

import json

from managed_assets import ROOT, Artifact, Problem
from managed_hardware import cuda_device
from managed_memory import MODEL_MEMORY, max_context
from suite_bundle import validate_spec

HF = "https://huggingface.co"
QWEN_REV = "4ca720788d1e01f1bff70c033e0d0028fd02e502"


def qwen_artifact(path, remote, size, digest):
    return Artifact(
        path, f"{HF}/unsloth/Qwen3.8-27B-GGUF/resolve/{QWEN_REV}/{remote}", size, digest
    )


MODELS = {
    "qwen27": {
        "name": "Qwen3.8-27B UD-Q4_K_S",
        "artifact": qwen_artifact(
            "Qwen3.8-27B-UD-Q4_K_S.gguf",
            "Qwen3.8-27B-UD-Q4_K_S.gguf",
            15358213024,
            "75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017",
        ),
        "vision": qwen_artifact(
            "mmproj-Qwen3.8-27B-F16.gguf",
            "mmproj-F16.gguf",
            927607488,
            "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e",
        ),
    },
    "flash": {
        "name": "Qwen3.8 Flash Next Q2",
        "artifact": Artifact(
            "Qwen3.8-Flash-Next-Q2.gguf",
            f"{HF}/antirez/qwen3.8-flash-next-gguf/resolve/main/Qwen3.8-Flash-Next-Q2.gguf",
            147207127040,
            "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9",
        ),
        "vision": Artifact(
            "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
            f"{HF}/antirez/qwen3.8-flash-next-gguf/resolve/main/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
            616703104,
            "b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad",
        ),
    },
}
SUITE_IDS = {
    "images": "Qwen/Qwen-Image-2.1-Turbo",
    "embeddings": "google/embeddinggemma-2",
}
SUITE_NAMES = {
    "images": "Qwen-Image-2.1-Turbo",
    "embeddings": "EmbeddingGemma 2",
}
SUITE_MANIFESTS = {
    "images": "qwen-image-2.1-turbo-bf16.json",
    "embeddings": "embeddinggemma-2-bf16.json",
}


def modes(model, host):
    """Expose qualified envelopes. The worker performs native admission at load."""
    metal = host["backend"] == "metal" and host["system"] == "Darwin"
    weights = MODELS[model]["artifact"].size
    vision = MODELS[model].get("vision")
    labels = {
        "ssd": "SSD streaming",
        "tiered": "CUDA streaming (VRAM, RAM and SSD)",
        "ram": "Fully in RAM",
        "vram": "Fully in VRAM",
        "unified": "Resident in unified memory",
    }
    reasons = {
        "ssd": "SSD streaming requires compatible Metal unified memory and enough room for the model's streaming cache and context.",
        "tiered": "Flash CUDA streaming requires an SM 8.6 GPU and enough free VRAM and RAM.",
        "ram": "This model has no managed CPU-only profile.",
        "vram": "This profile requires a compatible CUDA model and enough free memory on the GPU selected by CUDA.",
        "unified": "This profile requires a compatible Metal model and enough unified memory for the weights and context.",
    }
    options = []
    for key, label in labels.items():
        if key == "unified" and not metal:
            continue
        if key == "tiered" and (model != "flash" or host["backend"] != "cuda"):
            continue
        limit = max_context(model, key, host, weights)
        vision_limit = (
            max_context(model, key, host, weights, vision.size)
            if vision and limit and (model != "flash" or key == "tiered")
            else 0
        )
        recommended = min(limit, 65536) if key == "tiered" else limit
        options.append(
            {
                "mode_id": key,
                "label": label,
                "available": bool(limit),
                "reason": None if limit else reasons[key],
                "estimated_max_context_tokens": limit or None,
                "recommended_context_tokens": recommended or None,
                "estimated_max_context_tokens_with_vision": vision_limit or None,
            }
        )
    return options


class ModelCatalog:
    def __init__(self, store, gemma_media, *, suite_store, native_stores):
        self._store = store
        self._gemma_media = gemma_media
        self._suite_store = suite_store
        self._native_stores = native_stores

    @staticmethod
    def has_model(model_id):
        return isinstance(model_id, str) and model_id in MODELS

    @staticmethod
    def has_suite_model(model_id):
        return isinstance(model_id, str) and model_id in SUITE_IDS

    @staticmethod
    def suite_model_name(model_id):
        return SUITE_NAMES[model_id]

    @staticmethod
    def model_file(model_id):
        return MODELS[model_id]["artifact"]

    @staticmethod
    def vision_file(model_id):
        return MODELS[model_id]["vision"]

    def suite_bundle(self, feature):
        try:
            spec = self._suite_manifest(feature)
            validate_spec(spec)
            artifacts = [self._suite_artifact(item) for item in spec["files"]]
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise Problem(
                "SUITE_MANIFEST_INVALID",
                "Core could not read the file manifest for this feature.",
            ) from exc
        native = self._native_stores.get(feature)
        if (
            native
            and not all(self._suite_store.installed(item) for item in artifacts)
            and all(native.has_complete_file(item) for item in artifacts)
        ):
            return native, artifacts
        return self._suite_store, artifacts

    def _describe_suite(self):
        inventory = {}
        for feature, name in SUITE_NAMES.items():
            entry = {"id": feature, "name": name}
            try:
                store, assets = self.suite_bundle(feature)
                states = [store.inspect(item) for item in assets]
                entry.update(
                    size_bytes=sum(item.size for item in assets),
                    missing_bytes=sum(state.missing_bytes for state in states),
                    installed=all(state.installed for state in states),
                    present=all(store.has_complete_file(item) for item in assets),
                    managed=store is self._suite_store,
                )
            except (OSError, Problem) as error:
                entry["error"] = str(error)
            inventory[feature] = entry
        return inventory

    def remove_suite_model(self, feature):
        store, assets = self.suite_bundle(feature)
        store.remove_all(assets)

    @staticmethod
    def parse_profile(payload):
        if not isinstance(payload, dict):
            raise Problem("PROFILE_INVALID", "Expected a profile object.")
        model, mode, context = (
            payload.get(key) for key in ("model_id", "mode_id", "context_tokens")
        )
        features = payload.get("features", [])
        if (
            not isinstance(model, str)
            or model not in MODELS
            or mode not in ("ssd", "ram", "vram", "unified", "tiered")
        ):
            raise Problem("PROFILE_INVALID", "Choose a known model and memory mode.")
        if type(context) is not int or context < 1:
            raise Problem("CONTEXT_INVALID", "Enter a positive integer context length.")
        if context > MODEL_MEMORY[model].max_tokens:
            raise Problem(
                "CONTEXT_LIMIT",
                f"This model supports at most {MODEL_MEMORY[model].max_tokens:,} tokens.",
            )
        if not isinstance(features, list) or any(
            feature not in ("vision", "images", "embeddings") for feature in features
        ):
            raise Problem("FEATURE_INVALID", "Choose supported features.")
        return {
            "model_id": model,
            "mode_id": mode,
            "context_tokens": context,
            "features": sorted(set(features)),
        }

    def _features(self, model_id, host, mode=None):
        device = cuda_device(host)
        metal = (
            host["backend"] == "metal"
            and host["system"] == "Darwin"
            and host.get("unified_memory")
        )
        cuda_suite_mode = "tiered" if model_id == "flash" else "vram"
        suite = bool(
            (device and device["compute"] == "8.6" and mode in (None, cuda_suite_mode))
            or (metal and model_id == "flash" and mode in (None, "ssd"))
        )
        suite_reason = (
            "Image generation and embeddings require a compatible CUDA SM 8.6 "
            "profile or Metal SSD streaming."
        )
        gemma_media = bool(self._gemma_media and self._gemma_media.is_file())
        vision = (
            bool(device and mode in (None, "tiered"))
            if model_id == "flash"
            else "vision" in MODELS[model_id] and bool(device or metal)
        )
        return {
            "vision": (
                vision,
                None
                if vision
                else "A qualified matching encoder is unavailable for this profile.",
            ),
            "images": (
                suite,
                None if suite else suite_reason,
            ),
            "embeddings": (
                suite and gemma_media,
                (
                    None
                    if gemma_media
                    else "Configure the prepared EmbeddingGemma 2 media asset in the native suite or set DS4_GEMMA_MEDIA_ASSET."
                )
                if suite
                else suite_reason,
            ),
        }

    def inventory(self, host):
        suite = self._describe_suite()
        data = []
        for key, model in MODELS.items():
            artifact = model["artifact"]
            state = self._store.inspect(artifact)
            options = modes(key, host)
            data.append(
                {
                    "id": key,
                    "name": model["name"],
                    "filename": artifact.path,
                    "size_bytes": artifact.size,
                    "max_context_tokens": MODEL_MEMORY[key].max_tokens,
                    "installed": state.installed,
                    "importable": not state.installed and state.present,
                    "modes": options,
                    "features": self._describe_features(key, host, options, suite),
                }
            )
        return data, list(suite.values())

    def _describe_features(self, model, host, options, suite):
        qualified = {
            option["mode_id"]: (option, self._features(model, host, option["mode_id"]))
            for option in options
            if option["available"]
        }
        features = []
        for name, (_, reason) in self._features(model, host).items():
            available_modes = [
                mode
                for mode, (option, features_for_mode) in qualified.items()
                if features_for_mode[name][0]
                and (
                    name != "vision"
                    or option["estimated_max_context_tokens_with_vision"]
                )
            ]
            features.append(
                self._feature_entry(model, name, available_modes, reason, suite)
            )
        return features

    def _feature_entry(self, model, name, available_modes, reason, suite):
        entry = {"id": name, "available_modes": available_modes}
        if not available_modes:
            return {
                **entry,
                "available": False,
                "reason": reason or "No available memory mode can use this feature.",
            }
        inventory = suite.get(name)
        if inventory is None:
            try:
                store, (asset,) = self.files_for(model, name)
                state = store.inspect(asset)
                inventory = {
                    "size_bytes": asset.size,
                    "missing_bytes": state.missing_bytes,
                    "installed": state.installed,
                }
            except (OSError, Problem) as error:
                inventory = {"error": str(error)}
        if error := inventory.get("error"):
            return {
                **entry,
                "available_modes": [],
                "available": False,
                "reason": error,
            }
        return {
            **entry,
            "available": True,
            "reason": None,
            "size_bytes": inventory["size_bytes"],
            "missing_bytes": inventory["missing_bytes"],
            "installed": inventory["installed"],
        }

    def inspect_profile_files(self, profile):
        files = []
        for feature in (None, *profile["features"]):
            store, artifacts = self.files_for(profile["model_id"], feature)
            for artifact in artifacts:
                state = store.inspect(artifact)
                files.append(
                    {
                        "name": artifact.path,
                        "feature": feature,
                        "size_bytes": artifact.size,
                        "missing_bytes": state.missing_bytes,
                        "installed": state.installed,
                    }
                )
        return files

    def files_for(self, model_id, feature):
        if feature is None:
            return self._store, [self.model_file(model_id)]
        if feature == "vision":
            return self._store, [self.vision_file(model_id)]
        return self.suite_bundle(feature)

    @staticmethod
    def model_name(model_id):
        return MODELS[model_id]["name"]

    @staticmethod
    def _suite_manifest(feature):
        return json.loads(
            (ROOT / "config" / "suite-bundles" / SUITE_MANIFESTS[feature]).read_text()
        )

    @staticmethod
    def _suite_artifact(item):
        return Artifact(item["path"], item["url"], item["bytes"], item["sha256"])

    def validate(self, profile, host):
        model = profile["model_id"]
        options = modes(model, host)
        option = next(
            (item for item in options if item["mode_id"] == profile["mode_id"]), None
        )
        if option is None or not option["available"]:
            raise Problem(
                "MODE_UNAVAILABLE",
                option["reason"] if option else "This memory mode is unavailable.",
            )
        features = self._features(model, host, profile["mode_id"])
        for name in profile["features"]:
            if not features[name][0]:
                raise Problem("FEATURE_UNAVAILABLE", features[name][1])
        key = (
            "estimated_max_context_tokens_with_vision"
            if "vision" in profile["features"]
            else "estimated_max_context_tokens"
        )
        limit = option[key] or 0
        if profile["context_tokens"] > limit:
            raise Problem(
                "CONTEXT_LIMIT",
                f"The current memory budget fits at most {limit:,} tokens with these settings.",
            )
