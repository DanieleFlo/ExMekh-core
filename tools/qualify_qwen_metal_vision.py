#!/usr/bin/env python3
"""Bounded Qwen vision encoder parity, without loading an LLM or downloading HF weights.

Runs Metal and an independent Transformers CPU vision tower serially with the
same GGUF weights. Reports implementation parity only, not quantization quality
or the multimodal LLM gate. Requires the local metal-venv dependencies.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from gguf import GGUFReader
from tools.qualify_metal_foundations import write_json
from tools.qwen_vision_reference import (
    compare,
    ds4_embeddings,
    gguf_state,
    read_embedding_dump,
    sha256_file,
)


class VisionReference:
    """Own GGUF family selection and the independent CPU encoder's lifecycle."""

    def __init__(self, mmproj: Path, expected_projection: int):
        reader = GGUFReader(str(mmproj))
        if (
            reader.fields["general.architecture"].contents() != "clip"
            or reader.fields["clip.projector_type"].contents() != "qwen3vl_merger"
        ):
            raise ValueError("unsupported Qwen vision GGUF architecture/projector")

        def value(name: str) -> int:
            return int(reader.fields[f"clip.vision.{name}"].contents())

        projection = value("projection_dim")
        families = {
            2560: (Qwen4ExpVisionConfig, Qwen4ExpVisionModel),
            5120: (Qwen3_5VisionConfig, Qwen3_5VisionModel),
        }
        if projection not in families or projection != expected_projection:
            raise ValueError(
                f"projector width {projection} does not match target {expected_projection}"
            )
        configuration, self._model_type = families[projection]
        pos = next(t for t in reader.tensors if t.name == "v.position_embd.weight")
        self._configuration = configuration(
            depth=value("block_count"),
            hidden_size=value("embedding_length"),
            intermediate_size=value("feed_forward_length"),
            num_heads=value("attention.head_count"),
            out_hidden_size=projection,
            patch_size=value("patch_size"),
            spatial_merge_size=value("spatial_merge_size"),
            num_position_embeddings=int(pos.shape[1]),
        )
        self._path = mmproj
        self._model = None

    def _load(self):
        if self._model is None:
            model = self._model_type(self._configuration).float().eval()
            model.set_attn_implementation("eager")
            missing, unexpected = model.load_state_dict(
                gguf_state(model, str(self._path)), strict=False
            )
            if missing or unexpected:
                raise ValueError(f"vision state mismatch: {missing}, {unexpected}")
            self._model = model
        return self._model

    @torch.no_grad()
    def encode(self, inputs):
        pixels, grid = inputs
        result = self._load()(pixels, grid_thw=grid)
        embedding = result.pooler_output
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        return embedding.numpy(), (int(grid[0, 1]) // 2, int(grid[0, 2]) // 2)


def native_inputs(binary: str, image: Path, directory: Path, tokens: int):
    path = directory / "patches.bin"
    subprocess.run(
        [str(binary), "--patches", str(image), str(path), str(tokens), str(tokens)],
        check=True,
        timeout=120,
    )
    patches, (height, cols) = read_embedding_dump(path)
    if patches.shape[1] != 768 or height % 2 or cols % 2:
        raise ValueError("invalid native patch layout")
    count = patches.shape[0]
    # The GGUF splits Conv3D's two identical temporal image frames into w0/w1.
    pixels = (
        torch.from_numpy(patches.copy())
        .reshape(count, 3, 1, 16, 16)
        .expand(-1, -1, 2, -1, -1)
    )
    return pixels.reshape(count, 1536), torch.tensor([[1, height, cols]])


def hf_inputs(image: Path, tokens: int):
    processor = Qwen2VLImageProcessorPil(
        min_pixels=tokens * 32 * 32,
        max_pixels=tokens * 32 * 32,
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    values = processor(images=Image.open(image).convert("RGB"), return_tensors="pt")
    return values["pixel_values"].float(), values["image_grid_thw"]


def qualify(args, report: dict) -> None:
    reference_model = VisionReference(args.mmproj, args.projection)
    torch.set_num_threads(4)
    report.update(
        {
            "kind": "encoder-implementation-parity",
            "backend": "metal",
            "tokens": args.tokens,
            "projection": args.projection,
            "min_cos": 0.99,
            "mmproj": str(args.mmproj),
            "mmproj_sha256": sha256_file(args.mmproj),
            "image": str(args.image),
            "image_sha256": sha256_file(args.image),
            "source_sha256": {
                name: sha256_file(name)
                for name in (
                    "ds4.c",
                    "ds4_image.c",
                    "ds4_image_resize.inc",
                    "ds4_metal.m",
                    "ds4_qwen4_vision.h",
                    "metal/qwen4_vision.metal",
                    "tests/qwen4_vision_ref.py",
                    "tests/test_qwen4_vision.c",
                    "tools/qualify_qwen_metal_vision.py",
                    "tools/qwen_vision_reference.py",
                )
            },
            "binary_sha256": sha256_file(args.binary),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy", "gguf", "pillow")
            },
            "llm_multimodal_gate": "NOT_VERIFIED",
        }
    )
    with tempfile.TemporaryDirectory(prefix="qwen-metal-vision-") as directory:
        actual, grid = ds4_embeddings(
            args.binary,
            str(args.mmproj),
            str(args.image),
            str(Path(directory) / "metal.bin"),
            args.tokens,
            args.tokens,
        )
        if not 64 <= actual.shape[0] <= 128:
            raise ValueError(
                f"image preprocessing produced {actual.shape[0]} tokens; require 64–128"
            )
        native = native_inputs(args.binary, args.image, Path(directory), args.tokens)
        pil = hf_inputs(args.image, args.tokens)
        reference, reference_grid = reference_model.encode(native)
        pipeline, pipeline_grid = reference_model.encode(pil)
        if native[0].shape != pil[0].shape or not torch.equal(native[1], pil[1]):
            raise ValueError("native and Transformers image grids differ")
        difference = (native[0] - pil[0]).abs()
        report.update(
            shape=list(actual.shape),
            grid=grid,
            implementation=compare(
                actual, reference, grid, reference_grid, report["min_cos"]
            ),
            image_pipeline=compare(
                actual, pipeline, grid, pipeline_grid, report["min_cos"]
            ),
            preprocessing={
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--binary", default="tests/test_qwen4_vision")
    parser.add_argument("--tokens", type=int, choices=(64, 128), default=128)
    parser.add_argument("--projection", type=int, choices=(2560, 5120), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "kind": "encoder-implementation-parity",
        "backend": "metal",
        "status": "IN_PROGRESS",
        "llm_multimodal_gate": "NOT_VERIFIED",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    try:
        qualify(args, report)
        passed = report["implementation"]["pass"] and report["image_pipeline"]["pass"]
        report["status"] = "PASS" if passed else "FAIL"
    except KeyboardInterrupt:
        report["status"] = "NOT_VERIFIED"
        report["error"] = "interrupted"
        return 130
    except subprocess.TimeoutExpired as error:
        report["status"] = "NOT_VERIFIED"
        report["error"] = str(error)
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
    finally:
        write_json(args.report, report)
    print(
        json.dumps(
            {
                name: report[name]
                for name in (
                    "status",
                    "implementation",
                    "image_pipeline",
                    "preprocessing",
                    "error",
                )
                if name in report
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
