"""Bounded weight transport for the independent PyTorch development oracle.

Keep the upstream model and arithmetic intact. Only tensor loading and accelerator
scheduling change: safetensors are read on demand and each weighted module drains
its commands before the next module can allocate weights.
"""

from collections.abc import Mapping

import torch
from accelerate import cpu_offload, init_empty_weights
from safetensors import safe_open


class SafetensorsWeights(Mapping):
    def __init__(self, directory, model):
        available = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as source:
                # safe_open is not iterable; keys() is its public metadata API.
                for name in source.keys():  # noqa: SIM118
                    if name in available:
                        raise ValueError(f"Duplicate oracle weight: {name}")
                    available[name] = (path, source.get_slice(name).get_shape())
        self._tensors = {}
        self._buffers = {}
        self.reads = 0
        self.bytes = 0
        parameters = dict(model.named_parameters())
        for name, tensor in model.state_dict().items():
            key = name if name in available else "model." + name
            if key not in available:
                if name not in parameters and tensor.device.type != "meta":
                    self._buffers[name] = tensor
                    continue
                raise ValueError(f"Missing oracle weight: {name}")
            path, shape = available[key]
            if list(tensor.shape) != shape:
                raise ValueError(f"Oracle shape mismatch: {name}: {shape}")
            self._tensors[name] = (path, key, tensor.dtype)

    def __getitem__(self, name):
        if name in self._buffers:
            return self._buffers[name]
        path, key, dtype = self._tensors[name]
        with safe_open(path, framework="pt", device="cpu") as source:
            tensor = source.get_tensor(key).to(dtype=dtype)
        self.reads += 1
        self.bytes += tensor.numel() * tensor.element_size()
        return tensor

    def __iter__(self):
        yield from self._tensors
        yield from self._buffers

    def __len__(self):
        return len(self._tensors) + len(self._buffers)


def load_offloaded_model(model_class, directory, device, *, transformers=False):
    """Construct parameters on meta, validate the checkpoint, attach lazy loading."""
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            if transformers:
                config = model_class.config_class.from_pretrained(
                    directory, local_files_only=True
                )
                model = model_class(config)
                model.set_attn_implementation("eager")
            else:
                model = model_class.from_config(model_class.load_config(directory))
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()
    weights = SafetensorsWeights(directory, model)
    cpu_offload(model, execution_device=torch.device(device), state_dict=weights)
    if device == "mps":

        def drain(module, arguments, output):
            torch.mps.synchronize()
            torch.mps.empty_cache()

        for module in model.modules():
            if any(True for _ in module.parameters(recurse=False)):
                module.register_forward_hook(drain)
    return model, weights
