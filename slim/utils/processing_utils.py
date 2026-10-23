# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pybase64
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def load_routing_replay_shape(name_or_path: str) -> tuple[int, int]:
    """Per-token shape `(num_layers, top_k)` of the routed experts recorded for routing replay."""
    cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    return text_cfg.num_hidden_layers, text_cfg.num_experts_per_tok


def encode_tensor_to_b64_envelope(value) -> dict:
    """Encode a CPU tensor as a JSON-safe base64 envelope."""
    tensor = value.detach().cpu().contiguous()
    # An empty tensor has stride 0, which `view` rejects.
    raw = tensor.view(torch.uint8).numpy() if tensor.numel() else b""
    return {
        "__tensor__": True,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": pybase64.b64encode(raw).decode("ascii"),
    }


def encode_array_to_b64_envelope(values, dtype: type[np.number], columns: int | None = None) -> dict:
    """Encode a list, or a list of equal-length rows, of numbers as a tensor envelope.

    ``columns`` fixes the row width, so that no rows still encode as ``[0, columns]``.
    """
    array = np.asarray(values, dtype=dtype)
    if columns is not None:
        array = array.reshape(-1, columns)
    return encode_tensor_to_b64_envelope(torch.from_numpy(array))


def decode_tensor_from_b64_envelope(value: dict) -> torch.Tensor:
    """Decode a tensor produced by ``encode_tensor_to_b64_envelope``."""
    raw = pybase64.b64decode(value["data"], validate=True)
    dtype = getattr(torch, value["dtype"])
    if not raw:
        # `torch.frombuffer` rejects an empty buffer.
        return torch.empty(value["shape"], dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(value["shape"])


def decode_tensor_envelopes(value):
    """Decode every tensor envelope nested in a JSON-decoded structure."""
    if isinstance(value, dict):
        if value.get("__tensor__"):
            return decode_tensor_from_b64_envelope(value)
        return {key: decode_tensor_envelopes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_tensor_envelopes(item) for item in value]
    return value


# SGLang names a modality's primary processor tensor after the modality itself.
_MODALITY_FEATURE_NAMES = {
    "IMAGE": "pixel_values",
    "VIDEO": "pixel_values_videos",
    "AUDIO": "input_features",
}


def _merge_processor_field(name: str, values: list) -> torch.Tensor:
    """Join one processor field across the multimodal items of a single prompt."""
    tensors = []
    for value in values:
        if isinstance(value, (list, tuple)):
            tensors.extend(torch.as_tensor(item) for item in value)
        else:
            tensors.append(torch.as_tensor(value))
    if name.endswith("_grid_thw"):
        tensors = [tensor.unsqueeze(0) if tensor.ndim == 1 else tensor for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    if all(tensor.ndim == 0 for tensor in tensors):
        return torch.stack(tensors)
    return torch.cat(tensors, dim=0)


def encode_processor_outputs(mm_inputs) -> dict[str, dict] | None:
    """Encode an SGLang ``MultimodalInputs``' processor tensors for JSON transport.

    Returns the same field names an ``AutoProcessor`` call produces, so a caller can
    train on an SGLang-tokenized prompt exactly as it trains on its own.
    """
    if mm_inputs is None:
        return None

    fields: dict[str, list] = {}
    for item in mm_inputs.mm_items:
        feature_name = _MODALITY_FEATURE_NAMES.get(item.modality.name)
        if feature_name is not None and item.feature is not None:
            fields.setdefault(feature_name, []).append(item.feature)
        for name, value in item.model_specific_data.items():
            fields.setdefault(name, []).append(value)

    return {
        name: encode_tensor_to_b64_envelope(_merge_processor_field(name, values))
        for name, values in fields.items()
    } or None
