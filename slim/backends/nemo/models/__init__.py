# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model families and their construction.

A model family module defines:

- `MODEL_TYPES`: the Hugging Face `model_type` values it trains.
- `ROLLOUT_STATIC_MODULES`: module types whose parameters the rollout engines keep from their own checkpoint, in memory that a colocated weights release does not free.
- `validate_config(config, args, topology)`: rejects unsupported arguments and topologies.
- `build_model(config, args, checkpoint, distributed_setup, *, routing_replay, torch_dtype, peft_config)`: builds the parallelized model.
- `build_model_batch(model, pack)`: the unsharded CPU model inputs of one physical pack.
- `build_packed_cp_sharder(model, device_mesh, model_batch, *, padding_token_id)`: the sharder of the model inputs and RL token fields.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType

import torch
from nemo_automodel.components._peft.lora import PeftConfig
from transformers import AutoConfig

from . import qwen3_5, qwen4

_FAMILIES = {model_type: family for family in (qwen3_5, qwen4) for model_type in family.MODEL_TYPES}

__all__ = ["build_model", "model_family"]


def model_family(config) -> ModuleType:
    family = _FAMILIES.get(config.model_type)
    if family is None:
        raise ValueError(
            f"first-party NeMo training does not support model_type={config.model_type!r}; "
            f"supported: {', '.join(_FAMILIES)}"
        )
    return family


def build_model(
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
    peft_config: PeftConfig | None,
    frozen: bool = False,
):
    """Build the model of `checkpoint`, with the LoRA adapters of `peft_config` on a frozen base if given."""
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    # AdamW keeps no master copy, so the trained weights are stored in --master-weight-dtype.
    # Frozen weights, the base under PEFT included, keep the checkpoint's dtype.
    trained_dtype = torch.float32 if args.master_weight_dtype == "fp32" and not frozen else None
    if peft_config is not None:
        # None stores the adapters in the dtype of the weight they adapt.
        peft_config = dataclasses.replace(peft_config, lora_dtype=trained_dtype)
    return model_family(config).build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
        torch_dtype="auto" if peft_config is not None or trained_dtype is None else trained_dtype,
        peft_config=peft_config,
    )
