# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model construction and packed-input dispatch."""

from __future__ import annotations

import dataclasses

import torch
from nemo_automodel.components._peft.lora import PeftConfig
from transformers import AutoConfig

from . import qwen3_5
from .qwen3_5 import build_packed_position_ids, validate_config

__all__ = ["build_model", "build_packed_position_ids", "validate_config"]


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
    return qwen3_5.build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
        torch_dtype="auto" if peft_config is not None or trained_dtype is None else trained_dtype,
        peft_config=peft_config,
    )
