# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model construction and packed-input dispatch."""

from __future__ import annotations

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
    peft_config: PeftConfig | None = None,
):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    return qwen3_5.build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
        peft_config=peft_config,
    )
