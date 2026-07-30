# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model construction and packed-input dispatch."""

from __future__ import annotations

from transformers import AutoConfig

from . import qwen3_5
from .qwen3_5 import build_packed_position_ids, validate_config


def build_model(
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    return qwen3_5.build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
    )
