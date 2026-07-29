# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model construction and packed-input dispatch."""

from __future__ import annotations

from transformers import AutoConfig

from . import qwen3_5


def _model_module(config):
    model_type = getattr(config, "model_type", "")
    if model_type in qwen3_5.MODEL_TYPES:
        return qwen3_5
    supported = ", ".join(qwen3_5.MODEL_TYPES)
    raise ValueError(
        f"first-party NeMo training does not support model_type={model_type!r}; supported: {supported}"
    )


def validate_config(config, topology) -> None:
    _model_module(config).validate_config(config, topology)


def build_model(
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    return _model_module(config).build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
    )


def build_packed_position_ids(model, pack: dict, model_batch: dict):
    return _model_module(model.config).build_packed_position_ids(model, pack, model_batch)
