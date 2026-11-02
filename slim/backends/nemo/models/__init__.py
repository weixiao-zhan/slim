# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo model families and their construction.

A model family module defines:

- `MODEL_TYPES`: the Hugging Face `model_type` values it trains.
- `ROLLOUT_STATIC_MODULES`: module types whose parameters the rollout engines keep from their own checkpoint.
- `validate_config(config, topology)`: rejects unsupported topologies.
- `build_model(config, args, checkpoint, distributed_setup, *, routing_replay)`: builds the parallelized model.
- `build_model_batch(model, pack)`: the unsharded CPU model inputs of one physical pack.
- `build_packed_cp_sharder(model, device_mesh, *, padding_token_id)`: the sharder of the model inputs and RL token fields.
"""

from __future__ import annotations

from types import ModuleType

from transformers import AutoConfig

from . import qwen3_5, qwen3_8_flash_next

_FAMILIES = {model_type: family for family in (qwen3_5, qwen3_8_flash_next) for model_type in family.MODEL_TYPES}

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
):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    return model_family(config).build_model(
        config,
        args,
        checkpoint,
        distributed_setup,
        routing_replay=routing_replay,
    )
