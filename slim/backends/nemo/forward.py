# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified packed context-parallel forward preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data_packing import build_model_batch, build_token_slot_fields
from .packed_cp import build_packed_cp_sharder, build_packed_position_ids


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {name: move_to_device(item, device) for name, item in value.items()}
    return value


@dataclass
class PreparedForward:
    context_factory: object
    model_batch: dict
    fields: dict[str, torch.Tensor]
    sharder: object

    def gather(self, tensor: torch.Tensor, *, fill=0) -> torch.Tensor:
        return self.sharder.gather_token_tensor(tensor, seq_dim=1, trim=True, fill=fill)


def prepare_forward(
    model,
    device_mesh,
    pack: dict,
    *,
    padding_token_id: int,
) -> PreparedForward:
    """Shard packed model inputs and RL fields through one block-diagonal sharder."""
    device = torch.device("cuda", torch.cuda.current_device())
    model_batch = move_to_device(build_model_batch(pack), device)
    model_batch["position_ids"] = build_packed_position_ids(model, pack, model_batch)
    full_fields = move_to_device(build_token_slot_fields(pack), device)
    sharder = build_packed_cp_sharder(device_mesh, padding_token_id=padding_token_id)
    context_factory, model_batch = sharder.shard(model_batch)
    labels = model_batch.pop("labels")
    fields = {}
    for name, value in full_fields.items():
        fill = 0
        if name == "mismatch_weights":
            fill = 1
        fields[name] = sharder.shard_token_tensor(value, seq_dim=1, fill=fill)
    fields["labels"] = labels
    return PreparedForward(context_factory, model_batch, fields, sharder)


def model_forward(model, model_batch: dict):
    from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

    return model(**filter_forward_kwargs(model, model_batch))
