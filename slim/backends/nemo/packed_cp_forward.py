# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified packed context-parallel forward preparation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from .data_packing import build_training_fields
from .models import model_family


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

    @cached_property
    def loss_positions(self) -> torch.Tensor:
        """Local positions with a nonzero loss mask, the only predictions a policy loss reads."""
        return self.fields["loss_masks"][0].nonzero().squeeze(-1)

    def scatter_loss_values(self, values: torch.Tensor) -> torch.Tensor:
        """Scatter one value per `loss_positions` entry into the local sequence, zero elsewhere."""
        return values.new_zeros(self.fields["loss_masks"].shape).index_copy(1, self.loss_positions, values)

    def gather(self, tensor: torch.Tensor, *, fill=0) -> torch.Tensor:
        return self.sharder.gather_token_tensor(tensor, seq_dim=1, trim=True, fill=fill)


def prepare_forward(
    model,
    device_mesh,
    pack: dict,
    *,
    padding_token_id: int,
) -> PreparedForward:
    """Shard packed model inputs and RL fields through the model family's sharder."""
    device = torch.device("cuda", torch.cuda.current_device())
    family = model_family(model.config)
    model_batch = move_to_device(family.build_model_batch(model, pack), device)
    full_fields = move_to_device(build_training_fields(pack), device)
    sharder = family.build_packed_cp_sharder(model, device_mesh, model_batch, padding_token_id=padding_token_id)
    context_factory, model_batch = sharder.shard(model_batch)
    labels = model_batch.pop("labels")
    fields = {}
    for name, value in full_fields.items():
        fields[name] = sharder.shard_token_tensor(value, seq_dim=1, fill=0)
    fields["labels"] = labels
    return PreparedForward(context_factory, model_batch, fields, sharder)
