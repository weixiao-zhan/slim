# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent packed context-parallel batch sharding."""

from __future__ import annotations

import contextlib
from functools import partial

import torch
from nemo_automodel.components.distributed.blockdiag_cp import (
    kernels,
    make_cp_blockdiag_batch_and_ctx,
)
from nemo_automodel.components.distributed.blockdiag_cp import state as state_module
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    contiguous_local_indices,
)
from torch.nn.attention import SDPBackend, sdpa_kernel


def _pad_sequence(tensor: torch.Tensor, *, seq_dim: int, pad_len: int, fill: float | int) -> torch.Tensor:
    if pad_len == 0:
        return tensor
    shape = list(tensor.shape)
    shape[seq_dim] = pad_len
    padding = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=seq_dim)


def make_packed_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    shard_primary: bool = False,
):
    """Apply the same padded block-diagonal state at every CP degree."""
    if cp_mesh.size() > 1:
        return make_cp_blockdiag_batch_and_ctx(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            shard_primary=shard_primary,
        )

    del tp_mesh, padding_token_id
    primary_key = "inputs_embeds" if "inputs_embeds" in batch else "input_ids"
    primary = batch[primary_key]
    batch_size, seq_len = primary.shape[:2]
    if batch_size != 1:
        raise ValueError(f"packed context parallelism requires local_batch_size=1, got {batch_size}")
    if shard_primary:
        raise ValueError("the model implementation must shard primary packed inputs after embedding")

    doc_ids = batch["_packed_seq_ids"].to(device=primary.device, dtype=torch.long)
    batch.pop("attention_mask", None)
    batch["padding_mask"] = doc_ids.eq(0)
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask

    pad_len = (-seq_len) % 2
    doc_ids = _pad_sequence(doc_ids, seq_dim=1, pad_len=pad_len, fill=0)
    fills = {
        "labels": -100,
        "_packed_seq_ids": 0,
        "loss_mask": 0,
        "padding_mask": True,
        "visual_pos_masks": False,
    }
    for key in (
        "labels",
        "position_ids",
        "_packed_seq_ids",
        "loss_mask",
        "padding_mask",
        "visual_pos_masks",
    ):
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        seq_dim = 2 if key == "position_ids" and tensor.dim() == 3 else 1
        batch[key] = _pad_sequence(tensor, seq_dim=seq_dim, pad_len=pad_len, fill=fills.get(key, 0))

    flat_doc_ids = doc_ids.reshape(-1)
    boundaries = torch.where(flat_doc_ids[1:] != flat_doc_ids[:-1])[0].to(torch.long) + 1
    packed_cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=primary.device),
            boundaries,
            torch.tensor([flat_doc_ids.numel()], dtype=torch.long, device=primary.device),
        )
    )
    group = cp_mesh.get_group()
    runtime_config = state_module.cp_varlen_runtime_config()
    step_state = {
        "group": group,
        "doc_ids": doc_ids,
        "packed_cu_seqlens": packed_cu_seqlens,
        "packed_cu_seqlens_cpu": packed_cu_seqlens.detach().cpu(),
        "row_offset": 0,
        "seq_dim": 2,
        "attn_backend": runtime_config["attn_backend"],
        "kv_exchange": runtime_config["kv_exchange"],
        "varlen_meta": kernels.precompute_blockdiag_varlen_meta(
            doc_ids,
            row_offset=0,
            local_len=doc_ids.shape[1],
            device=primary.device,
        ),
    }
    step_state["model_state"] = state_module.BlockdiagCpModelState(
        group=group,
        packed_cu_seqlens=packed_cu_seqlens,
        packed_cu_seqlens_cpu=step_state["packed_cu_seqlens_cpu"],
    )

    @contextlib.contextmanager
    def context():
        token = state_module._CP_BLOCKDIAG_STATE.set(step_state)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            try:
                yield
            finally:
                state_module._CP_BLOCKDIAG_STATE.reset(token)

    layout = ShardLayout(original_seq_len=seq_len, padded_seq_len=seq_len + pad_len)
    return context, batch, layout


def build_packed_cp_sharder(device_mesh, *, padding_token_id: int):
    """Build the common contiguous block-diagonal sharder."""
    return ContextParallelSharder(
        device_mesh=device_mesh,
        shard_batch=partial(make_packed_cp_batch_and_ctx, shard_primary=False),
        local_token_global_indices=contiguous_local_indices,
        padding_token_id=padding_token_id,
    )
