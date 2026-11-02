# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next construction and model-owned packed CP inputs.

AutoModel trains the language model only: the policy has no vision tower and no MTP layer.
The QSA indexer is frozen in AutoModel and recomputes its top-k on every forward.
The n-gram PLE table is frozen here, so the optimizer and the weight sync skip its 51.2B parameters.
"""

from __future__ import annotations

from functools import partial

import torch
from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.context_parallel.sharder import (
    contiguous_local_indices,
    identity_local_indices,
    shard_batch_identity,
)
from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.cp import shard_batch_for_qwen3_8_flash_next_cp
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextOwnerShardedEmbedding

from ..data_packing import build_source_labels

MODEL_TYPES = ("qwen4_exp",)
# The engines load the frozen n-gram table from the rollout checkpoint and never receive it.
ROLLOUT_STATIC_MODULES: tuple[type, ...] = (Qwen3_8_FlashNextOwnerShardedEmbedding,)


def validate_config(config, topology) -> None:
    # AutoModel converts the owner-sharded n-gram table to a DTensor only in its MoE parallelizer,
    # which runs when EP is greater than one.
    if topology.expert_model_parallel_size == 1:
        raise ValueError("Qwen3.8-Flash-Next requires expert_model_parallel_size > 1")
    num_experts = int(config.get_text_config().num_experts)
    if num_experts % topology.expert_model_parallel_size:
        raise ValueError(
            f"num_experts {num_experts} must be divisible by expert_model_parallel_size "
            f"{topology.expert_model_parallel_size}"
        )


def build_backend_config(args) -> Qwen3_8_FlashNextBackendConfig:
    # QSA runs its sparse GQA through FlexAttention; the model owns its dispatch.
    return Qwen3_8_FlashNextBackendConfig(
        attn="flex",
        linear=args.nemo_linear_backend,
        rms_norm=args.nemo_rms_norm_backend,
        rope_fusion=False,
        experts=args.nemo_experts_backend,
        dispatcher=args.nemo_dispatcher,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )


def build_model(
    config,
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
):
    kwargs = {
        # AdamW keeps no master copy, so the stored weights are the master weights.
        "torch_dtype": torch.float32 if args.master_weight_dtype == "fp32" else "auto",
        "distributed_setup": distributed_setup,
        "backend": build_backend_config(args),
        "attn_implementation": "sdpa",
        "has_packed_sequence": True,
        "trust_remote_code": True,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "language_model_only": True,
        "text_config": {"router_aux_loss_coef": 0.0},
    }
    if routing_replay:
        kwargs["moe_overrides"] = {"enable_routing_replay": True}
    model = NeMoAutoModelForImageTextToText.from_pretrained(checkpoint, **kwargs)
    for module in model.modules():
        if isinstance(module, ROLLOUT_STATIC_MODULES):
            module.requires_grad_(False)
    return model


def build_model_batch(model: torch.nn.Module, pack: dict) -> dict:
    """Build the unsharded CPU model inputs of one text pack.

    The model takes document boundaries as `cu_seqlens`: its CP sharder moves them into the CP
    context, and without CP its QSA, Gated DeltaNet, and PLE layers read them directly.
    """
    tokens = pack["tokens"]
    return {
        "input_ids": tokens.unsqueeze(0),
        "labels": build_source_labels(tokens, pack["cu_seqlens"]).unsqueeze(0),
        "position_ids": pack["position_ids"].unsqueeze(0),
        "cu_seqlens": pack["cu_seqlens"],
    }


def build_packed_cp_sharder(model: torch.nn.Module, device_mesh, *, padding_token_id: int) -> ContextParallelSharder:
    """Shard packs contiguously with the model's CP sharder, or keep them whole at CP1.

    The model's CP sharder pads every pack to a multiple of the QSA compression ratio. Outside CP,
    the packed Gated DeltaNet path requires boundaries that end at the row length, so CP1 keeps the
    pack unpadded.
    """
    if device_mesh["cp"].size() == 1:
        return ContextParallelSharder(
            device_mesh=device_mesh,
            shard_batch=shard_batch_identity,
            local_token_global_indices=identity_local_indices,
            padding_token_id=padding_token_id,
        )
    return ContextParallelSharder(
        device_mesh=device_mesh,
        shard_batch=partial(
            shard_batch_for_qwen3_8_flash_next_cp,
            pad_multiple=model.config.text_config.indexer_compress_ratio,
        ),
        local_token_global_indices=contiguous_local_indices,
        padding_token_id=padding_token_id,
    )
