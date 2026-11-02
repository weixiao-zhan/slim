# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen4 construction and model-owned packed CP inputs.

AutoModel trains the language model only: the policy has no vision tower and no MTP layer.
The QSA indexer is frozen in AutoModel and recomputes its top-k on every forward.
The n-gram PLE table is frozen here and stored in BF16 at every master dtype, so the optimizer skips its 51.2B
parameters, and the weight sync skips them as rollout-static parameters.
The family therefore requires `--freeze-vision-tower` and `--freeze-ngram`.
"""

from __future__ import annotations

import torch
from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextOwnerShardedEmbedding

from ..data_packing import build_source_labels

MODEL_TYPES = ("qwen4_exp",)
# The engines load the frozen n-gram table from the rollout checkpoint into pinned host memory
# (`--sglang-ple-offload-embedding`) and never receive it.
ROLLOUT_STATIC_MODULES: tuple[type, ...] = (Qwen3_8_FlashNextOwnerShardedEmbedding,)


def validate_config(config, args, topology) -> None:
    if not (args.freeze_vision_tower and args.freeze_ngram):
        raise ValueError("Qwen4 requires --freeze-vision-tower and --freeze-ngram")
    if args.freeze_language_model:
        raise ValueError("Qwen4 trains its language model, so it does not support --freeze-language-model")
    # AutoModel converts the owner-sharded n-gram table to a DTensor only in its MoE parallelizer,
    # which runs when EP is greater than one.
    if topology.expert_model_parallel_size == 1:
        raise ValueError("Qwen4 requires expert_model_parallel_size > 1")
    num_experts = int(config.get_text_config().num_experts)
    if num_experts % topology.expert_model_parallel_size:
        raise ValueError(
            f"num_experts {num_experts} must be divisible by expert_model_parallel_size "
            f"{topology.expert_model_parallel_size}"
        )


def build_model(
    config,
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
    torch_dtype: torch.dtype | str,
    peft_config: PeftConfig | None,
):
    kwargs = {
        "torch_dtype": torch_dtype,
        "peft_config": peft_config,
        "distributed_setup": distributed_setup,
        # QSA runs its sparse GQA through FlexAttention.
        "backend": Qwen3_8_FlashNextBackendConfig(
            attn="flex",
            linear=args.nemo_linear_backend,
            rms_norm=args.nemo_rms_norm_backend,
            rope_fusion=False,
            experts=args.nemo_experts_backend,
            dispatcher=args.nemo_dispatcher,
            enable_hf_state_dict_adapter=True,
            enable_fsdp_optimizations=True,
        ),
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
    # The checkpoint's table is BF16, and the PLE layer casts its rows to the BF16 block activations.
    for module in model.modules():
        if isinstance(module, ROLLOUT_STATIC_MODULES):
            module.requires_grad_(False).to(torch.bfloat16)
    return model


def build_model_batch(model: torch.nn.Module, pack: dict) -> dict:
    """Build the unsharded CPU model inputs of one text pack.

    The model takes document boundaries as `cu_seqlens`: its CP sharder moves them into the CP
    context, and without CP its QSA, Gated DeltaNet, and PLE layers read them directly.
    """
    if "multimodal_inputs" in pack:
        raise ValueError("Qwen4 trains its language model only, so packs must be text")
    tokens = pack["tokens"]
    return {
        "input_ids": tokens.unsqueeze(0),
        "labels": build_source_labels(tokens, pack["cu_seqlens"]).unsqueeze(0),
        "position_ids": pack["position_ids"].unsqueeze(0),
        "cu_seqlens": pack["cu_seqlens"],
    }


def build_packed_cp_sharder(
    model: torch.nn.Module,
    device_mesh,
    model_batch: dict,
    *,
    padding_token_id: int,
) -> ContextParallelSharder:
    """Resolve the sharder that AutoModel selects for the model.

    Under CP it is the model's contiguous sharder, which pads every pack so that each rank's shard is a
    multiple of the QSA compression ratio. Outside CP it is the identity sharder, which keeps the pack unpadded, because
    the packed Gated DeltaNet path requires boundaries that end at the row length.
    """
    return ContextParallelSharder(model, device_mesh, model_batch, padding_token_id=padding_token_id)
