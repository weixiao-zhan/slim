# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 NeMo AutoModel construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def text_config(config):
    return getattr(config, "text_config", config)


def is_moe_config(config) -> bool:
    config = text_config(config)
    return int(getattr(config, "num_experts", 0) or 0) > 0


def validate_model_config(config, topology) -> None:
    model_type = getattr(config, "model_type", "")
    if model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(f"first-party NeMo training supports Qwen3.5, got model_type={model_type!r}")

    if not is_moe_config(config):
        if topology.expert_model_parallel_size != 1:
            raise ValueError("dense Qwen3.5 requires expert_model_parallel_size=1")
        return

    num_experts = int(text_config(config).num_experts)
    if num_experts % topology.expert_model_parallel_size:
        raise ValueError(
            f"num_experts {num_experts} must be divisible by expert_model_parallel_size "
            f"{topology.expert_model_parallel_size}"
        )


def build_backend_config(args):
    from nemo_automodel.components.models.common import BackendConfig

    return BackendConfig(
        attn="sdpa",
        linear=args.nemo_linear_backend,
        rms_norm=args.nemo_rms_norm_backend,
        rope_fusion=False,
        experts=args.nemo_experts_backend,
        dispatcher=args.nemo_dispatcher,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )


def register_qwen3_5_moe_parallel_strategy() -> None:
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        Qwen3_5ParallelizationStrategy,
        register_parallel_strategy,
    )

    model_name = "Qwen3_5MoeForConditionalGeneration"
    if model_name not in PARALLELIZATION_STRATEGIES:
        register_parallel_strategy(name=model_name)(Qwen3_5ParallelizationStrategy)


def _disable_fsdp_backward_prefetch(model: nn.Module) -> int:
    from torch.distributed.fsdp import FSDPModule

    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_modules_to_backward_prefetch([module])
            count += 1
    return count


def build_policy_model(
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
):
    from nemo_automodel import NeMoAutoModelForImageTextToText

    register_qwen3_5_moe_parallel_strategy()
    kwargs = {
        "distributed_setup": distributed_setup,
        "backend": build_backend_config(args),
        "attn_implementation": "sdpa",
        "has_packed_sequence": True,
        "trust_remote_code": True,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "num_nextn_predict_layers": 0,
        "text_config": {"router_aux_loss_coef": 0.0},
        "freeze_config": {
            "freeze_vision_tower": args.freeze_vision_tower,
            "freeze_audio_tower": args.freeze_audio_tower,
            "freeze_language_model": args.freeze_language_model,
        },
    }
    if routing_replay:
        kwargs["moe_overrides"] = {"enable_routing_replay": True}
    model = NeMoAutoModelForImageTextToText.from_pretrained(checkpoint, **kwargs)
    _disable_fsdp_backward_prefetch(model)
    from .packed_cp import install_qwen3_5_packed_cp

    install_qwen3_5_packed_cp(model, distributed_setup.mesh_context.device_mesh)
    return model


def build_optimizer(args, model, device_mesh, *, param_groups: list[dict] | None = None):
    from nemo_automodel.components.optim import AdamWConfig

    config = AdamWConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    if param_groups is not None:
        return config.build_from_param_groups(param_groups, device_mesh=device_mesh)
    optimizers = config.build(model, device_mesh=device_mesh)
    if len(optimizers) != 1:
        raise RuntimeError(f"expected one optimizer without pipeline parallelism, got {len(optimizers)}")
    return optimizers[0]


class ScalarValueHead(nn.Module):
    """Token-level scalar projection for the PPO critic."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).squeeze(-1).float()


def final_hidden_state(output) -> torch.Tensor:
    """Normalize AutoModel dense and MoE final-hidden-state output contracts."""
    final_state = get_final_hidden_states(output)
    if final_state is None:
        raise RuntimeError("Qwen3.5 critic backbone did not return final hidden states")
    if not isinstance(final_state, torch.Tensor):
        raise TypeError(f"expected final hidden state tensor, got {type(final_state).__name__}")
    return final_state


@dataclass
class CriticModules:
    backbone: nn.Module
    value_head: nn.Module


class CriticModel(nn.Module):
    """Checkpointable AutoModel backbone and scalar value head."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, **kwargs) -> torch.Tensor:
        output = self.backbone(logits_to_keep=1, output_hidden_states=True, **kwargs)
        return self.value_head(final_hidden_state(output))


def build_critic_model(args, checkpoint: str, distributed_setup) -> CriticModules:
    backbone = build_policy_model(
        args,
        checkpoint,
        distributed_setup,
        routing_replay=False,
    )
    for name, parameter in backbone.named_parameters():
        if name.endswith("lm_head.weight"):
            parameter.requires_grad_(False)

    config = text_config(backbone.config)
    storage_dtype = next(parameter.dtype for parameter in backbone.parameters() if parameter.is_floating_point())
    head = ScalarValueHead(
        hidden_size=config.hidden_size,
        dtype=storage_dtype,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

    strategy = distributed_setup.strategy_config
    from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh

    fsdp_mesh = get_fsdp_dp_mesh(distributed_setup.mesh_context.device_mesh)
    fully_shard_by_dtype(
        head,
        fsdp_mesh,
        strategy.mp_policy,
        strategy.offload_policy,
        reshard_after_forward=False,
    )
    return CriticModules(backbone=backbone, value_head=head)


def resolve_state_dict_adapter(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        adapter = getattr(candidate, "state_dict_adapter", None)
        if adapter is not None:
            return adapter
    raise RuntimeError(f"{type(model).__name__} does not expose a NeMo state_dict_adapter")
