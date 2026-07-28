# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent NeMo construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.optim import AdamWConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def text_config(config):
    return getattr(config, "text_config", config)


def build_optimizer(args, model, device_mesh, *, param_groups: list[dict] | None = None):
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
        raise RuntimeError("critic backbone did not return final hidden states")
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


def build_critic_model(backbone: nn.Module, distributed_setup) -> CriticModules:
    output_embeddings = backbone.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.requires_grad_(False)

    config = text_config(backbone.config)
    storage_dtype = next(parameter.dtype for parameter in backbone.parameters() if parameter.is_floating_point())
    head = ScalarValueHead(
        hidden_size=config.hidden_size,
        dtype=storage_dtype,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    strategy = distributed_setup.strategy_config
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
