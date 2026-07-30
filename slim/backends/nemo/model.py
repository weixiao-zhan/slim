# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent NeMo construction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.optim import AdamWConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def build_optimizer(args, device_mesh, *, param_groups: list[dict]):
    config = AdamWConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    return config.build_from_param_groups(param_groups, device_mesh=device_mesh)


class ScalarValueHead(nn.Module):
    """Token-level scalar projection for the PPO critic."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).squeeze(-1).float()


class CriticModel(nn.Module):
    """Checkpointable AutoModel backbone and scalar value head."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, **kwargs) -> torch.Tensor:
        output = self.backbone(logits_to_keep=1, output_hidden_states=True, **kwargs)
        return self.value_head(get_final_hidden_states(output))


def build_value_head(backbone: nn.Module, distributed_setup) -> nn.Module:
    backbone.get_output_embeddings().requires_grad_(False)

    config = backbone.config.get_text_config()
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
    return head
