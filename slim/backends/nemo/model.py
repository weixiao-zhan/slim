# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent NeMo construction helpers."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from dion import Muon
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.optim import AdamWConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


@torch.compile(fullgraph=True)
def _moonshot_newton_schulz(G: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Moonlight's quintic Newton-Schulz iteration over the last two dimensions.

    Dion passes the AdamW ``epsilon``; the normalization keeps Moonlight's 1e-7.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _matrix_sharded_off_group(parameter: torch.Tensor, group: dist.ProcessGroup) -> bool:
    if not isinstance(parameter, DTensor):
        return False
    mesh = parameter.device_mesh
    return any(
        placement.is_shard()
        and placement.dim >= parameter.ndim - 2
        and mesh.size(index) > 1
        and mesh.get_group(index) != group
        for index, placement in enumerate(parameter.placements)
    )


def _muon_parameter_ids(model: nn.Module, group: dist.ProcessGroup) -> set[int]:
    """Select hidden-layer matrices and grouped expert matrices for the Muon update."""
    output_embedding_ids = {id(parameter) for parameter in model.get_output_embeddings().parameters()}
    muon_ids = set()
    off_group_experts = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            continue
        is_experts = isinstance(module, (GroupedExperts, GroupedExpertsDeepEP))
        for name, parameter in module.named_parameters(recurse=False):
            if (
                not parameter.requires_grad
                or id(parameter) in output_embedding_ids
                or name.endswith("bias")
                or parameter.ndim != (3 if is_experts else 2)
            ):
                continue
            if _matrix_sharded_off_group(parameter, group):
                off_group_experts += 1
                continue
            muon_ids.add(id(parameter))
    if off_group_experts and dist.get_rank() == 0:
        logger.info(
            "Muon: %d expert parameters are FSDP-sharded over ep_shard > 1; they use AdamW.",
            off_group_experts,
        )
    return muon_ids


def build_optimizer(args, device_mesh, model: nn.Module, *, param_groups: list[dict]):
    """Build the optimizer over ``param_groups``.

    With ``--optimizer muon``, each group splits into a Muon subgroup for the matrices of
    ``model`` and an AdamW subgroup for the other parameters, one per parameter dtype
    because Dion runs one fused update per subgroup. All subgroups keep the group's
    scheduler keys.
    """
    if args.optimizer == "adamw":
        config = AdamWConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        return config.build_from_param_groups(param_groups, device_mesh=device_mesh)

    # Dion matches DTensor process groups by identity, so take the mesh FSDP sharded the model with.
    dion_mesh = model.get_input_embeddings().weight.device_mesh
    muon_ids = _muon_parameter_ids(model, dion_mesh.get_group())
    groups = []
    for group in param_groups:
        subgroups = {}
        for parameter in group["params"]:
            algorithm = "muon" if id(parameter) in muon_ids else "adamw"
            subgroups.setdefault((algorithm, parameter.dtype), []).append(parameter)
        for (algorithm, _), parameters in subgroups.items():
            groups.append({**group, "params": parameters, "algorithm": algorithm})
    return Muon(
        groups,
        distributed_mesh=dion_mesh,
        lr=args.lr,
        mu=args.muon_momentum,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        epsilon=args.adam_eps,
        nesterov=True,
        adjust_lr="rms_norm",
        newton_schulz_func=_moonshot_newton_schulz,
    )


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
