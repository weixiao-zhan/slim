# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient clipping for FSDP2 CPU-offloaded gradients."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate


def _all_reduce_scalar(value: torch.Tensor, op: dist.ReduceOp.RedOpType, group) -> None:
    """Reduce a CPU scalar through a CUDA-only process group."""
    if value.device.type != "cpu" or dist.get_backend(group) != "nccl":
        dist.all_reduce(value, op=op, group=group)
        return

    staged = value.to(device=torch.cuda.current_device())
    dist.all_reduce(staged, op=op, group=group)
    value.copy_(staged.cpu())


def _combine_l2_norms(norms: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not norms:
        return torch.zeros((), dtype=torch.float64, device=device)

    stacked = torch.stack([norm.to(device=device, dtype=torch.float64) for norm in norms])
    scale = stacked.abs().max()
    if scale == 0 or not torch.isfinite(scale):
        return scale
    return scale * stacked.div(scale).square().sum().sqrt()


@torch.no_grad()
def clip_cpu_offloaded_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> float:
    """Return the global pre-clip L2 norm and clip CPU-offloaded DTensor gradients."""
    sharding_groups: dict[tuple[object, ...], list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if isinstance(parameter, DTensor):
            key = (id(parameter.device_mesh), *(str(placement) for placement in parameter.placements))
        else:
            key = ("regular",)
        sharding_groups.setdefault(key, []).append(parameter)

    target_device = torch.device("cpu")
    group_norms = []
    for group_parameters in sharding_groups.values():
        first = group_parameters[0]
        is_dtensor = isinstance(first, DTensor)
        if is_dtensor and any(isinstance(placement, Partial) for placement in first.placements):
            raise RuntimeError("CPU-offloaded partial DTensor gradients are not supported")

        local_max = torch.zeros((), dtype=torch.float64, device=target_device)
        for parameter in group_parameters:
            grad = parameter.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            if grad is None or grad.numel() == 0:
                continue
            if grad.device.type != "cpu":
                raise RuntimeError(f"expected a CPU-offloaded gradient, got {grad.device}")
            local_max = torch.maximum(local_max, grad.detach().abs().max().double())

        if is_dtensor:
            for dim_index, placement in enumerate(first.placements):
                if not isinstance(placement, Replicate):
                    _all_reduce_scalar(
                        local_max,
                        dist.ReduceOp.MAX,
                        first.device_mesh.get_group(mesh_dim=dim_index),
                    )

        if local_max == 0 or not torch.isfinite(local_max):
            group_norms.append(local_max)
            continue

        local_square_sum = torch.zeros((), dtype=torch.float64, device=target_device)
        for parameter in group_parameters:
            grad = parameter.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            if grad is None or grad.numel() == 0:
                continue
            scaled_grad = grad.detach().abs().div(local_max)
            local_square_sum.add_(scaled_grad.square().sum(dtype=torch.float64))

        if is_dtensor:
            for dim_index, placement in enumerate(first.placements):
                if not isinstance(placement, Replicate):
                    _all_reduce_scalar(
                        local_square_sum,
                        dist.ReduceOp.SUM,
                        first.device_mesh.get_group(mesh_dim=dim_index),
                    )
        group_norms.append(local_max * local_square_sum.sqrt())

    total_norm = _combine_l2_norms(group_norms, target_device)
    clip_coefficient = min(1.0, float(max_norm) / (float(total_norm) + 1e-6))
    if math.isfinite(clip_coefficient) and clip_coefficient < 1.0:
        for group_parameters in sharding_groups.values():
            for parameter in group_parameters:
                parameter.grad.mul_(clip_coefficient)
    return float(total_norm)
