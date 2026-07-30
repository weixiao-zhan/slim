# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt Slim routing targets to NeMo router replay."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from nemo_automodel.components.moe.router_replay import RouterReplay


@contextmanager
def replay_router_targets(routed_experts: torch.Tensor | None):
    instances = RouterReplay.instances()
    if routed_experts is None or not instances:
        yield
        return
    if routed_experts.ndim != 4:
        raise ValueError("routing replay tensor must have shape [batch, tokens, layers, top_k]")
    if routed_experts.shape[2] != len(instances):
        raise ValueError(
            f"routing replay has {routed_experts.shape[2]} layers but the model has {len(instances)} routers"
        )
    per_layer = [
        routed_experts[:, :, index, :].reshape(-1, routed_experts.shape[-1]).contiguous()
        for index in range(len(instances))
    ]
    with RouterReplay.replay(per_layer):
        yield
