# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local ownership for NeMo router replay handles."""

from __future__ import annotations

from contextlib import contextmanager

import torch


class RouterReplayRegistry:
    """Capture and activate the replay handles belonging to one model."""

    def __init__(self, instances=()) -> None:
        self.instances = list(instances)

    @classmethod
    def begin_model_build(cls) -> None:
        from nemo_automodel.components.moe.router_replay import RouterReplay

        RouterReplay.clear_registry()

    @classmethod
    def finish_model_build(cls) -> RouterReplayRegistry:
        from nemo_automodel.components.moe.router_replay import RouterReplay

        registry = cls(RouterReplay.instances())
        RouterReplay.clear_registry()
        return registry

    @contextmanager
    def activate(self):
        from nemo_automodel.components.moe.router_replay import RouterReplay

        previous = RouterReplay._registry
        RouterReplay._registry = self.instances
        try:
            yield RouterReplay
        finally:
            RouterReplay._registry = previous

    @contextmanager
    def replay(self, routed_experts: torch.Tensor | None):
        if routed_experts is None or not self.instances:
            yield
            return
        if routed_experts.ndim != 4:
            raise ValueError("routing replay tensor must have shape [batch, tokens, layers, top_k]")
        if routed_experts.shape[2] != len(self.instances):
            raise ValueError(
                f"routing replay has {routed_experts.shape[2]} layers but the model has {len(self.instances)} routers"
            )
        per_layer = [
            routed_experts[:, :, index, :].reshape(-1, routed_experts.shape[-1]).contiguous()
            for index in range(len(self.instances))
        ]
        with self.activate() as replay_type, replay_type.replay(per_layer):
            yield
