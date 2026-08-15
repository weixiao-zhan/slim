# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo mesh construction and logical data-parallel coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes


@dataclass(frozen=True)
class NeMoTopology:
    world_size: int
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    @classmethod
    def from_args(cls, args, world_size: int) -> NeMoTopology:
        return cls(
            world_size=world_size,
            context_parallel_size=args.context_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
        )

    @property
    def logical_dp_size(self) -> int:
        return self.world_size // self.context_parallel_size

    def build(self, args):
        activation_checkpointing = args.activation_checkpointing
        strategy = FSDP2Config(
            sequence_parallel=False,
            patch_is_packed_sequence=False,
            offload_policy=None,
            activation_checkpointing=activation_checkpointing,
            defer_fsdp_grad_sync=args.defer_fsdp_grad_sync,
            reshard_after_forward=True,
        )
        sizes = ParallelismSizes(
            dp_size=None,
            dp_replicate_size=1,
            tp_size=1,
            pp_size=1,
            cp_size=self.context_parallel_size,
            ep_size=self.expert_model_parallel_size,
        )
        return DistributedSetup.build(
            strategy=strategy,
            parallelism_sizes=sizes,
            activation_checkpointing=activation_checkpointing,
            world_size=self.world_size,
            timeout_minutes=args.distributed_timeout_minutes,
        )
