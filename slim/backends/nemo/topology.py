# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo mesh construction and logical data-parallel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NeMoTopology:
    world_size: int
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = field(init=False)

    def __post_init__(self) -> None:
        for name, size in (
            ("world_size", self.world_size),
            ("context_parallel_size", self.context_parallel_size),
            ("expert_model_parallel_size", self.expert_model_parallel_size),
            ("dp_replicate_size", self.dp_replicate_size),
        ):
            if size < 1:
                raise ValueError(f"{name} must be at least 1")

        divisor = self.context_parallel_size * self.dp_replicate_size
        if self.world_size % divisor:
            raise ValueError(
                f"world size {self.world_size} is not divisible by CP {self.context_parallel_size} "
                f"times replicated DP {self.dp_replicate_size}"
            )
        dp_shard_size = self.world_size // divisor
        object.__setattr__(self, "dp_shard_size", dp_shard_size)
        if (dp_shard_size * self.context_parallel_size) % self.expert_model_parallel_size:
            raise ValueError(
                f"expert parallel size {self.expert_model_parallel_size} must divide "
                f"DP-shard times CP ({dp_shard_size * self.context_parallel_size})"
            )

    @classmethod
    def from_args(cls, args, world_size: int) -> NeMoTopology:
        return cls(
            world_size=world_size,
            context_parallel_size=args.context_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            dp_replicate_size=args.dp_replicate_size,
        )

    @property
    def logical_dp_size(self) -> int:
        return self.world_size // self.context_parallel_size

    def build(self, args):
        from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from torch.distributed.fsdp import CPUOffloadPolicy

        activation_checkpointing = args.activation_checkpointing or args.gradient_checkpointing
        strategy = FSDP2Config(
            sequence_parallel=False,
            patch_is_packed_sequence=False,
            offload_policy=CPUOffloadPolicy() if args.nemo_cpu_offload else None,
            activation_checkpointing=activation_checkpointing,
            defer_fsdp_grad_sync=args.defer_fsdp_grad_sync,
        )
        sizes = ParallelismSizes(
            dp_size=None,
            dp_replicate_size=self.dp_replicate_size,
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


def flat_mesh(device_mesh, name: str):
    from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

    return get_flat_mesh(device_mesh, name)


def mesh_rank(mesh) -> int:
    if mesh is None or mesh.size() == 1:
        return 0
    return mesh.get_local_rank()
