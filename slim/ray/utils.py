# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import dataclasses

import ray
from ray.util.placement_group import PlacementGroup


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def train_span(args) -> int:
    """Number of placement-group bundles occupied by the actor and critic trainers."""
    if args.critic_train_only:
        return args.critic_num_gpus
    # Critic-colocate folds the critic onto the actor GPUs, so the span is just the actor's.
    if args.critic_colocate or not args.use_critic:
        return args.actor_num_gpus
    return args.actor_num_gpus + args.critic_num_gpus


@dataclasses.dataclass(frozen=True)
class Placement:
    """Where the actor, critic, and rollout roles run in the placement group.

    The bundles are sorted by (node, GPU id), and each role holds a range of the sorted positions.
    With ``--rollout-colocate`` the rollout range overlaps the training ranges, and rollout time-shares these GPUs
    through offload.
    """

    pg: PlacementGroup
    bundle_indices: list[int]  # bundle index at each sorted position
    gpu_ids: list[int]  # physical GPU id at each sorted position
    actor: range
    critic: range
    rollout: range

    def role_pg(self, role: str) -> tuple[PlacementGroup, list[int], list[int]]:
        """``(pg, bundle_indices, gpu_ids)`` of the GPUs of ``role``, in sorted order."""
        positions = getattr(self, role)
        return self.pg, [self.bundle_indices[p] for p in positions], [self.gpu_ids[p] for p in positions]

    def actor_ranks(self, rollout_gpus: range) -> list[int] | None:
        """Actor ranks on the rollout GPUs ``rollout_gpus``, or None if not all of them are actor GPUs."""
        start = self.rollout.start + rollout_gpus.start
        stop = self.rollout.start + rollout_gpus.stop
        if start < self.actor.start or stop > self.actor.stop:
            return None
        return list(range(start - self.actor.start, stop - self.actor.start))


def rollout_node_ids(pg) -> list[str]:
    """Distinct node ids of the rollout bundles, in sorted bundle order."""
    placement_group, reordered_bundle_indices, _ = pg
    bundles_to_node_id = ray.util.placement_group_table(placement_group)["bundles_to_node_id"]
    return list(dict.fromkeys(bundles_to_node_id[index] for index in reordered_bundle_indices))
