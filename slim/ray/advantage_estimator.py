# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-independent construction of source-token-aligned training targets."""

from __future__ import annotations

from itertools import chain

import ray
import torch

from slim.utils.misc import load_function
from slim.utils.ppo_utils import vanilla_gae
from slim.utils.trajectory_batch import TrajectoryBatch, group_by_episode
from slim.utils.types import Trajectory


class AdvantageEstimator:
    """Prepare training targets before role-specific packing.

    Trajectories arrive already split across DP ranks, each stamped with the
    attempt it belongs to. The estimator regroups them by `episode_index` so
    baselines see an attempt as a unit, then writes targets back onto the same
    trajectory objects the role-local packs already index.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.custom_reward_post_process = None
        if args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process = load_function(args.custom_reward_post_process_path)

    def compute_training_targets(
        self,
        rollout_data_refs: list,
        value_payloads: list[dict] | None = None,
    ) -> list:
        batches = ray.get(rollout_data_refs)
        self.compute_partition_training_targets(batches, value_payloads)
        return [ray.put(batch) for batch in batches]

    def compute_partition_training_targets(
        self,
        batches: list[TrajectoryBatch],
        value_payloads: list[dict] | None = None,
    ) -> None:
        """Write targets onto every trajectory of every DP rank's batch."""
        trajectories = list(chain.from_iterable(batch.trajectories for batch in batches))
        episodes = group_by_episode(trajectories)

        if self.custom_reward_post_process is not None:
            self.custom_reward_post_process(self.args, episodes)

        if self.args.advantage_estimator in ("grpo", "gspo"):
            self._compute_group_targets(episodes)
        else:
            self._compute_ppo_targets(episodes, self._index_values(batches, value_payloads))

        for trajectory in trajectories:
            if trajectory.episode_index is None:
                _padding_targets(trajectory, values=self.args.advantage_estimator == "ppo_gae")

    def _compute_group_targets(self, episodes: list[list[Trajectory]]) -> None:
        """Center each trajectory's reward on its prompt group's episode-level statistics.

        The baseline and scale are episode-level, so an attempt that made 20
        generation calls does not outvote one that made a single call. The
        numerator stays the trajectory's own reward, so within-attempt differentiation
        survives when rewards are genuinely per-trajectory.
        """
        episode_rewards = torch.tensor(
            [sum(t.reward for t in trajectories) / len(trajectories) for trajectories in episodes],
            dtype=torch.float32,
        )
        offsets = torch.zeros_like(episode_rewards)
        scales = torch.ones_like(episode_rewards)

        if self.args.group_advantage_normalization:
            group_indices = [trajectories[0].group_index for trajectories in episodes]
            if any(index is None for index in group_indices):
                raise ValueError("group advantage normalization requires a group_index on every episode")
            groups, inverse, counts = torch.tensor(group_indices, dtype=torch.long).unique(
                return_inverse=True,
                return_counts=True,
            )
            counts = counts.to(episode_rewards.dtype)
            sums = torch.zeros(groups.numel()).scatter_add_(0, inverse, episode_rewards)
            offsets = (sums / counts)[inverse]
            if self.args.group_advantage_std_normalization:
                squares = torch.zeros(groups.numel()).scatter_add_(0, inverse, (episode_rewards - offsets) ** 2)
                scales = (squares / (counts - 1).clamp_min(1))[inverse].sqrt() + 1e-6

        for trajectories, offset, scale in zip(episodes, offsets.tolist(), scales.tolist(), strict=True):
            for trajectory in trajectories:
                advantage = (trajectory.reward - offset) / scale
                targets = torch.full((len(trajectory.token_ids),), advantage, dtype=torch.float32)
                targets[-1] = 0
                trajectory.set_train_targets(targets)

    @staticmethod
    def _index_values(batches: list[TrajectoryBatch], payloads: list[dict]) -> dict[int, object]:
        """Match critic values to trajectories by rank-local order."""
        values_by_dp = {payload["dp_rank"]: payload["values"] for payload in payloads if payload["cp_rank"] == 0}
        values_by_trajectory = {}
        for dp_rank, batch in enumerate(batches):
            for trajectory, value in zip(batch.trajectories, values_by_dp[dp_rank], strict=True):
                values_by_trajectory[id(trajectory)] = value
        return values_by_trajectory

    def _compute_ppo_targets(
        self,
        episodes: list[list[Trajectory]],
        values_by_trajectory: dict[int, object],
    ) -> None:
        """Deposit each trajectory's reward at its last active position and run GAE per trajectory.

        A trajectory is contiguous, so the recurrence within one is sound. It does
        not cross trajectory boundaries: bootstrapping $V$ from one into another
        would need to know which follows which.
        """
        trajectories = list(chain.from_iterable(episodes))
        max_tokens = max(len(trajectory.token_ids) for trajectory in trajectories)
        rewards = torch.zeros(len(trajectories), max_tokens)
        values = torch.zeros_like(rewards)
        masks = torch.zeros_like(rewards, dtype=torch.bool)
        for index, trajectory in enumerate(trajectories):
            mask = torch.as_tensor(trajectory.loss_mask, dtype=torch.bool)
            token_count = len(trajectory.token_ids)
            active = mask.nonzero().flatten()
            if active.numel() == 0:
                raise ValueError(
                    f"a trajectory of episode {trajectory.episode_index} has no policy-controlled predictions"
                )
            rewards[index, active[-1]] = trajectory.reward
            values[index, :token_count] = torch.as_tensor(
                values_by_trajectory[id(trajectory)], dtype=torch.float32
            )
            masks[index, :token_count] = mask

        advantages, value_targets = vanilla_gae(
            rewards,
            values,
            masks,
            self.args.gamma,
            self.args.lambd,
        )
        if self.args.normalize_advantages:
            selected = advantages[masks]
            mean = selected.mean()
            std = selected.std(correction=0).clamp_min(1e-8)
            advantages = torch.where(masks, (advantages - mean) / std, 0)

        for index, trajectory in enumerate(trajectories):
            token_count = len(trajectory.token_ids)
            trajectory.set_train_targets(
                advantages[index, :token_count],
                values=values[index, :token_count],
                value_targets=value_targets[index, :token_count],
            )


def _padding_targets(trajectory: Trajectory, *, values: bool) -> None:
    zeros = torch.zeros(len(trajectory.token_ids), dtype=torch.float32)
    if values:
        trajectory.set_train_targets(zeros, values=zeros.clone(), value_targets=zeros.clone())
    else:
        trajectory.set_train_targets(zeros)



RayAdvantageEstimator = ray.remote(num_cpus=1)(AdvantageEstimator)
