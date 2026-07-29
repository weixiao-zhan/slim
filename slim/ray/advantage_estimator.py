# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-independent construction of edge-aligned training targets."""

from __future__ import annotations

from itertools import chain

import ray
import torch

from slim.utils.misc import load_function
from slim.utils.ppo_utils import vanilla_gae
from slim.utils.types import Episode


class AdvantageEstimator:
    """Prepare training targets before role-specific packing."""

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
        partitions = ray.get(rollout_data_refs)
        self.compute_partition_training_targets(partitions, value_payloads)
        return [ray.put(episodes) for episodes in partitions]

    def compute_partition_training_targets(
        self,
        partitions: list[list[Episode]],
        value_payloads: list[dict] | None = None,
    ) -> None:
        episodes = list(chain.from_iterable(partitions))
        if any(episode.episode_index is None for episode in episodes):
            raise ValueError("every training episode must have an episode_index")
        episodes.sort(key=lambda episode: episode.episode_index)
        episode_indices = [episode.episode_index for episode in episodes]
        if episode_indices != list(range(len(episodes))):
            raise ValueError(f"episode_index values must be contiguous and unique, got {episode_indices}")

        if self.args.advantage_estimator in ("grpo", "gspo"):
            if value_payloads is not None:
                raise ValueError(f"{self.args.advantage_estimator} does not accept critic values")
            advantages = self._compute_group_advantages(episodes)
            for episode, advantage in zip(episodes, advantages, strict=True):
                episode.set_train_targets([advantage] * episode.num_edges)
            return

        if self.args.advantage_estimator != "ppo_gae":
            raise NotImplementedError(self.args.advantage_estimator)
        if value_payloads is None:
            raise ValueError("PPO GAE requires critic values")

        values_by_episode_index = self._index_values(partitions, value_payloads)
        self._compute_ppo_targets(episodes, values_by_episode_index)

    def _compute_group_advantages(self, episodes: list[Episode]) -> list[float]:
        if self.custom_reward_post_process is not None:
            self.custom_reward_post_process(self.args, episodes)

        rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float32)
        if not self.args.group_advantage_normalization:
            return rewards.tolist()

        expected = self.args.n_samples_per_prompt * self.args.rollout_batch_size
        if rewards.numel() == expected:
            rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
        else:
            rewards = rewards.view(1, -1)
        rewards -= rewards.mean(dim=-1, keepdim=True)
        if self.args.group_advantage_std_normalization:
            rewards /= rewards.std(dim=-1, keepdim=True) + 1e-6

        return rewards.flatten().tolist()

    @staticmethod
    def _index_values(partitions: list[list[Episode]], payloads: list[dict]) -> dict[int, object]:
        values_by_dp = {}
        for payload in payloads:
            if payload["cp_rank"] != 0:
                continue
            dp_rank = payload["dp_rank"]
            if dp_rank in values_by_dp:
                raise ValueError(f"multiple critic value payloads for DP rank {dp_rank}")
            values_by_dp[dp_rank] = payload["values"]

        if set(values_by_dp) != set(range(len(partitions))):
            raise ValueError(
                f"critic value DP ranks {sorted(values_by_dp)} do not match rollout partitions "
                f"{list(range(len(partitions)))}"
            )

        values_by_episode_index = {}
        for dp_rank, episodes in enumerate(partitions):
            values = values_by_dp[dp_rank]
            if len(values) != len(episodes):
                raise ValueError(
                    f"DP rank {dp_rank} has {len(episodes)} episodes but {len(values)} critic values"
                )
            for episode, value in zip(episodes, values, strict=True):
                if len(value) != episode.num_edges:
                    raise ValueError(
                        f"episode {episode.episode_index} has {episode.num_edges} edges but {len(value)} values"
                    )
                values_by_episode_index[episode.episode_index] = value
        return values_by_episode_index

    def _compute_ppo_targets(
        self,
        episodes: list[Episode],
        values_by_episode_index: dict[int, object],
    ) -> None:
        max_edges = max(episode.num_edges for episode in episodes)
        rewards = torch.zeros(len(episodes), max_edges)
        values = torch.zeros_like(rewards)
        masks = torch.zeros_like(rewards, dtype=torch.bool)
        for index, episode in enumerate(episodes):
            mask = torch.as_tensor(episode.loss_mask, dtype=torch.bool)
            active = mask.nonzero().flatten()
            if active.numel() == 0:
                raise ValueError(f"episode {episode.episode_index} has no policy-controlled edges")
            rewards[index, active[-1]] = episode.reward
            episode_values = values_by_episode_index[episode.episode_index]
            values[index, : episode.num_edges] = torch.as_tensor(episode_values, dtype=torch.float32)
            masks[index, : episode.num_edges] = mask

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

        for index, episode in enumerate(episodes):
            edge_count = episode.num_edges
            episode.set_train_targets(
                advantages[index, :edge_count].tolist(),
                values=values[index, :edge_count].tolist(),
                value_targets=value_targets[index, :edge_count].tolist(),
            )


RayAdvantageEstimator = ray.remote(num_cpus=1)(AdvantageEstimator)
