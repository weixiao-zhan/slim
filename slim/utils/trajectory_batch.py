# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flatten episodes into the per-DP-rank trajectory batches training consumes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from slim.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slim.utils.types import Episode, Trajectory

# Per-document loss weight $w_d$ for each normalization unit. `episode` spreads one
# unit of weight over an attempt's spans so every attempt counts once; `trajectory`
# weights every span equally; `token` denominates the policy term in tokens instead.
LOSS_NORMALIZATION_UNITS = ("episode", "trajectory", "token")

# Optional sequence fields a padding document must mirror when the real spans carry them.
_OPTIONAL_FIELD_DTYPES = {
    "rollout_log_probs": torch.float32,
    "rollout_routed_experts": torch.int32,
}


@dataclass
class TrajectoryBatch:
    """One DP rank's share of a rollout, evenly divided into optimizer steps."""

    trajectories: list[Trajectory] = field(default_factory=list)
    num_steps: int = 1

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {self.num_steps}")
        if len(self.trajectories) % self.num_steps:
            raise ValueError(
                f"{len(self.trajectories)} trajectories do not divide evenly into "
                f"{self.num_steps} optimizer steps"
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def step_size(self) -> int:
        return len(self.trajectories) // self.num_steps

    def steps(self):
        """Yield each optimizer step's slice of the rank-local trajectory list."""
        size = self.step_size
        for start in range(0, len(self.trajectories), size):
            yield start, self.trajectories[start : start + size]


def _desugar_reward(episode: Episode) -> None:
    """Broadcast an attempt-level reward onto every span of the attempt.

    Broadcast is the only sound distribution: trajectories carry no order
    relation, so there is no distinguished span to receive the reward. Under
    `--loss-normalization-unit episode` each span weighs $1/k$, so an attempt
    scored $R$ still contributes total credit $R$.
    """
    span_rewards = [trajectory.reward for trajectory in episode.trajectories]
    if episode.reward is None:
        if any(reward is None for reward in span_rewards):
            raise ValueError(
                f"episode {episode.episode_index} needs a reward on the episode or on every trajectory"
            )
        return
    if any(reward is not None for reward in span_rewards):
        raise ValueError(
            f"episode {episode.episode_index} sets both episode and trajectory rewards; "
            "reward level is exclusive"
        )
    for trajectory in episode.trajectories:
        trajectory.reward = episode.reward
    episode.reward = None


def _padding_trajectory(pad_token_id: int, optional_shapes: dict[str, tuple[int, ...]]) -> Trajectory:
    r"""An inert two-token document that lets every rank hold an equal share.

    Two tokens is the minimum `pack_sequences` accepts. The all-zero `loss_mask`
    gives it a zero numerator and denominator in every reduction, and the zero
    `loss_weight` keeps it out of $\sum_d w_d$.

    `pack_sequences` requires each optional sequence field to be present for every
    document of a pack or for none, so padding mirrors whichever fields the real
    spans carry.
    """
    trajectory = Trajectory(
        token_ids=[pad_token_id, pad_token_id],
        loss_mask=[0],
        reward=0.0,
        loss_weight=0.0,
    )
    for name, trailing_shape in optional_shapes.items():
        setattr(trajectory, name, torch.zeros((1, *trailing_shape), dtype=_OPTIONAL_FIELD_DTYPES[name]))
    trajectory.finalize_source_token_alignment()
    return trajectory


def flatten_episodes(episodes: list[Episode], *, loss_normalization_unit: str) -> list[Trajectory]:
    """Desugar rewards and stamp every span with its attempt's identity and weight."""
    if loss_normalization_unit not in LOSS_NORMALIZATION_UNITS:
        raise ValueError(f"unknown loss normalization unit {loss_normalization_unit!r}")

    trajectories = []
    for episode in episodes:
        if not episode.trajectories:
            raise ValueError(f"episode {episode.episode_index} has no trajectories")
        _desugar_reward(episode)
        weight = 1.0 / len(episode.trajectories) if loss_normalization_unit == "episode" else 1.0
        for trajectory in episode.trajectories:
            trajectory.episode_index = episode.episode_index
            trajectory.group_index = episode.group_index
            trajectory.loss_weight = weight
            trajectories.append(trajectory)
    return trajectories


def _optional_field_shapes(trajectories: list[Trajectory]) -> dict[str, tuple[int, ...]]:
    """Collect the trailing shape of each optional sequence field the spans carry."""
    shapes = {}
    for name in _OPTIONAL_FIELD_DTYPES:
        present = {
            tuple(getattr(trajectory, name).shape[1:])
            for trajectory in trajectories
            if getattr(trajectory, name) is not None
        }
        if not present:
            continue
        if len(present) > 1:
            raise ValueError(f"{name} shapes disagree across trajectories: {sorted(present)}")
        shapes[name] = present.pop()
    return shapes


def build_dp_batches(
    episodes: list[Episode],
    *,
    dp_size: int,
    num_steps: int,
    loss_normalization_unit: str,
    pad_token_id: int,
    balance_data: bool,
) -> list[TrajectoryBatch]:
    """Split an episode batch into one balanced trajectory batch per DP rank.

    The optimizer step count is denominated in episodes, so the LR decay horizon
    does not move with how many generation calls a rollout made. The unit of work
    is the trajectory: each is its own packed document and cannot be concatenated
    with its siblings, so distributing spans rather than attempts is what gives
    every rank an equal document count.
    """
    trajectories = flatten_episodes(episodes, loss_normalization_unit=loss_normalization_unit)

    padding_count = -len(trajectories) % (dp_size * num_steps)
    if padding_count:
        optional_shapes = _optional_field_shapes(trajectories)
        trajectories += [
            _padding_trajectory(pad_token_id, optional_shapes) for _ in range(padding_count)
        ]

    lengths = [len(trajectory.token_ids) for trajectory in trajectories]
    rank_indices: list[list[int]] = [[] for _ in range(dp_size)]
    for step_indices in _partition(lengths, num_steps, balance_data):
        step_lengths = [lengths[index] for index in step_indices]
        for rank, positions in enumerate(_partition(step_lengths, dp_size, balance_data)):
            rank_indices[rank].extend(step_indices[position] for position in positions)

    return [
        TrajectoryBatch(
            trajectories=[trajectories[index] for index in indices],
            num_steps=num_steps,
        )
        for indices in rank_indices
    ]


def _partition(lengths: list[int], count: int, balance_data: bool) -> list[list[int]]:
    if balance_data:
        return get_seqlen_balanced_partitions(lengths, count, equal_size=True)
    return [list(range(offset, len(lengths), count)) for offset in range(count)]


def group_by_episode(trajectories: list[Trajectory]) -> list[list[Trajectory]]:
    """Regroup flattened spans by the attempt they belong to, in rollout order.

    Padding trajectories carry no `episode_index` and form no group.
    """
    episodes: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        if trajectory.episode_index is None:
            continue
        episodes.setdefault(trajectory.episode_index, []).append(trajectory)
    indices = sorted(episodes)
    if indices != list(range(len(indices))):
        raise ValueError(f"episode_index values must be contiguous and unique, got {indices}")
    return [episodes[index] for index in indices]
