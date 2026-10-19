# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.ray.advantage_estimator import AdvantageEstimator
from slim.utils.trajectory_batch import TrajectoryBatch
from slim.utils.types import Episode, Trajectory


NUM_GPUS = 0


def _args(advantage_estimator: str, **overrides):
    values = {
        "advantage_estimator": advantage_estimator,
        "custom_reward_post_process_path": None,
        "gamma": 1.0,
        "group_advantage_normalization": True,
        "lambd": 1.0,
        "normalize_advantages": False,
        "group_advantage_std_normalization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _trajectory(reward=None, *, loss_mask=None, episode_index=0, group_index=0, loss_weight=1.0):
    loss_mask = [1, 1] if loss_mask is None else loss_mask
    trajectory = Trajectory(
        token_ids=list(range(len(loss_mask) + 1)),
        loss_mask=loss_mask,
        reward=reward,
        episode_index=episode_index,
        group_index=group_index,
        loss_weight=loss_weight,
    )
    trajectory.finalize_source_token_alignment()
    return trajectory


def _episode(index: int, reward: float, *, group_index: int = 0, trajectory_count: int = 1) -> Episode:
    episode = Episode(
        trajectories=[Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1]) for _ in range(trajectory_count)],
        reward=reward,
        episode_index=index,
        group_index=group_index,
    )
    episode.finalize_source_token_alignment()
    return episode


def _batch(trajectories: list[Trajectory]) -> TrajectoryBatch:
    return TrajectoryBatch(trajectories=trajectories)


@pytest.mark.unit
def test_trajectory_finalization_is_a_single_rollout_boundary():
    trajectory = Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1], rollout_log_probs=[0.0, -0.5])

    trajectory.finalize_source_token_alignment()

    assert trajectory.loss_mask.tolist() == [0, 1, 0]
    assert trajectory.rollout_log_probs.tolist() == [0.0, -0.5, 0.0]
    with pytest.raises(ValueError, match="prediction count"):
        trajectory.finalize_source_token_alignment()


@pytest.mark.unit
def test_grpo_normalizes_rewards_by_group_across_dp_ranks():
    trajectories = [
        _trajectory(1.0, episode_index=0, group_index=0),
        _trajectory(3.0, episode_index=1, group_index=0),
        _trajectory(10.0, episode_index=2, group_index=1),
        _trajectory(14.0, episode_index=3, group_index=1),
    ]
    batches = [_batch([trajectories[2], trajectories[0]]), _batch([trajectories[3], trajectories[1]])]

    AdvantageEstimator(_args("grpo")).compute_partition_training_targets(batches)

    assert [trajectory.reward for trajectory in trajectories] == [1.0, 3.0, 10.0, 14.0]
    assert [trajectory.advantages.tolist() for trajectory in trajectories] == [
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
    ]
    assert all(t.values is None and t.value_targets is None for t in trajectories)


@pytest.mark.unit
def test_grpo_baseline_weights_each_episode_once_regardless_of_trajectory_count():
    # One attempt made three calls scoring 0, the other made one call scoring 4.
    # The group mean is 2 (the mean of the attempts), not 1 (the mean of the trajectories).
    trajectories = [
        _trajectory(0.0, episode_index=0, group_index=0, loss_weight=1 / 3),
        _trajectory(0.0, episode_index=0, group_index=0, loss_weight=1 / 3),
        _trajectory(0.0, episode_index=0, group_index=0, loss_weight=1 / 3),
        _trajectory(4.0, episode_index=1, group_index=0),
    ]

    AdvantageEstimator(_args("grpo")).compute_partition_training_targets([_batch(trajectories)])

    assert [trajectory.advantages[0].item() for trajectory in trajectories] == [-2.0, -2.0, -2.0, 2.0]


@pytest.mark.unit
def test_grpo_keeps_within_episode_reward_differences():
    trajectories = [
        _trajectory(1.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(3.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(2.0, episode_index=1, group_index=0),
    ]

    AdvantageEstimator(_args("grpo")).compute_partition_training_targets([_batch(trajectories)])

    # Both attempts mean 2, so the group baseline is 2 and each trajectory keeps its own offset.
    assert [trajectory.advantages[0].item() for trajectory in trajectories] == [-1.0, 1.0, 0.0]


@pytest.mark.unit
def test_grpo_std_normalization_scales_by_the_episode_level_spread():
    trajectories = [
        _trajectory(0.0, episode_index=0, group_index=0),
        _trajectory(2.0, episode_index=1, group_index=0),
    ]

    AdvantageEstimator(
        _args("grpo", group_advantage_std_normalization=True)
    ).compute_partition_training_targets([_batch(trajectories)])

    advantages = [trajectory.advantages[0].item() for trajectory in trajectories]
    assert advantages == pytest.approx([-0.7071, 0.7071], abs=1e-3)


@pytest.mark.unit
def test_ppo_gae_selects_cp_zero_values_by_dp_rank():
    first = _trajectory(2.0, loss_mask=[0, 1, 0], episode_index=0)
    second = _trajectory(3.0, episode_index=1)
    batches = [_batch([first]), _batch([second])]
    payloads = [
        {"dp_rank": 1, "cp_rank": 1, "values": [torch.full((3,), 99.0)]},
        {"dp_rank": 1, "cp_rank": 0, "values": [torch.zeros(3)]},
        {"dp_rank": 0, "cp_rank": 0, "values": [torch.zeros(4)]},
        {"dp_rank": 0, "cp_rank": 1, "values": [torch.full((4,), 99.0)]},
    ]

    AdvantageEstimator(_args("ppo_gae")).compute_partition_training_targets(batches, payloads)

    assert first.values.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert first.advantages.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert first.value_targets.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert second.values.tolist() == [0.0, 0.0, 0.0]
    assert second.advantages.tolist() == [3.0, 3.0, 0.0]
    assert second.value_targets.tolist() == [3.0, 3.0, 0.0]


@pytest.mark.unit
def test_ppo_gae_does_not_bootstrap_across_trajectory_boundaries():
    # Two trajectories of one attempt. Each computes its own return from its own reward,
    # so the first trajectory's advantage does not see the second trajectory's value.
    trajectories = [
        _trajectory(5.0, episode_index=0, loss_weight=0.5),
        _trajectory(0.0, episode_index=0, loss_weight=0.5),
    ]
    payloads = [{"dp_rank": 0, "cp_rank": 0, "values": [torch.zeros(3), torch.zeros(3)]}]

    AdvantageEstimator(_args("ppo_gae")).compute_partition_training_targets([_batch(trajectories)], payloads)

    assert trajectories[0].advantages.tolist() == [5.0, 5.0, 0.0]
    assert trajectories[1].advantages.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_custom_reward_post_process_sees_episodes_as_groups_of_trajectories(monkeypatch):
    trajectories = [
        _trajectory(1.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(1.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(1.0, episode_index=1, group_index=0),
    ]
    seen = {}

    def post_process(args, episodes):
        seen["trajectory_counts"] = [len(trajectories) for trajectories in episodes]
        for trajectories in episodes:
            for index, trajectory in enumerate(trajectories):
                trajectory.reward = float(index)

    monkeypatch.setattr(
        "slim.ray.advantage_estimator.load_function",
        lambda _path: post_process,
    )
    estimator = AdvantageEstimator(
        _args("grpo", custom_reward_post_process_path="custom.shape", group_advantage_normalization=False)
    )

    estimator.compute_partition_training_targets([_batch(trajectories)])

    assert seen["trajectory_counts"] == [2, 1]
    assert [trajectory.advantages[0].item() for trajectory in trajectories] == [0.0, 1.0, 0.0]
