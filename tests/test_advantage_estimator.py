# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.ray.advantage_estimator import AdvantageEstimator
from slim.utils.types import Episode


NUM_GPUS = 0


def _args(advantage_estimator: str, **overrides):
    values = {
        "advantage_estimator": advantage_estimator,
        "custom_reward_post_process_path": None,
        "gamma": 1.0,
        "group_advantage_normalization": True,
        "lambd": 1.0,
        "n_samples_per_prompt": 2,
        "normalize_advantages": False,
        "group_advantage_std_normalization": False,
        "rollout_batch_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _episode(index: int, reward: float, *, loss_mask=None):
    loss_mask = [1, 1] if loss_mask is None else loss_mask
    return Episode(
        tokens=list(range(len(loss_mask) + 1)),
        loss_mask=loss_mask,
        reward=reward,
        rollout_index=index,
    )


@pytest.mark.unit
def test_episode_set_train_targets_validates_edge_alignment():
    episode = _episode(0, 1.0)

    with pytest.raises(ValueError, match="advantages length"):
        episode.set_train_targets([1.0])

    episode.set_train_targets([1.0, 2.0], values=[0.1, 0.2], value_targets=[1.1, 2.2])

    assert episode.advantages == [1.0, 2.0]
    assert episode.values == [0.1, 0.2]
    assert episode.value_targets == [1.1, 2.2]


@pytest.mark.unit
def test_grpo_normalizes_rewards_in_rollout_order_across_partitions():
    episodes = [
        _episode(0, 1.0),
        _episode(1, 3.0),
        _episode(2, 10.0),
        _episode(3, 14.0),
    ]
    partitions = [[episodes[2], episodes[0]], [episodes[3], episodes[1]]]

    AdvantageEstimator(_args("grpo")).prepare_partitions(partitions)

    assert [episode.reward for episode in episodes] == [1.0, 3.0, 10.0, 14.0]
    assert [episode.advantages for episode in episodes] == [
        [-1.0, -1.0],
        [1.0, 1.0],
        [-2.0, -2.0],
        [2.0, 2.0],
    ]
    assert all(episode.values is None and episode.value_targets is None for episode in episodes)


@pytest.mark.unit
def test_grpo_can_use_raw_rewards_as_reinforce_advantages():
    episodes = [_episode(0, -1.0), _episode(1, 2.0)]

    AdvantageEstimator(
        _args("grpo", group_advantage_normalization=False, rollout_batch_size=1)
    ).prepare_partitions([episodes])

    assert [episode.reward for episode in episodes] == [-1.0, 2.0]
    assert [episode.advantages for episode in episodes] == [
        [-1.0, -1.0],
        [2.0, 2.0],
    ]


@pytest.mark.unit
def test_estimator_requires_canonical_rollout_indices():
    episodes = [_episode(0, 1.0), _episode(2, 2.0)]

    with pytest.raises(ValueError, match="contiguous and unique"):
        AdvantageEstimator(_args("grpo")).prepare_partitions([episodes])


@pytest.mark.unit
def test_ppo_gae_selects_cp_zero_values_by_dp_rank():
    first = _episode(0, 2.0, loss_mask=[0, 1, 0])
    second = _episode(1, 3.0)
    partitions = [[first], [second]]
    payloads = [
        {"dp_rank": 1, "cp_rank": 1, "values": [torch.full((2,), 99.0)]},
        {"dp_rank": 1, "cp_rank": 0, "values": [torch.zeros(2)]},
        {"dp_rank": 0, "cp_rank": 0, "values": [torch.zeros(3)]},
        {"dp_rank": 0, "cp_rank": 1, "values": [torch.full((3,), 99.0)]},
    ]

    AdvantageEstimator(_args("ppo_gae")).prepare_partitions(
        partitions,
        payloads,
    )

    assert first.values == [0.0, 0.0, 0.0]
    assert first.advantages == [0.0, 2.0, 0.0]
    assert first.value_targets == [0.0, 2.0, 0.0]
    assert second.values == [0.0, 0.0]
    assert second.advantages == [3.0, 3.0]
    assert second.value_targets == [3.0, 3.0]
