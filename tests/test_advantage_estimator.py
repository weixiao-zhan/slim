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
    episode = Episode(
        tokens=list(range(len(loss_mask) + 1)),
        loss_mask=loss_mask,
        reward=reward,
        episode_index=index,
    )
    episode.finalize_source_token_alignment()
    return episode


@pytest.mark.unit
def test_episode_requires_loss_mask_before_finalization():
    episode = Episode(tokens=[1, 2])

    with pytest.raises(ValueError, match="loss_mask must be present"):
        episode.finalize_source_token_alignment()


@pytest.mark.unit
def test_episode_finalization_is_a_single_rollout_boundary():
    episode = Episode(tokens=[1, 2, 3], loss_mask=[0, 1], rollout_log_probs=[0.0, -0.5])

    episode.finalize_source_token_alignment()

    assert episode.loss_mask.tolist() == [0, 1, 0]
    assert episode.rollout_log_probs.tolist() == [0.0, -0.5, 0.0]
    with pytest.raises(ValueError, match="prediction count"):
        episode.finalize_source_token_alignment()


@pytest.mark.unit
def test_episode_finalizes_source_token_alignment_and_validates_training_targets():
    episode = _episode(0, 1.0)

    with pytest.raises(ValueError, match="advantages length"):
        episode.set_train_targets([1.0])

    episode.set_train_targets(
        [1.0, 2.0, 0.0],
        values=[0.1, 0.2, 0.0],
        value_targets=[1.1, 2.2, 0.0],
    )

    assert episode.loss_mask.tolist() == [1, 1, 0]
    assert episode.advantages.tolist() == [1.0, 2.0, 0.0]
    assert episode.values.tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert episode.value_targets.tolist() == pytest.approx([1.1, 2.2, 0.0])

    with pytest.raises(ValueError, match="terminal source-token slot"):
        episode.set_train_targets([1.0, 2.0, 3.0])


@pytest.mark.unit
def test_grpo_normalizes_rewards_in_rollout_order_across_partitions():
    episodes = [
        _episode(0, 1.0),
        _episode(1, 3.0),
        _episode(2, 10.0),
        _episode(3, 14.0),
    ]
    partitions = [[episodes[2], episodes[0]], [episodes[3], episodes[1]]]

    AdvantageEstimator(_args("grpo")).compute_partition_training_targets(partitions)

    assert [episode.reward for episode in episodes] == [1.0, 3.0, 10.0, 14.0]
    assert [episode.advantages.tolist() for episode in episodes] == [
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
    ]
    assert all(episode.values is None and episode.value_targets is None for episode in episodes)


@pytest.mark.unit
def test_grpo_can_use_raw_rewards_as_reinforce_advantages():
    episodes = [_episode(0, -1.0), _episode(1, 2.0)]

    AdvantageEstimator(
        _args("grpo", group_advantage_normalization=False, rollout_batch_size=1)
    ).compute_partition_training_targets([episodes])

    assert [episode.reward for episode in episodes] == [-1.0, 2.0]
    assert [episode.advantages.tolist() for episode in episodes] == [
        [-1.0, -1.0, 0.0],
        [2.0, 2.0, 0.0],
    ]


@pytest.mark.unit
def test_estimator_requires_canonical_rollout_indices():
    episodes = [_episode(0, 1.0), _episode(2, 2.0)]

    with pytest.raises(ValueError, match="contiguous and unique"):
        AdvantageEstimator(_args("grpo")).compute_partition_training_targets([episodes])


@pytest.mark.unit
def test_ppo_gae_selects_cp_zero_values_by_dp_rank():
    first = _episode(0, 2.0, loss_mask=[0, 1, 0])
    second = _episode(1, 3.0)
    partitions = [[first], [second]]
    payloads = [
        {"dp_rank": 1, "cp_rank": 1, "values": [torch.full((3,), 99.0)]},
        {"dp_rank": 1, "cp_rank": 0, "values": [torch.zeros(3)]},
        {"dp_rank": 0, "cp_rank": 0, "values": [torch.zeros(4)]},
        {"dp_rank": 0, "cp_rank": 1, "values": [torch.full((4,), 99.0)]},
    ]

    AdvantageEstimator(_args("ppo_gae")).compute_partition_training_targets(
        partitions,
        payloads,
    )

    assert first.values.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert first.advantages.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert first.value_targets.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert second.values.tolist() == [0.0, 0.0, 0.0]
    assert second.advantages.tolist() == [3.0, 3.0, 0.0]
    assert second.value_targets.tolist() == [3.0, 3.0, 0.0]
