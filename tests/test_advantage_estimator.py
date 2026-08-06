# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.ray.advantage_estimator import AdvantageEstimator
from slim.utils.trajectory_batch import TrajectoryBatch, build_dp_batches
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


def _episode(index: int, reward: float, *, group_index: int = 0, span_count: int = 1) -> Episode:
    episode = Episode(
        trajectories=[Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1]) for _ in range(span_count)],
        reward=reward,
        episode_index=index,
        group_index=group_index,
    )
    episode.finalize_source_token_alignment()
    return episode


def _batch(trajectories: list[Trajectory]) -> TrajectoryBatch:
    return TrajectoryBatch(trajectories=trajectories)


@pytest.mark.unit
def test_trajectory_requires_loss_mask_before_finalization():
    trajectory = Trajectory(token_ids=[1, 2])

    with pytest.raises(ValueError, match="loss_mask must be present"):
        trajectory.finalize_source_token_alignment()


@pytest.mark.unit
def test_trajectory_finalization_is_a_single_rollout_boundary():
    trajectory = Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1], rollout_log_probs=[0.0, -0.5])

    trajectory.finalize_source_token_alignment()

    assert trajectory.loss_mask.tolist() == [0, 1, 0]
    assert trajectory.rollout_log_probs.tolist() == [0.0, -0.5, 0.0]
    with pytest.raises(ValueError, match="prediction count"):
        trajectory.finalize_source_token_alignment()


@pytest.mark.unit
def test_trajectory_finalizes_source_token_alignment_and_validates_training_targets():
    trajectory = _trajectory(1.0)

    with pytest.raises(ValueError, match="advantages length"):
        trajectory.set_train_targets([1.0])

    trajectory.set_train_targets(
        [1.0, 2.0, 0.0],
        values=[0.1, 0.2, 0.0],
        value_targets=[1.1, 2.2, 0.0],
    )

    assert trajectory.loss_mask.tolist() == [1, 1, 0]
    assert trajectory.advantages.tolist() == [1.0, 2.0, 0.0]
    assert trajectory.values.tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert trajectory.value_targets.tolist() == pytest.approx([1.1, 2.2, 0.0])

    with pytest.raises(ValueError, match="terminal source-token slot"):
        trajectory.set_train_targets([1.0, 2.0, 3.0])


@pytest.mark.unit
def test_episode_finalization_covers_every_span():
    episode = Episode(
        trajectories=[
            Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1]),
            Trajectory(token_ids=[4, 5], loss_mask=[1]),
        ],
        _sampling_params={"temperature": 1.0},
    )

    episode.finalize_source_token_alignment()

    assert [t.loss_mask.tolist() for t in episode.trajectories] == [[0, 1, 0], [1, 0]]
    assert episode._sampling_params is None


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
def test_grpo_baseline_weights_each_episode_once_regardless_of_span_count():
    # One attempt made three calls scoring 0, the other made one call scoring 4.
    # The group mean is 2 (the mean of the attempts), not 1 (the mean of the spans).
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

    # Both attempts mean 2, so the group baseline is 2 and each span keeps its own offset.
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
def test_grpo_can_use_raw_rewards_as_reinforce_advantages():
    trajectories = [
        _trajectory(-1.0, episode_index=0, group_index=0),
        _trajectory(2.0, episode_index=1, group_index=0),
    ]

    AdvantageEstimator(
        _args("grpo", group_advantage_normalization=False)
    ).compute_partition_training_targets([_batch(trajectories)])

    assert [trajectory.reward for trajectory in trajectories] == [-1.0, 2.0]
    assert [trajectory.advantages.tolist() for trajectory in trajectories] == [
        [-1.0, -1.0, 0.0],
        [2.0, 2.0, 0.0],
    ]


@pytest.mark.unit
def test_estimator_requires_canonical_episode_indices():
    trajectories = [_trajectory(1.0, episode_index=0), _trajectory(2.0, episode_index=2)]

    with pytest.raises(ValueError, match="contiguous and unique"):
        AdvantageEstimator(_args("grpo")).compute_partition_training_targets([_batch(trajectories)])


@pytest.mark.unit
def test_padding_trajectories_receive_inert_targets():
    real = _trajectory(1.0, episode_index=0, group_index=0)
    padding = Trajectory(token_ids=[0, 0], loss_mask=[0], loss_weight=0.0)
    padding.finalize_source_token_alignment()

    AdvantageEstimator(
        _args("grpo", group_advantage_normalization=False)
    ).compute_partition_training_targets([_batch([real, padding])])

    assert padding.advantages.tolist() == [0.0, 0.0]


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
    # Two spans of one attempt. Each computes its own return from its own reward,
    # so the first span's advantage does not see the second span's value.
    spans = [
        _trajectory(5.0, episode_index=0, loss_weight=0.5),
        _trajectory(0.0, episode_index=0, loss_weight=0.5),
    ]
    payloads = [{"dp_rank": 0, "cp_rank": 0, "values": [torch.zeros(3), torch.zeros(3)]}]

    AdvantageEstimator(_args("ppo_gae")).compute_partition_training_targets([_batch(spans)], payloads)

    assert spans[0].advantages.tolist() == [5.0, 5.0, 0.0]
    assert spans[1].advantages.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_custom_reward_post_process_sees_episodes_as_groups_of_spans(monkeypatch):
    trajectories = [
        _trajectory(1.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(1.0, episode_index=0, group_index=0, loss_weight=0.5),
        _trajectory(1.0, episode_index=1, group_index=0),
    ]
    seen = {}

    def post_process(args, episodes):
        seen["span_counts"] = [len(spans) for spans in episodes]
        for spans in episodes:
            for index, trajectory in enumerate(spans):
                trajectory.reward = float(index)

    monkeypatch.setattr(
        "slim.ray.advantage_estimator.load_function",
        lambda _path: post_process,
    )
    estimator = AdvantageEstimator(
        _args("grpo", custom_reward_post_process_path="custom.shape", group_advantage_normalization=False)
    )

    estimator.compute_partition_training_targets([_batch(trajectories)])

    assert seen["span_counts"] == [2, 1]
    assert [trajectory.advantages[0].item() for trajectory in trajectories] == [0.0, 1.0, 0.0]


@pytest.mark.unit
def test_episode_reward_broadcasts_to_every_span():
    episode = _episode(0, 1.5, span_count=3)

    trajectories = build_dp_batches(
        [episode],
        dp_size=1,
        num_steps=1,
        loss_normalization_unit="episode",
        pad_token_id=0,
        balance_data=False,
    )[0].trajectories

    assert [trajectory.reward for trajectory in trajectories] == [1.5, 1.5, 1.5]
    assert [trajectory.loss_weight for trajectory in trajectories] == pytest.approx([1 / 3] * 3)
    assert episode.reward is None


@pytest.mark.unit
def test_setting_both_reward_levels_is_rejected():
    episode = _episode(0, 1.0)
    episode.trajectory.reward = 2.0

    with pytest.raises(ValueError, match="reward level is exclusive"):
        build_dp_batches(
            [episode],
            dp_size=1,
            num_steps=1,
            loss_normalization_unit="episode",
            pad_token_id=0,
            balance_data=False,
        )


@pytest.mark.unit
def test_missing_reward_at_both_levels_is_rejected():
    episode = Episode(trajectories=[Trajectory(token_ids=[1, 2], loss_mask=[1])], episode_index=0)
    episode.finalize_source_token_alignment()

    with pytest.raises(ValueError, match="needs a reward"):
        build_dp_batches(
            [episode],
            dp_size=1,
            num_steps=1,
            loss_normalization_unit="episode",
            pad_token_id=0,
            balance_data=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("unit", "expected"),
    [("episode", 0.25), ("trajectory", 1.0), ("token", 1.0)],
)
def test_loss_weight_follows_the_normalization_unit(unit, expected):
    episode = _episode(0, 1.0, span_count=4)

    trajectories = build_dp_batches(
        [episode],
        dp_size=1,
        num_steps=1,
        loss_normalization_unit=unit,
        pad_token_id=0,
        balance_data=False,
    )[0].trajectories

    assert [trajectory.loss_weight for trajectory in trajectories] == pytest.approx([expected] * 4)


@pytest.mark.unit
def test_dp_split_gives_every_rank_the_same_document_count_per_step():
    # Lopsided span counts: 4 spans, then three single-span attempts.
    episodes = [_episode(0, 1.0, span_count=4)] + [_episode(index, 1.0) for index in range(1, 4)]

    batches = build_dp_batches(
        episodes,
        dp_size=2,
        num_steps=2,
        loss_normalization_unit="episode",
        pad_token_id=0,
        balance_data=True,
    )

    assert [len(batch) for batch in batches] == [4, 4]
    assert all(batch.num_steps == 2 and batch.step_size == 2 for batch in batches)
    assert sum(len(batch) for batch in batches) == 8


@pytest.mark.unit
def test_dp_split_pads_to_a_multiple_of_ranks_and_steps():
    episodes = [_episode(index, 1.0) for index in range(3)]

    batches = build_dp_batches(
        episodes,
        dp_size=2,
        num_steps=1,
        loss_normalization_unit="episode",
        pad_token_id=7,
        balance_data=True,
    )

    trajectories = [t for batch in batches for t in batch.trajectories]
    padding = [t for t in trajectories if t.episode_index is None]
    assert [len(batch) for batch in batches] == [2, 2]
    assert len(padding) == 1
    assert padding[0].token_ids.tolist() == [7, 7]
    assert padding[0].loss_mask.tolist() == [0, 0]
    assert padding[0].loss_weight == 0.0


@pytest.mark.unit
def test_padding_carries_routing_replay_data_when_the_batch_does():
    episode = _episode(0, 1.0)
    episode.trajectory.rollout_routed_experts = torch.zeros((3, 4, 2), dtype=torch.int32)

    batches = build_dp_batches(
        [episode],
        dp_size=2,
        num_steps=1,
        loss_normalization_unit="episode",
        pad_token_id=0,
        balance_data=False,
    )

    padding = next(t for batch in batches for t in batch.trajectories if t.episode_index is None)
    assert padding.rollout_routed_experts.shape == (2, 4, 2)


@pytest.mark.unit
def test_trajectory_batch_rejects_a_step_count_that_does_not_divide_evenly():
    with pytest.raises(ValueError, match="do not divide evenly"):
        TrajectoryBatch(trajectories=[_trajectory(1.0), _trajectory(1.0), _trajectory(1.0)], num_steps=2)
