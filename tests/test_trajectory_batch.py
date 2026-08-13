# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flatten-pad-partition boundary between rollout and training."""

import itertools

import pytest
import torch

from slim.backends.nemo.data_packing import pack_sequences, unpack_sequences
from slim.backends.nemo.loss import count_global_denominators, reduce_weighted_sequence_mean
from slim.utils.trajectory_batch import build_dp_batches, group_by_episode
from slim.utils.types import Episode, Trajectory


NUM_GPUS = 0


def _episode(index: int, span_lengths: list[int], reward: float = 1.0, group_index: int = 0) -> Episode:
    episode = Episode(
        trajectories=[
            Trajectory(token_ids=list(range(length)), loss_mask=[1] * (length - 1))
            for length in span_lengths
        ],
        reward=reward,
        episode_index=index,
        group_index=group_index,
    )
    episode.finalize_source_token_alignment()
    return episode


def _build(episodes, *, dp_size, num_steps, unit="episode", balance_data=True):
    return build_dp_batches(
        episodes,
        dp_size=dp_size,
        num_steps=num_steps,
        loss_normalization_unit=unit,
        pad_token_id=0,
        balance_data=balance_data,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dp_size", "num_steps", "balance_data"),
    list(itertools.product([1, 2, 4], [1, 2], [True, False])),
)
def test_every_rank_holds_the_same_document_count_per_step(dp_size, num_steps, balance_data):
    """The synchronized pack count in `_packed_data` requires equal per-step work."""
    # Deliberately lopsided: one attempt made seven calls, the rest made one or two.
    episodes = [
        _episode(0, [3] * 7),
        _episode(1, [5, 9]),
        _episode(2, [4]),
        _episode(3, [11]),
        _episode(4, [6, 7, 8]),
        _episode(5, [12]),
        _episode(6, [3]),
        _episode(7, [10]),
    ]

    batches = _build(episodes, dp_size=dp_size, num_steps=num_steps, balance_data=balance_data)

    assert len(batches) == dp_size
    assert len({len(batch) for batch in batches}) == 1
    assert all(batch.num_steps == num_steps for batch in batches)
    step_sizes = {len(step) for batch in batches for _, step in batch.steps()}
    assert len(step_sizes) == 1

    # Every real span survives the split exactly once, and padding is bounded.
    real = [t for batch in batches for t in batch.trajectories if t.episode_index is not None]
    assert len(real) == sum(len(episode.trajectories) for episode in episodes)
    assert len({id(t) for t in real}) == len(real)
    padding = sum(1 for batch in batches for t in batch.trajectories if t.episode_index is None)
    assert padding < dp_size * num_steps


@pytest.mark.unit
def test_steps_slice_the_rank_local_list_contiguously():
    batches = _build([_episode(index, [4]) for index in range(8)], dp_size=2, num_steps=2)

    for batch in batches:
        starts = [start for start, _ in batch.steps()]
        assert starts == [0, batch.step_size]
        assert [t for _, step in batch.steps() for t in step] == batch.trajectories


@pytest.mark.unit
def test_regrouping_recovers_every_attempt_after_the_split():
    episodes = [_episode(0, [3, 4, 5]), _episode(1, [6]), _episode(2, [7, 8])]

    batches = _build(episodes, dp_size=2, num_steps=1)
    regrouped = group_by_episode([t for batch in batches for t in batch.trajectories])

    assert [len(spans) for spans in regrouped] == [3, 1, 2]
    assert all(len({t.episode_index for t in spans}) == 1 for spans in regrouped)


@pytest.mark.unit
def test_episode_normalization_makes_the_loss_denominator_count_attempts():
    """`count_global_denominators` must agree with the unit the baseline centers on."""
    episodes = [_episode(0, [3] * 4), _episode(1, [3]), _episode(2, [3]), _episode(3, [3])]

    batches = _build(episodes, dp_size=1, num_steps=1)
    for trajectory in batches[0].trajectories:
        trajectory.set_train_targets(torch.zeros(len(trajectory.token_ids)))
    packs = pack_sequences(batches[0].trajectories)

    weight_sum, _ = count_global_denominators(packs, dp_group=None, device="cpu")

    # Seven documents, four attempts.
    assert sum(len(pack["loss_weights"]) for pack in packs) == 7
    assert weight_sum.item() == pytest.approx(4.0)


@pytest.mark.unit
def test_a_broadcast_reward_gives_an_attempt_the_same_credit_at_any_span_count():
    """Under `episode`, reward R over k spans still contributes total credit R."""
    single = _build([_episode(0, [4])], dp_size=1, num_steps=1)[0].trajectories
    split = _build([_episode(0, [4, 4, 4, 4])], dp_size=1, num_steps=1)[0].trajectories

    def reduce(trajectories):
        # One value per token, so each document's mean is its reward.
        values = torch.cat(
            [torch.full((len(t.token_ids),), t.reward) for t in trajectories]
        ).unsqueeze(0)
        mask = torch.cat([t.loss_mask.float() for t in trajectories]).unsqueeze(0)
        document_ids = torch.cat(
            [
                torch.full((len(t.token_ids),), index + 1, dtype=torch.long)
                for index, t in enumerate(trajectories)
            ]
        ).unsqueeze(0)
        weights = torch.tensor([t.loss_weight for t in trajectories])
        return reduce_weighted_sequence_mean(
            values,
            mask,
            document_ids,
            num_documents=len(trajectories),
            global_sequences=weights.sum(),
            cp_group=None,
            loss_weights=weights,
        )

    torch.testing.assert_close(reduce(single), reduce(split))
    torch.testing.assert_close(reduce(single), torch.tensor(1.0))


@pytest.mark.unit
def test_padding_documents_pack_and_reduce_inertly():
    episodes = [_episode(0, [4]), _episode(1, [4]), _episode(2, [4])]

    batches = _build(episodes, dp_size=2, num_steps=1)
    trajectories = batches[0].trajectories + batches[1].trajectories
    for trajectory in trajectories:
        trajectory.set_train_targets(torch.zeros(len(trajectory.token_ids)))
    packs = pack_sequences(trajectories)

    weight_sum, token_count = count_global_denominators(packs, dp_group=None, device="cpu")

    assert len(trajectories) == 4
    assert weight_sum.item() == pytest.approx(3.0)
    # Three real spans of four tokens each contribute three unmasked predictions apiece.
    assert token_count.item() == 9


@pytest.mark.unit
def test_group_statistics_survive_spans_landing_on_different_ranks():
    episodes = [
        _episode(0, [3, 4], reward=0.0, group_index=0),
        _episode(1, [5], reward=1.0, group_index=0),
        _episode(2, [6], reward=0.0, group_index=1),
        _episode(3, [7, 8, 9], reward=1.0, group_index=1),
    ]

    batches = _build(episodes, dp_size=2, num_steps=1)
    regrouped = group_by_episode([t for batch in batches for t in batch.trajectories])

    for spans in regrouped:
        assert len({t.group_index for t in spans}) == 1
    by_group: dict[int, set[int]] = {}
    for spans in regrouped:
        by_group.setdefault(spans[0].group_index, set()).add(spans[0].episode_index)
    assert by_group == {0: {0, 1}, 1: {2, 3}}


@pytest.mark.unit
def test_critic_value_round_trip_preserves_rank_local_order():
    """`compute_values` reconstructs positionally, so the two sides must agree.

    Each span is given a distinctive constant, so any shift in the mapping between
    a pack's `_document_indices` and the rank-local list surfaces as a mismatch.
    """
    from types import SimpleNamespace

    from slim.backends.nemo.data_packing import fill_document_terminal_slots
    from slim.ray.advantage_estimator import AdvantageEstimator
    from slim.utils.seqlen_balancing import build_token_budget_partitions

    episodes = [_episode(0, [5, 7]), _episode(1, [4]), _episode(2, [6, 9, 5])]
    batches = _build(episodes, dp_size=2, num_steps=1)

    payloads = []
    for rank, batch in enumerate(batches):
        lengths = [len(t.token_ids) for t in batch.trajectories]
        packs = pack_sequences(
            batch.trajectories,
            partitions=build_token_budget_partitions(lengths, 16),
        )
        for pack in packs:
            # The critic zeroes each document's terminal slot, which has no prediction.
            pack["cur_values"] = fill_document_terminal_slots(
                torch.cat(
                    [torch.full((lengths[index],), float(index + 100 * rank)) for index in pack["_document_indices"]]
                ),
                pack["cu_seqlens"],
            )
        values = [None] * len(batch)
        for pack in packs:
            for index, document in zip(pack["_document_indices"], unpack_sequences(pack), strict=True):
                values[index] = document["cur_values"]
        assert all(value is not None for value in values)
        payloads.append({"dp_rank": rank, "cp_rank": 0, "values": values})

    args = SimpleNamespace(
        advantage_estimator="ppo_gae",
        custom_reward_post_process_path=None,
        gamma=1.0,
        lambd=1.0,
        group_advantage_normalization=True,
        group_advantage_std_normalization=False,
        normalize_advantages=False,
    )
    AdvantageEstimator(args).compute_partition_training_targets(batches, payloads)

    for rank, batch in enumerate(batches):
        for slot, trajectory in enumerate(batch.trajectories):
            marker = float(slot + 100 * rank)
            assert trajectory.values[:-1].unique().tolist() == [marker]
            assert trajectory.values[-1] == 0.0


@pytest.mark.unit
def test_multi_trajectory_generate_produces_two_flattenable_spans(monkeypatch):
    """The agentic generate function must leave every span edge-aligned."""
    import asyncio

    import tests.multi_trajectory_generate as multi_trajectory

    async def fake_span(state, episode):
        trajectory = episode.trajectory
        generated = list(range(900, 910))
        trajectory.token_ids.extend(generated)
        trajectory.loss_mask.extend([1] * len(generated))
        trajectory.rollout_log_probs.extend([-0.1] * len(generated))
        episode.status = Episode.Status.COMPLETED
        return episode

    monkeypatch.setattr(multi_trajectory, "generate_span", fake_span)

    episode = Episode.from_example({"prompt": "p"})
    episode.trajectory.token_ids = list(range(200))
    episode.trajectory.loss_mask = [0] * 199
    episode.trajectory.rollout_log_probs = [0.0] * 199
    episode.max_tokens = 4096

    episode = asyncio.run(multi_trajectory.generate(None, episode))

    first, second = episode.trajectories
    # The second span starts from a compressed view of the first, not an extension.
    assert second.token_ids[:64] == first.token_ids[-64:]
    for trajectory in episode.trajectories:
        assert len(trajectory.loss_mask) == len(trajectory.token_ids) - 1
        assert len(trajectory.rollout_log_probs) == len(trajectory.token_ids) - 1

    episode.reward = 1.0
    episode.episode_index = 0
    episode.group_index = 0
    episode.finalize_source_token_alignment()
    batch = _build([episode], dp_size=1, num_steps=1)[0]

    assert [t.loss_weight for t in batch.trajectories] == [0.5, 0.5]
    assert [t.reward for t in batch.trajectories] == [1.0, 1.0]


@pytest.mark.unit
def test_flatten_rejects_an_episode_with_no_spans():
    episode = Episode(trajectories=[], reward=1.0, episode_index=0)

    with pytest.raises(ValueError, match="has no trajectories"):
        _build([episode], dp_size=1, num_steps=1)


@pytest.mark.unit
def test_flatten_rejects_an_unknown_normalization_unit():
    with pytest.raises(ValueError, match="unknown loss normalization unit"):
        _build([_episode(0, [3])], dp_size=1, num_steps=1, unit="sequence")
