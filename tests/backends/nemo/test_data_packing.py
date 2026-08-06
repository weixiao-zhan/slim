# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from slim.backends.nemo.data_packing import (
    build_document_ids,
    build_model_batch,
    build_source_labels,
    build_token_budget_partitions,
    build_training_fields,
    fill_document_terminal_slots,
    pack_sequences,
    unpack_sequences,
    update_packed_targets,
)
from slim.utils.types import Trajectory


NUM_GPUS = 0


def _trajectory(token_ids, loss_mask, reward=1.0, loss_weight=1.0):
    trajectory = Trajectory(
        token_ids=token_ids,
        loss_mask=loss_mask,
        reward=reward,
        loss_weight=loss_weight,
    )
    trajectory.finalize_source_token_alignment()
    trajectory.set_train_targets(
        [0.5] * (len(token_ids) - 1) + [0.0],
        values=[0.25] * (len(token_ids) - 1) + [0.0],
        value_targets=[1.5] * (len(token_ids) - 1) + [0.0],
    )
    return trajectory


@pytest.mark.unit
def test_source_labels_do_not_cross_document_boundaries():
    tokens = torch.tensor([10, 11, 12, 20, 21])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    labels = build_source_labels(tokens, cu_seqlens)

    assert labels.tolist() == [11, 12, -100, 21, -100]


@pytest.mark.unit
def test_fill_document_terminal_slots_preserves_trailing_dimensions():
    values = torch.arange(21).reshape(7, 3)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    slots = fill_document_terminal_slots(values, cu_seqlens, fill=-1)

    assert slots.shape == (7, 3)
    assert slots[2].tolist() == [-1, -1, -1]
    assert slots[6].tolist() == [-1, -1, -1]
    torch.testing.assert_close(slots[:2], values[:2])
    torch.testing.assert_close(slots[3:6], values[3:6])


@pytest.mark.unit
def test_custom_mismatch_metrics_preserve_source_token_alignment():
    pack = {
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "_mismatch_metrics": {
            "rs_keep": torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0]),
        },
    }

    fields = build_training_fields(pack)

    assert fields["mismatch/rs_keep"].tolist() == [[1.0, 0.0, 0.0, 0.5, 0.0]]


@pytest.mark.unit
def test_document_ids_include_zero_only_for_physical_padding():
    ids = build_document_ids(torch.tensor([0, 2, 5], dtype=torch.int32), total_length=7)

    assert ids.tolist() == [1, 1, 2, 2, 2, 0, 0]


@pytest.mark.unit
def test_pack_sequences_stays_on_cpu_and_emits_indexed_mask_for_multiple_documents():
    trajectories = [
        _trajectory([1, 2, 3], [0, 1]),
        _trajectory([4, 5], [1], reward=2.0),
    ]

    pack = pack_sequences(trajectories)[0]
    batch = build_model_batch(pack)

    assert all(not value.is_cuda for value in pack.values() if isinstance(value, torch.Tensor))
    assert [document["reward"] for document in unpack_sequences(pack)] == [1.0, 2.0]
    assert batch["input_ids"].shape == (1, 5)
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 2, 2]]
    assert batch["labels"].tolist() == [[2, 3, -100, 5, -100]]


@pytest.mark.unit
def test_single_document_uses_the_same_indexed_batch_contract():
    pack = pack_sequences([_trajectory([1, 2, 3], [1, 1])])[0]

    batch = build_model_batch(pack)

    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1]]
    assert batch["labels"].tolist() == [[2, 3, -100]]


@pytest.mark.unit
def test_pack_sequences_rejects_unfinalized_trajectory_fields():
    trajectory = Trajectory(token_ids=[1, 2, 3], loss_mask=[1])

    with pytest.raises(TypeError, match="finalized tensors"):
        pack_sequences([trajectory])


@pytest.mark.unit
def test_pack_sequences_allows_targets_to_be_attached_after_precompute():
    trajectories = [
        Trajectory(token_ids=[1, 2, 3], loss_mask=[0, 1], reward=1.0),
        Trajectory(token_ids=[4, 5], loss_mask=[1], reward=2.0),
    ]
    for trajectory in trajectories:
        trajectory.finalize_source_token_alignment()
    packs = pack_sequences(trajectories, partitions=[[1, 0]])

    assert "advantages" not in packs[0]
    assert "old_values" not in packs[0]
    assert "value_targets" not in packs[0]

    trajectories[0].set_train_targets(
        [0.0, 0.5, 0.0],
        values=[0.1, 0.2, 0.0],
        value_targets=[0.3, 0.4, 0.0],
    )
    trajectories[1].set_train_targets([1.0, 0.0], values=[0.6, 0.0], value_targets=[0.7, 0.0])
    update_packed_targets(packs, trajectories)

    assert packs[0]["advantages"].tolist() == [1.0, 0.0, 0.0, 0.5, 0.0]
    assert packs[0]["old_values"].tolist() == pytest.approx([0.6, 0.0, 0.1, 0.2, 0.0])
    assert packs[0]["value_targets"].tolist() == pytest.approx([0.7, 0.0, 0.3, 0.4, 0.0])


@pytest.mark.unit
def test_update_packed_targets_rejects_partial_targets():
    trajectories = [
        Trajectory(token_ids=[1, 2], loss_mask=[1], reward=1.0),
        Trajectory(token_ids=[3, 4], loss_mask=[1], reward=2.0),
    ]
    for trajectory in trajectories:
        trajectory.finalize_source_token_alignment()
    packs = pack_sequences(trajectories)
    trajectories[0].set_train_targets([1.0, 0.0])

    with pytest.raises(ValueError, match="advantages must be present for every trajectory"):
        update_packed_targets(packs, trajectories)


@pytest.mark.unit
def test_pack_carries_per_document_loss_weights_in_partition_order():
    trajectories = [
        _trajectory([1, 2, 3], [0, 1], loss_weight=0.5),
        _trajectory([4, 5], [1], loss_weight=1.0),
    ]

    pack = pack_sequences(trajectories, partitions=[[1, 0]])[0]

    assert pack["loss_weights"] == [1.0, 0.5]
    assert [document["loss_weights"] for document in unpack_sequences(pack)] == [1.0, 0.5]


@pytest.mark.unit
def test_token_budget_partitions_remain_safe_when_pack_count_is_synchronized():
    lengths = [2, 7, 7, 8, 8]

    partitions = build_token_budget_partitions(lengths, 16, num_packs=3)

    assert len(partitions) == 3
    assert sorted(index for partition in partitions for index in partition) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in partition) <= 16 for partition in partitions)
