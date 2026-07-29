# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from slim.backends.nemo.data_packing import (
    build_document_ids,
    build_model_batch,
    build_source_labels,
    build_token_budget_partitions,
    build_token_slot_fields,
    edge_to_token_slots,
    pack_sequences,
    token_slots_to_edges,
    unpack_sequences,
)
from slim.utils.types import Episode


NUM_GPUS = 0


def _episode(tokens, loss_mask, reward=1.0):
    episode = Episode(tokens=tokens, loss_mask=loss_mask, reward=reward)
    episode._advantages = [0.5] * episode.num_edges
    episode._returns = [1.5] * episode.num_edges
    return episode


@pytest.mark.unit
def test_source_labels_do_not_cross_document_boundaries():
    tokens = torch.tensor([10, 11, 12, 20, 21])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    labels = build_source_labels(tokens, cu_seqlens)

    assert labels.tolist() == [11, 12, -100, 21, -100]


@pytest.mark.unit
def test_edge_token_slot_round_trip_preserves_trailing_dimensions():
    values = torch.arange(15).reshape(5, 3)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    slots = edge_to_token_slots(values, cu_seqlens, fill=-1, total_length=8)

    assert slots.shape == (8, 3)
    assert slots[2].tolist() == [-1, -1, -1]
    assert slots[6].tolist() == [-1, -1, -1]
    assert slots[7].tolist() == [-1, -1, -1]
    torch.testing.assert_close(token_slots_to_edges(slots, cu_seqlens), values)


@pytest.mark.unit
def test_custom_mismatch_metrics_are_built_as_token_slot_fields():
    pack = {
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "_mismatch_metrics": {
            "rs_keep": torch.tensor([1.0, 0.0, 0.5]),
        },
    }

    fields = build_token_slot_fields(pack)

    assert fields["mismatch/rs_keep"].tolist() == [[1.0, 0.0, 0.0, 0.5, 0.0]]


@pytest.mark.unit
def test_document_ids_include_zero_only_for_physical_padding():
    ids = build_document_ids(torch.tensor([0, 2, 5], dtype=torch.int32), total_length=7)

    assert ids.tolist() == [1, 1, 2, 2, 2, 0, 0]


@pytest.mark.unit
def test_pack_sequences_stays_on_cpu_and_emits_indexed_mask_for_multiple_documents():
    episodes = [
        _episode([1, 2, 3], [0, 1]),
        _episode([4, 5], [1], reward=2.0),
    ]

    pack = pack_sequences(episodes)[0]
    batch = build_model_batch(pack)

    assert all(not value.is_cuda for value in pack.values() if isinstance(value, torch.Tensor))
    assert [episode["reward"] for episode in unpack_sequences(pack)] == [1.0, 2.0]
    assert batch["input_ids"].shape == (1, 5)
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 2, 2]]
    assert batch["labels"].tolist() == [[2, 3, -100, 5, -100]]


@pytest.mark.unit
def test_single_document_uses_the_same_indexed_batch_contract():
    pack = pack_sequences([_episode([1, 2, 3], [1, 1])])[0]

    batch = build_model_batch(pack)

    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1]]
    assert batch["labels"].tolist() == [[2, 3, -100]]


@pytest.mark.unit
def test_pack_sequences_validates_every_edge_field():
    episode = _episode([1, 2, 3], [1])

    with pytest.raises(ValueError, match="loss_mask length"):
        pack_sequences([episode])


@pytest.mark.unit
def test_token_budget_partitions_remain_safe_when_pack_count_is_synchronized():
    lengths = [2, 7, 7, 8, 8]

    partitions = build_token_budget_partitions(lengths, 16, num_packs=3)

    assert len(partitions) == 3
    assert sorted(index for partition in partitions for index in partition) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in partition) <= 16 for partition in partitions)
