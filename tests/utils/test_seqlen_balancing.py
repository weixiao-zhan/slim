# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from slim.utils.seqlen_balancing import build_token_budget_partitions


@pytest.mark.unit
def test_token_budget_partitions_remain_safe_when_pack_count_is_synchronized():
    lengths = [2, 7, 7, 8, 8]

    partitions = build_token_budget_partitions(lengths, 16, num_packs=3)

    assert len(partitions) == 3
    assert sorted(index for partition in partitions for index in partition) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in partition) <= 16 for partition in partitions)
