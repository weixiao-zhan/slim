# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
import torch

from slim.ray.rollout import _load_debug_rollout_episodes
from slim.utils.types import Episode


@pytest.mark.unit
def test_load_debug_rollout_episodes_round_trip(tmp_path):
    episode = Episode(
        tokens=torch.tensor([11, 12, 13]),
        loss_mask=torch.tensor([0, 1], dtype=torch.int32),
        reward=1.0,
        rollout_log_probs=torch.tensor([0.0, -0.5]),
        status=Episode.Status.COMPLETED,
    )
    path_template = str(tmp_path / "rollout_{rollout_id}.pt")
    torch.save(
        {
            "rollout_id": 3,
            "episodes": [dataclasses.asdict(episode)],
        },
        path_template.format(rollout_id=3),
    )

    episodes = _load_debug_rollout_episodes(path_template, 3)

    assert len(episodes) == 1
    torch.testing.assert_close(episodes[0].tokens, episode.tokens)
    torch.testing.assert_close(episodes[0].loss_mask, episode.loss_mask)
    torch.testing.assert_close(episodes[0].rollout_log_probs, episode.rollout_log_probs)
    assert episodes[0].reward == 1.0
