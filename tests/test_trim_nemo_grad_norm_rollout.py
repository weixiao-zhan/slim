# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import torch

from slim.utils.types import Episode, Trajectory
from tests.trim_nemo_grad_norm_rollout import trim_record


def _trajectory(offset: int = 0) -> Trajectory:
    return Trajectory(
        token_ids=torch.arange(offset, offset + 8),
        loss_mask=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        rollout_log_probs=torch.arange(7, dtype=torch.float32),
        rollout_routed_experts=torch.arange(14).reshape(7, 2),
        reward=1.0,
        text="full",
        generated_text="full",
    )


def test_trim_record_preserves_prompt_and_slices_prediction_fields():
    episode = Episode(trajectories=[_trajectory()], status=Episode.Status.COMPLETED)

    trimmed, token_count = trim_record(dataclasses.asdict(episode), max_response_tokens=2)
    trajectory = trimmed["trajectories"][0]

    assert token_count == 6
    assert trajectory["token_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert trajectory["loss_mask"].tolist() == [0, 0, 0, 1, 1]
    assert trajectory["rollout_log_probs"].tolist() == [0, 1, 2, 3, 4]
    assert trajectory["rollout_routed_experts"].shape == (5, 2)
    assert trajectory["reward"] == 1.0
    assert trajectory["text"] is None
    assert trajectory["generated_text"] is None
    assert trimmed["max_tokens"] == 6
    assert trimmed["status"] == Episode.Status.TRUNCATED


def test_trim_record_bounds_every_trajectory_of_a_multi_trajectory_attempt():
    episode = Episode(trajectories=[_trajectory(), _trajectory(offset=100)], status=Episode.Status.COMPLETED)

    trimmed, token_count = trim_record(dataclasses.asdict(episode), max_response_tokens=2)

    assert token_count == 12
    assert [len(trajectory["token_ids"]) for trajectory in trimmed["trajectories"]] == [6, 6]
    assert trimmed["trajectories"][1]["token_ids"].tolist() == [100, 101, 102, 103, 104, 105]
    assert trimmed["max_tokens"] == 6
