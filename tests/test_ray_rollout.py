# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from slim.ray import rollout
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


@pytest.mark.unit
def test_log_rollout_data_reports_reward(monkeypatch):
    logged = {}
    monkeypatch.setattr(rollout, "_compute_episode_metrics", lambda args, episodes: {})
    monkeypatch.setattr(rollout, "_compute_perf_metrics", lambda episodes, rollout_time: {})
    monkeypatch.setattr(rollout, "_compute_rollout_log_probs_metric", lambda episodes: None)
    monkeypatch.setattr(rollout, "compute_rollout_step", lambda args, rollout_id: rollout_id)
    monkeypatch.setattr(rollout.logging_utils, "log", lambda args, metrics: logged.update(metrics))

    rollout._log_rollout_data(
        rollout_id=0,
        args=SimpleNamespace(custom_rollout_log_function_path=None),
        episodes=[Episode(reward=0.25), Episode(reward=0.75)],
        rollout_extra_metrics=None,
        rollout_time=1.0,
    )

    assert logged["rollout/reward"] == 0.5
