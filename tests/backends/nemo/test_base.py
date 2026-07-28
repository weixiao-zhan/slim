# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace

import pytest

from slim.backends.nemo import base
from slim.backends.nemo.base import NeMoTrainer


@pytest.mark.unit
def test_debug_rollout_only_train_does_not_touch_trainer_state():
    class PlaceholderTrainer:
        args = SimpleNamespace(debug_rollout_only=True)

        def wake_up(self):
            raise AssertionError("debug rollout-only trainer must remain uninitialized")

    NeMoTrainer.train(PlaceholderTrainer(), rollout_id=0, rollout_data_ref=None)


@pytest.mark.unit
def test_train_pairs_critic_values_by_global_rank(monkeypatch):
    captured = {}

    class PlaceholderTrainer:
        args = SimpleNamespace(debug_rollout_only=False)
        dp_rank = 1
        dp_size = 2
        _pending_episodes = ["episode"]
        _pending_packed_batches = ["pack"]
        _pending_grad_accum = [1]

        def wake_up(self):
            pass

        def sleep(self):
            pass

        def _train_core(self, rollout_id, episodes, values, packed_batches, grad_accum):
            captured["values"] = values

    monkeypatch.setattr(base.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(base.ray, "get", lambda value: value)
    monkeypatch.setattr(base, "inverse_timer", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(base, "timer", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(base.train_metric_utils, "log_perf_data_raw", lambda **_kwargs: None)
    monkeypatch.setattr(base, "clear_memory", lambda: None)

    NeMoTrainer.train(
        PlaceholderTrainer(),
        rollout_id=0,
        rollout_data_ref=None,
        values_refs=["rank-0", "rank-1", "rank-2", "rank-3"],
    )

    assert captured["values"] == "rank-2"
