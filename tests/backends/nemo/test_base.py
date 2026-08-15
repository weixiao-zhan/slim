# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace

import pytest

from slim.backends.nemo import base
from slim.backends.nemo.base import (
    NeMoTrainer,
    _assigned_cuda_device,
    _bind_default_process_group_device,
    _clear_inactive_optimizer_grads,
)


@pytest.mark.unit
def test_assigned_cuda_device_uses_local_id_for_single_visible_gpu(monkeypatch):
    monkeypatch.setattr(base.ray, "get_gpu_ids", lambda: [6])
    monkeypatch.setattr(base.torch.cuda, "device_count", lambda: 1)

    assert _assigned_cuda_device(rank=1) == 0


@pytest.mark.unit
def test_assigned_cuda_device_falls_back_without_ray_gpu(monkeypatch):
    monkeypatch.setattr(base.ray, "get_gpu_ids", lambda: [])
    monkeypatch.setattr(base.torch.cuda, "device_count", lambda: 8)

    assert _assigned_cuda_device(rank=10) == 2


@pytest.mark.unit
def test_assigned_cuda_device_rejects_invalid_ray_gpu_id(monkeypatch):
    monkeypatch.setattr(base.ray, "get_gpu_ids", lambda: [8])
    monkeypatch.setattr(base.torch.cuda, "device_count", lambda: 8)

    with pytest.raises(RuntimeError, match="Ray assigned GPU 8"):
        _assigned_cuda_device(rank=0)


@pytest.mark.unit
def test_bind_default_process_group_device_uses_cuda_tensor(monkeypatch):
    calls = {}
    binding = object()

    monkeypatch.setattr(
        base.torch,
        "zeros",
        lambda size, device: calls.update(size=size, device=device) or binding,
    )
    monkeypatch.setattr(base.dist, "all_reduce", lambda tensor: calls.update(tensor=tensor))

    device = base.torch.device("cuda", 6)
    _bind_default_process_group_device(device)

    assert calls == {"size": 1, "device": device, "tensor": binding}


@pytest.mark.unit
def test_inactive_optimizer_group_does_not_accumulate_adam_state():
    active = base.torch.nn.Parameter(base.torch.ones(1))
    delayed = base.torch.nn.Parameter(base.torch.ones(1))
    optimizer = base.torch.optim.AdamW(
        [
            {"params": [active], "start_step": 0},
            {"params": [delayed], "start_step": 1},
        ]
    )
    (active + delayed).sum().backward()

    _clear_inactive_optimizer_grads(optimizer, lr_step=0)
    optimizer.step()

    assert optimizer.state[active]
    assert delayed not in optimizer.state


@pytest.mark.unit
def test_debug_rollout_only_train_does_not_touch_trainer_state():
    class PlaceholderTrainer:
        args = SimpleNamespace(debug_rollout_only=True)

        def wake_up(self):
            raise AssertionError("debug rollout-only trainer must remain uninitialized")

    NeMoTrainer.train(PlaceholderTrainer(), rollout_id=0, rollout_data_ref=None)


@pytest.mark.unit
def test_train_updates_precomputed_packs_with_train_targets(monkeypatch):
    captured = {}
    episodes_with_targets = [object()]

    class PlaceholderTrainer:
        args = SimpleNamespace(debug_rollout_only=False)
        dp_rank = 1
        dp_size = 2
        _precomputed_packed_data = (["pack"], [1])

        def wake_up(self):
            pass

        def sleep(self):
            pass

        _take_packed_data = NeMoTrainer._take_packed_data

        def _train_core(self, rollout_id, packed_batches, grad_accum):
            captured["packed_batches"] = packed_batches
            captured["grad_accum"] = grad_accum

    monkeypatch.setattr(base, "process_rollout_data", lambda refs, dp_rank, dp_size: episodes_with_targets)
    monkeypatch.setattr(
        base,
        "update_packed_targets",
        lambda packs, episodes: captured.update(target_packs=packs, target_episodes=episodes),
    )
    monkeypatch.setattr(base.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(base, "inverse_timer", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(base, "timer", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(base.train_metric_utils, "log_perf_data_raw", lambda **_kwargs: None)
    monkeypatch.setattr(base, "clear_memory", lambda: None)

    NeMoTrainer.train(
        PlaceholderTrainer(),
        rollout_id=0,
        rollout_data_ref=None,
    )

    assert captured == {
        "packed_batches": ["pack"],
        "grad_accum": [1],
        "target_packs": ["pack"],
        "target_episodes": episodes_with_targets,
    }
