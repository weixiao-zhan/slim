# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.checkpoint import (
    _initialize_missing_adamw_state,
    resolve_checkpoint_dir,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_resolve_checkpoint_dir_uses_tracker_or_explicit_step(tmp_path):
    first = tmp_path / "iter_0000003"
    second = tmp_path / "iter_0000007"
    first.mkdir()
    second.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")

    assert resolve_checkpoint_dir(str(tmp_path), SimpleNamespace(ckpt_step=None)) == second
    assert resolve_checkpoint_dir(str(tmp_path), SimpleNamespace(ckpt_step=3)) == first
    assert resolve_checkpoint_dir(str(first), SimpleNamespace(ckpt_step=None)) == first


@pytest.mark.unit
def test_checkpoint_initializes_adamw_state_for_parameters_without_gradients():
    used = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.AdamW([used, unused])
    used.sum().backward()
    optimizer.step()

    assert optimizer.state[used]
    assert not optimizer.state[unused]

    _initialize_missing_adamw_state(optimizer)

    assert set(optimizer.state[unused]) == {"step", "exp_avg", "exp_avg_sq"}
    assert optimizer.state[unused]["step"].item() == 0
    torch.testing.assert_close(optimizer.state[unused]["exp_avg"], torch.zeros_like(unused))
    torch.testing.assert_close(optimizer.state[unused]["exp_avg_sq"], torch.zeros_like(unused))


@pytest.mark.unit
def test_checkpoint_does_not_reallocate_existing_adamw_state(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()

    def fail_zeros_like(*args, **kwargs):
        raise AssertionError("existing AdamW state must not be reallocated")

    monkeypatch.setattr("slim.backends.nemo.checkpoint.torch.zeros_like", fail_zeros_like)

    _initialize_missing_adamw_state(optimizer)
