# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from slim.backends.nemo.lr_scheduler import NeMoLRScheduler


NUM_GPUS = 0


@pytest.mark.unit
def test_scheduler_starts_parameter_groups_at_independent_steps():
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameters[0]], "max_lr": 3e-4, "start_step": 0},
            {"params": [parameters[1]], "max_lr": 2e-5, "start_step": 1},
            {"params": [parameters[2]], "max_lr": 4e-6, "start_step": 2},
        ],
        lr=1e-6,
    )
    scheduler = NeMoLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_decay_style="linear",
        decay_start=3,
    )

    assert scheduler.get_last_lr() == [3e-4, 0.0, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 4e-6]


def _scheduler(lr_warmup_steps, lr_decay_steps, decay_start, lr_decay_style="linear"):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    return NeMoLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.0,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=lr_decay_style,
        decay_start=decay_start,
    )


def _run(scheduler, num_steps):
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(num_steps):
        scheduler.optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


@pytest.mark.unit
def test_scheduler_runs_warmup_constant_and_decay_phases():
    lrs = _run(_scheduler(lr_warmup_steps=2, lr_decay_steps=4, decay_start=5), 10)
    assert lrs == pytest.approx([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0])


@pytest.mark.unit
def test_scheduler_keeps_max_lr_without_decay_steps():
    lrs = _run(_scheduler(lr_warmup_steps=0, lr_decay_steps=0, decay_start=3), 6)
    assert lrs == pytest.approx([1.0] * 7)


@pytest.mark.unit
def test_start_lr_decay_moves_decay_start_earlier_only():
    scheduler = _scheduler(lr_warmup_steps=0, lr_decay_steps=4, decay_start=20)
    _run(scheduler, 3)
    assert scheduler.start_lr_decay() == 4
    _run(scheduler, 1)
    # A later call keeps the decay start and returns the remaining decay steps.
    assert scheduler.start_lr_decay() == 3
    assert scheduler.decay_start == 3
    assert _run(scheduler, 3)[1:] == pytest.approx([0.5, 0.25, 0.0])


@pytest.mark.unit
def test_start_lr_decay_waits_for_warmup_end():
    scheduler = _scheduler(lr_warmup_steps=4, lr_decay_steps=2, decay_start=20)
    _run(scheduler, 1)
    assert scheduler.start_lr_decay() == 5
    assert scheduler.decay_start == 4


@pytest.mark.unit
def test_start_lr_decay_survives_state_dict_reload():
    scheduler = _scheduler(lr_warmup_steps=0, lr_decay_steps=4, decay_start=20)
    _run(scheduler, 2)
    scheduler.start_lr_decay()
    _run(scheduler, 1)

    resumed = _scheduler(lr_warmup_steps=0, lr_decay_steps=4, decay_start=20)
    resumed.load_state_dict(scheduler.state_dict())
    assert resumed.start_lr_decay() == 3
    assert _run(resumed, 2)[1:] == _run(scheduler, 2)[1:]
