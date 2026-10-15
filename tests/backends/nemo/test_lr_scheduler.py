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
        lr_decay_steps=3,
        lr_decay_style="constant",
    )

    assert scheduler.get_last_lr() == [3e-4, 0.0, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 4e-6]
