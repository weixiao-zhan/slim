# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from slim.backends.nemo import base
from slim.backends.nemo.base import (
    _clear_inactive_optimizer_grads,
)


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
