# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from slim.backends.nemo.grad_clip import _all_reduce_scalar, clip_cpu_offloaded_grad_norm


@pytest.mark.unit
def test_clip_cpu_offloaded_grad_norm_reports_preclip_norm():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    norm = clip_cpu_offloaded_grad_norm([first, second], max_norm=6.5)

    assert norm == pytest.approx(13.0)
    assert torch.linalg.vector_norm(torch.cat([first.grad, second.grad])).item() == pytest.approx(6.5)


@pytest.mark.unit
def test_all_reduce_scalar_stages_cpu_value_for_nccl():
    value = torch.tensor(3.0)
    staged = torch.tensor(3.0)

    with (
        patch("slim.backends.nemo.grad_clip.dist.get_backend", return_value="nccl"),
        patch("slim.backends.nemo.grad_clip.torch.cuda.current_device", return_value=0),
        patch.object(value, "to", return_value=staged) as to_device,
        patch("slim.backends.nemo.grad_clip.dist.all_reduce") as all_reduce,
    ):
        _all_reduce_scalar(value, torch.distributed.ReduceOp.SUM, object())

    to_device.assert_called_once_with(device=0)
    all_reduce.assert_called_once()
    assert value.item() == 3.0
