# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.topology import NeMoTopology


NUM_GPUS = 0


@pytest.mark.unit
def test_topology_derives_logical_data_parallel_size():
    topology = NeMoTopology(
        world_size=16,
        context_parallel_size=2,
        expert_model_parallel_size=4,
    )

    assert topology.logical_dp_size == 8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"world_size": 8, "context_parallel_size": 3}, "not divisible"),
        (
            {
                "world_size": 8,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 3,
            },
            "expert parallel size",
        ),
        ({"world_size": 0}, "world_size"),
    ],
)
def test_topology_rejects_invalid_meshes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NeMoTopology(**kwargs)


@pytest.mark.unit
def test_topology_uses_automodel_default_mixed_precision(monkeypatch):
    from nemo_automodel.components.distributed.config import DistributedSetup

    captured = {}

    def fake_build(cls, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(DistributedSetup, "build", classmethod(fake_build))
    topology = NeMoTopology(world_size=8, context_parallel_size=2)
    args = SimpleNamespace(
        activation_checkpointing=False,
        defer_fsdp_grad_sync=True,
        distributed_timeout_minutes=10,
    )

    topology.build(args)

    policy = captured["strategy"].mp_policy
    assert policy.param_dtype is torch.bfloat16
    assert policy.output_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.cast_forward_inputs is True
    assert captured["strategy"].reshard_after_forward is True
    assert captured["parallelism_sizes"].dp_replicate_size == 1
