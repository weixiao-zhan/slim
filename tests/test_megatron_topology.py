from __future__ import annotations

from types import SimpleNamespace

import pytest

from slim.backends.megatron.topology import (
    ParallelTopology,
    require_single_model_chunk,
    validate_provider_topology,
)


pytestmark = pytest.mark.unit


def _provider(**overrides):
    values = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": True,
        "virtual_pipeline_model_parallel_size": None,
        "num_moe_experts": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dense_topology_computes_data_parallel_size():
    topology = validate_provider_topology(_provider(), world_size=16)

    assert topology.data_parallel_size == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline"),
        ({"num_layers_per_virtual_pipeline_stage": 4}, "virtual pipeline"),
        ({"num_virtual_stages_per_pipeline_rank": 2}, "virtual pipeline"),
    ],
)
def test_provider_topology_rejects_every_virtual_pipeline_setting(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_provider_topology(_provider(**overrides), world_size=16)


def test_topology_rejects_invalid_world_size_and_sequence_parallel():
    with pytest.raises(ValueError, match="world_size must be divisible"):
        ParallelTopology(world_size=7, tensor_model_parallel_size=2)
    with pytest.raises(ValueError, match="sequence_parallel requires"):
        ParallelTopology(world_size=2, sequence_parallel=True)
    with pytest.raises(ValueError, match="context_model_parallel_size|context_parallel_size"):
        validate_provider_topology(_provider(context_parallel_size=0), world_size=16)


def test_provider_topology_rejects_ep_for_dense_model():
    with pytest.raises(ValueError, match="requires an MoE provider"):
        validate_provider_topology(_provider(expert_model_parallel_size=2), world_size=16)


def test_provider_topology_rejects_invalid_expert_parallel_grid():
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        validate_provider_topology(
            _provider(
                expert_model_parallel_size=3,
                expert_tensor_parallel_size=1,
                num_moe_experts=8,
            ),
            world_size=16,
        )


def test_provider_topology_validates_gdn_head_exchange():
    provider = _provider(
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        experimental_attention_variant="gated_delta_net",
        linear_num_key_heads=16,
        linear_num_value_heads=48,
    )
    topology = validate_provider_topology(provider, world_size=8)
    assert topology.tensor_model_parallel_size * topology.context_parallel_size == 8

    provider.context_parallel_size = 16
    with pytest.raises(ValueError, match="Gated DeltaNet requires"):
        validate_provider_topology(provider, world_size=32)


def test_require_single_model_chunk_rejects_virtual_chunks():
    chunk = object()
    assert require_single_model_chunk([chunk]) is chunk
    with pytest.raises(ValueError, match="exactly one model chunk"):
        require_single_model_chunk([chunk, object()])
