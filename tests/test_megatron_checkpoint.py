from dataclasses import replace

import pytest

from slim.backends.megatron.checkpoint import CheckpointMetadata, MegatronCheckpointManager


def _metadata(**kwargs):
    metadata = CheckpointMetadata(
        rollout_id=2,
        next_rollout_id=3,
        global_step=10,
        world_size=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=2,
        sequence_parallel=True,
    )
    return replace(metadata, **kwargs)


def test_checkpoint_rejects_multiple_model_chunks():
    with pytest.raises(ValueError, match="exactly one model chunk"):
        MegatronCheckpointManager(
            [object(), object()],
            None,
            None,
            use_distributed_optimizer=True,
            pg_collection=object(),
        )


def test_checkpoint_topology_must_match_by_default():
    with pytest.raises(ValueError, match="context_model_parallel_size|context_parallel_size"):
        MegatronCheckpointManager._validate_topology(
            _metadata(),
            _metadata(context_parallel_size=1),
        )


def test_checkpoint_rejects_data_parallel_world_size_changes():
    with pytest.raises(ValueError, match="world_size"):
        MegatronCheckpointManager._validate_topology(
            _metadata(),
            _metadata(world_size=16),
        )
