"""Distributed checkpoints for the Megatron backend."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    """Slim state stored alongside MCore sharded training state."""

    rollout_id: int
    next_rollout_id: int
    global_step: int
    world_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    megatron_core_revision: str | None = None
    megatron_bridge_revision: str | None = None


def _require_single_chunk(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
    if len(model) != 1:
        raise ValueError(
            "The slim Megatron backend requires exactly one model chunk per pipeline stage; "
            f"received {len(model)}."
        )
    return model[0]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    seen: set[int] = set()
    while hasattr(model, "module") and id(model) not in seen:
        seen.add(id(model))
        wrapped = model.module
        if wrapped is None or wrapped is model:
            break
        model = wrapped
    return model


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
        try:
            from megatron.core.tensor_parallel import get_cuda_rng_tracker

            state["mcore_cuda_rng_tracker"] = get_cuda_rng_tracker().get_states()
        except (ImportError, RuntimeError):
            pass
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
    if "mcore_cuda_rng_tracker" in state:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker

        get_cuda_rng_tracker().set_states(state["mcore_cuda_rng_tracker"])


def build_sharded_state_dict(
    *,
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    scheduler: Any,
    metadata: CheckpointMetadata,
    is_loading: bool,
    use_distributed_optimizer: bool,
    pg_collection: Any,
) -> dict[str, Any]:
    """Build the local MCore sharded-state template."""

    from megatron.core.dist_checkpointing.mapping import ShardedObject

    model_chunk = _unwrap_model(_require_single_chunk(model))
    if not hasattr(model_chunk, "sharded_state_dict"):
        raise TypeError(f"{type(model_chunk).__name__} does not provide sharded_state_dict().")
    if pg_collection is None or getattr(pg_collection, "dp_cp", None) is None:
        raise ValueError("Checkpoint sharded state requires a DP-CP process group.")

    sharded_metadata = {
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": pg_collection.dp_cp,
        "singleton_local_shards": False,
    }
    if use_distributed_optimizer:
        sharded_metadata["distrib_optim_sharding_type"] = "dp_reshardable"
    state: dict[str, Any] = {
        "model": model_chunk.sharded_state_dict(metadata=sharded_metadata),
        "slim": asdict(metadata),
    }
    if optimizer is not None:
        kwargs = {
            "is_loading": is_loading,
            "metadata": sharded_metadata,
        }
        state["optimizer"] = optimizer.sharded_state_dict(state, **kwargs)
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    state["rng_state"] = ShardedObject(
        "slim_rng_state",
        _rng_state(),
        (world_size,),
        (rank,),
        replica_id=0,
    )
    return state


class MegatronCheckpointManager:
    """Save and restore one non-interleaved MCore model chunk."""

    def __init__(
        self,
        model: Sequence[torch.nn.Module],
        optimizer: Any,
        scheduler: Any,
        *,
        use_distributed_optimizer: bool,
        pg_collection: Any,
    ):
        _require_single_chunk(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_distributed_optimizer = use_distributed_optimizer
        self.pg_collection = pg_collection

    def save(self, path: str | Path, metadata: CheckpointMetadata) -> None:
        from megatron.core import dist_checkpointing

        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state = build_sharded_state_dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=metadata,
            is_loading=False,
            use_distributed_optimizer=self.use_distributed_optimizer,
            pg_collection=self.pg_collection,
        )
        dist_checkpointing.save(state, str(checkpoint_path))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load(
        self,
        path: str | Path,
        expected_topology: CheckpointMetadata,
    ) -> CheckpointMetadata:
        from megatron.core import dist_checkpointing

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        common_state = dist_checkpointing.load_common_state_dict(str(checkpoint_path))
        if "slim" not in common_state:
            raise ValueError(f"Checkpoint {checkpoint_path} has no slim metadata.")
        loaded_metadata = CheckpointMetadata(**common_state["slim"])
        self._validate_topology(expected_topology, loaded_metadata)

        template = build_sharded_state_dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=expected_topology,
            is_loading=True,
            use_distributed_optimizer=self.use_distributed_optimizer,
            pg_collection=self.pg_collection,
        )
        state = dist_checkpointing.load(template, str(checkpoint_path))

        _unwrap_model(_require_single_chunk(self.model)).load_state_dict(state["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng_state"])
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return loaded_metadata

    @staticmethod
    def _validate_topology(
        expected: CheckpointMetadata,
        loaded: CheckpointMetadata,
    ) -> None:
        fields = (
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "sequence_parallel",
        )
        changed = {field: (getattr(loaded, field), getattr(expected, field)) for field in fields if getattr(loaded, field) != getattr(expected, field)}
        if changed:
            details = ", ".join(f"{field}: {old} -> {new}" for field, (old, new) in changed.items())
            raise ValueError(f"Checkpoint topology differs from the requested topology ({details}).")
