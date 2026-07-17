"""Topology validation for the non-interleaved Megatron backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from typing import Any


_VIRTUAL_PIPELINE_FIELDS = (
    "virtual_pipeline_model_parallel_size",
    "num_layers_per_virtual_pipeline_stage",
    "num_virtual_stages_per_pipeline_rank",
)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _provider_parallel_size(provider: Any, field_name: str) -> Any:
    value = getattr(provider, field_name, None)
    return 1 if value is None else value


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    """The MCore parallel dimensions used by one training world."""

    world_size: int
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        ):
            _positive_int(field_name, getattr(self, field_name))
        if not isinstance(self.sequence_parallel, bool):
            raise ValueError(f"sequence_parallel must be a bool, got {self.sequence_parallel!r}.")
        if self.sequence_parallel and self.tensor_model_parallel_size == 1:
            raise ValueError("sequence_parallel requires tensor_model_parallel_size greater than 1.")

        model_parallel_size = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )
        if self.world_size % model_parallel_size:
            raise ValueError(
                "world_size must be divisible by tensor_model_parallel_size * "
                "pipeline_model_parallel_size * context_parallel_size; "
                f"got world_size={self.world_size} and model_parallel_size={model_parallel_size}."
            )
        expert_parallel_size = (
            self.expert_tensor_parallel_size
            * self.expert_model_parallel_size
            * self.pipeline_model_parallel_size
        )
        if self.world_size % expert_parallel_size:
            raise ValueError(
                "world_size must be divisible by expert_tensor_parallel_size * "
                "expert_model_parallel_size * pipeline_model_parallel_size; "
                f"got world_size={self.world_size} and expert_parallel_size={expert_parallel_size}."
            )

    @property
    def data_parallel_size(self) -> int:
        """Return the data-parallel degree outside TP, PP, and CP."""

        return self.world_size // (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )


def validate_no_virtual_pipeline(provider: Any) -> None:
    """Reject every Bridge provider setting that enables virtual pipelines."""

    configured = {
        field_name: getattr(provider, field_name)
        for field_name in _VIRTUAL_PIPELINE_FIELDS
        if getattr(provider, field_name, None) is not None
    }
    if configured:
        details = ", ".join(f"{name}={value!r}" for name, value in configured.items())
        raise ValueError(f"The slim Megatron backend does not support virtual pipeline parallelism ({details}).")


def validate_gdn_head_topology(provider: Any, topology: ParallelTopology) -> None:
    """Validate the sequence-to-head exchange used by Gated DeltaNet."""

    if getattr(provider, "experimental_attention_variant", None) != "gated_delta_net":
        return

    num_key_heads = _positive_int("linear_num_key_heads", getattr(provider, "linear_num_key_heads", None))
    num_value_heads = _positive_int("linear_num_value_heads", getattr(provider, "linear_num_value_heads", None))
    head_parallel_size = topology.tensor_model_parallel_size * topology.context_parallel_size
    available_head_factor = gcd(num_key_heads, num_value_heads)
    if available_head_factor % head_parallel_size:
        raise ValueError(
            "Gated DeltaNet requires tensor_model_parallel_size * context_parallel_size "
            "to divide gcd(linear_num_key_heads, linear_num_value_heads); "
            f"got {head_parallel_size} and gcd={available_head_factor}."
        )


def validate_provider_topology(provider: Any, *, world_size: int) -> ParallelTopology:
    """Validate a finalized Bridge provider and return its immutable topology."""

    validate_no_virtual_pipeline(provider)
    topology = ParallelTopology(
        world_size=world_size,
        tensor_model_parallel_size=_provider_parallel_size(provider, "tensor_model_parallel_size"),
        pipeline_model_parallel_size=_provider_parallel_size(provider, "pipeline_model_parallel_size"),
        context_parallel_size=_provider_parallel_size(provider, "context_parallel_size"),
        expert_model_parallel_size=_provider_parallel_size(provider, "expert_model_parallel_size"),
        expert_tensor_parallel_size=_provider_parallel_size(
            provider,
            "expert_tensor_parallel_size",
        ),
        sequence_parallel=getattr(provider, "sequence_parallel", False),
    )
    if topology.expert_model_parallel_size > 1 and getattr(provider, "num_moe_experts", None) is None:
        raise ValueError("expert_model_parallel_size greater than 1 requires an MoE provider.")
    validate_gdn_head_topology(provider, topology)
    return topology


def require_single_model_chunk(model_chunks: Sequence[Any]) -> Any:
    """Return the only local model chunk and reject interleaved model lists."""

    if not isinstance(model_chunks, Sequence) or isinstance(model_chunks, (str, bytes)):
        raise TypeError("Megatron model chunks must be provided as a sequence.")
    if len(model_chunks) != 1:
        raise ValueError(
            "The slim Megatron backend requires exactly one model chunk per pipeline stage; "
            f"received {len(model_chunks)}."
        )
    return model_chunks[0]
