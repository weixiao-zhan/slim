"""Episode packing for the Megatron backend.

This module has no import-time dependency on Megatron Core or Megatron Bridge.
When Bridge is available, its THD row-packing and packed-sequence helpers are
used directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import torch

from slim.utils.types import Episode


DEFAULT_IGNORE_INDEX = -100

_VISUAL_INPUT_FIELDS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "image_sizes",
    "image_position_ids",
    "mm_token_type_ids",
)
_IMAGE_INPUT_FIELDS = frozenset(("pixel_values", "image_grid_thw", "image_sizes", "image_position_ids"))
_VIDEO_INPUT_FIELDS = frozenset(("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"))
_EDGE_FEATURE_SOURCES = {
    "advantages": "_advantages",
    "returns": "_returns",
    "rollout_log_probs": "rollout_log_probs",
    "actor_old_log_probs": "_actor_old_log_probs",
    "ref_log_probs": "_ref_log_probs",
    "old_values": "_values",
}


@dataclass(frozen=True)
class PackedRange:
    """A half-open range in a packed tensor."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"Invalid packed range [{self.start}, {self.stop}).")

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class NamedRange:
    """A named half-open range."""

    name: str
    span: PackedRange


@dataclass(frozen=True)
class TrajectoryLayout:
    """Logical, physical, and multimodal offsets for one input episode."""

    episode_index: int
    logical_tokens: PackedRange
    tokens: PackedRange
    edges: PackedRange
    physical_tokens: PackedRange
    media_ranges: tuple[NamedRange, ...]
    placeholder_ranges: tuple[NamedRange, ...]


@dataclass(frozen=True)
class _FallbackGenericVisualInputs:
    """Dependency-free equivalent of Bridge's ``GenericVisualInputs``."""

    pixel_values: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    second_per_grid_ts: torch.Tensor | None = None
    image_sizes: torch.Tensor | None = None
    image_position_ids: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        kwargs = self.as_model_kwargs()
        for key in ("pixel_values", "pixel_values_videos"):
            value = kwargs.get(key)
            if value is not None and value.dim() == 5:
                kwargs[key] = value.flatten(0, 1)
        for key in ("image_grid_thw", "video_grid_thw"):
            value = kwargs.get(key)
            if value is not None and value.dim() == 3:
                kwargs[key] = value.flatten(0, 1)
        return kwargs


@dataclass(frozen=True)
class PackedLayout:
    """One physical THD stream and all tensors aligned to it."""

    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    token_mask: torch.Tensor
    edge_mask: torch.Tensor
    edge_features: Mapping[str, torch.Tensor]
    rollout_routed_experts: torch.Tensor | None
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    cu_seqlens_q_padded: torch.Tensor | None
    cu_seqlens_kv_padded: torch.Tensor | None
    max_seqlen_q: torch.Tensor
    max_seqlen_kv: torch.Tensor
    total_tokens: int
    logical_lengths: tuple[int, ...]
    physical_lengths: tuple[int, ...]
    alignment: int
    trajectories: tuple[TrajectoryLayout, ...]
    visual_inputs: Any | None

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode], **kwargs: Any) -> PackedLayout:
        return build_packed_layout(episodes, **kwargs)

    def packed_sequence_metadata(self) -> dict[str, torch.Tensor | int]:
        metadata: dict[str, torch.Tensor | int] = {
            "cu_seqlens_q": self.cu_seqlens_q,
            "cu_seqlens_kv": self.cu_seqlens_kv,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "total_tokens": self.total_tokens,
        }
        if self.cu_seqlens_q_padded is not None:
            metadata["cu_seqlens_q_padded"] = self.cu_seqlens_q_padded
        if self.cu_seqlens_kv_padded is not None:
            metadata["cu_seqlens_kv_padded"] = self.cu_seqlens_kv_padded
        return metadata

    def as_mcore_batch(
        self,
        *,
        token_key: str = "tokens",
        include_rl_features: bool = True,
    ) -> dict[str, Any]:
        """Return the tensor dictionary consumed by a Megatron forward step."""
        if token_key not in {"tokens", "input_ids"}:
            raise ValueError("token_key must be 'tokens' or 'input_ids'.")
        batch: dict[str, Any] = {
            token_key: self.tokens,
            "labels": self.labels,
            "loss_mask": self.loss_mask,
            "attention_mask": None,
            "position_ids": self.position_ids,
            "token_mask": self.token_mask,
            "edge_mask": self.edge_mask,
            "visual_inputs": self.visual_inputs,
            **self.packed_sequence_metadata(),
        }
        if include_rl_features:
            batch.update(self.edge_features)
            if self.rollout_routed_experts is not None:
                batch["rollout_routed_experts"] = self.rollout_routed_experts
        return batch

    def get_packed_seq_params(self) -> Any:
        """Build Bridge's native ``PackedSeqParams`` object."""
        try:
            from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
        except ImportError as exc:
            raise RuntimeError("Megatron Bridge is required to build PackedSeqParams.") from exc
        return get_packed_seq_params(self.packed_sequence_metadata())

    def get_context_parallel_indices(
        self,
        *,
        cp_size: int,
        cp_rank: int,
        device: torch.device,
        cp_group: Any | None = None,
    ) -> torch.Tensor:
        """Return Bridge and MCore's native THD context-parallel indices."""
        try:
            from megatron.bridge.training.utils.packed_seq_utils import (
                get_packed_seq_cp_partition_indices,
            )
        except ImportError as exc:
            raise RuntimeError("Megatron Bridge is required to build context-parallel indices.") from exc
        return get_packed_seq_cp_partition_indices(
            self.get_packed_seq_params(),
            total_tokens=self.total_tokens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
            cp_group=cp_group,
        )


def packed_sequence_alignment(
    *,
    tensor_model_parallel_size: int,
    context_parallel_size: int,
    sequence_parallel: bool,
) -> int:
    """Return per-trajectory alignment for SP and THD zigzag CP."""
    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1.")
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be >= 1.")
    if sequence_parallel and tensor_model_parallel_size == 1:
        raise ValueError("sequence_parallel requires tensor_model_parallel_size > 1.")
    cp_alignment = 2 * context_parallel_size if context_parallel_size > 1 else 1
    sp_alignment = (
        context_parallel_size * tensor_model_parallel_size
        if sequence_parallel
        else 1
    )
    return math.lcm(cp_alignment, sp_alignment)


def build_packed_layout(
    episodes: Sequence[Episode],
    *,
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    pad_token_id: int = 0,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    sequence_length: int | None = None,
    placeholder_token_ids: Mapping[str, int] | None = None,
) -> PackedLayout:
    """Pack episodes into one MCore THD stream without rerunning a processor.

    ``placeholder_token_ids`` maps modality names such as ``image`` and
    ``video`` to the provider's placeholder token IDs. It is required for each
    visual modality present in the batch.
    """
    if not episodes:
        raise ValueError("Cannot build a PackedLayout from an empty episode list.")
    if sequence_length is not None and sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    alignment = packed_sequence_alignment(
        tensor_model_parallel_size=tensor_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )
    token_rows = _normalize_token_rows(episodes, sequence_length=sequence_length)
    device = token_rows[0].device
    edge_feature_rows = _collect_edge_feature_rows(episodes, token_rows, device=device)
    routing_rows = _collect_routing_rows(episodes, token_rows, device=device)

    rows: list[dict[str, torch.Tensor]] = []
    for index, (episode, tokens) in enumerate(zip(episodes, token_rows, strict=True)):
        edge_count = tokens.numel() - 1
        loss_edges = _as_edge_tensor(
            episode.loss_mask,
            edge_count=edge_count,
            name=f"episodes[{index}].loss_mask",
            device=device,
            dtype=torch.float32,
            default_value=1,
        )
        labels = torch.full_like(tokens, ignore_index)
        if edge_count:
            labels[:-1] = tokens[1:]
        loss_mask = torch.zeros(tokens.numel(), dtype=torch.float32, device=device)
        if edge_count:
            loss_mask[:edge_count] = loss_edges

        row = {
            "tokens": tokens,
            "position_ids": torch.arange(tokens.numel(), dtype=torch.long, device=device),
            "labels": labels,
            "loss_mask": loss_mask,
            "token_mask": torch.ones(tokens.numel(), dtype=torch.bool, device=device),
            "edge_mask": torch.cat(
                (
                    torch.ones(edge_count, dtype=torch.bool, device=device),
                    torch.zeros(1, dtype=torch.bool, device=device),
                )
            ),
        }
        for name, feature_rows in edge_feature_rows.items():
            edge_values = feature_rows[index]
            row[name] = torch.cat((edge_values, edge_values.new_zeros(1)))
        rows.append(row)

    sequence_tensor_pad_values: dict[str, int | float] = {
        "token_mask": 0,
        "edge_mask": 0,
        **{name: 0 for name in edge_feature_rows},
    }
    packed = _get_sequence_batch_builder()(
        rows,
        token_key="tokens",
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        pad_to_multiple_of=alignment,
        sequence_tensor_pad_values=sequence_tensor_pad_values,
    )

    logical_lengths = tuple(tokens.numel() for tokens in token_rows)
    physical_lengths = tuple(_ceil_to_multiple(length, alignment) for length in logical_lengths)
    physical_starts = _cumulative_starts(physical_lengths)
    media_ranges, visual_inputs = _pack_visual_inputs(episodes)
    normalized_placeholder_ids = _normalize_placeholder_token_ids(placeholder_token_ids)
    _validate_visual_placeholder_configuration(episodes, token_rows, normalized_placeholder_ids)
    trajectories = _build_trajectory_layouts(
        token_rows=token_rows,
        logical_lengths=logical_lengths,
        physical_lengths=physical_lengths,
        physical_starts=physical_starts,
        media_ranges=media_ranges,
        placeholder_token_ids=normalized_placeholder_ids,
    )
    routed_experts = _pack_routing_rows(
        routing_rows,
        physical_starts=physical_starts,
        total_tokens=packed["total_tokens"],
        device=device,
    )

    return PackedLayout(
        tokens=packed["tokens"],
        labels=packed["labels"],
        loss_mask=packed["loss_mask"],
        position_ids=packed["position_ids"],
        token_mask=packed["token_mask"],
        edge_mask=packed["edge_mask"],
        edge_features=MappingProxyType({name: packed[name] for name in edge_feature_rows}),
        rollout_routed_experts=routed_experts,
        cu_seqlens_q=packed["cu_seqlens_q"],
        cu_seqlens_kv=packed["cu_seqlens_kv"],
        cu_seqlens_q_padded=packed.get("cu_seqlens_q_padded"),
        cu_seqlens_kv_padded=packed.get("cu_seqlens_kv_padded"),
        max_seqlen_q=packed["max_seqlen_q"],
        max_seqlen_kv=packed["max_seqlen_kv"],
        total_tokens=packed["total_tokens"],
        logical_lengths=logical_lengths,
        physical_lengths=physical_lengths,
        alignment=alignment,
        trajectories=trajectories,
        visual_inputs=visual_inputs,
    )


def _normalize_token_rows(
    episodes: Sequence[Episode],
    *,
    sequence_length: int | None,
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    device: torch.device | None = None
    for index, episode in enumerate(episodes):
        tokens = torch.as_tensor(episode.tokens, dtype=torch.long)
        if tokens.dim() != 1:
            raise ValueError(f"episodes[{index}].tokens must be a 1D tensor or sequence.")
        if tokens.numel() == 0:
            raise ValueError(f"episodes[{index}].tokens cannot be empty.")
        if sequence_length is not None and tokens.numel() > sequence_length:
            raise ValueError(
                f"episodes[{index}] token length {tokens.numel()} exceeds sequence_length {sequence_length}."
            )
        if device is None:
            device = tokens.device
        elif tokens.device != device:
            raise ValueError("All episode token tensors must be on the same device.")
        rows.append(tokens)
    return rows


def _collect_edge_feature_rows(
    episodes: Sequence[Episode],
    token_rows: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    result: dict[str, list[torch.Tensor]] = {}
    for output_name, source_name in _EDGE_FEATURE_SOURCES.items():
        values = [getattr(episode, source_name, None) for episode in episodes]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"Edge feature '{source_name}' must be present for every episode or omitted from all.")
        result[output_name] = [
            _as_edge_tensor(
                value,
                edge_count=tokens.numel() - 1,
                name=f"episodes[{index}].{source_name}",
                device=device,
                dtype=torch.float32,
            )
            for index, (value, tokens) in enumerate(zip(values, token_rows, strict=True))
        ]
    return result


def _collect_routing_rows(
    episodes: Sequence[Episode],
    token_rows: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> list[torch.Tensor] | None:
    values = [episode.rollout_routed_experts for episode in episodes]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("rollout_routed_experts must be present for every episode or omitted from all.")

    rows: list[torch.Tensor] = []
    tail_shape: torch.Size | None = None
    for index, (value, tokens) in enumerate(zip(values, token_rows, strict=True)):
        row = torch.as_tensor(value, dtype=torch.int32, device=device)
        edge_count = tokens.numel() - 1
        if row.dim() < 2 or row.size(0) != edge_count:
            raise ValueError(
                f"episodes[{index}].rollout_routed_experts must have leading edge dimension {edge_count}."
            )
        if tail_shape is None:
            tail_shape = row.shape[1:]
        elif row.shape[1:] != tail_shape:
            raise ValueError("rollout_routed_experts trailing dimensions must match across episodes.")
        rows.append(row)
    return rows


def _as_edge_tensor(
    value: Any | None,
    *,
    edge_count: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    default_value: int | float | None = None,
) -> torch.Tensor:
    if value is None:
        if default_value is None:
            raise ValueError(f"{name} is required.")
        return torch.full((edge_count,), default_value, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() != 1 or tensor.numel() != edge_count:
        raise ValueError(f"{name} length {tensor.numel()} does not match edge count {edge_count}.")
    return tensor


def _pack_routing_rows(
    rows: list[torch.Tensor] | None,
    *,
    physical_starts: Sequence[int],
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if rows is None:
        return None
    packed = torch.zeros(
        (1, total_tokens, *rows[0].shape[1:]),
        dtype=torch.int32,
        device=device,
    )
    for start, row in zip(physical_starts, rows, strict=True):
        packed[0, start : start + row.size(0)] = row
    return packed


def _pack_visual_inputs(
    episodes: Sequence[Episode],
) -> tuple[list[tuple[NamedRange, ...]], Any | None]:
    per_episode = [[] for _ in episodes]
    concatenated: dict[str, torch.Tensor] = {}
    observed_keys: set[str] = set()

    for index, episode in enumerate(episodes):
        multimodal_inputs = episode.multimodal_inputs
        if multimodal_inputs is None:
            continue
        if not isinstance(multimodal_inputs, Mapping):
            raise ValueError(f"episodes[{index}].multimodal_inputs must be a mapping or None.")
        unsupported = set(multimodal_inputs) - set(_VISUAL_INPUT_FIELDS)
        if unsupported:
            keys = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported multimodal processor output keys: {keys}.")
        observed_keys.update(key for key, value in multimodal_inputs.items() if value is not None)

    for key in _VISUAL_INPUT_FIELDS:
        if key not in observed_keys:
            continue
        parts: list[torch.Tensor] = []
        offset = 0
        for index, episode in enumerate(episodes):
            value = (episode.multimodal_inputs or {}).get(key)
            if value is None:
                length = 0
            else:
                if not isinstance(value, torch.Tensor) or value.dim() < 1:
                    raise ValueError(f"episodes[{index}].multimodal_inputs['{key}'] must be a tensor with dim >= 1.")
                length = value.size(0)
                parts.append(value)
            per_episode[index].append(NamedRange(key, PackedRange(offset, offset + length)))
            offset += length
        if parts:
            try:
                concatenated[key] = torch.cat(parts, dim=0)
            except RuntimeError as exc:
                raise ValueError(f"Multimodal tensor '{key}' cannot be concatenated along dim 0.") from exc

    if not concatenated:
        return [tuple(ranges) for ranges in per_episode], None
    visual_inputs_type = _get_visual_inputs_type()
    return (
        [tuple(ranges) for ranges in per_episode],
        visual_inputs_type(**concatenated),
    )


def _normalize_placeholder_token_ids(
    placeholder_token_ids: Mapping[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    if placeholder_token_ids is None:
        return ()
    normalized = tuple((str(name), int(token_id)) for name, token_id in placeholder_token_ids.items())
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("Placeholder modality names must be unique.")
    if len({token_id for _, token_id in normalized}) != len(normalized):
        raise ValueError("Placeholder token IDs must be unique.")
    return normalized


def _validate_visual_placeholder_configuration(
    episodes: Sequence[Episode],
    token_rows: Sequence[torch.Tensor],
    placeholder_token_ids: tuple[tuple[str, int], ...],
) -> None:
    configured = dict(placeholder_token_ids)
    for index, (episode, tokens) in enumerate(zip(episodes, token_rows, strict=True)):
        keys = {key for key, value in (episode.multimodal_inputs or {}).items() if value is not None}
        for modality, modality_fields in (("image", _IMAGE_INPUT_FIELDS), ("video", _VIDEO_INPUT_FIELDS)):
            if not keys.intersection(modality_fields):
                continue
            if modality not in configured:
                raise ValueError(f"episodes[{index}] has {modality} inputs but no '{modality}' placeholder token ID.")
            if not torch.any(tokens == configured[modality]):
                raise ValueError(
                    f"episodes[{index}] has {modality} inputs but no matching '{modality}' placeholder tokens."
                )


def _build_trajectory_layouts(
    *,
    token_rows: Sequence[torch.Tensor],
    logical_lengths: Sequence[int],
    physical_lengths: Sequence[int],
    physical_starts: Sequence[int],
    media_ranges: Sequence[tuple[NamedRange, ...]],
    placeholder_token_ids: tuple[tuple[str, int], ...],
) -> tuple[TrajectoryLayout, ...]:
    logical_starts = _cumulative_starts(logical_lengths)
    layouts = []
    for index, (tokens, logical_start, physical_start, logical_length, physical_length) in enumerate(
        zip(
            token_rows,
            logical_starts,
            physical_starts,
            logical_lengths,
            physical_lengths,
            strict=True,
        )
    ):
        placeholder_ranges = []
        for name, token_id in placeholder_token_ids:
            indices = torch.nonzero(tokens == token_id, as_tuple=False).flatten().tolist()
            for start, stop in _contiguous_ranges(indices):
                placeholder_ranges.append(
                    NamedRange(name, PackedRange(physical_start + start, physical_start + stop))
                )
        placeholder_ranges.sort(key=lambda item: item.span.start)
        layouts.append(
            TrajectoryLayout(
                episode_index=index,
                logical_tokens=PackedRange(logical_start, logical_start + logical_length),
                tokens=PackedRange(physical_start, physical_start + logical_length),
                edges=PackedRange(physical_start, physical_start + logical_length - 1),
                physical_tokens=PackedRange(physical_start, physical_start + physical_length),
                media_ranges=media_ranges[index],
                placeholder_ranges=tuple(placeholder_ranges),
            )
        )
    return tuple(layouts)


def _contiguous_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges


def _cumulative_starts(lengths: Sequence[int]) -> tuple[int, ...]:
    starts = []
    offset = 0
    for length in lengths:
        starts.append(offset)
        offset += length
    return tuple(starts)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _get_sequence_batch_builder() -> Callable[..., dict[str, Any]]:
    try:
        from megatron.bridge.data.packing.in_batch import (
            build_mcore_thd_sequence_batch_from_rows,
        )
    except ImportError:
        return _fallback_build_mcore_thd_sequence_batch_from_rows
    return build_mcore_thd_sequence_batch_from_rows


def _get_visual_inputs_type() -> type:
    try:
        from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs
    except ImportError:
        return _FallbackGenericVisualInputs
    return GenericVisualInputs


def _fallback_build_mcore_thd_sequence_batch_from_rows(
    rows: Sequence[Mapping[str, torch.Tensor]],
    *,
    token_key: str,
    sequence_length: int | None,
    pad_token_id: int,
    ignore_index: int,
    pad_to_multiple_of: int,
    sequence_tensor_pad_values: Mapping[str, int | float],
) -> dict[str, Any]:
    """Dependency-free implementation of Bridge's THD row helper contract."""
    lengths = [row[token_key].numel() for row in rows]
    padded_lengths = [_ceil_to_multiple(length, pad_to_multiple_of) for length in lengths]
    logical_boundaries = [0]
    physical_boundaries = [0]
    for length, padded_length in zip(lengths, padded_lengths, strict=True):
        logical_boundaries.append(logical_boundaries[-1] + length)
        physical_boundaries.append(physical_boundaries[-1] + padded_length)

    total_tokens = physical_boundaries[-1]
    first = rows[0]
    output: dict[str, Any] = {
        token_key: torch.full(
            (1, total_tokens),
            pad_token_id,
            dtype=first[token_key].dtype,
            device=first[token_key].device,
        ),
        "position_ids": torch.zeros(
            (1, total_tokens),
            dtype=first["position_ids"].dtype,
            device=first["position_ids"].device,
        ),
        "labels": torch.full(
            (1, total_tokens),
            ignore_index,
            dtype=first["labels"].dtype,
            device=first["labels"].device,
        ),
        "loss_mask": torch.zeros(
            (1, total_tokens),
            dtype=first["loss_mask"].dtype,
            device=first["loss_mask"].device,
        ),
        "attention_mask": None,
    }
    for key, pad_value in sequence_tensor_pad_values.items():
        output[key] = torch.full(
            (1, total_tokens),
            pad_value,
            dtype=first[key].dtype,
            device=first[key].device,
        )

    for row, length, start in zip(rows, lengths, physical_boundaries[:-1], strict=True):
        output[token_key][0, start : start + length] = row[token_key]
        output["position_ids"][0, start : start + length] = row["position_ids"]
        output["labels"][0, start : start + length] = row["labels"]
        output["loss_mask"][0, start : start + length] = row["loss_mask"]
        for key in sequence_tensor_pad_values:
            output[key][0, start : start + length] = row[key]
        padded_stop = start + _ceil_to_multiple(length, pad_to_multiple_of)
        if padded_stop > start + length:
            output["position_ids"][0, start + length : padded_stop] = torch.arange(
                row["position_ids"][-1] + 1,
                row["position_ids"][-1] + 1 + padded_stop - start - length,
                dtype=row["position_ids"].dtype,
                device=row["position_ids"].device,
            )

    cu_seqlens = torch.tensor(logical_boundaries, dtype=torch.int32, device=first[token_key].device)
    cu_seqlens_padded = torch.tensor(physical_boundaries, dtype=torch.int32, device=first[token_key].device)
    output.update(
        {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens,
            "cu_seqlens_q_padded": cu_seqlens_padded,
            "cu_seqlens_kv_padded": cu_seqlens_padded,
            "max_seqlen_q": torch.tensor(max(padded_lengths), dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(max(padded_lengths), dtype=torch.int32),
            "total_tokens": total_tokens,
        }
    )
    return output
