# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode packing and source-token-aligned NeMo batches."""

from __future__ import annotations

import torch

from slim.utils.types import Episode

EDGE_FIELDS = frozenset(
    {
        "loss_masks",
        "advantages",
        "value_targets",
        "rollout_log_probs",
        "rollout_routed_experts",
        "actor_old_log_probs",
        "ref_log_probs",
        "cur_log_probs",
        "entropy",
        "old_values",
        "cur_values",
        "mismatch_weights",
        "mismatch_masks",
    }
)
TOKEN_FIELDS = frozenset({"tokens", "position_ids"})
TOKEN_SLOT_FILLS = {
    "loss_masks": 0,
    "advantages": 0.0,
    "value_targets": 0.0,
    "rollout_log_probs": 0.0,
    "rollout_routed_experts": 0,
    "actor_old_log_probs": 0.0,
    "ref_log_probs": 0.0,
    "old_values": 0.0,
    "mismatch_weights": 1.0,
    "mismatch_masks": 0,
}


def _as_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _validate_edge_field(episode: Episode, name: str, value) -> None:
    if value is None:
        raise ValueError(f"{name} must be present for every training episode")
    if len(value) != episode.num_edges:
        raise ValueError(f"{name} length {len(value)} does not match num_edges {episode.num_edges}")


def _optional_edge_field(episodes: list[Episode], indices: list[int], name: str, dtype: torch.dtype):
    values = [getattr(episodes[index], name, None) for index in indices]
    present = [value is not None and len(value) > 0 for value in values]
    if any(present) and not all(present):
        raise ValueError(f"{name} must be present for every episode in a pack or for none")
    if not all(present):
        return None
    for index, value in zip(indices, values, strict=True):
        _validate_edge_field(episodes[index], name, value)
    return torch.cat([_as_tensor(value, dtype=dtype) for value in values])


def build_token_budget_partitions(
    lengths: list[int],
    max_tokens_per_pack: int,
    *,
    num_packs: int | None = None,
) -> list[list[int]]:
    """Partition sequences without exceeding the physical-pack token budget."""
    if max_tokens_per_pack < 1:
        raise ValueError("max_tokens_per_pack must be at least 1")
    if not lengths:
        return []
    if any(length < 1 for length in lengths):
        raise ValueError("sequence lengths must be at least 1")
    oversized = [index for index, length in enumerate(lengths) if length > max_tokens_per_pack]
    if oversized:
        index = oversized[0]
        raise ValueError(
            f"sequence {index} has {lengths[index]} tokens, exceeding the physical-pack budget "
            f"{max_tokens_per_pack}"
        )

    partitions: list[list[int]] = []
    totals: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda item: (-lengths[item], item)):
        for partition_index, total in enumerate(totals):
            if total + lengths[index] <= max_tokens_per_pack:
                partitions[partition_index].append(index)
                totals[partition_index] += lengths[index]
                break
        else:
            partitions.append([index])
            totals.append(lengths[index])

    if num_packs is None:
        return partitions
    if num_packs < len(partitions):
        raise ValueError(
            f"num_packs {num_packs} is below the minimum budget-safe pack count {len(partitions)}"
        )
    if num_packs > len(lengths):
        raise ValueError(f"num_packs {num_packs} exceeds sequence count {len(lengths)}")

    while len(partitions) < num_packs:
        splittable = [index for index, partition in enumerate(partitions) if len(partition) > 1]
        if not splittable:
            raise RuntimeError(f"cannot split {len(partitions)} packs into {num_packs}")
        partition_index = max(splittable, key=lambda index: totals[index])
        partition = partitions[partition_index]
        total = totals[partition_index]
        split_position = min(
            range(len(partition)),
            key=lambda position: (
                abs(total - 2 * lengths[partition[position]]),
                partition[position],
            ),
        )
        sequence_index = partition.pop(split_position)
        totals[partition_index] -= lengths[sequence_index]
        partitions.append([sequence_index])
        totals.append(lengths[sequence_index])

    return partitions


def pack_sequences(
    episodes: list[Episode],
    partitions: list[list[int]] | None = None,
) -> list[dict]:
    """Build CPU-resident physical packs from explicit episode partitions."""
    if not episodes:
        return []

    lengths = [len(episode.tokens) for episode in episodes]
    if any(length < 2 for length in lengths):
        raise ValueError("every training episode must contain at least two tokens")
    if partitions is None:
        partitions = [list(range(len(episodes)))]
    else:
        partitions = [list(partition) for partition in partitions]
        if not partitions or any(not partition for partition in partitions):
            raise ValueError("partitions must contain one or more non-empty packs")
        indices = [index for partition in partitions for index in partition]
        if sorted(indices) != list(range(len(episodes))):
            raise ValueError("partitions must contain every episode index exactly once")

    packs: list[dict] = []
    for indices in partitions:
        cu_seqlens = [0]
        token_parts = []
        position_parts = []
        loss_mask_parts = []
        for index in indices:
            episode = episodes[index]
            tokens = _as_tensor(episode.tokens, dtype=torch.long)
            _validate_edge_field(episode, "loss_mask", episode.loss_mask)
            token_parts.append(tokens)
            position_parts.append(torch.arange(tokens.numel(), dtype=torch.long))
            loss_mask_parts.append(_as_tensor(episode.loss_mask, dtype=torch.int32))
            cu_seqlens.append(cu_seqlens[-1] + tokens.numel())

        pack = {
            "tokens": torch.cat(token_parts),
            "position_ids": torch.cat(position_parts),
            "loss_masks": torch.cat(loss_mask_parts),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "response_lengths": [episodes[index].response_length for index in indices],
            "reward": [episodes[index].reward for index in indices],
            "_episode_indices": list(indices),
        }

        for name, dtype in (
            ("advantages", torch.float32),
            ("value_targets", torch.float32),
            ("rollout_log_probs", torch.float32),
            ("rollout_routed_experts", torch.int32),
        ):
            value = _optional_edge_field(episodes, indices, name, dtype)
            if value is not None:
                pack[name] = value

        values = [episodes[index].values for index in indices]
        if any(value is not None for value in values):
            if not all(value is not None for value in values):
                raise ValueError("values must be present for every episode in a pack or for none")
            for index, value in zip(indices, values, strict=True):
                _validate_edge_field(episodes[index], "values", value)
            pack["old_values"] = torch.cat([_as_tensor(value, dtype=torch.float32) for value in values])

        multimodal_inputs: dict[str, torch.Tensor] = {}
        multimodal_counts: dict[str, list[int]] = {}
        for episode_offset, index in enumerate(indices):
            inputs = episodes[index].multimodal_inputs or {}
            for name in inputs:
                multimodal_counts.setdefault(name, [0] * episode_offset)
            for name, counts in multimodal_counts.items():
                tensor = inputs.get(name)
                if tensor is None:
                    counts.append(0)
                    continue
                tensor = tensor.detach().cpu()
                multimodal_inputs[name] = (
                    tensor if name not in multimodal_inputs else torch.cat((multimodal_inputs[name], tensor), dim=0)
                )
                counts.append(tensor.shape[0])
        if multimodal_inputs:
            pack["multimodal_inputs"] = multimodal_inputs
            pack["multimodal_num_items"] = multimodal_counts
        packs.append(pack)
    return packs


def unpack_sequences(pack: dict) -> list[dict]:
    """Return per-episode views from an edge-packed batch."""
    cu_seqlens = pack["cu_seqlens"].tolist()
    edge_offsets = [0]
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        edge_offsets.append(edge_offsets[-1] + end - start - 1)

    multimodal_offsets = {}
    for name, counts in pack.get("multimodal_num_items", {}).items():
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        multimodal_offsets[name] = offsets

    episodes = []
    for index, (token_start, token_end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        edge_start, edge_end = edge_offsets[index : index + 2]
        episode = {}
        for name, value in pack.items():
            if name == "multimodal_num_items":
                continue
            if name == "multimodal_inputs":
                episode[name] = {}
                for media_name, media_tensor in value.items():
                    start, end = multimodal_offsets[media_name][index : index + 2]
                    if end > start:
                        episode[name][media_name] = media_tensor[start:end]
            elif name in EDGE_FIELDS:
                episode[name] = value[edge_start:edge_end]
            elif name in TOKEN_FIELDS:
                episode[name] = value[token_start:token_end]
            elif not isinstance(value, torch.Tensor):
                episode[name] = value[index]
        episodes.append(episode)
    return episodes


def update_packed_targets(packs: list[dict], episodes: list[Episode]) -> None:
    for pack in packs:
        indices = pack["_episode_indices"]
        for episode_field, pack_field in (
            ("advantages", "advantages"),
            ("value_targets", "value_targets"),
            ("values", "old_values"),
        ):
            values = [getattr(episodes[index], episode_field) for index in indices]
            if any(value is not None for value in values) and not all(value is not None for value in values):
                raise ValueError(f"{episode_field} must be present for every episode in a pack or for none")
            if all(value is not None for value in values):
                for index, value in zip(indices, values, strict=True):
                    _validate_edge_field(episodes[index], episode_field, value)
                pack[pack_field] = torch.cat([_as_tensor(value, dtype=torch.float32) for value in values])
            else:
                pack.pop(pack_field, None)


def edge_to_token_slots(
    edge_values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    fill: float | int,
    total_length: int | None = None,
) -> torch.Tensor:
    """Place each edge value at its source-token position."""
    boundaries = cu_seqlens.tolist()
    packed_length = boundaries[-1]
    total_length = packed_length if total_length is None else total_length
    if total_length < packed_length:
        raise ValueError("total_length cannot be shorter than the packed token stream")
    output = torch.full(
        (total_length, *edge_values.shape[1:]),
        fill,
        dtype=edge_values.dtype,
        device=edge_values.device,
    )
    cursor = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        edge_count = end - start - 1
        output[start : end - 1] = edge_values[cursor : cursor + edge_count]
        cursor += edge_count
    if cursor != edge_values.shape[0]:
        raise ValueError(f"consumed {cursor} edge values but received {edge_values.shape[0]}")
    return output


def token_slots_to_edges(token_values: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    boundaries = cu_seqlens.tolist()
    pieces = [token_values[start : end - 1] for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
    return torch.cat(pieces) if pieces else token_values.new_empty((0, *token_values.shape[1:]))


def build_document_ids(cu_seqlens: torch.Tensor, total_length: int | None = None) -> torch.Tensor:
    boundaries = cu_seqlens.tolist()
    total_length = boundaries[-1] if total_length is None else total_length
    if total_length < boundaries[-1]:
        raise ValueError("total_length cannot be shorter than the packed token stream")
    document_ids = torch.zeros(total_length, dtype=torch.long)
    for document_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), start=1):
        document_ids[start:end] = document_id
    return document_ids


def build_source_labels(tokens: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(tokens, -100)
    for start, end in zip(cu_seqlens.tolist()[:-1], cu_seqlens.tolist()[1:], strict=True):
        labels[start : end - 1] = tokens[start + 1 : end]
    return labels


def build_model_batch(pack: dict) -> dict:
    tokens = pack["tokens"]
    batch = {
        "input_ids": tokens.unsqueeze(0),
        "labels": build_source_labels(tokens, pack["cu_seqlens"]).unsqueeze(0),
        "_packed_seq_ids": build_document_ids(pack["cu_seqlens"]).unsqueeze(0),
    }
    batch.update(pack.get("multimodal_inputs") or {})
    return batch


def build_token_slot_fields(pack: dict) -> dict[str, torch.Tensor]:
    fields = {
        "document_ids": build_document_ids(pack["cu_seqlens"]).unsqueeze(0),
    }
    for name, fill in TOKEN_SLOT_FILLS.items():
        value = pack.get(name)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            fields[name] = edge_to_token_slots(value, pack["cu_seqlens"], fill=fill).unsqueeze(0)
    for name, value in pack.get("_mismatch_metrics", {}).items():
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            fields[f"mismatch/{name}"] = edge_to_token_slots(
                value,
                pack["cu_seqlens"],
                fill=0,
            ).unsqueeze(0)
    return fields
