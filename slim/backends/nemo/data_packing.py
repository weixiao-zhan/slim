# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory packing for source-token-aligned NeMo batches.

A :class:`~slim.utils.types.Trajectory` is one packed document: its tokens form a
contiguous block delimited by ``cu_seqlens``, and every reduction keyed on
``document_ids`` addresses it individually.
"""

from __future__ import annotations

from itertools import accumulate

import torch

from slim.utils.types import TOKEN_FIELDS, Trajectory, as_cpu_tensor

SEQUENCE_FIELDS = frozenset(
    {
        "tokens",
        "position_ids",
        "loss_masks",
        "advantages",
        "value_targets",
        "rollout_log_probs",
        "rollout_topk_log_probs",
        "rollout_topk_token_ids",
        "rollout_routed_experts",
        "ref_log_probs",
        "cur_log_probs",
        "entropy",
        "old_values",
        "cur_values",
    }
)
TRAINING_FIELDS = SEQUENCE_FIELDS - {"tokens", "position_ids", "cur_log_probs", "cur_values", "entropy"}
# Pack field of each trajectory training target.
_TARGET_PACK_FIELDS = {"advantages": "advantages", "value_targets": "value_targets", "values": "old_values"}
# Optional per-token trajectory fields a pack can carry. A role packs the subset it reads.
PACK_FIELDS = (
    "advantages",
    "values",
    "value_targets",
    "rollout_log_probs",
    "rollout_topk_log_probs",
    "rollout_topk_token_ids",
    "rollout_routed_experts",
)


def _optional_source_field(trajectories: list[Trajectory], indices: list[int], name: str):
    values = [getattr(trajectories[index], name, None) for index in indices]
    present = [value is not None and len(value) > 0 for value in values]
    if any(present) and not all(present):
        raise ValueError(f"{name} must be present for every trajectory in a pack or for none")
    if not all(present):
        return None
    return torch.cat([as_cpu_tensor(value, dtype=TOKEN_FIELDS[name].dtype) for value in values])


def pack_sequences(
    trajectories: list[Trajectory],
    partitions: list[list[int]] | None = None,
    fields: tuple[str, ...] = PACK_FIELDS,
) -> list[dict]:
    """Build CPU-resident physical packs from explicit trajectory partitions.

    `fields` names the optional per-token trajectory fields to pack, from `PACK_FIELDS`.
    """
    if not trajectories:
        return []

    lengths = [len(trajectory.token_ids) for trajectory in trajectories]
    if any(length < 2 for length in lengths):
        raise ValueError("every training trajectory must contain at least two tokens")
    if partitions is None:
        partitions = [list(range(len(trajectories)))]

    packs: list[dict] = []
    for indices in partitions:
        cu_seqlens = [0]
        token_parts = []
        position_parts = []
        loss_mask_parts = []
        for index in indices:
            trajectory = trajectories[index]
            tokens = as_cpu_tensor(trajectory.token_ids, dtype=TOKEN_FIELDS["token_ids"].dtype)
            token_parts.append(tokens)
            position_parts.append(torch.arange(tokens.numel(), dtype=torch.long))
            loss_mask_parts.append(as_cpu_tensor(trajectory.loss_mask, dtype=TOKEN_FIELDS["loss_mask"].dtype))
            cu_seqlens.append(cu_seqlens[-1] + tokens.numel())

        pack = {
            "tokens": torch.cat(token_parts),
            "position_ids": torch.cat(position_parts),
            "loss_masks": torch.cat(loss_mask_parts),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "response_lengths": [trajectories[index].response_length for index in indices],
            "reward": [trajectories[index].reward for index in indices],
            "loss_weights": [trajectories[index].loss_weight for index in indices],
            "_document_indices": list(indices),
        }

        for name in fields:
            value = _optional_source_field(trajectories, indices, name)
            if value is not None:
                pack[_TARGET_PACK_FIELDS.get(name, name)] = value

        multimodal_inputs: dict[str, list[torch.Tensor]] = {}
        multimodal_counts: dict[str, list[int]] = {}
        for document_offset, index in enumerate(indices):
            inputs = trajectories[index].multimodal_inputs or {}
            for name in inputs:
                multimodal_counts.setdefault(name, [0] * document_offset)
            for name, counts in multimodal_counts.items():
                tensor = inputs.get(name)
                if tensor is None:
                    counts.append(0)
                    continue
                tensor = tensor.detach().cpu()
                multimodal_inputs.setdefault(name, []).append(tensor)
                counts.append(tensor.shape[0])
        if multimodal_inputs:
            pack["multimodal_inputs"] = {name: torch.cat(parts, dim=0) for name, parts in multimodal_inputs.items()}
            pack["multimodal_num_items"] = multimodal_counts
        packs.append(pack)
    return packs


def multimodal_item_offsets(pack: dict) -> dict[str, list[int]]:
    """Start offset of each document's items in every multimodal input, followed by the item total."""
    return {name: [0, *accumulate(counts)] for name, counts in pack.get("multimodal_num_items", {}).items()}


def unpack_sequences(pack: dict) -> list[dict]:
    """Return per-document views from a source-token-aligned batch."""
    cu_seqlens = pack["cu_seqlens"].tolist()
    multimodal_offsets = multimodal_item_offsets(pack)

    documents = []
    for index, (token_start, token_end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        document = {}
        for name, value in pack.items():
            if name == "multimodal_num_items":
                continue
            if name == "multimodal_inputs":
                document[name] = {}
                for media_name, media_tensor in value.items():
                    start, end = multimodal_offsets[media_name][index : index + 2]
                    if end > start:
                        document[name][media_name] = media_tensor[start:end]
            elif name in SEQUENCE_FIELDS:
                document[name] = value[token_start:token_end]
            elif not isinstance(value, torch.Tensor):
                document[name] = value[index]
        documents.append(document)
    return documents


def update_packed_targets(
    packs: list[dict],
    trajectories: list[Trajectory],
    fields: tuple[str, ...] = PACK_FIELDS,
) -> None:
    targets = [name for name in fields if name in _TARGET_PACK_FIELDS]
    for pack in packs:
        for name in targets:
            value = _optional_source_field(trajectories, pack["_document_indices"], name)
            if value is None:
                pack.pop(_TARGET_PACK_FIELDS[name], None)
            else:
                pack[_TARGET_PACK_FIELDS[name]] = value


def fill_document_terminal_slots(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    fill: float | int = 0,
) -> torch.Tensor:
    """Fill the source-token slot without a prediction target in each document."""
    output = values.clone()
    terminal_indices = cu_seqlens[1:].to(device=values.device, dtype=torch.long) - 1
    output[terminal_indices] = fill
    return output


def build_document_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    boundaries = cu_seqlens.tolist()
    document_ids = torch.zeros(boundaries[-1], dtype=torch.long)
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


def build_training_fields(pack: dict) -> dict[str, torch.Tensor]:
    fields = {
        "document_ids": build_document_ids(pack["cu_seqlens"]).unsqueeze(0),
    }
    for name in TRAINING_FIELDS:
        value = pack.get(name)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            fields[name] = value.unsqueeze(0)
    return fields
