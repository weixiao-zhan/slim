"""Data packing utilities for FSDP backend to reduce padding overhead."""

import math

import torch

from slim.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slim.utils.types import Episode


# Keys sliced by edge offsets (edge-aligned data, length = total_tokens - num_sequences)
_EDGE_KEYS = frozenset([
    "actor_old_log_probs", "ref_log_probs", "cur_log_probs", "entropy", "cur_values",
    "rollout_log_probs", "loss_masks", "advantages", "returns", "old_values",
    "rollout_routed_experts",
])
# Keys sliced by token offsets (cu_seqlens)
_TOKEN_KEYS = frozenset(["tokens", "position_ids"])


def pack_sequences(
    episodes: list[Episode],
    max_tokens_per_gpu: int | None = None,
    num_packs: int | None = None,
) -> list[dict]:
    """Pack frozen (tensor-backed) episodes into dense batches."""
    if not episodes:
        return []

    seq_lengths = [len(ep.tokens) for ep in episodes]

    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_gpu:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
    else:
        k_partitions = 1

    partitions = get_seqlen_balanced_partitions(
        seq_lengths, k_partitions=k_partitions, equal_size=False
    )

    result = []
    for indices in partitions:
        token_parts = []
        mask_parts = []
        posid_parts = []
        advantage_parts = []
        return_parts = []
        logprob_parts = []
        old_value_parts = []
        routed_experts_parts = []
        cu_seqlens = [0]
        edge_lengths = []

        for i in indices:
            ep = episodes[i]
            n = len(ep.tokens)

            token_parts.append(ep.tokens)
            posid_parts.append(torch.arange(n, dtype=torch.int))
            mask_parts.append(ep.loss_mask)
            advantage_parts.append(torch.tensor(ep._advantages, dtype=torch.float32))
            return_parts.append(torch.tensor(ep._returns, dtype=torch.float32))
            edge_lengths.append(ep.num_edges)

            if ep.rollout_log_probs is not None and len(ep.rollout_log_probs):
                logprob_parts.append(ep.rollout_log_probs)
            if ep.rollout_routed_experts is not None and len(ep.rollout_routed_experts):
                routed_experts_parts.append(ep.rollout_routed_experts)
            if hasattr(ep, "_values") and ep._values is not None:
                old_value_parts.append(torch.tensor(ep._values, dtype=torch.float32))
            cu_seqlens.append(cu_seqlens[-1] + n)

        packed_batch = {
            "tokens": torch.cat(token_parts),
            "loss_masks": torch.cat(mask_parts),
            "position_ids": torch.cat(posid_parts),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "rewards": torch.tensor([episodes[i].reward for i in indices], dtype=torch.float32),
            "raw_reward": [episodes[i].reward for i in indices],
            "response_lengths": [episodes[i].response_length for i in indices],
            "edge_lengths": edge_lengths,
            "advantages": torch.cat(advantage_parts),
            "returns": torch.cat(return_parts),
            "rollout_log_probs": (
                torch.cat(logprob_parts).to(torch.cuda.current_device())
                if logprob_parts
                else torch.tensor([], dtype=torch.float32, device=torch.cuda.current_device())
            ),
        }

        if old_value_parts:
            packed_batch["old_values"] = torch.cat(old_value_parts)

        if routed_experts_parts:
            # [total_edges, num_layers, top_k] int32 — kept on CPU until the
            # actor pushes it into the routing-replay buffer.
            packed_batch["rollout_routed_experts"] = torch.cat(routed_experts_parts, dim=0)

        has_multimodal = any(episodes[i].multimodal_inputs is not None for i in indices)
        if has_multimodal:
            multimodal_data = {}
            multimodal_num_items = {}
            for episode_idx, i in enumerate(indices):
                mm = episodes[i].multimodal_inputs
                mm = mm or {}
                for key in mm:
                    if key not in multimodal_num_items:
                        multimodal_num_items[key] = [0] * episode_idx

                for key, counts in multimodal_num_items.items():
                    mm_tensor = mm.get(key)
                    if mm_tensor is None:
                        counts.append(0)
                        continue
                    # All remaining mm fields (pixel_values, image_grid_thw, etc.)
                    # are non-token-aligned and concat along dim=0.
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                    counts.append(mm_tensor.size(0))
            packed_batch["multimodal_inputs"] = multimodal_data
            packed_batch["multimodal_num_items"] = multimodal_num_items

        packed_batch["_episode_indices"] = list(indices)
        result.append(packed_batch)

    return result


def unpack_sequences(packed_batch: dict) -> list[dict]:
    """Unpack sequences from a packed batch."""
    cu_seqlens = packed_batch["cu_seqlens"]
    num_sequences = len(cu_seqlens) - 1
    edge_lengths = packed_batch["edge_lengths"]
    multimodal_num_items = packed_batch.get("multimodal_num_items", {})

    # Precompute cumulative offsets for O(1) slicing
    edge_offsets = [0]
    for el in edge_lengths:
        edge_offsets.append(edge_offsets[-1] + el)

    mm_offsets = {}
    for mm_key, num_items_list in multimodal_num_items.items():
        offsets = [0]
        for n in num_items_list:
            offsets.append(offsets[-1] + n)
        mm_offsets[mm_key] = offsets

    instances = []
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        edge_start = edge_offsets[i]
        edge_end = edge_offsets[i + 1]
        instance = {}

        for key, value in packed_batch.items():
            if key in instance or key == "multimodal_num_items":
                continue
            if key == "multimodal_inputs":
                instance[key] = {}
                for mm_key, mm_tensor in value.items():
                    if mm_key in mm_offsets:
                        s, e = mm_offsets[mm_key][i], mm_offsets[mm_key][i + 1]
                        if e > s:
                            instance[key][mm_key] = mm_tensor[s:e]
            elif key in _EDGE_KEYS:
                instance[key] = value[edge_start:edge_end]
            elif key in _TOKEN_KEYS:
                instance[key] = value[start_idx:end_idx]
            elif not isinstance(value, torch.Tensor):
                instance[key] = value[i]

        instances.append(instance)

    return instances


def strip_cross_boundary(flat_edges: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Remove cross-boundary entries from a flat edge tensor computed via [:-1].

    When multiple sequences are packed and we compute logits[:-1], the result
    has total_tokens - 1 entries, including cross-boundary entries between
    adjacent sequences.  This function extracts only the valid within-sequence
    edges (total_tokens - num_sequences entries).
    """
    if len(cu_seqlens) <= 2:
        # Single sequence — no cross-boundary entries
        return flat_edges
    indices = torch.cat([
        torch.arange(cu_seqlens[i], cu_seqlens[i + 1] - 1, device=flat_edges.device)
        for i in range(len(cu_seqlens) - 1)
    ])
    return flat_edges[indices]


def init_dummy_advantages(episodes: list[Episode]) -> None:
    """Set zero advantages/returns on episodes so pack_sequences can proceed."""
    for ep in episodes:
        ep._advantages = [0.0] * ep.num_edges
        ep._returns = [0.0] * ep.num_edges


def update_packed_advantages(packed_batches: list[dict], episodes: list[Episode]) -> None:
    """Update advantages/returns/old_values in pre-packed batches from episodes.

    After advantages are computed on episodes (e.g. via GAE), this function
    overwrites the dummy values that were used during initial packing.
    """
    for batch in packed_batches:
        ep_indices = batch["_episode_indices"]
        adv_parts = []
        ret_parts = []
        val_parts = []
        for idx in ep_indices:
            ep = episodes[idx]
            adv_parts.append(torch.tensor(ep._advantages, dtype=torch.float32))
            ret_parts.append(torch.tensor(ep._returns, dtype=torch.float32))
            values = getattr(ep, "_values", None)
            if values is not None:
                val_parts.append(torch.tensor(values, dtype=torch.float32))
        batch["advantages"] = torch.cat(adv_parts)
        batch["returns"] = torch.cat(ret_parts)
        if val_parts:
            batch["old_values"] = torch.cat(val_parts)
