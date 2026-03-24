"""Data packing utilities for FSDP backend to reduce padding overhead."""

import math

import torch

from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Episode


def pack_sequences(
    episodes: list[Episode],
    max_tokens_per_gpu: int | None = None,
    num_packs: int | None = None,
) -> list[dict]:
    """Pack episodes into dense batches with cumulative sequence lengths.

    Args:
        episodes: List of Episode objects to pack
        max_tokens_per_gpu: Maximum tokens per GPU pack
        num_packs: Explicit number of packs to create

    Returns:
        List of packed batches
    """
    if not episodes:
        return []

    seq_lengths = [len(ep.tokens) for ep in episodes]

    # Determine number of packs and use balanced partitioning
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
        cu_seqlens = [0]
        flat_tokens = []
        flat_masks = []
        flat_positionids = []
        flat_advantages = []
        flat_returns = []
        flat_rollout_log_probs = []

        for i in indices:
            ep = episodes[i]
            flat_tokens.extend(ep.tokens)
            flat_positionids.extend(range(len(ep.tokens)))
            flat_masks.extend(ep.loss_mask)
            flat_advantages.extend(ep._advantages)
            flat_returns.extend(ep._returns)
            if ep.rollout_log_probs:
                flat_rollout_log_probs.extend(ep.rollout_log_probs)
            cu_seqlens.append(cu_seqlens[-1] + len(ep.tokens))

        packed_batch = {
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "position_ids": torch.tensor(flat_positionids, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "rewards": torch.tensor([episodes[i].reward for i in indices], dtype=torch.float32),
            "raw_reward": [episodes[i].reward for i in indices],
            "response_lengths": [episodes[i].response_length for i in indices],
            "advantages": torch.tensor(flat_advantages, dtype=torch.float32),
            "returns": torch.tensor(flat_returns, dtype=torch.float32),
            "rollout_log_probs": torch.tensor(
                flat_rollout_log_probs, dtype=torch.float32, device=torch.cuda.current_device()
            ),
        }

        # Collect multimodal training tensors
        has_multimodal = any(episodes[i].multimodal_train_inputs is not None for i in indices)
        if has_multimodal:
            multimodal_data = {}
            multimodal_num_items = {}
            for i in indices:
                mm = episodes[i].multimodal_train_inputs
                if mm is None:
                    continue
                for key, mm_tensor in mm.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key].append(mm_tensor.size(0))
            packed_batch["multimodal_train_inputs"] = multimodal_data
            packed_batch["multimodal_num_items"] = multimodal_num_items

        result.append(packed_batch)

    return result


def unpack_sequences(packed_batch: dict) -> list[dict]:
    """Unpack sequences from a packed batch.

    Args:
        packed_batch: Packed batch

    Returns:
        List of unpacked batches
    """
    cu_seqlens = packed_batch["cu_seqlens"]
    num_sequences = len(cu_seqlens) - 1
    response_lengths = packed_batch["response_lengths"]
    multimodal_num_items = packed_batch.get("multimodal_num_items", {})

    instances = []

    # Calculate pad_length by counting trailing zeros
    tokens = packed_batch["tokens"]
    nonzero_indices = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero_indices) > 0:
        pad_length = len(tokens) - nonzero_indices[-1].item() - 1
    else:
        pad_length = 0

    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        instance = {}

        for key, value in packed_batch.items():
            if key in instance:
                continue
            if key == "multimodal_num_items":
                continue
            elif key == "multimodal_train_inputs" and isinstance(value, dict):
                instance[key] = {}
                for mm_key, mm_tensor in value.items():
                    if mm_key in multimodal_num_items:
                        num_items_list = multimodal_num_items[mm_key]
                        start_mm_idx = sum(num_items_list[:i])
                        end_mm_idx = start_mm_idx + num_items_list[i]
                        if num_items_list[i] > 0:
                            instance[key][mm_key] = mm_tensor[start_mm_idx:end_mm_idx]
            elif isinstance(value, torch.Tensor):
                if key in ["log_probs", "ref_log_probs", "cur_log_probs", "entropy"]:
                    instance[key] = value[
                        end_idx - 1 - response_lengths[i] - pad_length : end_idx - 1 - pad_length
                    ]
                elif key == "rollout_log_probs":
                    instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                elif key in ["tokens", "position_ids"]:
                    if len(value) > start_idx:
                        instance[key] = value[start_idx:end_idx]
                    else:
                        raise ValueError(f"Attribute {key} is not found in the packed batch")
                elif key in ["loss_masks", "advantages", "returns"]:
                    instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
            elif isinstance(value, list):
                instance[key] = value[i]
            else:
                raise ValueError(f"Attribute {key} is not found in the packed batch")

        instances.append(instance)

    return instances
