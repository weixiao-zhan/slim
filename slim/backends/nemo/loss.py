# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CP-local RL loss reductions with AutoModel gradient scaling."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce


_LOGPROB_BLOCK_SIZE = 32768


@triton.jit
def _selective_log_probs_forward_kernel(
    logits,
    labels,
    output,
    num_columns,
    temperature,
    WRITE_GRADIENT: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    logits += row * num_columns
    target = tl.load(labels + row)

    if target == IGNORE_INDEX:
        tl.store(output + row, 0.0)
        if WRITE_GRADIENT:
            for start in range(0, num_columns, BLOCK_SIZE):
                columns = start + tl.arange(0, BLOCK_SIZE)
                tl.store(logits + columns, 0.0, mask=columns < num_columns)
        return

    row_max = float("-inf")
    denominator = 0.0
    for start in range(0, num_columns, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(logits + columns, mask=columns < num_columns, other=float("-inf"))
        values = values.to(tl.float32) / temperature
        block_max = tl.max(values)
        new_max = tl.maximum(row_max, block_max)
        denominator = denominator * tl.exp(row_max - new_max) + tl.sum(tl.exp(values - new_max))
        row_max = new_max

    log_denominator = row_max + tl.log(denominator)
    selected = tl.load(logits + target).to(tl.float32) / temperature
    tl.store(output + row, selected - log_denominator)

    if WRITE_GRADIENT:
        for start in range(0, num_columns, BLOCK_SIZE):
            columns = start + tl.arange(0, BLOCK_SIZE)
            values = tl.load(logits + columns, mask=columns < num_columns, other=float("-inf"))
            values = values.to(tl.float32) / temperature
            gradient = -tl.exp(values - row_max) / denominator
            gradient += columns == target
            tl.store(logits + columns, gradient / temperature, mask=columns < num_columns)


@triton.jit
def _scale_log_probs_gradient_kernel(
    gradient,
    output_gradient,
    num_columns,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gradient += row * num_columns
    scale = tl.load(output_gradient + row)
    for start in range(0, num_columns, BLOCK_SIZE):
        columns = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(gradient + columns, mask=columns < num_columns)
        tl.store(gradient + columns, values * scale, mask=columns < num_columns)


class _SelectiveLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
        rows, num_columns = logits.shape
        output = torch.empty(rows, dtype=torch.float32, device=logits.device)
        write_gradient = logits.requires_grad
        _selective_log_probs_forward_kernel[(rows,)](
            logits,
            labels,
            output,
            num_columns,
            temperature,
            WRITE_GRADIENT=write_gradient,
            IGNORE_INDEX=-100,
            BLOCK_SIZE=_LOGPROB_BLOCK_SIZE,
            num_warps=32,
        )
        if write_gradient:
            ctx.save_for_backward(logits.detach())
            ctx.num_columns = num_columns
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        output_gradient = output_gradient.contiguous()
        _scale_log_probs_gradient_kernel[(gradient.shape[0],)](
            gradient,
            output_gradient,
            ctx.num_columns,
            BLOCK_SIZE=_LOGPROB_BLOCK_SIZE,
            num_warps=32,
        )
        return gradient, None, None


def selective_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float | None = None,
) -> torch.Tensor:
    """Select source-aligned target log probabilities."""
    temperature = 1.0 if temperature is None else temperature
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits and labels shapes do not align: {logits.shape} and {labels.shape}")

    if logits.is_cuda:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        if not flat_logits.is_contiguous():
            flat_logits = flat_logits.contiguous()
        flat_labels = labels.reshape(-1).contiguous()
        return _SelectiveLogProbs.apply(flat_logits, flat_labels, temperature).reshape_as(labels)

    logits = logits / temperature
    valid = labels != -100
    targets = labels.masked_fill(~valid, 0).long()
    selected = logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~valid, 0)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits.float().log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def count_global_denominators(packs: list[dict], dp_group, device) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Reduce the document weight sum and loss-token count over logical DP ranks.

    The sequence-level denominator is $\sum_d w_d$ rather than a document count,
    so it agrees with whichever unit the advantage baseline uses. Under
    `--loss-normalization-unit episode` an attempt's trajectories each weigh $1/k$ and the
    sum counts attempts; under `trajectory` each weighs $1$ and the sum counts
    trajectories. Padding documents weigh $0$ and drop out of both.
    """
    weight_sum = 0.0
    token_count = 0
    for pack in packs:
        boundaries = pack["cu_seqlens"].tolist()
        masks = pack["loss_masks"]
        weights = pack["loss_weights"]
        if len(weights) != len(boundaries) - 1:
            raise ValueError(
                f"{len(weights)} loss weights do not match {len(boundaries) - 1} packed documents"
            )
        weight_sum += sum(weights)
        expected_length = boundaries[-1]
        if masks.ndim != 1 or masks.shape[0] != expected_length:
            raise ValueError(
                f"source-aligned loss_masks shape {tuple(masks.shape)} does not match "
                f"packed token count {expected_length}"
            )
        token_count += int(masks.sum())

    counts = torch.tensor([weight_sum, token_count], dtype=torch.float64, device=device)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size(group=dp_group) > 1:
        torch.distributed.all_reduce(counts, group=dp_group)
    return counts[0].clamp_min(1), counts[1].clamp_min(1)


def cp_sum(tensor: torch.Tensor, cp_group, *, differentiable: bool) -> torch.Tensor:
    if (
        cp_group is None
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size(group=cp_group) == 1
    ):
        return tensor
    if differentiable:
        return differentiable_all_reduce(tensor, group=cp_group)
    output = tensor.clone()
    torch.distributed.all_reduce(output, group=cp_group)
    return output


def _document_sums(
    values: torch.Tensor,
    mask: torch.Tensor,
    document_ids: torch.Tensor,
    num_documents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.squeeze(0)
    mask = mask.squeeze(0).to(values.dtype)
    document_ids = document_ids.squeeze(0).long()
    numerators = values.new_zeros(num_documents)
    denominators = values.new_zeros(num_documents)
    numerators.scatter_add_(0, (document_ids - 1).clamp_min(0), values * mask * (document_ids > 0))
    denominators.scatter_add_(0, (document_ids - 1).clamp_min(0), mask * (document_ids > 0))
    return numerators, denominators


def sequence_mean_at_tokens(
    values: torch.Tensor,
    mask: torch.Tensor,
    document_ids: torch.Tensor,
    num_documents: int,
    cp_group,
) -> torch.Tensor:
    """Expand each differentiable CP-global document mean to local tokens."""
    numerators, denominators = _document_sums(values, mask, document_ids, num_documents)
    numerators = cp_sum(numerators, cp_group, differentiable=True)
    denominators = cp_sum(denominators, cp_group, differentiable=False).clamp_min(1)
    means = numerators / denominators
    local_ids = document_ids.squeeze(0).long()
    output = values.new_zeros(local_ids.shape)
    valid = local_ids > 0
    output[valid] = means[local_ids[valid] - 1]
    return output.unsqueeze(0)


def reduce_token_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    global_tokens: torch.Tensor,
) -> torch.Tensor:
    """Sum local token values and normalize by the global valid-token count."""
    return (values * mask.to(values.dtype)).sum() / global_tokens


def reduce_weighted_sequence_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    document_ids: torch.Tensor,
    num_documents: int,
    global_sequences: torch.Tensor,
    cp_group,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Weight each document mean by $w_d$ and divide by the global $\sum_d w_d$.

    $$L = \frac{1}{\sum_d w_d} \sum_d w_d \cdot
           \frac{\sum_t v_{d,t} m_{d,t}}{\sum_t m_{d,t}}$$

    $\sum_d w_d$ is what `count_global_denominators` reduces, so the weights are not
    optional: omitting them would denominate in attempts while numerating in documents.
    """
    numerators, denominators = _document_sums(values, mask, document_ids, num_documents)
    denominators = cp_sum(denominators, cp_group, differentiable=False).clamp_min(1)
    means = numerators / denominators * loss_weights.to(device=values.device, dtype=values.dtype)
    return means.sum() / global_sequences
