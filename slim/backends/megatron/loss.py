"""RL losses on TP-sharded logits and CP-local packed tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from slim.utils.ppo_utils import compute_approx_kl, compute_policy_loss


class _AllReduceSumForwardIdentityBackward(torch.autograd.Function):
    """Sum TP values in forward without summing replicated loss gradients."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        result = tensor.clone()
        torch.distributed.all_reduce(
            result,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


def _all_reduce_sum_forward_identity_backward(
    tensor: torch.Tensor,
    *,
    group: Any,
) -> torch.Tensor:
    return _AllReduceSumForwardIdentityBackward.apply(tensor, group)


def selected_log_probs(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_group: Any = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return selected-token log probabilities without gathering the vocabulary."""

    if vocab_parallel_logits.shape[:-1] != labels.shape:
        raise ValueError(
            "labels must match every non-vocabulary logits dimension; "
            f"got logits={tuple(vocab_parallel_logits.shape)} and labels={tuple(labels.shape)}."
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

    logits = vocab_parallel_logits if temperature == 1.0 else vocab_parallel_logits / temperature
    return -vocab_parallel_cross_entropy(logits, labels, tp_group=tp_group)


def vocab_parallel_entropy(
    vocab_parallel_logits: torch.Tensor,
    *,
    tp_group: Any = None,
) -> torch.Tensor:
    """Compute entropy with differentiable TP sum reductions."""

    logits = vocab_parallel_logits.float()
    local_max = logits.max(dim=-1).values.detach()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

    shifted = logits - local_max.unsqueeze(-1)
    local_z = shifted.exp().sum(dim=-1)
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        from torch.distributed.nn.functional import all_reduce

        z = all_reduce(local_z, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    else:
        z = local_z
    log_z = z.log()
    probabilities = shifted.exp() / z.unsqueeze(-1)
    local_entropy = -(probabilities * (shifted - log_z.unsqueeze(-1))).sum(dim=-1)
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        return _all_reduce_sum_forward_identity_backward(
            local_entropy,
            group=tp_group,
        )
    return local_entropy


def objective_weights(
    *,
    loss_mask: torch.Tensor,
    sequence_ids: torch.Tensor,
    sequence_active_counts: torch.Tensor,
    global_active_tokens: int | torch.Tensor,
    global_num_sequences: int,
    calculate_per_token_loss: bool,
) -> torch.Tensor:
    """Convert Slim reduction semantics to MCore global-token normalization."""

    if loss_mask.shape != sequence_ids.shape:
        raise ValueError("loss_mask and sequence_ids must have identical shapes.")
    if global_num_sequences < 1:
        raise ValueError("global_num_sequences must be positive.")
    if sequence_active_counts.ndim != 1:
        raise ValueError("sequence_active_counts must be one-dimensional.")
    if sequence_ids.numel() and (
        int(sequence_ids.min()) < 0 or int(sequence_ids.max()) >= sequence_active_counts.numel()
    ):
        raise ValueError("sequence_ids contains an index outside sequence_active_counts.")

    mask = loss_mask.to(dtype=torch.float32)
    active_tokens = torch.as_tensor(global_active_tokens, dtype=torch.float32, device=mask.device)
    base = active_tokens / global_num_sequences
    if calculate_per_token_loss:
        return mask * base

    counts = sequence_active_counts.to(device=mask.device, dtype=torch.float32).clamp_min(1)
    return mask * base / counts.index_select(0, sequence_ids.reshape(-1)).reshape_as(sequence_ids)


@dataclass(frozen=True)
class PolicyLossResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    num_active_tokens: torch.Tensor
    current_log_probs: torch.Tensor


def policy_loss(
    *,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    objective_weight: torch.Tensor,
    sample_mean_weight: torch.Tensor | None = None,
    eps_clip: float,
    eps_clip_high: float,
    policy_surrogate: str,
    eps_clip_c: float | None = None,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    reference_log_probs: torch.Tensor | None = None,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "k3",
    use_unbiased_kl: bool = False,
) -> PolicyLossResult:
    """Compute Slim's actor objective on already aligned rank-local tensors."""

    tensors = (old_log_probs, advantages, loss_mask, objective_weight)
    if any(tensor.shape != current_log_probs.shape for tensor in tensors):
        raise ValueError("All policy-loss tensors must have identical shapes.")
    if entropy_coef != 0.0 and entropy is None:
        raise ValueError("entropy is required when entropy_coef is nonzero.")
    if kl_loss_coef != 0.0 and reference_log_probs is None:
        raise ValueError("reference_log_probs is required when kl_loss_coef is nonzero.")

    old_log_probs = old_log_probs.to(device=current_log_probs.device)
    advantages = advantages.to(device=current_log_probs.device)
    weights = objective_weight.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
    sample_weights = (
        weights
        if sample_mean_weight is None
        else sample_mean_weight.to(
            device=current_log_probs.device,
            dtype=current_log_probs.dtype,
        )
    )
    if sample_weights.shape != current_log_probs.shape:
        raise ValueError("sample_mean_weight must match current_log_probs.")
    ppo_kl = old_log_probs - current_log_probs
    per_token_pg, per_token_clipfrac = compute_policy_loss(
        ppo_kl,
        current_log_probs,
        advantages,
        eps_clip,
        eps_clip_high,
        policy_surrogate,
        eps_clip_c,
    )
    pg_loss = (per_token_pg * weights).sum()
    loss = pg_loss
    metrics = {
        "pg_loss": pg_loss.detach(),
        "pg_clipfrac": (per_token_clipfrac * weights).sum().detach(),
        "pg_kl_k3": ((torch.exp(-ppo_kl) + ppo_kl - 1) * weights).sum().detach(),
    }

    if entropy is not None:
        entropy_loss = (
            entropy.to(device=current_log_probs.device) * sample_weights
        ).sum()
        loss = loss - entropy_coef * entropy_loss
        metrics["entropy"] = entropy_loss.detach()

    if reference_log_probs is not None:
        importance_ratio = torch.exp(current_log_probs - old_log_probs) if use_unbiased_kl else None
        per_token_kl = compute_approx_kl(
            current_log_probs,
            reference_log_probs.to(device=current_log_probs.device),
            kl_loss_type=kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = (per_token_kl * sample_weights).sum()
        loss = loss + kl_loss_coef * kl_loss
        metrics["kl_loss"] = kl_loss.detach()

    return PolicyLossResult(
        loss=loss,
        metrics=metrics,
        num_active_tokens=loss_mask.sum(),
        current_log_probs=current_log_probs,
    )
