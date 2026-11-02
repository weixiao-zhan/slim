# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from collections.abc import Callable

import torch


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_rollout) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    else:
        # k3 and low_var_kl: the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


@torch.compile(dynamic=True)
def compute_ppo_clip_surrogate_weight(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    # The clipped branch of the max is constant in r, so a clipped token carries no gradient.
    positive = advantages > 0
    clipped = torch.where(positive, ratio > 1 + eps_clip_high, ratio < 1 - eps_clip) & (advantages != 0)
    kept = ~clipped
    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        # Dual clip caps the loss of a negative advantage at -eps_clip_c * A.
        kept = kept & (positive | (ratio <= eps_clip_c))
    return ratio * kept, clipped.float()


@torch.compile(dynamic=True)
def compute_mis_surrogate_weight(
    ratio: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    # The clamp has no gradient outside its range, so a clipped token carries none.
    kept = (ratio >= 1 - eps_clip) & (ratio <= 1 + eps_clip_high)
    return ratio * kept, (~kept).float()


@torch.compile(dynamic=True)
def compute_cis_surrogate_weight(
    ratio: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    weights = ratio.clamp(1 - eps_clip, 1 + eps_clip_high)
    if eps_clip == 0 and eps_clip_high == 0:
        # The clamp pins the weight to 1: plain REINFORCE, with no token clipped.
        return weights, torch.zeros_like(ratio)
    return weights, ((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip_high)).float()


def compute_surrogate_weight(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    policy_surrogate: str,
    eps_clip_c: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Detached weight $w(r, A)$ of each token's score, and whether the token is clipped.

    Every surrogate has the gradient $-A w \nabla \log p$, so the policy loss is
    $-A \operatorname{sg}[w] \log p$. `ppo_clip` and `mis` differentiate the ratio,
    and $\nabla r = r \nabla \log p$ makes their weight $r$ where the ratio is not
    clipped and $0$ where it is. `ratio` and `advantages` broadcast, so the same call
    weighs the sampled token `[T]` and the sampler top-K `[T, K]`.
    """
    ratio = ratio.detach()
    if policy_surrogate == "ppo_clip":
        return compute_ppo_clip_surrogate_weight(ratio, advantages, eps_clip, eps_clip_high, eps_clip_c)
    if policy_surrogate == "is":
        return ratio, torch.zeros_like(ratio)
    if policy_surrogate == "mis":
        return compute_mis_surrogate_weight(ratio, eps_clip, eps_clip_high)
    return compute_cis_surrogate_weight(ratio, eps_clip, eps_clip_high)


@torch.compile(dynamic=True)
def compute_score_centering_loss(
    topk_log_probs: torch.Tensor,
    rollout_topk_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    surrogate_weight: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    mass_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Per-token additive term that removes the sampler-expected score from the policy gradient.

    Score centering (Marek & Ryabinin, arXiv 2609.20807) uses the sampler's top-$k$
    tokens $H$ with log probabilities $\log q_v$ and models the tail with the trainer's
    shape, $\hat q_v = \rho p_v$:

    $$L_t = A_t \sum_{v \in H} \operatorname{sg}[q_v w_v - \alpha p_v] \log p_v$$

    $$\rho = \frac{1 - \sum_H q_v}{1 - \sum_H p_v}$$

    $$\alpha = \rho w(1/\rho, A_t)$$

    $w_v = w(p_v / q_v, A_t)$ is the surrogate's weight. `surrogate_weight(ratio, advantages)`
    is `compute_surrogate_weight` with the clip range bound, the same $w$ that weighs the
    sampled token. Added to the policy loss, the update becomes $A (w_x s_x - \mathbb{E}_{\hat q}[w s])$.

    Returns the per-token term, the sampler top-K mass $\sum_H q_v$, and $\rho$.
    """
    trainer_topk_log_probs = topk_log_probs.detach()
    trainer_topk = trainer_topk_log_probs.exp()
    sampler_topk = rollout_topk_log_probs.exp()
    topk_mass = sampler_topk.sum(dim=-1)
    rho = (1 - topk_mass).clamp_min(mass_floor) / (1 - trainer_topk.sum(dim=-1)).clamp_min(mass_floor)
    # SGLang reports an underflowed sampler probability as the float32 minimum. Capping the
    # exponent keeps its ratio finite, and its zero sampler mass then zeroes its term.
    topk_ratio = (trainer_topk_log_probs - rollout_topk_log_probs).clamp(max=80).exp()
    weights, _ = surrogate_weight(topk_ratio, advantages.unsqueeze(-1))
    tail_weights, _ = surrogate_weight(rho.reciprocal(), advantages)
    alpha = rho * tail_weights
    coefficients = sampler_topk * weights - alpha.unsqueeze(-1) * trainer_topk
    values = advantages * (coefficients * topk_log_probs).sum(dim=-1)
    return values, topk_mass, rho


def compute_value_loss(
    cur_values: torch.Tensor,
    old_values: torch.Tensor,
    value_targets: torch.Tensor,
    value_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped value loss for PPO critic training.

    Uses the standard PPO value clipping approach: the value prediction is
    clipped to be within [old_value - clip, old_value + clip], and we take
    the maximum of the clipped and unclipped squared errors.

    Args:
        cur_values: Current value predictions from the critic.
        old_values: Value predictions from the previous iteration.
        value_targets: Regression targets computed from GAE.
        value_clip: Clipping range for value predictions.

    Returns:
        Per-token value loss and per-token clip fraction (1 where the clipped
        squared error wins the max), neither yet reduced.
    """
    values_clipped = old_values + (cur_values - old_values).clamp(-value_clip, value_clip)
    vf_loss1 = (cur_values - value_targets) ** 2
    vf_loss2 = (values_clipped - value_targets) ** 2
    clipfrac = torch.gt(vf_loss2, vf_loss1).float()
    return 0.5 * torch.max(vf_loss1, vf_loss2), clipfrac


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lambd: float,
):
    """Compute GAE over policy-controlled edges.

    Masked edges, such as prompt and tool-observation tokens, do not advance
    the temporal recurrence. Their output advantages are zero.
    """
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    active_edges = loss_mask.to(device=device, dtype=torch.bool)

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    next_value = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        candidate_gaelam = delta + gamma * lambd * lastgaelam
        active = active_edges[:, t]
        lastgaelam = torch.where(active, candidate_gaelam, lastgaelam)
        next_value = torch.where(active, values[:, t], next_value)
        adv_rev.append(torch.where(active, lastgaelam, torch.zeros_like(lastgaelam)))

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    value_targets = full_advantages + values  # [B, max_len]
    return full_advantages, value_targets
