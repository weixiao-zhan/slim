# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PPO advantage utilities."""

import pytest
import torch

from slim.utils.ppo_utils import compute_surrogate_weight, vanilla_gae


NUM_GPUS = 0


def _active_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()]


@pytest.mark.unit
def test_vanilla_gae_skips_observation_edges():
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0]])
    values = torch.tensor([[1.0, 100.0, -100.0, 2.0, 4.0]])
    loss_mask = torch.tensor([[1, 0, 0, 1, 1]])

    advantages, value_targets = vanilla_gae(rewards, values, loss_mask, gamma=0.9, lambd=0.8)

    compact_mask = torch.ones(1, 3, dtype=torch.bool)
    compact_advantages, compact_value_targets = vanilla_gae(
        torch.tensor([[0.0, 0.0, 3.0]]),
        torch.tensor([[1.0, 2.0, 4.0]]),
        compact_mask,
        gamma=0.9,
        lambd=0.8,
    )

    torch.testing.assert_close(
        _active_values(advantages, loss_mask),
        compact_advantages.flatten(),
    )
    torch.testing.assert_close(
        _active_values(value_targets, loss_mask),
        compact_value_targets.flatten(),
    )
    assert torch.equal(advantages[~loss_mask.bool()], torch.zeros(2))


@pytest.mark.unit
def test_vanilla_gae_is_invariant_to_observation_length_and_values():
    short_mask = torch.tensor([[1, 0, 1, 1]])
    short_advantages, _ = vanilla_gae(
        torch.tensor([[0.0, 0.0, 0.0, 2.0]]),
        torch.tensor([[0.5, 1000.0, 1.0, 1.5]]),
        short_mask,
        gamma=0.95,
        lambd=0.9,
    )

    long_mask = torch.tensor([[1, 0, 0, 0, 1, 1]])
    long_advantages, _ = vanilla_gae(
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
        torch.tensor([[0.5, -10.0, 20.0, -30.0, 1.0, 1.5]]),
        long_mask,
        gamma=0.95,
        lambd=0.9,
    )

    torch.testing.assert_close(
        _active_values(short_advantages, short_mask),
        _active_values(long_advantages, long_mask),
    )


def _ratio_loss(ratio, advantages, policy_surrogate, eps_clip, eps_clip_high, eps_clip_c):
    """Original surrogate losses, which differentiate the ratio itself."""
    clamped = ratio.clamp(1 - eps_clip, 1 + eps_clip_high)
    if policy_surrogate == "is":
        return -ratio * advantages
    if policy_surrogate == "mis":
        return -clamped * advantages
    if policy_surrogate == "cis":
        return -clamped.detach() * advantages * ratio.log()
    losses = torch.maximum(-ratio * advantages, -clamped * advantages)
    if eps_clip_c is None:
        return losses
    return torch.where(advantages < 0, torch.minimum(-eps_clip_c * advantages, losses), losses)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("policy_surrogate", "eps_clip_c"),
    [("ppo_clip", None), ("ppo_clip", 1.5), ("is", None), ("mis", None), ("cis", None)],
)
def test_surrogate_weights_reproduce_the_ratio_gradient(policy_surrogate, eps_clip_c):
    ratios = torch.tensor([0.5, 0.9, 1.1, 1.4, 2.0])
    advantages = torch.tensor([1.0, -1.0]).repeat_interleave(len(ratios))
    log_probs = ratios.log().repeat(2).requires_grad_(True)
    reference_log_probs = log_probs.detach().clone().requires_grad_(True)

    weights, _ = compute_surrogate_weight(log_probs.exp(), advantages, 0.2, 0.3, policy_surrogate, eps_clip_c)
    (-advantages * weights * log_probs).sum().backward()
    _ratio_loss(reference_log_probs.exp(), advantages, policy_surrogate, 0.2, 0.3, eps_clip_c).sum().backward()

    torch.testing.assert_close(log_probs.grad, reference_log_probs.grad)
