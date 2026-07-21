# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PPO advantage utilities."""

from types import SimpleNamespace

import pytest
import torch

from slim.backends.fsdp_utils.base import FSDPTrainer
from slim.utils.ppo_utils import vanilla_gae
from slim.utils.types import Episode


NUM_GPUS = 0


def _active_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()]


@pytest.mark.unit
def test_vanilla_gae_skips_observation_edges():
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0]])
    values = torch.tensor([[1.0, 100.0, -100.0, 2.0, 4.0]])
    loss_mask = torch.tensor([[1, 0, 0, 1, 1]])

    advantages, returns = vanilla_gae(rewards, values, loss_mask, gamma=0.9, lambd=0.8)

    compact_mask = torch.ones(1, 3, dtype=torch.bool)
    compact_advantages, compact_returns = vanilla_gae(
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
        _active_values(returns, loss_mask),
        compact_returns.flatten(),
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


@pytest.mark.unit
def test_vanilla_gae_validates_shapes():
    with pytest.raises(ValueError, match="must have the same shape"):
        vanilla_gae(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.ones(1, 2),
            gamma=1.0,
            lambd=1.0,
        )


@pytest.mark.unit
def test_ppo_reward_is_assigned_to_last_policy_edge():
    trainer = object.__new__(FSDPTrainer)
    trainer.args = SimpleNamespace(gamma=1.0, lambd=1.0, normalize_advantages=False)
    episode = Episode(
        tokens=[10, 11, 12, 13],
        loss_mask=[0, 1, 0],
        reward=2.0,
    )

    trainer._compute_ppo_advantages([episode], [torch.zeros(3)])

    assert episode._advantages == [0.0, 2.0, 0.0]
    assert episode._returns == [0.0, 2.0, 0.0]
