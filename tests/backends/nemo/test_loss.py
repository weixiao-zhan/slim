# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from slim.backends.nemo.actor import ActorNeMoTrainer
from slim.backends.nemo.loss import (
    count_global_denominators,
    reduce_token_mean,
    reduce_weighted_sequence_mean,
    selective_log_probs,
    sequence_mean_at_tokens,
)


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [None, 0.7])
def test_selective_log_probs_matches_fp32_reference(temperature):
    torch.manual_seed(11)
    logits = torch.randn(2, 3, 17, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, -100, 7], [4, 3, 2]])
    output_gradient = torch.randn(2, 3)
    scale = 1.0 if temperature is None else temperature

    actual = selective_log_probs(logits, labels, temperature)
    targets = labels.masked_fill(labels == -100, 0)
    expected = (
        (reference_logits / scale)
        .float()
        .log_softmax(dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(labels == -100, 0)
    )
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_fused_selective_log_probs_preserves_fp32_values_and_bf16_gradients():
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 251, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, -100, 7], [4, 3, 2]], device="cuda")
    output_gradient = torch.randn(2, 3, device="cuda")

    actual = selective_log_probs(logits, labels, temperature=0.7)
    targets = labels.masked_fill(labels == -100, 0)
    expected = (
        (reference_logits.float() / 0.7)
        .log_softmax(dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(labels == -100, 0)
    )
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=2e-2, rtol=1e-2)


@pytest.mark.unit
def test_global_denominators_sum_document_weights_and_masked_tokens():
    packs = [
        {
            "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
            "loss_masks": torch.tensor([0, 1, 0, 0, 1, 0]),
            "loss_weights": [1.0, 1.0],
        },
        {
            "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
            "loss_masks": torch.tensor([0, 0]),
            "loss_weights": [1.0],
        },
    ]

    sequences, tokens = count_global_denominators(packs, dp_group=None, device="cpu")

    assert sequences.item() == 3
    assert tokens.item() == 2


@pytest.mark.unit
def test_global_denominators_count_an_episode_once_across_its_trajectories():
    packs = [
        {
            "cu_seqlens": torch.tensor([0, 3, 6, 9], dtype=torch.int32),
            "loss_masks": torch.ones(9, dtype=torch.int32),
            # Two spans of one attempt, then a whole single-span attempt.
            "loss_weights": [0.5, 0.5, 1.0],
        },
    ]

    sequences, _ = count_global_denominators(packs, dp_group=None, device="cpu")

    assert sequences.item() == 2


@pytest.mark.unit
def test_global_denominators_exclude_padding_documents():
    packs = [
        {
            "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
            "loss_masks": torch.tensor([1, 1, 0, 0, 0]),
            "loss_weights": [1.0, 0.0],
        },
    ]

    sequences, tokens = count_global_denominators(packs, dp_group=None, device="cpu")

    assert sequences.item() == 1
    assert tokens.item() == 2


@pytest.mark.unit
def test_global_denominators_reject_non_source_aligned_masks():
    packs = [
        {
            "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
            "loss_masks": torch.tensor([0, 1, 0, 1]),
            "loss_weights": [1.0, 1.0],
        },
    ]

    with pytest.raises(ValueError, match="packed token count 6"):
        count_global_denominators(packs, dp_group=None, device="cpu")


@pytest.mark.unit
def test_global_denominators_reject_weights_that_do_not_match_documents():
    packs = [
        {
            "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
            "loss_masks": torch.ones(6, dtype=torch.int32),
            "loss_weights": [1.0],
        },
    ]

    with pytest.raises(ValueError, match="do not match 2 packed documents"):
        count_global_denominators(packs, dp_group=None, device="cpu")


@pytest.mark.unit
def test_policy_reductions_use_matching_global_denominators():
    values = torch.tensor([[2.0, 4.0, 10.0, 14.0]])
    mask = torch.ones_like(values)
    document_ids = torch.tensor([[1, 1, 2, 2]])

    token_loss = reduce_token_mean(values, mask, torch.tensor(8.0))
    sequence_loss = reduce_weighted_sequence_mean(
        values,
        mask,
        document_ids,
        num_documents=2,
        global_sequences=torch.tensor(4.0),
        cp_group=None,
        loss_weights=torch.ones(2),
    )

    torch.testing.assert_close(token_loss, torch.tensor(3.75))
    torch.testing.assert_close(sequence_loss, torch.tensor(3.75))


@pytest.mark.unit
def test_sequence_reduction_weights_an_episode_by_its_trajectory_count():
    # Two spans of one attempt (means 3 and 12) beside a single-span attempt (mean 20).
    values = torch.tensor([[2.0, 4.0, 10.0, 14.0, 20.0, 20.0]])
    mask = torch.ones_like(values)
    document_ids = torch.tensor([[1, 1, 2, 2, 3, 3]])

    per_episode = reduce_weighted_sequence_mean(
        values,
        mask,
        document_ids,
        num_documents=3,
        global_sequences=torch.tensor(2.0),
        cp_group=None,
        loss_weights=torch.tensor([0.5, 0.5, 1.0]),
    )
    per_trajectory = reduce_weighted_sequence_mean(
        values,
        mask,
        document_ids,
        num_documents=3,
        global_sequences=torch.tensor(3.0),
        cp_group=None,
        loss_weights=torch.tensor([1.0, 1.0, 1.0]),
    )

    # (0.5*3 + 0.5*12 + 20) / 2 weights each attempt once.
    torch.testing.assert_close(per_episode, torch.tensor(13.75))
    # (3 + 12 + 20) / 3 weights each span once.
    torch.testing.assert_close(per_trajectory, torch.tensor(35.0 / 3))


@pytest.mark.unit
def test_padding_documents_do_not_shift_the_sequence_reduction():
    values = torch.tensor([[2.0, 4.0, 99.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    document_ids = torch.tensor([[1, 1, 2, 2]])

    reduced = reduce_weighted_sequence_mean(
        values,
        mask,
        document_ids,
        num_documents=2,
        global_sequences=torch.tensor(1.0),
        cp_group=None,
        loss_weights=torch.tensor([1.0, 0.0]),
    )

    torch.testing.assert_close(reduced, torch.tensor(3.0))


@pytest.mark.unit
def test_gspo_sequence_means_preserve_gradient_flow():
    values = torch.tensor([[1.0, 3.0, 10.0, 14.0]], requires_grad=True)
    mask = torch.ones_like(values)
    document_ids = torch.tensor([[1, 1, 2, 2]])

    means = sequence_mean_at_tokens(
        values,
        mask,
        document_ids,
        num_documents=2,
        cp_group=None,
    )
    means.sum().backward()

    torch.testing.assert_close(means, torch.tensor([[2.0, 2.0, 12.0, 12.0]]))
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("normalization_unit", "expected_pg_loss"),
    [
        ("episode", 17.0 / 3.0),
        ("token", 7.5),
    ],
)
def test_policy_normalization_keeps_entropy_and_kl_sequence_normalized(
    monkeypatch,
    normalization_unit,
    expected_pg_loss,
):
    trainer = ActorNeMoTrainer.__new__(ActorNeMoTrainer)
    trainer.args = type(
        "Args",
        (),
        {
            "loss_normalization_unit": normalization_unit,
            "rollout_temperature": 1.0,
            "old_logprob_source": "rollout",
            "advantage_estimator": "grpo",
            "eps_clip": 0.2,
            "eps_clip_high": None,
            "policy_surrogate": "dual-clip",
            "eps_clip_c": 3.0,
            "entropy_coef": 1.0,
            "kl_loss_coef": 1.0,
            "use_unbiased_kl": False,
            "kl_loss_type": "k3",
        },
    )()
    trainer.cp_group = None

    monkeypatch.setattr(
        "slim.backends.nemo.actor.selective_log_probs",
        lambda logits, labels, temperature: torch.zeros_like(labels, dtype=torch.float32),
    )
    monkeypatch.setattr(
        "slim.backends.nemo.actor.compute_policy_loss",
        lambda *args, **kwargs: (
            torch.tensor([[2.0, 4.0, 10.0, 14.0]]),
            torch.zeros(1, 4),
        ),
    )
    monkeypatch.setattr(
        "slim.backends.nemo.actor.entropy_from_logits",
        lambda logits: torch.tensor([[2.0, 10.0, 10.0, 10.0]]),
    )
    monkeypatch.setattr(
        "slim.backends.nemo.actor.compute_approx_kl",
        lambda *args, **kwargs: torch.tensor([[4.0, 20.0, 20.0, 20.0]]),
    )

    fields = {
        "labels": torch.ones(1, 4, dtype=torch.long),
        "loss_masks": torch.ones(1, 4),
        "document_ids": torch.tensor([[1, 2, 2, 2]]),
        "rollout_log_probs": torch.zeros(1, 4),
        "advantages": torch.ones(1, 4),
        "ref_log_probs": torch.zeros(1, 4),
    }
    _, metrics = trainer._policy_loss(
        torch.zeros(1, 4, 2),
        fields,
        num_documents=2,
        global_sequences=torch.tensor(2.0),
        global_tokens=torch.tensor(4.0),
        loss_weights=torch.ones(2),
    )

    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(expected_pg_loss))
    torch.testing.assert_close(metrics["entropy_loss"], torch.tensor(6.0))
    torch.testing.assert_close(metrics["kl_loss"], torch.tensor(12.0))
