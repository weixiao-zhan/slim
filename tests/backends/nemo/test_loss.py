# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from functools import partial
from types import SimpleNamespace

from slim.backends.nemo.actor import ActorNeMoTrainer
from slim.backends.nemo.loss import (
    count_global_denominators,
    reduce_token_mean,
    reduce_weighted_sequence_mean,
    selective_log_probs,
    sequence_mean_at_tokens,
)
from slim.utils.ppo_utils import compute_score_centering_loss, compute_surrogate_weight


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_fused_selective_log_probs_preserves_fp32_values_and_bf16_gradients():
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 251, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, 5, 7], [4, 3, 2]], device="cuda")
    output_gradient = torch.randn(2, 3, device="cuda")

    actual = selective_log_probs(logits, labels, temperature=0.7)
    expected = (reference_logits.float() / 0.7).log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=2e-2, rtol=1e-2)


def _topk_columns(device):
    labels = torch.tensor([[1, 5, 7], [4, 3, 2]], device=device)
    # The top-K repeats an id and contains the label, as a sampler top-K usually does.
    topk = torch.tensor(
        [
            [[1, 5, 5], [6, 5, 0], [7, 2, 9]],
            [[4, 0, 8], [3, 6, 1], [2, 2, 11]],
        ],
        device=device,
    )
    return torch.cat((labels.unsqueeze(-1), topk), dim=-1)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_fused_selective_log_probs_reads_extra_columns_from_one_softmax():
    torch.manual_seed(23)
    logits = torch.randn(2, 3, 251, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    columns = _topk_columns("cuda")
    output_gradient = torch.randn(2, 3, 4, device="cuda")

    actual = selective_log_probs(logits, columns, temperature=0.7)
    expected = (reference_logits.float() / 0.7).log_softmax(dim=-1).gather(-1, columns)
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=2e-2, rtol=1e-2)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_fused_selective_log_probs_reads_every_column_before_writing_the_gradient():
    # Enough rows that the gathered column reads race the in-place gradient writes without a barrier.
    torch.manual_seed(47)
    rows, vocab = 2048, 32768
    logits = (torch.randn(1, rows, vocab, device="cuda") * 3).to(torch.bfloat16)
    columns = torch.randint(0, vocab, (1, rows, 33), device="cuda")

    expected = logits.float().log_softmax(dim=-1).gather(-1, columns)
    actual = selective_log_probs(logits.clone().requires_grad_(True), columns)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_fused_selective_log_probs_keeps_bf16_precision_when_the_topk_contains_the_label():
    # Score centering on a confident token: the label and its top-K column carry nearly opposite
    # gradients, so the true logit gradient is small and must not be the residue of a bf16 cancellation.
    torch.manual_seed(53)
    rows, vocab = 256, 4096
    logits = (torch.randn(rows, vocab, device="cuda") * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (rows,), device="cuda")
    logits[torch.arange(rows), labels] = logits.float().max(dim=-1).values.to(torch.bfloat16) + 12
    others = logits.float().scatter(-1, labels[:, None], float("-inf")).topk(7, dim=-1).indices
    columns = torch.cat((labels[:, None], labels[:, None], others), dim=-1)
    output_gradient = torch.cat((torch.ones(rows, 1), -torch.ones(rows, 1), 1e-3 * torch.randn(rows, 7)), dim=-1)
    output_gradient = output_gradient.to("cuda")

    fused = logits.clone().requires_grad_(True)
    selective_log_probs(fused, columns).backward(output_gradient)
    reference = logits.double().requires_grad_(True)
    reference.log_softmax(dim=-1).gather(-1, columns).backward(output_gradient.double())

    error = (fused.grad.double() - reference.grad).norm() / reference.grad.norm()
    assert error < 1e-2


def _surrogate_weight(policy_surrogate, eps_clip, eps_clip_high, eps_clip_c=None):
    return partial(
        compute_surrogate_weight,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
        policy_surrogate=policy_surrogate,
        eps_clip_c=eps_clip_c,
    )


def _score_centering_gradient(
    logits, label, topk_ids, sampler_log_probs, advantage, eps_clip=0.0, policy_surrogate="cis"
):
    """Gradient of the score-centered policy loss of one token with respect to its logits."""
    logits = logits.detach().cuda().requires_grad_(True)
    topk_ids, sampler_log_probs = topk_ids.cuda(), sampler_log_probs.cuda()
    scored_token_ids = torch.cat((torch.tensor([label], device="cuda"), topk_ids))
    all_log_probs = selective_log_probs(logits.unsqueeze(0), scored_token_ids.unsqueeze(0))[0]
    log_prob, topk_log_probs = all_log_probs[0], all_log_probs[1:]
    advantage = torch.tensor(advantage, device="cuda")
    surrogate_weight = _surrogate_weight(policy_surrogate, eps_clip, eps_clip)
    weight, _ = surrogate_weight((log_prob - sampler_log_probs[label]).exp(), advantage)
    policy_loss = -advantage * weight * log_prob
    values, topk_mass, rho = compute_score_centering_loss(
        topk_log_probs,
        sampler_log_probs[topk_ids],
        advantage,
        surrogate_weight,
    )
    (policy_loss + values).backward()
    return logits.grad.cpu(), topk_mass.cpu(), rho.cpu()


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_full_vocabulary_score_centering_steps_toward_the_sampled_token_from_the_sampler():
    torch.manual_seed(29)
    vocab = 11
    logits = torch.randn(vocab)
    sampler_log_probs = torch.randn(vocab).log_softmax(dim=-1)

    gradient, _, rho = _score_centering_gradient(
        logits, 3, torch.arange(vocab), sampler_log_probs, 0.8
    )

    # With the whole vocabulary as top-K, $-A (s_x - \bar s)$ in logit space is $-A (1_x - q)$.
    expected = -0.8 * (torch.nn.functional.one_hot(torch.tensor(3), vocab) - sampler_log_probs.exp())
    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(rho, torch.tensor(1.0))


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
@pytest.mark.parametrize(
    ("policy_surrogate", "eps_clip", "advantage"),
    [
        ("cis", 0.0, -1.3),
        ("cis", 0.2, -1.3),
        ("mis", 0.2, -1.3),
        ("ppo_clip", 0.2, -1.3),
        ("ppo_clip", 0.2, 0.9),
    ],
)
def test_top_k_score_centering_is_exact_when_the_sampler_tail_matches_the_trainer_shape(
    policy_surrogate, eps_clip, advantage
):
    torch.manual_seed(31)
    vocab = 13
    logits = torch.randn(vocab)
    trainer = logits.softmax(dim=-1)
    topk_ids = trainer.argsort(descending=True)[:4]
    in_topk = torch.zeros(vocab, dtype=torch.bool)
    in_topk[topk_ids] = True
    sampler = torch.where(in_topk, torch.rand(vocab), trainer)
    sampler[in_topk] = sampler[in_topk] / sampler[in_topk].sum() * 0.7
    sampler[~in_topk] = trainer[~in_topk] / trainer[~in_topk].sum() * 0.3

    top_k, _, _ = _score_centering_gradient(
        logits, 5, topk_ids, sampler.log(), advantage, eps_clip, policy_surrogate
    )
    full, _, _ = _score_centering_gradient(
        logits, 5, torch.arange(vocab), sampler.log(), advantage, eps_clip, policy_surrogate
    )

    torch.testing.assert_close(top_k, full)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("policy_surrogate", "eps_clip"),
    [("cis", 0.0), ("cis", 0.2), ("mis", 0.2), ("ppo_clip", 0.2)],
)
def test_score_centering_vanishes_on_policy(policy_surrogate, eps_clip):
    torch.manual_seed(37)
    logits = torch.randn(9)
    topk_log_probs = logits.log_softmax(dim=-1)[:4]

    values, topk_mass, rho = compute_score_centering_loss(
        topk_log_probs.requires_grad_(True),
        topk_log_probs.detach(),
        torch.tensor(2.0),
        _surrogate_weight(policy_surrogate, eps_clip, eps_clip),
    )
    values.backward()

    torch.testing.assert_close(topk_log_probs.grad, torch.zeros(4))
    torch.testing.assert_close(topk_mass, topk_log_probs.detach().exp().sum())
    torch.testing.assert_close(rho, torch.tensor(1.0))


@pytest.mark.unit
@pytest.mark.parametrize("policy_surrogate", ["is", "cis"])
def test_score_centering_vanishes_under_unclipped_importance_weights(policy_surrogate):
    torch.manual_seed(41)
    topk_log_probs = torch.randn(9).log_softmax(dim=-1)[:4].requires_grad_(True)
    sampler_topk_log_probs = torch.randn(9).log_softmax(dim=-1)[:4]

    # A clamp wide enough to never bind turns `cis` into exact IS, $w(r) = r$.
    values, _, _ = compute_score_centering_loss(
        topk_log_probs,
        sampler_topk_log_probs,
        torch.tensor(1.0),
        _surrogate_weight(policy_surrogate, 1.0, 1e6),
    )
    values.backward()

    torch.testing.assert_close(topk_log_probs.grad, torch.zeros(4), atol=1e-6, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("policy_surrogate", "eps_clip_c"),
    [("cis", None), ("mis", None), ("ppo_clip", None), ("ppo_clip", 3.0)],
)
@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_score_centering_stays_finite_when_the_sampler_reports_an_underflowed_probability(
    policy_surrogate, eps_clip_c, advantage
):
    topk_log_probs = torch.tensor([-0.5, -2.0, -3.0, -4.0], requires_grad=True)
    # SGLang reports a sampler probability that underflows float32 as the float32 minimum.
    sampler_topk_log_probs = torch.tensor([-0.6, -1.9, -3.2, torch.finfo(torch.float32).min])

    values, _, _ = compute_score_centering_loss(
        topk_log_probs,
        sampler_topk_log_probs,
        torch.tensor(advantage),
        _surrogate_weight(policy_surrogate, 0.2, 0.2, eps_clip_c),
    )
    values.backward()

    assert torch.isfinite(values).all()
    assert torch.isfinite(topk_log_probs.grad).all()


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused kernel")
def test_policy_loss_adds_score_centering_and_reports_topk_statistics():
    torch.manual_seed(43)
    vocab = 7
    trainer = ActorNeMoTrainer.__new__(ActorNeMoTrainer)
    trainer.args = SimpleNamespace(
        loss_normalization_unit="token",
        rollout_temperature=1.0,
        old_logprob_source="rollout",
        advantage_estimator="grpo",
        eps_clip=0.0,
        eps_clip_high=0.0,
        eps_clip_c=None,
        policy_surrogate="cis",
        use_score_centering=vocab,
        entropy_coef=0.0,
        kl_loss_coef=0.0,
    )
    trainer.cp_group = None
    # The trainer passes the fields at the loss positions only.
    logits = torch.randn(1, 2, vocab, device="cuda", requires_grad=True)
    sampler_log_probs = torch.randn(1, 2, vocab, device="cuda").log_softmax(dim=-1)
    labels = torch.tensor([[2, 5]], device="cuda")
    fields = {
        "labels": labels,
        "loss_masks": torch.ones(1, 2, device="cuda"),
        "document_ids": torch.ones(1, 2, dtype=torch.long, device="cuda"),
        "rollout_log_probs": sampler_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        "rollout_topk_log_probs": sampler_log_probs,
        "rollout_topk_token_ids": torch.arange(vocab, device="cuda").expand(1, 2, vocab),
        "advantages": torch.tensor([[0.5, -1.0]], device="cuda"),
    }

    loss, metrics = trainer._policy_loss(
        logits,
        fields,
        num_documents=1,
        global_sequences=torch.tensor(1.0, device="cuda"),
        global_tokens=torch.tensor(2.0, device="cuda"),
        loss_weights=torch.ones(1, device="cuda"),
    )
    loss.backward()

    sampled = torch.nn.functional.one_hot(labels, vocab)
    expected = -fields["advantages"].unsqueeze(-1) * (sampled - sampler_log_probs.exp()) / 2
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(metrics["score_centering/topk_mass"], torch.tensor(1.0, device="cuda"))
    torch.testing.assert_close(metrics["score_centering/rho"], torch.tensor(1.0, device="cuda"))


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
            # Two trajectories of one attempt, then a whole single-trajectory attempt.
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
    # Two trajectories of one attempt (means 3 and 12) beside a single-trajectory attempt (mean 20).
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
    # (3 + 12 + 20) / 3 weights each trajectory once.
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
            "use_score_centering": None,
            "entropy_coef": 1.0,
            "kl_loss_coef": 1.0,
            "use_unbiased_kl": False,
            "kl_loss_type": "k3",
        },
    )()
    trainer.cp_group = None

    monkeypatch.setattr(
        "slim.backends.nemo.actor.selective_log_probs",
        lambda logits, labels, temperature: torch.ones_like(labels, dtype=torch.float32),
    )
    # With a unit score and A = -1, the mocked weights are the per-token policy losses.
    monkeypatch.setattr(
        "slim.backends.nemo.actor.compute_surrogate_weight",
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
        "advantages": -torch.ones(1, 4),
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
