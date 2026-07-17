import torch

from slim.backends.megatron.loss import (
    _all_reduce_sum_forward_identity_backward,
    objective_weights,
    policy_loss,
    vocab_parallel_entropy,
)


def test_sample_mean_weights_survive_global_token_normalization():
    mask = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.float32)
    sequence_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    counts = torch.tensor([2, 3])
    weights = objective_weights(
        loss_mask=mask,
        sequence_ids=sequence_ids,
        sequence_active_counts=counts,
        global_active_tokens=5,
        global_num_sequences=2,
        calculate_per_token_loss=False,
    )
    values = torch.tensor([2.0, 4.0, 99.0, 3.0, 6.0, 9.0])
    mcore_normalized = (values * weights).sum() / 5
    expected = ((2.0 + 4.0) / 2 + (3.0 + 6.0 + 9.0) / 3) / 2
    assert torch.isclose(mcore_normalized, torch.tensor(expected))


def test_token_sum_weights_survive_global_token_normalization():
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
    weights = objective_weights(
        loss_mask=mask,
        sequence_ids=torch.tensor([0, 0, 0, 1]),
        sequence_active_counts=torch.tensor([2, 1]),
        global_active_tokens=3,
        global_num_sequences=2,
        calculate_per_token_loss=True,
    )
    values = torch.tensor([2.0, 4.0, 99.0, 6.0])
    assert torch.isclose((values * weights).sum() / 3, torch.tensor(6.0))


def test_vocab_parallel_entropy_matches_dense_entropy_without_tp():
    logits = torch.tensor([[1.0, 2.0, -1.0], [0.0, 0.5, 0.5]], requires_grad=True)
    expected = torch.distributions.Categorical(logits=logits).entropy()
    actual = vocab_parallel_entropy(logits)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert logits.grad is not None


def test_final_tp_entropy_sum_has_identity_backward(monkeypatch):
    def fake_all_reduce(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group == "tp"
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    value = torch.tensor([3.0], requires_grad=True)
    reduced = _all_reduce_sum_forward_identity_backward(value, group="tp")

    assert reduced.item() == 6.0
    reduced.backward()
    assert value.grad.item() == 1.0


def test_policy_loss_requires_reference_when_kl_is_enabled():
    values = torch.ones(2)
    try:
        policy_loss(
            current_log_probs=values,
            old_log_probs=values,
            advantages=values,
            loss_mask=values,
            objective_weight=values,
            eps_clip=0.2,
            eps_clip_high=0.2,
            policy_surrogate="ppo_clip",
            kl_loss_coef=0.1,
        )
    except ValueError as exc:
        assert "reference_log_probs" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_entropy_uses_sample_mean_weight_independently_from_policy_weight():
    current = torch.zeros(3)
    result = policy_loss(
        current_log_probs=current,
        old_log_probs=current,
        advantages=torch.ones(3),
        loss_mask=torch.ones(3),
        objective_weight=torch.tensor([2.0, 2.0, 2.0]),
        sample_mean_weight=torch.tensor([0.5, 0.5, 1.0]),
        eps_clip=0.2,
        eps_clip_high=0.2,
        policy_surrogate="ppo_clip",
        entropy=torch.tensor([1.0, 3.0, 5.0]),
        entropy_coef=0.1,
    )

    assert torch.isclose(result.metrics["entropy"], torch.tensor(7.0))
