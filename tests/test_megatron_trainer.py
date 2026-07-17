from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import slim.backends.megatron.trainer as trainer_module
from slim.backends.megatron.batch_layout import PackedRange, TrajectoryLayout
from slim.backends.megatron.loss import PolicyLossResult
from slim.backends.megatron.trainer import (
    MegatronTrainer,
    _PreparedBatch,
    _rebalance_microbatches,
    _resolve_distributed_checkpoint,
    _split_by_capacity,
)

pytestmark = pytest.mark.unit


class _Group:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _trainer(*, is_vlm, sequence_parallel=False, tp_rank=0):
    trainer = MegatronTrainer.__new__(MegatronTrainer)
    trainer.is_vlm = is_vlm
    trainer.bundle = SimpleNamespace(
        topology=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            tensor_model_parallel_size=2 if sequence_parallel else 1,
            context_parallel_size=2,
        )
    )
    trainer.pg_collection = SimpleNamespace(
        tp=_Group(2 if sequence_parallel else 1, tp_rank),
        cp=_Group(2, 0),
    )
    trainer.router_replay = None
    trainer.args = SimpleNamespace(
        rollout_temperature=1.0,
        calculate_per_token_loss=False,
        old_logprob_source="actor",
        entropy_coef=0.0,
    )
    return trainer


def _batch(*, cp_indices=None, visual_kwargs=None, routes=None):
    cp_indices = cp_indices if cp_indices is not None else torch.tensor([0, 2])
    tensors = {
        "tokens": torch.tensor([[10, 11, 12, 13]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
        "labels": torch.tensor([[11, 12, 13, -100]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "advantages": torch.tensor([[2.0, 3.0, 4.0, 0.0]]),
        "actor_old_log_probs": torch.tensor([[-1.0, -2.0, -3.0, 0.0]]),
    }
    if routes is not None:
        tensors["rollout_routed_experts"] = routes
    return _PreparedBatch(
        layout=SimpleNamespace(total_tokens=4),
        episodes=[],
        tensors=tensors,
        visual_kwargs=visual_kwargs or {},
        packed_seq_params=object(),
        cp_indices=cp_indices,
        sequence_ids=torch.tensor([[0, 0, 1, 1]]),
        sequence_active_counts=torch.tensor([1.0, 1.0]),
    )


def test_split_by_capacity_uses_first_fit_and_rejects_oversize():
    episodes = [
        SimpleNamespace(tokens=range(length))
        for length in (100, 200, 300)
    ]

    assert [list(map(lambda item: len(item.tokens), batch)) for batch in _split_by_capacity(episodes, 300)] == [
        [100, 200],
        [300],
    ]
    with pytest.raises(ValueError, match="exceeds"):
        _split_by_capacity(episodes, 299)


def test_split_by_capacity_charges_physical_alignment():
    episodes = [SimpleNamespace(tokens=range(5)), SimpleNamespace(tokens=range(3))]

    assert [
        [len(item.tokens) for item in batch]
        for batch in _split_by_capacity(episodes, 8, alignment=4)
    ] == [[5], [3]]
    with pytest.raises(ValueError, match="Aligned episode length 8"):
        _split_by_capacity([episodes[0]], 7, alignment=4)


def test_rebalance_microbatches_splits_bins_without_empty_batches():
    episodes = [SimpleNamespace(tokens=[index]) for index in range(4)]
    batches = _rebalance_microbatches([episodes[:3], episodes[3:]], 3)

    assert len(batches) == 3
    assert sorted(item.tokens[0] for batch in batches for item in batch) == [0, 1, 2, 3]
    assert all(batches)


def test_resolve_distributed_checkpoint_accepts_direct_or_latest_root(tmp_path):
    direct = tmp_path / "direct"
    old = tmp_path / "save" / "rollout_00000001"
    latest = tmp_path / "save" / "rollout_00000003"
    incomplete = tmp_path / "save" / "rollout_00000004"
    for path in (direct, old, latest, incomplete):
        path.mkdir(parents=True, exist_ok=True)
    complete = {str(direct), str(old), str(latest)}

    assert _resolve_distributed_checkpoint(
        direct,
        is_checkpoint=complete.__contains__,
    ) == direct
    assert _resolve_distributed_checkpoint(
        tmp_path / "save",
        is_checkpoint=complete.__contains__,
    ) == latest
    with pytest.raises(ValueError, match="neither"):
        _resolve_distributed_checkpoint(
            incomplete,
            is_checkpoint=complete.__contains__,
        )


def test_vlm_forward_keeps_full_thd_input_for_internal_cp_slicing():
    trainer = _trainer(is_vlm=True)
    batch = _batch(visual_kwargs={"pixel_values": torch.ones(2, 3)})
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return torch.ones(1, 2, 4)

    trainer._model_forward(model, batch, training=False, replay_routes=False)

    assert calls[0]["input_ids"].tolist() == [[10, 11, 12, 13]]
    assert calls[0]["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert calls[0]["pixel_values"].shape == (2, 3)


def test_text_forward_slices_tokens_and_positions_with_cp_index():
    trainer = _trainer(is_vlm=False)
    batch = _batch(cp_indices=torch.tensor([3, 1]))
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return torch.ones(1, 2, 4)

    trainer._model_forward(model, batch, training=False, replay_routes=False)

    assert calls[0]["input_ids"].tolist() == [[13, 11]]
    assert calls[0]["position_ids"].tolist() == [[3, 1]]


def test_routing_replay_applies_cp_before_sp():
    trainer = _trainer(is_vlm=True, sequence_parallel=True, tp_rank=1)
    replay = SimpleNamespace(prepared=[], hooks=[])
    replay.prepare_forward = replay.prepared.append
    replay.register_backward_hook = replay.hooks.append
    trainer.router_replay = replay
    routes = torch.arange(8, dtype=torch.int32).reshape(1, 4, 2, 1)
    batch = _batch(cp_indices=torch.tensor([3, 0, 2, 1]), routes=routes)
    output = torch.ones(1, 4, 8, requires_grad=True)

    trainer._model_forward(
        lambda **_kwargs: output,
        batch,
        training=True,
        replay_routes=True,
    )

    expected_cp = routes.index_select(1, batch.cp_indices).squeeze(0)
    assert torch.equal(replay.prepared[0], expected_cp.tensor_split(2, dim=0)[1])
    assert replay.hooks == [output]


def test_local_policy_inputs_share_the_exact_cp_index(monkeypatch):
    trainer = _trainer(is_vlm=False)
    batch = _batch(cp_indices=torch.tensor([2, 0]))
    captured = {}

    def fake_selected(logits, labels, **_kwargs):
        captured["labels"] = labels
        return torch.tensor([[-0.3, -0.1]])

    monkeypatch.setattr(trainer_module, "selected_log_probs", fake_selected)
    inputs = trainer._local_policy_inputs(
        torch.ones(1, 2, 8),
        batch,
        global_active_tokens=2,
        global_num_sequences=2,
    )

    assert captured["labels"].tolist() == [[13, 11]]
    assert inputs["loss_mask"].tolist() == [[1.0, 1.0]]
    assert inputs["advantages"].tolist() == [[4.0, 2.0]]
    assert inputs["old_log_probs"].tolist() == [[-3.0, -1.0]]
    assert inputs["objective_weight"].tolist() == [[1.0, 1.0]]
    assert inputs["sample_mean_weight"].tolist() == [[1.0, 1.0]]


def test_entropy_reads_logits_before_selected_log_probs_mutates_them(monkeypatch):
    trainer = _trainer(is_vlm=False)
    trainer.args.entropy_coef = 0.1
    batch = _batch()
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    original = logits.clone()
    events = []

    def fake_entropy(value, **_kwargs):
        events.append("entropy")
        assert torch.equal(value, original)
        return torch.ones(1, 2)

    def fake_selected(value, _labels, **_kwargs):
        events.append("selected")
        value.zero_()
        return torch.zeros(1, 2)

    monkeypatch.setattr(trainer_module, "vocab_parallel_entropy", fake_entropy)
    monkeypatch.setattr(trainer_module, "selected_log_probs", fake_selected)

    inputs = trainer._local_policy_inputs(
        logits,
        batch,
        global_active_tokens=2,
        global_num_sequences=2,
    )

    assert events == ["entropy", "selected"]
    assert inputs["entropy"].tolist() == [[1.0, 1.0]]


def test_forward_loss_returns_mcore_per_token_contract(monkeypatch):
    trainer = _trainer(is_vlm=False)
    trainer.args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.2,
        policy_surrogate="ppo_clip",
        eps_clip_c=None,
        entropy_coef=0.0,
    )
    batch = _batch()
    logits = torch.ones(1, 2, 8, requires_grad=True)
    monkeypatch.setattr(
        trainer,
        "_model_forward",
        lambda *_args, **_kwargs: logits,
    )
    monkeypatch.setattr(
        trainer,
        "_local_policy_inputs",
        lambda *_args, **_kwargs: {
            "current_log_probs": torch.ones(1, 2),
            "old_log_probs": torch.ones(1, 2),
            "advantages": torch.ones(1, 2),
            "loss_mask": torch.tensor([[1.0, 0.0]]),
            "objective_weight": torch.ones(1, 2),
            "sample_mean_weight": torch.ones(1, 2),
        },
    )
    result = PolicyLossResult(
        loss=torch.tensor(3.0, requires_grad=True),
        metrics={"pg_loss": torch.tensor(2.0)},
        num_active_tokens=torch.tensor(1.0),
        current_log_probs=torch.ones(1, 2),
    )
    monkeypatch.setattr(trainer_module, "policy_loss", lambda **_kwargs: result)

    output, loss_func = trainer._forward_step(
        global_active_tokens=1,
        global_num_sequences=1,
    )(iter([batch]), object())
    loss, num_tokens, metrics = loss_func(output)

    assert output is logits
    assert loss is result.loss
    assert num_tokens.dtype == torch.int32
    assert num_tokens.item() == 1
    assert metrics["loss"].item() == 3.0
    assert metrics["pg_loss"].item() == 2.0


def test_attach_log_probs_restores_edge_aligned_episode_values():
    trainer = _trainer(is_vlm=False)
    trainer.pg_collection.pp = _Group(1, 0)
    episodes = [SimpleNamespace(), SimpleNamespace()]
    trajectories = (
        TrajectoryLayout(
            episode_index=0,
            logical_tokens=PackedRange(0, 3),
            tokens=PackedRange(0, 3),
            edges=PackedRange(0, 2),
            physical_tokens=PackedRange(0, 4),
            media_ranges=(),
            placeholder_ranges=(),
        ),
        TrajectoryLayout(
            episode_index=1,
            logical_tokens=PackedRange(3, 5),
            tokens=PackedRange(4, 6),
            edges=PackedRange(4, 5),
            physical_tokens=PackedRange(4, 6),
            media_ranges=(),
            placeholder_ranges=(),
        ),
    )
    batch = _batch()
    batch.episodes = episodes
    batch.layout = SimpleNamespace(trajectories=trajectories)

    trainer._attach_log_probs(
        [batch],
        [{"log_probs": torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0, 0.0]])}],
        attribute="_actor_old_log_probs",
    )

    assert episodes[0]._actor_old_log_probs.tolist() == [1.0, 2.0]
    assert episodes[1]._actor_old_log_probs.tolist() == [3.0]
