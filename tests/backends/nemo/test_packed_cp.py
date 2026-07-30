# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from nemo_automodel.components.distributed.blockdiag_cp import make_cp_blockdiag_batch_and_ctx

from slim.backends.nemo.base import NeMoTrainer
from slim.backends.nemo.models.qwen3_5 import (
    _dummy_visual_inputs,
    _global_media_presence,
    _install_attention_dispatch,
    _install_moe_cp,
    _install_primary_shard,
    _vision_sync_group,
    build_packed_position_ids,
    install_packed_cp,
)
from slim.backends.nemo.packed_cp_forward import (
    build_packed_cp_sharder,
)
from slim.utils.types import Episode
from tests.backends.nemo.run_packed_cp_qualification import _compare_with_baseline
from tests.backends.nemo.run_training_qualification import (
    _heterogeneous_rollout_partition,
    _uses_local_vision,
)
from tests.prepare_nemo_heterogeneous_rollout import build_heterogeneous_records


NUM_GPUS = 0


@pytest.mark.unit
def test_fixed_heterogeneous_records_assign_vision_to_dp1_and_center_every_reward():
    records = [{"reward": 0.0, "multimodal_inputs": None} for _ in range(8)]
    records.extend({"reward": 1.0, "multimodal_inputs": {"pixel_values": torch.ones(1)}} for _ in range(4))

    batch, vision_indices = build_heterogeneous_records(
        records,
        batch_size=32,
        vision_episodes=4,
        layout_stride=8,
    )

    assert vision_indices == [1, 9, 17, 25]
    assert [index for index, record in enumerate(batch) if record["multimodal_inputs"]] == vision_indices
    for start in range(0, len(batch), 8):
        rewards = torch.tensor([record["reward"] for record in batch[start : start + 8]])
        advantages = rewards - rewards.mean()
        assert rewards.min() == 0
        assert rewards.max() == 1
        assert torch.all(advantages != 0)


@pytest.mark.unit
def test_heterogeneous_multimodal_assignment_keeps_cp_peers_aligned():
    cli = Namespace(heterogeneous_multimodal=True, multimodal=False)

    assert [_uses_local_vision(cli, dp_rank) for dp_rank in range(4)] == [False, True, False, False]


@pytest.mark.unit
@pytest.mark.parametrize("dp_size", [8, 4])
def test_heterogeneous_rollout_partition_assigns_vision_only_to_dp1(dp_size):
    vision_indices = {1, 9, 17, 25}
    episodes = [
        Episode(
            tokens=[index, index + 1],
            loss_mask=[1],
            multimodal_inputs={"pixel_values": torch.ones(1, 2)} if index in vision_indices else None,
        )
        for index in range(256)
    ]
    for episode in episodes:
        episode.finalize_source_token_alignment()
        episode.set_train_targets([1.0, 0.0])

    partitions = [_heterogeneous_rollout_partition(episodes, dp_rank=dp_rank, dp_size=dp_size) for dp_rank in range(dp_size)]

    assert sum(len(partition) for partition, _ in partitions) == 256
    for dp_rank, (partition, uses_vision) in enumerate(partitions):
        vision_count = sum(bool(episode.multimodal_inputs) for episode in partition)
        assert uses_vision == (dp_rank == 1)
        assert vision_count == (4 if uses_vision else 0)


@pytest.mark.unit
def test_global_media_presence_uses_remote_modality_flags(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_reduce(flags, *, op, group):
        flags[0] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert _global_media_presence(
        pixel_values=None,
        pixel_values_videos=torch.ones(1),
        device=torch.device("cpu"),
        group=object(),
    ) == (True, True)


@pytest.mark.unit
def test_vision_sync_uses_flat_dp_shard_cp_group(monkeypatch):
    device_mesh = object()
    group = object()
    flat_dp_shard_cp_mesh = SimpleNamespace(get_group=lambda: group)
    calls = []

    def fake_flat_mesh(mesh, name):
        calls.append((mesh, name))
        return flat_dp_shard_cp_mesh

    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5.get_flat_mesh", fake_flat_mesh)

    assert _vision_sync_group(device_mesh) is group
    assert calls == [(device_mesh, "dp_shard_cp")]


@pytest.mark.unit
def test_dummy_visual_inputs_form_one_merged_token():
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                in_channels=3,
                temporal_patch_size=2,
                patch_size=16,
                spatial_merge_size=2,
            )
        )
    )

    pixels, grid = _dummy_visual_inputs(model, torch.device("cpu"))

    assert pixels.shape == (4, 1536)
    assert grid.tolist() == [[1, 2, 2]]


@pytest.mark.unit
def test_packed_runtime_attaches_cp_mesh_to_moe_routers(monkeypatch):
    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cp_mesh = None

    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5.MoE", FakeMoE)
    model = torch.nn.Sequential(FakeMoE())
    cp_mesh = object()

    _install_moe_cp(model, cp_mesh)

    assert model[0].cp_mesh is cp_mesh


@pytest.mark.unit
def test_packed_runtime_defaults_to_halo_kv_exchange(monkeypatch):
    calls = []
    model = nn.Module()
    model.backend = SimpleNamespace(attn="sdpa")
    model.config = SimpleNamespace(model_type="qwen3_5")
    cp_mesh = SimpleNamespace(size=lambda: 1, get_group=lambda: object())
    device_mesh = {"cp": cp_mesh}

    monkeypatch.setattr(
        "slim.backends.nemo.models.qwen3_5.configure_cp_varlen",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5._vision_sync_group", lambda mesh: object())
    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5._install_attention_dispatch", lambda model: None)
    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5._install_gdn_dispatch", lambda model, mesh: None)
    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5._install_moe_cp", lambda model, mesh: None)
    monkeypatch.setattr(
        "slim.backends.nemo.models.qwen3_5._install_synchronized_vision",
        lambda model, sync_group, cp_group: None,
    )
    monkeypatch.setattr("slim.backends.nemo.models.qwen3_5._install_primary_shard", lambda model, mesh: None)

    install_packed_cp(model, device_mesh)

    assert calls == [{"attn_backend": "flash", "kv_exchange": "halo"}]
    assert model.cp_mesh is cp_mesh


class _FakeCPMesh:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self):
        return object()


class _FakeDeviceMesh:
    mesh_dim_names = ("cp",)

    def __init__(self, cp_size=2):
        self.cp = _FakeCPMesh(cp_size)

    def __getitem__(self, name):
        assert name == "cp"
        return self.cp


def _episode(start):
    episode = Episode(tokens=[start, start + 1, start + 2], loss_mask=[1, 1], reward=1.0)
    episode.finalize_source_token_alignment()
    episode.set_train_targets([0.5, 0.5, 0.0], value_targets=[1.0, 1.0, 0.0])
    return episode


def _episode_with_length(start, length):
    episode = Episode(
        tokens=list(range(start, start + length)),
        loss_mask=[1] * (length - 1),
        reward=1.0,
    )
    episode.finalize_source_token_alignment()
    episode.set_train_targets(
        [0.5] * (len(episode.tokens) - 1) + [0.0],
        value_targets=[1.0] * (len(episode.tokens) - 1) + [0.0],
    )
    return episode


@pytest.mark.unit
def test_common_sharder_is_contiguous_block_diagonal():
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices

    sharder = build_packed_cp_sharder(_FakeDeviceMesh(), padding_token_id=17)

    assert sharder.shard_batch.func is make_cp_blockdiag_batch_and_ctx
    assert sharder.shard_batch.keywords == {"shard_primary": False}
    assert sharder.local_token_global_indices is contiguous_local_indices
    assert sharder._padding_token_id == 17


@pytest.mark.unit
def test_text_positions_restart_for_every_document():
    pack = {
        "position_ids": torch.tensor([0, 1, 0, 1, 2]),
        "multimodal_inputs": None,
    }
    model_batch = {"input_ids": torch.tensor([[10, 11, 20, 21, 22]])}

    positions = build_packed_position_ids(SimpleNamespace(), pack, model_batch)

    assert positions.tolist() == [[0, 1, 0, 1, 2]]


@pytest.mark.unit
def test_multimodal_positions_are_built_per_document_with_local_media():
    calls = []

    class Model:
        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            calls.append(batch)
            length = batch["input_ids"].shape[1]
            positions = torch.arange(length).view(1, 1, length).expand(3, 1, length)
            return {"position_ids": positions}

    pack = {
        "cu_seqlens": torch.tensor([0, 2, 5], dtype=torch.int32),
        "multimodal_inputs": {
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
        },
        "multimodal_num_items": {
            "image_grid_thw": [1, 0],
            "video_grid_thw": [0, 1],
        },
    }
    model_batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21, 22]]),
        **pack["multimodal_inputs"],
    }

    positions = build_packed_position_ids(Model(), pack, model_batch)

    assert positions.shape == (3, 1, 5)
    assert positions[0].tolist() == [[0, 1, 0, 1, 2]]
    assert calls[0]["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert "video_grid_thw" not in calls[0]
    assert "image_grid_thw" not in calls[1]
    assert calls[1]["video_grid_thw"].tolist() == [[2, 2, 2]]


@pytest.mark.unit
def test_image_grid_hws_is_promoted_for_positions_and_forward():
    calls = []

    class Model:
        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            calls.append(batch)
            length = batch["input_ids"].shape[1]
            return {"position_ids": torch.zeros(3, 1, length, dtype=torch.long)}

    pack = {
        "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
        "multimodal_inputs": {"image_grid_hws": torch.tensor([[2, 3]])},
        "multimodal_num_items": {"image_grid_hws": [1]},
    }
    model_batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "image_grid_hws": pack["multimodal_inputs"]["image_grid_hws"].clone(),
    }

    build_packed_position_ids(Model(), pack, model_batch)

    assert "image_grid_hws" not in model_batch
    assert model_batch["image_grid_thw"].tolist() == [[1, 2, 3]]
    assert calls[0]["image_grid_thw"].tolist() == [[1, 2, 3]]


@pytest.mark.unit
def test_dense_attention_dispatches_to_block_diagonal_runtime(monkeypatch):
    from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention

    attention = Qwen3NextAttention.__new__(Qwen3NextAttention)
    nn.Module.__init__(attention)
    attention.backend = SimpleNamespace(attn="sdpa")
    attention.attn_func = lambda query, key, value, **kwargs: "base"
    model = nn.Module()
    model.add_module("self_attn", attention)
    _install_attention_dispatch(model)

    monkeypatch.setattr(
        "slim.backends.nemo.models.qwen3_5.current_blockdiag_cp_state",
        lambda: object(),
    )
    monkeypatch.setattr(
        "slim.backends.nemo.models.qwen3_5.cp_blockdiag_sdpa",
        lambda *args, **kwargs: "packed",
    )

    tensor = torch.zeros(1)
    assert attention.attn_func(tensor, tensor, tensor) == "packed"


@pytest.mark.unit
@pytest.mark.parametrize("model_kind", ["dense", "moe"])
def test_primary_shard_is_contiguous_for_dense_and_moe(model_kind):
    from nemo_automodel.components.distributed.blockdiag_cp import state

    calls = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cp_mesh = _FakeCPMesh(size=2, rank=0)
            self.model_kind = model_kind

        def _embed_and_splice_for_cp(self, input_ids, **media):
            calls.append((input_ids.clone(), media))
            return input_ids.unsqueeze(-1)

        def forward(self, *, input_ids=None, inputs_embeds=None, **kwargs):
            return input_ids, inputs_embeds, kwargs

    values = torch.arange(8).view(1, 8)
    model = Model()
    _install_primary_shard(model, model.cp_mesh)
    token = state._CP_BLOCKDIAG_STATE.set({"model_state": object()})
    try:
        input_ids, local, kwargs = model(
            input_ids=values,
            pixel_values=torch.ones(1),
            image_grid_thw=torch.ones(1, 3),
            mm_token_type_ids=torch.ones_like(values),
        )
    finally:
        state._CP_BLOCKDIAG_STATE.reset(token)

    assert input_ids is None
    assert local.flatten().tolist() == [0, 1, 2, 3]
    assert kwargs == {}
    assert calls[0][0].tolist() == values.tolist()
    assert calls[0][1]["pixel_values"].tolist() == [1.0]


@pytest.mark.unit
def test_primary_shard_does_not_modify_automodel_modules():
    from nemo_automodel.components.distributed.blockdiag_cp import state
    from nemo_automodel.components.models.qwen3_5 import model as dense_model_module

    original = dense_model_module.shard_sequence_for_cp_round_robin

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cp_mesh = _FakeCPMesh(size=2, rank=0)

        def _embed_and_splice_for_cp(self, input_ids, **_media):
            return input_ids.unsqueeze(-1)

        def forward(self, *, input_ids=None, inputs_embeds=None):
            assert dense_model_module.shard_sequence_for_cp_round_robin is original
            raise RuntimeError("forward failed")

    model = Model()
    _install_primary_shard(model, model.cp_mesh)
    token = state._CP_BLOCKDIAG_STATE.set({"model_state": object()})
    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            model(input_ids=torch.arange(8).view(1, 8))
    finally:
        state._CP_BLOCKDIAG_STATE.reset(token)

    assert dense_model_module.shard_sequence_for_cp_round_robin is original


@pytest.mark.unit
def test_cp1_uses_padded_block_diagonal_state(monkeypatch):
    from nemo_automodel.components.distributed.blockdiag_cp import kernels, state

    monkeypatch.setattr(kernels, "precompute_blockdiag_varlen_meta", lambda *args, **kwargs: "meta")
    batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21, 22]]),
        "labels": torch.tensor([[11, -100, 21, 22, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2, 2]]),
    }

    context_factory, sharded, layout = make_cp_blockdiag_batch_and_ctx(
        _FakeCPMesh(size=1),
        None,
        batch,
        shard_primary=False,
    )

    assert layout.original_seq_len == 5
    assert layout.padded_seq_len == 6
    assert sharded["labels"].tolist() == [[11, -100, 21, 22, -100, -100]]
    assert sharded["_packed_seq_ids"].tolist() == [[1, 1, 2, 2, 2, 0]]
    with context_factory():
        step_state = state._CP_BLOCKDIAG_STATE.get()
        assert step_state["doc_ids"].tolist() == [[1, 1, 2, 2, 2, 0]]
        assert step_state["packed_cu_seqlens"].tolist() == [0, 2, 5, 6]
        assert step_state["varlen_meta"] == "meta"


@pytest.mark.unit
def test_cp1_kv_gather_is_differentiable_identity(monkeypatch):
    from nemo_automodel.components.distributed.blockdiag_cp.exchange import _AllGatherSeqDiff

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("CP1 K/V gather must not launch a collective")

    monkeypatch.setattr(torch.distributed, "all_gather", unexpected_collective)
    monkeypatch.setattr(torch.distributed, "reduce_scatter", unexpected_collective)

    tensor = torch.randn(1, 2, 3, 4, requires_grad=True)
    gathered = _AllGatherSeqDiff.apply(tensor, object(), 2)
    gathered.sum().backward()

    assert torch.equal(gathered, tensor)
    assert torch.equal(tensor.grad, torch.ones_like(tensor))


@pytest.mark.unit
def test_physical_pack_count_is_model_independent(monkeypatch):
    trainer = NeMoTrainer.__new__(NeMoTrainer)
    trainer.args = Namespace(
        global_batch_size=4,
        micro_batch_size=2,
        use_dynamic_batch_size=False,
    )
    trainer.dp_size = 1
    trainer.dp_group = object()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    packs, boundaries = trainer._packed_data([_episode(10), _episode(20), _episode(30), _episode(40)])

    assert len(packs) == 2
    assert boundaries == [2]
    assert [len(pack["_episode_dp_indices"]) for pack in packs] == [2, 2]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cp_size", "expected_pack_count", "expected_pack_lengths"),
    [
        (1, 4, [4, 4, 4, 4]),
        (2, 2, [8, 8]),
    ],
)
def test_dynamic_pack_budget_is_gpu_local_after_cp(
    monkeypatch,
    cp_size,
    expected_pack_count,
    expected_pack_lengths,
):
    trainer = NeMoTrainer.__new__(NeMoTrainer)
    trainer.args = Namespace(
        global_batch_size=4,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
    )
    trainer.dp_size = 1
    trainer.dp_group = object()
    trainer.cp_size = cp_size
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    packs, boundaries = trainer._packed_data(
        [
            _episode_with_length(10, 4),
            _episode_with_length(20, 4),
            _episode_with_length(30, 4),
            _episode_with_length(40, 4),
        ]
    )

    assert len(packs) == expected_pack_count
    assert boundaries == [expected_pack_count]
    assert sorted(len(pack["tokens"]) for pack in packs) == expected_pack_lengths


@pytest.mark.unit
def test_qualification_comparison_enforces_first_layer_parity(tmp_path):
    baseline = {
        "log_probs": torch.ones(1, 2),
        "mean_loss": 1.0,
        "gradient_name": "norm.weight",
        "gradient": torch.ones(2),
        "layer_outputs": {"layer.0.linear_attn": torch.ones(1, 2, 2)},
    }
    baseline_path = tmp_path / "baseline.pt"
    torch.save(baseline, baseline_path)

    metrics, failures = _compare_with_baseline(baseline, baseline_path)
    assert not failures
    assert metrics["first_layer_relative_l2"] == 0
    assert metrics["mean_loss_abs_delta"] == 0

    candidate = dict(baseline)
    candidate["layer_outputs"] = {"layer.0.linear_attn": torch.zeros(1, 2, 2)}
    _, failures = _compare_with_baseline(candidate, baseline_path)
    assert any("first_layer_relative_l2" in failure for failure in failures)

    candidate = dict(baseline, expert_parallel_size=8)
    with pytest.raises(RuntimeError, match="different expert-parallel sizes"):
        _compare_with_baseline(candidate, baseline_path)
