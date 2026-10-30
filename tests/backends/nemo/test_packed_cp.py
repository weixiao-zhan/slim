# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from nemo_automodel.components.distributed.blockdiag_cp import make_cp_blockdiag_batch_and_ctx

from slim.backends.nemo.base import NeMoTrainer
from slim.backends.nemo.models.qwen3_5 import _global_media_presence, build_packed_position_ids
from slim.utils.trajectory_batch import TrajectoryBatch
from slim.utils.types import Trajectory

NUM_GPUS = 0


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


def _trajectory(start, length=3):
    trajectory = Trajectory(
        token_ids=list(range(start, start + length)),
        loss_mask=[1] * (length - 1),
        reward=1.0,
        loss_weight=1.0,
    )
    trajectory.finalize_source_token_alignment()
    trajectory.set_train_targets(
        [0.5] * (length - 1) + [0.0],
        value_targets=[1.0] * (length - 1) + [0.0],
    )
    return trajectory


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
def test_dense_block_uses_native_packed_attention(monkeypatch):
    from nemo_automodel.components.distributed import blockdiag_cp
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5DenseBlock, _dense_moe_config
    from nemo_automodel.components.models.qwen3_5_moe.model import _Qwen3_5MoeAttention
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    block = Qwen3_5DenseBlock(0, config, _dense_moe_config(config, torch.float32), backend)
    attention = block.self_attn
    assert isinstance(attention, _Qwen3_5MoeAttention)

    query = torch.randn(1, 2, 5, 16)
    key, value = torch.randn(2, 1, 2, 5, 16)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.testing.assert_close(attention.attn_func(query, key, value, is_causal=True), expected)

    monkeypatch.setattr(blockdiag_cp, "current_blockdiag_cp_state", lambda: object())
    monkeypatch.setattr(blockdiag_cp, "cp_blockdiag_sdpa", lambda *args, **kwargs: "packed")
    assert attention.attn_func(query, key, value) == "packed"


@pytest.mark.unit
@pytest.mark.parametrize(("cp_size", "packed"), [(None, False), (1, False), (2, False), (1, True), (2, True)])
def test_native_gdn_selects_cp_from_mesh_and_packed_state(monkeypatch, cp_size, packed):
    from nemo_automodel.components.distributed import blockdiag_cp
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

    model = CPAwareGatedDeltaNet.__new__(CPAwareGatedDeltaNet)
    nn.Module.__init__(model)
    model._cp_mesh = None if cp_size is None else _FakeCPMesh(size=cp_size)
    active_state = object() if packed else None
    monkeypatch.setattr(blockdiag_cp, "current_blockdiag_cp_state", lambda: active_state)
    monkeypatch.setattr(model, "_forward_no_cp", lambda hidden, **kwargs: ("base", kwargs))
    monkeypatch.setattr(model, "_forward_with_cp", lambda hidden, **kwargs: ("cp", kwargs))
    positions = torch.arange(5).view(1, 5)
    path, kwargs = model(torch.randn(1, 5, 8), position_ids=positions)
    if packed or cp_size == 2:
        assert path == "cp"
        assert kwargs["position_ids"] is positions
        assert kwargs["blockdiag_state"] is active_state
    else:
        assert path == "base"
    if packed:
        with pytest.raises(ValueError, match="does not support a Gated DeltaNet cache"):
            model(torch.randn(1, 5, 8), cache_params=object())


class _DecoderInputCaptured(Exception):
    pass


def _native_primary_model(model_kind, cp_mesh):
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration

    class DecoderInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.embed_tokens = nn.Embedding(64, 3)

        def forward(self, **kwargs):
            self.inputs = kwargs
            raise _DecoderInputCaptured

    cls = Qwen3_5ForConditionalGeneration if model_kind == "dense" else Qwen3_5MoeForConditionalGeneration
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(output_hidden_states=False),
        image_token_id=0,
        video_token_id=61,
        vision_start_token_id=63,
    )
    model.cp_mesh = cp_mesh
    model.model = DecoderInput()
    model.lm_head = nn.Identity()
    model.mtp = None
    model.vision_scale = nn.Parameter(torch.tensor(0.5))
    model.embed_calls = []

    def embed_and_splice(self, input_ids, **media):
        self.embed_calls.append((input_ids, media))
        return self.model.language_model.embed_tokens(input_ids) + self.vision_scale * media["pixel_values"].mean()

    model._embed_and_splice_for_cp = MethodType(embed_and_splice, model)
    return model


@pytest.mark.unit
@pytest.mark.parametrize("model_kind", ["dense", "moe"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("length", [5, 8, 9])
def test_native_primary_shards_preserve_contiguous_values_and_gradients(model_kind, cp_size, length):
    from nemo_automodel.components.distributed.blockdiag_cp import state

    values = torch.arange(length).view(1, length)
    for rank in range(cp_size):
        model = _native_primary_model(model_kind, _FakeCPMesh(size=cp_size, rank=rank))
        pixels = torch.tensor([2.0])
        token = state._CP_BLOCKDIAG_STATE.set({"model_state": object()})
        try:
            with pytest.raises(_DecoderInputCaptured):
                model(
                    input_ids=values,
                    pixel_values=pixels,
                    image_grid_thw=torch.ones(1, 3),
                    mm_token_type_ids=torch.ones_like(values),
                )
        finally:
            state._CP_BLOCKDIAG_STATE.reset(token)

        inputs = model.model.inputs
        assert inputs["input_ids"] is None
        for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            assert inputs.get(name) is None
        assert len(model.embed_calls) == 1
        assert model.embed_calls[0][0] is values
        assert model.embed_calls[0][1]["pixel_values"] is pixels
        assert not model._forward_pre_hooks

        embedding = model.model.language_model.embed_tokens.weight
        ref_weight = embedding.detach().clone().requires_grad_()
        ref_vision = model.vision_scale.detach().clone().requires_grad_()
        full = torch.nn.functional.embedding(values, ref_weight) + ref_vision * pixels.mean()
        padded = torch.nn.functional.pad(full, (0, 0, 0, (-length) % (2 * cp_size)))
        expected = padded.chunk(cp_size, dim=1)[rank]
        local = inputs["inputs_embeds"]
        torch.testing.assert_close(local, expected, rtol=0, atol=0)
        local.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(embedding.grad, ref_weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(model.vision_scale.grad, ref_vision.grad, rtol=0, atol=0)


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
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4,
    )
    trainer.dp_size = 1
    trainer.dp_group = object()
    trainer.cp_size = cp_size
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    batch = TrajectoryBatch(
        trajectories=[_trajectory(start, length=4) for start in (10, 20, 30, 40)],
    )
    packs, boundaries = trainer._packed_data(batch)

    assert len(packs) == expected_pack_count
    assert boundaries == [expected_pack_count]
    assert sorted(len(pack["tokens"]) for pack in packs) == expected_pack_lengths
