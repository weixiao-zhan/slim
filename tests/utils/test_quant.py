# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for slim/utils/quant.py — the shared weight quantizer used by both
the offline converter (tools/convert_hf_to_fp8.py) and the online rollout weight sync.

The central guarantee: the online quantizer produces byte-identical output to the
offline converter, so training-time FP8 rollout matches a deployed FP8 checkpoint.
"""


import pytest
import torch

from slim.utils.quant import (
    QuantizerFP8,
    block_fp8,
    module_path_match,
    split_fused_experts,
)


def test_online_matches_offline_bit_identical():
    """QuantizerFP8.quantize == block_fp8, bit-for-bit (fp32 scales)."""
    torch.manual_seed(0)
    w = torch.randn(384, 256, dtype=torch.bfloat16)
    q_off, s_off = block_fp8(w, [128, 128])

    qz = QuantizerFP8([128, 128], ["lm_head", "linear_attn", "visual"])
    out = qz.quantize("model.language_model.layers.5.self_attn.o_proj.weight", w)

    assert [n for n, _ in out] == [
        "model.language_model.layers.5.self_attn.o_proj.weight",
        "model.language_model.layers.5.self_attn.o_proj.weight_scale_inv",
    ]
    q_on, s_on = out[0][1], out[1][1]
    assert q_on.dtype == torch.float8_e4m3fn and s_on.dtype == torch.float32
    assert torch.equal(q_off.view(torch.uint8), q_on.view(torch.uint8))
    assert torch.equal(s_off, s_on)


def test_block_fp8_shapes_and_dequant():
    w = torch.randn(256, 384, dtype=torch.bfloat16)
    q, s = block_fp8(w, [128, 128])
    assert q.shape == (256, 384) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 3) and s.dtype == torch.float32
    # dequant = q * scale (multiplicative); within e4m3 block error
    deq = (q.to(torch.float32).reshape(2, 128, 3, 128) * s.reshape(2, 1, 3, 1)).reshape(256, 384)
    rel = (deq - w.float()).abs().max() / w.float().abs().max()
    assert rel < 0.1


def test_block_fp8_grouped_experts_matches_individual_weights():
    weight = torch.randn(3, 256, 384, dtype=torch.bfloat16)
    grouped_weight, grouped_scale = block_fp8(weight, [128, 128])

    for expert_id in range(weight.shape[0]):
        expert_weight, expert_scale = block_fp8(weight[expert_id], [128, 128])
        assert torch.equal(grouped_weight[expert_id].view(torch.uint8), expert_weight.view(torch.uint8))
        assert torch.equal(grouped_scale[expert_id], expert_scale)


def test_offline_split_experts_match_online_grouped_sync():
    """Offline per-expert split (converter) == online grouped-expert sync, bytes and scale names.

    Offline checkpoints store fused MoE experts split per expert so sglang loads their scales;
    the online path quantizes the fused tensor and emits per-expert scales. Both must agree.
    """
    torch.manual_seed(0)
    qz = QuantizerFP8([128, 128], [])
    prefix = "model.language_model.layers.0.mlp.experts"
    for projection, shape in (("gate_up_proj", (3, 512, 384)), ("down_proj", (3, 384, 256))):
        weight = torch.randn(*shape, dtype=torch.bfloat16)
        (_, q_online), *online_scales = qz.quantize(f"{prefix}.{projection}", weight)

        offline = {}
        for key, expert_weight in split_fused_experts(f"{prefix}.{projection}", weight):
            offline[key], offline[key.replace(".weight", ".weight_scale_inv")] = block_fp8(expert_weight, [128, 128])
        assert [name for name, _ in online_scales] == [k for k in offline if k.endswith("weight_scale_inv")]
        for name, scale in online_scales:
            assert torch.equal(offline[name], scale)

        q_offline = [q for k, q in offline.items() if k.endswith(".weight")]
        if projection == "gate_up_proj":  # online keeps [E, gate;up, in], offline stores gate, up per expert
            q_offline = [torch.cat(q_offline[i : i + 2]) for i in range(0, len(q_offline), 2)]
        assert torch.equal(torch.stack(q_offline).view(torch.uint8), q_online.view(torch.uint8))


def test_block_fp8_ue8m0_scales_are_powers_of_two():
    """ue8m0 scales are float32 powers of two; weights stay e4m3, same shape as fp32 path."""
    w = torch.randn(256, 384, dtype=torch.bfloat16)
    q, s = block_fp8(w, [128, 128], scale_fmt="ue8m0")
    assert q.shape == (256, 384) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 3) and s.dtype == torch.float32
    # every scale is exactly 2**k: mantissa bits are zero
    bits = s.view(torch.int32)
    assert torch.equal(bits & 0x7FFFFF, torch.zeros_like(bits))
    deq = (q.to(torch.float32).reshape(2, 128, 3, 128) * s.reshape(2, 1, 3, 1)).reshape(256, 384)
    rel = (deq - w.float()).abs().max() / w.float().abs().max()
    assert rel < 0.2  # power-of-two rounding is lossier than fp32 scales


@pytest.mark.parametrize(
    "pattern,name,expected",
    [
        ("linear_attn", "model.language_model.layers.5.linear_attn.conv1d", True),
        ("visual", "model.visual.blocks.0.attn.qkv", True),
        ("lm_head", "lm_head", True),
        ("mlp.gate", "model.layers.0.mlp.gate", True),
        ("mlp.gate", "model.layers.0.mlp.gate_proj", False),  # segment boundary, not substring
        ("linear_attn", "model.layers.0.self_attn.q_proj", False),
    ],
)
def test_module_path_match_segment_semantics(pattern, name, expected):
    """Must mirror sglang's _module_path_match dotted-segment matching exactly."""
    assert module_path_match(pattern, name) is expected


def test_keep_predicate_matches_language_model_container():
    qz = QuantizerFP8([128, 128], ["model.embed_tokens"])
    weight = torch.randn(256, 128, dtype=torch.bfloat16)

    assert qz.quantize("model.language_model.embed_tokens.weight", weight) == (
        ("model.language_model.embed_tokens.weight", weight),
    )


@pytest.mark.parametrize(
    ("name", "scale_names", "scale_shape"),
    [
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            [
                f"model.language_model.layers.0.mlp.experts.{expert_id}.{projection}.weight_scale_inv"
                for expert_id in range(4)
                for projection in ("gate_proj", "up_proj")
            ],
            (4, 16),
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj",
            [
                f"model.language_model.layers.0.mlp.experts.{expert_id}.down_proj.weight_scale_inv"
                for expert_id in range(4)
            ],
            (16, 4),
        ),
    ],
)
def test_quantize_grouped_experts(name, scale_names, scale_shape):
    shape = (4, 1024, 2048) if name.endswith("gate_up_proj") else (4, 2048, 512)
    qz = QuantizerFP8([128, 128], [])
    output = qz.quantize(name, torch.randn(*shape, dtype=torch.bfloat16))

    assert [tensor_name for tensor_name, _ in output] == [name, *scale_names]
    assert output[0][1].dtype == torch.float8_e4m3fn
    assert all(tensor.shape == scale_shape for _, tensor in output[1:])


