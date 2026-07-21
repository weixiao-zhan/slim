# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for slim/utils/quant.py — the shared weight quantizer used by both
the offline converter (tools/convert_hf_to_fp8.py) and the online rollout weight sync.

The central guarantee: the online quantizer produces byte-identical output to the
offline converter, so training-time FP8 rollout matches a deployed FP8 checkpoint.
"""

import json
import os
import tempfile

import pytest
import torch

from slim.utils.quant import (
    Quantizer,
    QuantizerFP8,
    block_fp8,
    make_keep_predicate,
    module_path_match,
    parse_quant_recipe,
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


def test_block_fp8_handles_non_128_multiple_via_padding():
    w = torch.randn(130, 100, dtype=torch.bfloat16)
    q, s = block_fp8(w, [128, 128])
    assert q.shape == (130, 100) and s.shape == (2, 1)


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


def test_block_fp8_ue8m0_requires_128_blocks():
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        block_fp8(w, [64, 128], scale_fmt="ue8m0")


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


def test_keep_predicate_generalizes_across_layers():
    is_kept = make_keep_predicate(["lm_head", "embed_tokens", "linear_attn", "visual"])
    for name in [
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.23.linear_attn.conv1d",
        "model.visual.blocks.7.mlp.linear_fc1",
        "model.language_model.embed_tokens",
    ]:
        assert is_kept(name)
    for name in [
        "model.language_model.layers.5.self_attn.q_proj",
        "model.language_model.layers.5.mlp.gate_proj",
    ]:
        assert not is_kept(name)


def test_quantize_keeps_listed_and_non_2d():
    qz = QuantizerFP8([128, 128], ["lm_head", "linear_attn"])
    # kept module -> single passthrough tensor, dtype unchanged
    out = qz.quantize("lm_head.weight", torch.randn(256, 128, dtype=torch.bfloat16))
    assert len(out) == 1 and out[0][1].dtype == torch.bfloat16
    out = qz.quantize("model.layers.0.linear_attn.conv1d.weight", torch.randn(128, 4, dtype=torch.bfloat16))
    assert len(out) == 1
    # 1D weight (norm) -> passthrough
    out = qz.quantize("model.norm.weight", torch.randn(128, dtype=torch.bfloat16))
    assert len(out) == 1
    # quantizable -> weight + scale
    out = qz.quantize("model.layers.0.mlp.up_proj.weight", torch.randn(256, 128, dtype=torch.bfloat16))
    assert len(out) == 2 and out[0][1].dtype == torch.float8_e4m3fn


def test_quantize_divisibility_assert():
    qz = QuantizerFP8([128, 128], [])
    with pytest.raises(AssertionError):
        qz.quantize("model.layers.0.mlp.down_proj.weight", torch.randn(130, 128, dtype=torch.bfloat16))


def test_parse_quant_recipe_fp8_block():
    recipe = parse_quant_recipe(
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        }
    )
    assert recipe["block_size"] == [128, 128]
    assert recipe["keep_patterns"] == ["lm_head"]
    assert recipe["scale_fmt"] is None


def _write_config(d, quantization_config):
    cfg = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
    if quantization_config is not None:
        cfg["quantization_config"] = quantization_config
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))


def test_maybe_from_checkpoint_non_fp8_returns_none():
    with tempfile.TemporaryDirectory() as d:
        _write_config(d, None)
        assert Quantizer.maybe_from_checkpoint(d) is None


def test_maybe_from_checkpoint_rejects_non_block_fp8():
    # Per-tensor FP8 (no weight_block_size) must fail-fast, not silently push BF16:
    # sglang transposes + requants per-tensor/channel weights at load time, which the
    # sync path's load_weights does not re-run, so it would desync.
    with tempfile.TemporaryDirectory() as d:
        _write_config(d, {"quant_method": "fp8"})
        with pytest.raises(AssertionError):
            Quantizer.maybe_from_checkpoint(d)


def test_maybe_from_checkpoint_rejects_static_activation():
    with tempfile.TemporaryDirectory() as d:
        _write_config(
            d,
            {"quant_method": "fp8", "weight_block_size": [128, 128], "activation_scheme": "static"},
        )
        with pytest.raises(AssertionError):
            Quantizer.maybe_from_checkpoint(d)


def test_maybe_from_checkpoint_accepts_ue8m0_block():
    # UE8M0 is no longer rejected at config read; the Blackwell caveat is documented in
    # QuantizerFP8.from_quant_config instead (sync stays correct on Ada/Hopper).
    with tempfile.TemporaryDirectory() as d:
        _write_config(d, {"quant_method": "fp8", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"})
        qz = Quantizer.maybe_from_checkpoint(d)
        assert isinstance(qz, QuantizerFP8) and qz.block_size == [128, 128]
        assert qz.scale_fmt == "ue8m0"


def test_maybe_from_checkpoint_builds_for_block_fp8():
    with tempfile.TemporaryDirectory() as d:
        _write_config(
            d,
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head", "linear_attn"],
            },
        )
        qz = Quantizer.maybe_from_checkpoint(d)
        assert isinstance(qz, QuantizerFP8) and qz.block_size == [128, 128]
        assert qz.is_kept("model.layers.0.linear_attn.conv1d")
        assert not qz.is_kept("model.layers.0.mlp.up_proj")
