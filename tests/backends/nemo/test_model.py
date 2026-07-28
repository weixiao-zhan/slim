# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.model import (
    _disable_fsdp_backward_prefetch,
    build_policy_model,
    final_hidden_state,
    register_qwen3_5_moe_parallel_strategy,
)


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("container", [lambda value: value, lambda value: (value,)])
def test_final_hidden_state_accepts_dense_and_moe_outputs(container):
    hidden = torch.randn(1, 4, 8)
    output = SimpleNamespace(hidden_states=container(hidden))

    assert final_hidden_state(output) is hidden


@pytest.mark.unit
def test_policy_model_disables_fsdp_backward_prefetch(monkeypatch):
    import torch.nn as nn

    calls = []

    class FakeFSDPModule(nn.Module):
        def set_modules_to_backward_prefetch(self, modules):
            calls.append(modules)

    monkeypatch.setattr("torch.distributed.fsdp.FSDPModule", FakeFSDPModule)
    first = FakeFSDPModule()
    second = FakeFSDPModule()
    model = nn.Sequential(first, second)

    assert _disable_fsdp_backward_prefetch(model) == 2
    assert calls == [[first], [second]]


@pytest.mark.unit
def test_qwen3_5_moe_uses_mixed_dtype_parallel_strategy(monkeypatch):
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        Qwen3_5ParallelizationStrategy,
    )

    registry = dict(PARALLELIZATION_STRATEGIES)
    registry.pop("Qwen3_5MoeForConditionalGeneration", None)
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.PARALLELIZATION_STRATEGIES",
        registry,
    )

    register_qwen3_5_moe_parallel_strategy()

    assert isinstance(registry["Qwen3_5MoeForConditionalGeneration"], Qwen3_5ParallelizationStrategy)


@pytest.mark.unit
def test_policy_model_disables_mtp_training_layers(monkeypatch):
    captured = {}
    model = SimpleNamespace()

    def fake_from_pretrained(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured["kwargs"] = kwargs
        return model

    def fake_backend_config(_args):
        return object()

    from nemo_automodel import NeMoAutoModelForImageTextToText

    monkeypatch.setattr(NeMoAutoModelForImageTextToText, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("slim.backends.nemo.model.build_backend_config", fake_backend_config)
    monkeypatch.setattr("slim.backends.nemo.model._disable_fsdp_backward_prefetch", lambda model: 0)
    monkeypatch.setattr("slim.backends.nemo.packed_cp.install_qwen3_5_packed_cp", lambda *args: None)

    args = SimpleNamespace(
        freeze_vision_tower=True,
        freeze_audio_tower=True,
        freeze_language_model=False,
    )
    distributed_setup = SimpleNamespace(
        mesh_context=SimpleNamespace(
            device_mesh={"cp": object()},
            parallelize_axis_kwargs=lambda: {
                "cp_axis_name": "cp",
                "ep_axis_name": None,
            },
        ),
    )
    result = build_policy_model(
        args,
        "/models/qwen3.5",
        distributed_setup=distributed_setup,
        routing_replay=True,
    )

    assert result is model
    assert captured["checkpoint"] == "/models/qwen3.5"
    assert "dtype" not in captured["kwargs"]
    assert "torch_dtype" not in captured["kwargs"]
    assert captured["kwargs"]["attn_implementation"] == "sdpa"
    assert captured["kwargs"]["distributed_setup"] is distributed_setup
    assert captured["kwargs"]["num_nextn_predict_layers"] == 0
    assert captured["kwargs"]["text_config"] == {"router_aux_loss_coef": 0.0}
    assert captured["kwargs"]["moe_overrides"] == {"enable_routing_replay": True}
    assert captured["kwargs"]["freeze_config"] == {
        "freeze_vision_tower": True,
        "freeze_audio_tower": True,
        "freeze_language_model": False,
    }
