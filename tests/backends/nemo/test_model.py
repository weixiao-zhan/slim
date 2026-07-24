# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from slim.backends.nemo.model import (
    _model_distributed_setup,
    _use_demand_fsdp_unsharding,
    build_policy_model,
    register_qwen3_5_moe_parallel_strategy,
)


NUM_GPUS = 0


@pytest.mark.unit
def test_policy_model_uses_demand_fsdp_unsharding(monkeypatch):
    import torch.nn as nn

    calls = []

    class FakeFSDPModule(nn.Module):
        def set_modules_to_backward_prefetch(self, modules):
            calls.append(modules)

    monkeypatch.setattr("torch.distributed.fsdp.FSDPModule", FakeFSDPModule)
    first = FakeFSDPModule()
    second = FakeFSDPModule()
    model = nn.Sequential(first, second)

    assert _use_demand_fsdp_unsharding(model) == 2
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
    monkeypatch.setattr("slim.backends.nemo.model._use_demand_fsdp_unsharding", lambda model: 0)
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
    assert captured["kwargs"]["num_nextn_predict_layers"] == 0
    assert captured["kwargs"]["text_config"] == {"router_aux_loss_coef": 0.0}
    assert captured["kwargs"]["moe_overrides"] == {"enable_routing_replay": True}
    assert captured["kwargs"]["freeze_config"] == {
        "freeze_vision_tower": True,
        "freeze_audio_tower": True,
        "freeze_language_model": False,
    }


@pytest.mark.unit
def test_moe_model_construction_reserves_cp_for_packed_runtime():
    from nemo_automodel.components.distributed.config import DistributedSetup

    class MeshContext:
        def parallelize_axis_kwargs(self):
            return {
                "dp_axis_names": ("dp_shard_cp",),
                "cp_axis_name": "cp",
                "tp_axis_name": None,
                "ep_axis_name": "ep",
                "ep_shard_axis_names": None,
            }

    mesh_context = MeshContext()
    setup = DistributedSetup(mesh_context=mesh_context)

    model_setup = _model_distributed_setup(setup)

    assert model_setup is not setup
    assert model_setup.mesh_context.parallelize_axis_kwargs()["cp_axis_name"] is None
    assert model_setup.mesh_context.parallelize_axis_kwargs()["ep_axis_name"] == "ep"
    assert model_setup.mesh_context.parallelize_axis_kwargs()["dp_axis_names"] == ("dp_shard_cp",)
    assert setup.mesh_context.parallelize_axis_kwargs()["cp_axis_name"] == "cp"
